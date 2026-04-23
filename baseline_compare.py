"""
baseline_compare.py — Always-YOLO vs Always-RT-DETR vs AdaptAlloc-RL
=====================================================================
Fixes vs original:
  1. Defaults to fine-tuned weights (yolo_cracks.pt / rtdetr_folder).
  2. Pixel-level Shannon entropy fed into RLScheduler.encode_state().
  3. Reward uses real IoU F1 (has_gt=True) when labels are present.
  4. Summary table now shows all three modes clearly.
  5. Cleaner chart layout with entropy scatter panel added.

Usage:
  python baseline_compare.py --source /path/to/images --labels /path/to/labels
  python baseline_compare.py --source video.mp4
  python baseline_compare.py --webcam
  python baseline_compare.py --roboflow --rf-key YOUR_KEY
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from policy import (
    RLScheduler, ModelAwareComplexityScorer,
    UncertaintyEstimate, HardwareCost,
    compute_pixel_entropy,
)
from telemetry import GPUProfiler, FrameMetrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default to fine-tuned models; fall back gracefully
_DEFAULT_YOLO   = "data/models/yolo_cracks.pt"
_DEFAULT_RTDETR = "data/models/rtdetr_folder"
if not Path(_DEFAULT_YOLO).exists():
    _DEFAULT_YOLO = "yolov8n.pt"
if not Path(_DEFAULT_RTDETR).exists():
    _DEFAULT_RTDETR = ""


# ══════════════════════════════════════════════════════════════════════════════
# Model wrappers (lightweight, standalone)
# ══════════════════════════════════════════════════════════════════════════════

class YOLOWrapper:
    def __init__(self, weights=_DEFAULT_YOLO, conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf  = conf
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self.model(dummy, verbose=False, device=DEVICE)

    def infer(self, frame, profiler=None):
        if profiler:
            with profiler.measure("yolo") as ref:
                results = self.model(frame, verbose=False,
                                     conf=self.conf, device=DEVICE)
            m = ref.value
        else:
            t0 = time.perf_counter()
            results = self.model(frame, verbose=False,
                                 conf=self.conf, device=DEVICE)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            m = FrameMetrics("yolo", gpu_ms=ms, wall_ms=ms)
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            dets = {"boxes": np.empty((0, 4)), "scores": np.empty(0), "n": 0}
        else:
            dets = {
                "boxes":  res.boxes.xyxy.cpu().numpy(),
                "scores": res.boxes.conf.cpu().numpy(),
                "n":      len(res.boxes),
            }
        m.n_detections   = dets["n"]
        m.avg_confidence = float(dets["scores"].mean()) if dets["n"] > 0 else 0.0
        return dets, m

    def infer_with_uncertainty(self, frame, profiler=None):
        dets, m = self.infer(frame, profiler)
        n, c = dets["n"], dets["scores"]
        if n == 0:
            unc = UncertaintyEstimate(0, 0, 1.0, 0, 1.0, 1.0, 0, 0, 0)
        else:
            mu  = float(c.mean())
            var = float(c.var()) if len(c) > 1 else 0.0
            hist, _ = np.histogram(c, bins=10, range=(0, 1))
            p   = hist / (hist.sum() + 1e-9)
            ent = float(-np.sum(p * np.log2(p + 1e-9))) / np.log2(10)
            unc = UncertaintyEstimate(mu, var, ent, n,
                                     float((c < 0.5).mean()), ent, var, var, mu)
        return dets, unc, m


class RTDETRWrapper:
    def __init__(self, model_path=_DEFAULT_RTDETR, conf=0.25):
        if not model_path:
            from ultralytics import RTDETR
            self._mode = "ul"
            self.model = RTDETR("rtdetr-l.pt")
        else:
            self._mode = "hf"
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            import zipfile, tempfile
            if model_path.endswith(".zip"):
                tmp = tempfile.mkdtemp()
                with zipfile.ZipFile(model_path, "r") as z:
                    z.extractall(tmp)
                for dp, _, f in os.walk(tmp):
                    if "config.json" in f:
                        model_path = dp
                        break
            self.proc  = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForObjectDetection.from_pretrained(model_path)
            self.model.to(DEVICE).eval()
        self.conf = conf
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self.infer(dummy)

    def infer(self, frame, profiler=None):
        if profiler:
            with profiler.measure("rtdetr") as ref:
                dets = self._fwd(frame)
            m = ref.value
        else:
            t0 = time.perf_counter()
            dets = self._fwd(frame)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            m  = FrameMetrics("rtdetr", gpu_ms=ms, wall_ms=ms)
        m.n_detections   = dets["n"]
        m.avg_confidence = float(dets["scores"].mean()) if dets["n"] > 0 else 0.0
        return dets, m

    def _fwd(self, frame):
        if self._mode == "ul":
            res = self.model(frame, verbose=False,
                             conf=self.conf, device=DEVICE)[0]
            if res.boxes is None or len(res.boxes) == 0:
                return {"boxes": np.empty((0, 4)), "scores": np.empty(0), "n": 0}
            return {"boxes":  res.boxes.xyxy.cpu().numpy(),
                    "scores": res.boxes.conf.cpu().numpy(),
                    "n":      len(res.boxes)}
        from PIL import Image as PILImage
        pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = frame.shape[:2]
        inp = self.proc(pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model(**inp)
        res = self.proc.post_process_object_detection(
            out, target_sizes=[(h, w)], threshold=self.conf)[0]
        return {
            "boxes":  res["boxes"].cpu().numpy(),
            "scores": res["scores"].cpu().numpy(),
            "n":      len(res["boxes"]),
        }


# ══════════════════════════════════════════════════════════════════════════════
# IoU / F1 helpers
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih   = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter    = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1e-9)


def compute_f1(dets, gt_boxes, iou_thr=0.30):
    n_pred, n_gt = dets["n"], len(gt_boxes)
    if n_pred == 0 and n_gt == 0:
        return 1.0, True      # true negative
    if n_pred == 0 or n_gt == 0:
        return 0.0, False
    matched, tp, fp = set(), 0, 0
    for idx in np.argsort(dets["scores"])[::-1]:
        pb = dets["boxes"][idx]
        best_iou, best_gi = 0.0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched:
                continue
            iou = _iou(pb, gb)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thr:
            tp += 1
            matched.add(best_gi)
        else:
            fp += 1
    fn = n_gt - len(matched)
    pr = tp / (tp + fp + 1e-9)
    rc = tp / (tp + fn + 1e-9)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    return float(f1), False


# ══════════════════════════════════════════════════════════════════════════════
# Ground-truth loader (YOLO format)
# ══════════════════════════════════════════════════════════════════════════════

def load_gt_boxes(label_path: str, img_w: int, img_h: int) -> np.ndarray:
    boxes = []
    if not label_path or not os.path.exists(label_path):
        return np.empty((0, 4))
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return (np.array(boxes, dtype=np.float32)
            if boxes else np.empty((0, 4)))


# ══════════════════════════════════════════════════════════════════════════════
# Roboflow helper
# ══════════════════════════════════════════════════════════════════════════════

def download_roboflow_dataset(api_key: str) -> str:
    from roboflow import Roboflow
    rf      = Roboflow(api_key=api_key)
    project = rf.workspace("dhathus-workspace").project("crack_detection-ibpez")
    version = project.version(1)
    dataset = version.download("yolov8")
    return str(Path(dataset.location) / "valid" / "images")


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluator
# ══════════════════════════════════════════════════════════════════════════════

class BaselineEvaluator:
    def __init__(self, yolo_weights=_DEFAULT_YOLO, rtdetr_path=_DEFAULT_RTDETR,
                 conf=0.25, iou_thr=0.30, lam=0.40,
                 policy_path="checkpoints/rl_policy.pt",
                 output_dir="outputs"):
        self.iou_thr    = iou_thr
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print("=" * 65)
        print("  AdaptAlloc Baseline Comparator")
        print("=" * 65)
        print(f"  YOLO:   {yolo_weights}")
        print(f"  RTDETR: {rtdetr_path or 'rtdetr-l.pt (auto)'}")
        print()

        self.profiler = GPUProfiler()
        print("[1/2] Loading YOLOv8 …")
        self.yolo = YOLOWrapper(yolo_weights, conf)
        print("[2/2] Loading RT-DETR …")
        self.rtdetr = RTDETRWrapper(rtdetr_path, conf)
        print("Done.\n")

        self.scorer = ModelAwareComplexityScorer()
        self.rl     = RLScheduler(lam=lam, epsilon=0.05)  # eval: low epsilon
        self.rl.load(policy_path)

    # ── single frame ─────────────────────────────────────────────────────────

    def eval_frame(self, frame: np.ndarray, gt_boxes: np.ndarray) -> dict:
        has_gt    = len(gt_boxes) > 0
        pixel_ent = compute_pixel_entropy(frame)

        # Always-YOLO
        dy, my   = self.yolo.infer(frame, self.profiler)
        f1_y, _  = compute_f1(dy, gt_boxes, self.iou_thr)

        # Always-RT-DETR
        dr, mr   = self.rtdetr.infer(frame, self.profiler)
        f1_r, _  = compute_f1(dr, gt_boxes, self.iou_thr)

        # AdaptAlloc-RL
        dy2, unc, my2 = self.yolo.infer_with_uncertainty(frame, self.profiler)
        max_c  = float(dy2["scores"].max()) if dy2["n"] > 0 else 0.0
        cplx   = self.scorer.score(unc, dy2["n"], max_c, pixel_ent)
        state  = self.rl.encode_state(cplx, unc, pixel_ent)
        action, _, _ = self.rl.select_action(state, training=False)
        run_rt = (action == 1)

        if run_rt:
            d_adapt, m_adapt = dr, mr
            active = "rtdetr"
        else:
            d_adapt, m_adapt = dy2, my2
            active = "yolo"

        f1_adapt, _ = compute_f1(d_adapt, gt_boxes, self.iou_thr)

        # Update policy with real F1 reward (so it keeps improving during eval)
        yolo_f1_val = f1_y if has_gt else my2.avg_confidence
        cost_obj = HardwareCost(
            m_adapt.gpu_ms, m_adapt.wall_ms,
            m_adapt.vram_delta_mb, 0.0, m_adapt.energy_mj, active,
            composite=m_adapt.gpu_ms + m_adapt.energy_mj,
        )
        reward = self.rl.compute_reward(
            f1_adapt if has_gt else m_adapt.avg_confidence,
            yolo_f1_val, cost_obj, run_rt,
            d_adapt["n"] > 0, has_gt=has_gt,
        )

        return {
            "yolo_ms":      my.gpu_ms,  "yolo_energy":  my.energy_mj,
            "yolo_f1":      f1_y,       "yolo_n":       dy["n"],
            "rtdetr_ms":    mr.gpu_ms,  "rtdetr_energy": mr.energy_mj,
            "rtdetr_f1":    f1_r,       "rtdetr_n":     dr["n"],
            "adapt_ms":     m_adapt.gpu_ms,
            "adapt_energy": m_adapt.energy_mj,
            "adapt_f1":     f1_adapt,
            "adapt_active": active,
            "adapt_ran_rt": run_rt,
            "has_gt":       has_gt,
            "pixel_entropy": pixel_ent,
            "reward":       reward,
        }

    # ── multi-frame run ───────────────────────────────────────────────────────

    def run(self, frames: List[np.ndarray],
            gt_list: Optional[List[np.ndarray]] = None) -> pd.DataFrame:
        if gt_list is None:
            gt_list = [np.empty((0, 4))] * len(frames)

        rows = []
        for i, (fr, gt) in enumerate(zip(frames, gt_list)):
            row = self.eval_frame(fr, gt)
            rows.append(row)
            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  [{i+1:3d}/{len(frames)}]  "
                    f"YOLO={row['yolo_f1']:.3f}  "
                    f"RTDETR={row['rtdetr_f1']:.3f}  "
                    f"Adapt={row['adapt_f1']:.3f}  "
                    f"active={row['adapt_active']}  "
                    f"entropy={row['pixel_entropy']:.3f}"
                )
        return pd.DataFrame(rows)

    # ── summary ───────────────────────────────────────────────────────────────

    def summarise(self, df: pd.DataFrame) -> pd.DataFrame:
        rt_pct = df["adapt_ran_rt"].mean() * 100
        table = {
            "Mode":       ["Always-YOLO", "Always-RT-DETR", "AdaptAlloc-RL"],
            "F1":         [df["yolo_f1"].mean(),
                           df["rtdetr_f1"].mean(),
                           df["adapt_f1"].mean()],
            "GPU-ms":     [df["yolo_ms"].mean(),
                           df["rtdetr_ms"].mean(),
                           df["adapt_ms"].mean()],
            "Energy-mJ":  [df["yolo_energy"].mean(),
                           df["rtdetr_energy"].mean(),
                           df["adapt_energy"].mean()],
            "RT-DETR-%":  [0.0, 100.0, rt_pct],
        }

        w = 78
        print("\n" + "=" * w)
        print(f"  {'Mode':<22} {'F1':>6} {'GPU-ms':>8} {'Energy mJ':>10} {'RT%':>6}")
        print("-" * w)
        for i in range(3):
            print(f"  {table['Mode'][i]:<22} "
                  f"{table['F1'][i]:>6.3f} "
                  f"{table['GPU-ms'][i]:>8.1f} "
                  f"{table['Energy-mJ'][i]:>10.2f} "
                  f"{table['RT-DETR-%'][i]:>5.1f}%")
        print("=" * w)

        adapt_f1   = df["adapt_f1"].mean()
        rtdetr_f1  = df["rtdetr_f1"].mean()
        yolo_f1    = df["yolo_f1"].mean()
        gpu_save   = (df["rtdetr_ms"].mean() - df["adapt_ms"].mean()) \
                     / (df["rtdetr_ms"].mean() + 1e-9) * 100
        cost_save  = (df["rtdetr_energy"].mean() - df["adapt_energy"].mean()) \
                     / (df["rtdetr_energy"].mean() + 1e-9) * 100

        print(f"\n  AdaptAlloc vs Always-RT-DETR:")
        print(f"    ΔF1 (vs RTDETR) = {adapt_f1 - rtdetr_f1:+.4f}")
        print(f"    ΔF1 (vs YOLO)   = {adapt_f1 - yolo_f1:+.4f}")
        print(f"    GPU saving      = {gpu_save:.1f}%")
        print(f"    Energy saving   = {cost_save:.1f}%")
        print(f"    RT-DETR ran on  = {rt_pct:.1f}% of frames")
        print("=" * w)

        return pd.DataFrame(table)

    # ── charts ────────────────────────────────────────────────────────────────

    def plot(self, df: pd.DataFrame, summary: pd.DataFrame):
        fig = plt.figure(figsize=(22, 14), facecolor="#0d0f14")
        fig.suptitle("AdaptAlloc — Baseline Comparison Report",
                     fontsize=16, color="#e8f4fd",
                     fontfamily="monospace", y=0.98)
        gs = gridspec.GridSpec(2, 4, figure=fig,
                               hspace=0.50, wspace=0.35,
                               left=0.05, right=0.97,
                               top=0.92, bottom=0.07)

        COLS  = ["#4ade80", "#60a5fa", "#f97316"]
        MODES = ["Always-YOLO", "Always-RT-DETR", "AdaptAlloc-RL"]
        BG    = "#1a1f2e"

        def _ax(pos, title):
            ax = fig.add_subplot(pos)
            ax.set_facecolor(BG)
            ax.spines[:].set_color("#2d3449")
            ax.tick_params(colors="#7a8aaa", labelsize=9)
            ax.set_title(title, color="#c5cde0", fontsize=10, pad=8)
            return ax

        # Cumulative energy
        ax1 = _ax(gs[0, 0], "Cumulative Energy (mJ)")
        vals = [df["yolo_energy"].sum(),
                df["rtdetr_energy"].sum(),
                df["adapt_energy"].sum()]
        bars = ax1.bar(MODES, vals, color=COLS, width=0.5, zorder=3)
        ax1.bar_label(bars, fmt="%.1f", color="#e8f4fd", fontsize=9, padding=3)
        ax1.set_ylabel("mJ", color="#7a8aaa", fontsize=9)
        ax1.tick_params(axis="x", rotation=12)
        ax1.grid(axis="y", color="#2d3449", linestyle="--", alpha=0.7, zorder=0)

        # Avg GPU latency
        ax2 = _ax(gs[0, 1], "Avg GPU Latency (ms)")
        vals = [df["yolo_ms"].mean(),
                df["rtdetr_ms"].mean(),
                df["adapt_ms"].mean()]
        bars = ax2.bar(MODES, vals, color=COLS, width=0.5, zorder=3)
        ax2.bar_label(bars, fmt="%.1f", color="#e8f4fd", fontsize=9, padding=3)
        ax2.set_ylabel("ms", color="#7a8aaa", fontsize=9)
        ax2.tick_params(axis="x", rotation=12)
        ax2.grid(axis="y", color="#2d3449", linestyle="--", alpha=0.7, zorder=0)

        # Mean F1
        ax3 = _ax(gs[0, 2], "Mean F1 Score")
        vals = [df["yolo_f1"].mean(),
                df["rtdetr_f1"].mean(),
                df["adapt_f1"].mean()]
        bars = ax3.bar(MODES, vals, color=COLS, width=0.5, zorder=3)
        ax3.bar_label(bars, fmt="%.3f", color="#e8f4fd", fontsize=9, padding=3)
        ax3.set_ylim(0, 1.1)
        ax3.set_ylabel("F1", color="#7a8aaa", fontsize=9)
        ax3.tick_params(axis="x", rotation=12)
        ax3.grid(axis="y", color="#2d3449", linestyle="--", alpha=0.7, zorder=0)

        # NEW: Entropy vs F1 scatter
        ax_s = _ax(gs[0, 3], "Scene Entropy vs F1 (AdaptAlloc)")
        yolo_pts  = df[~df["adapt_ran_rt"]]
        rt_pts    = df[df["adapt_ran_rt"]]
        ax_s.scatter(yolo_pts["pixel_entropy"], yolo_pts["adapt_f1"],
                     c="#4ade80", s=18, alpha=0.7, label="YOLO chosen")
        ax_s.scatter(rt_pts["pixel_entropy"],   rt_pts["adapt_f1"],
                     c="#60a5fa", s=18, alpha=0.7, label="RT-DETR chosen")
        ax_s.set_xlabel("Pixel Shannon Entropy", color="#7a8aaa", fontsize=9)
        ax_s.set_ylabel("F1", color="#7a8aaa", fontsize=9)
        ax_s.legend(fontsize=8, facecolor=BG,
                    labelcolor="#c5cde0", edgecolor="#2d3449")
        ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1.05)
        ax_s.grid(color="#2d3449", linestyle="--", alpha=0.5)

        # Per-frame F1 rolling
        ax4 = _ax(gs[1, 0], "Per-Frame F1 (rolling mean w=10)")
        w = 10
        for col, key, lbl in zip(COLS,
                                  ["yolo_f1", "rtdetr_f1", "adapt_f1"],
                                  MODES):
            ax4.plot(df[key].rolling(w, min_periods=1).mean().values,
                     color=col, linewidth=1.5, label=lbl)
        ax4.set_ylim(0, 1.1)
        ax4.set_xlabel("Frame", color="#7a8aaa", fontsize=9)
        ax4.set_ylabel("F1", color="#7a8aaa", fontsize=9)
        ax4.legend(fontsize=8, facecolor=BG,
                   labelcolor="#c5cde0", edgecolor="#2d3449")
        ax4.grid(color="#2d3449", linestyle="--", alpha=0.5)

        # Per-frame energy
        ax5 = _ax(gs[1, 1], "Per-Frame Energy (mJ)")
        for col, key, lbl in zip(COLS,
                                  ["yolo_energy", "rtdetr_energy", "adapt_energy"],
                                  MODES):
            ax5.plot(df[key].values, color=col, linewidth=1.2,
                     alpha=0.85, label=lbl)
        ax5.set_xlabel("Frame", color="#7a8aaa", fontsize=9)
        ax5.set_ylabel("mJ", color="#7a8aaa", fontsize=9)
        ax5.legend(fontsize=8, facecolor=BG,
                   labelcolor="#c5cde0", edgecolor="#2d3449")
        ax5.grid(color="#2d3449", linestyle="--", alpha=0.5)

        # Model selection pie
        ax6 = _ax(gs[1, 2], "AdaptAlloc Model Selection")
        rt_pct = df["adapt_ran_rt"].mean() * 100
        yo_pct = 100 - rt_pct
        wedges, texts, autotexts = ax6.pie(
            [yo_pct, rt_pct],
            labels=["YOLO-only", "RT-DETR"],
            colors=["#4ade80", "#60a5fa"],
            autopct="%1.1f%%",
            pctdistance=0.75,
            startangle=90,
            wedgeprops={"edgecolor": "#0d0f14", "linewidth": 2},
        )
        for t in texts:      t.set_color("#c5cde0"); t.set_fontsize(9)
        for at in autotexts: at.set_color("#0d0f14"); at.set_fontsize(9)

        # NEW: Reward over time
        ax7 = _ax(gs[1, 3], "RL Reward over Frames")
        ax7.plot(df["reward"].rolling(10, min_periods=1).mean().values,
                 color="#f97316", linewidth=1.5)
        ax7.axhline(0, color="#5a6a8a", linestyle="--", linewidth=0.8)
        ax7.set_xlabel("Frame", color="#7a8aaa", fontsize=9)
        ax7.set_ylabel("Reward", color="#7a8aaa", fontsize=9)
        ax7.grid(color="#2d3449", linestyle="--", alpha=0.5)

        out_path = self.output_dir / "baseline_comparison.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor="#0d0f14")
        plt.close(fig)
        print(f"\n[Chart] Saved → {out_path}")
        return out_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AdaptAlloc Baseline Comparator")
    parser.add_argument("--source",    default="",
                        help="Image folder or video file")
    parser.add_argument("--labels",    default="",
                        help="YOLO-format labels folder (optional but recommended)")
    parser.add_argument("--webcam",    action="store_true")
    parser.add_argument("--roboflow",  action="store_true",
                        help="Auto-download crack_detection dataset")
    parser.add_argument("--rf-key",    default=os.getenv("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--yolo",      default=_DEFAULT_YOLO)
    parser.add_argument("--rtdetr",    default=_DEFAULT_RTDETR)
    parser.add_argument("--policy",    default="checkpoints/rl_policy.pt")
    parser.add_argument("--conf",      type=float, default=0.25)
    parser.add_argument("--iou",       type=float, default=0.30)
    parser.add_argument("--lam",       type=float, default=0.40)
    parser.add_argument("--max-frames", type=int,  default=200)
    parser.add_argument("--output",    default="outputs")
    args = parser.parse_args()

    frames: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []

    if args.roboflow:
        if not args.rf_key:
            print("[ERROR] --rf-key required for Roboflow download")
            return
        print("[Roboflow] Downloading crack_detection-ibpez v1 …")
        img_dir     = download_roboflow_dataset(args.rf_key)
        args.source = img_dir
        args.labels = str(Path(img_dir).parent / "labels")
        print(f"[Roboflow] Dataset ready at {img_dir}\n")

    if args.webcam:
        cap = cv2.VideoCapture(0)
        print("Capturing from webcam … (Ctrl-C to stop)")
        for _ in range(args.max_frames):
            ret, fr = cap.read()
            if not ret:
                break
            frames.append(fr)
            gt_list.append(np.empty((0, 4)))
        cap.release()

    elif args.source:
        src = Path(args.source)
        if src.is_dir():
            img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            paths    = sorted(
                p for p in src.iterdir()
                if p.suffix.lower() in img_exts
            )[:args.max_frames]
            lbl_dir = Path(args.labels) if args.labels else None
            for p in paths:
                fr = cv2.imread(str(p))
                if fr is None:
                    continue
                frames.append(fr)
                if lbl_dir:
                    lp = lbl_dir / (p.stem + ".txt")
                    gt_list.append(
                        load_gt_boxes(str(lp), fr.shape[1], fr.shape[0]))
                else:
                    gt_list.append(np.empty((0, 4)))
        else:
            cap = cv2.VideoCapture(str(src))
            for _ in range(args.max_frames):
                ret, fr = cap.read()
                if not ret:
                    break
                frames.append(fr)
                gt_list.append(np.empty((0, 4)))
            cap.release()

    if not frames:
        print("[ERROR] No frames loaded. Use --source, --webcam, or --roboflow")
        return

    n_gt = sum(len(g) > 0 for g in gt_list)
    print(f"[Data] {len(frames)} frames, {n_gt} with ground-truth boxes.\n")
    if n_gt == 0:
        print("[WARN] No ground-truth labels found — F1 will be 0 for "
              "everything and reward will use confidence proxy.\n"
              "       Pass --labels <dir> or --roboflow for real F1.\n")

    ev      = BaselineEvaluator(args.yolo, args.rtdetr, args.conf,
                                args.iou, args.lam, args.policy, args.output)
    df      = ev.run(frames, gt_list)
    summary = ev.summarise(df)
    ev.plot(df, summary)

    csv_path = Path(args.output) / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[CSV] Results → {csv_path}")


if __name__ == "__main__":
    main()
