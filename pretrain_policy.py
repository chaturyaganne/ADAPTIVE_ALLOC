"""
pretrain_policy.py — Offline pre-train the REINFORCE policy on a labelled
                     crack-detection dataset BEFORE running the live demo.
===========================================================================
Why this matters:
  The live demo only has ~8 frames per REINFORCE update and starts with
  ε=0.20. Without pre-training, the policy is essentially random for the
  first few hundred frames, making "adaptive" look no better than guessing.

  This script generates synthetic episodes from your validation set, runs
  many REINFORCE updates offline, then saves a checkpoint that the live demo
  loads at startup.

Usage:
  python pretrain_policy.py --images data/crack_val/images \
                            --labels data/crack_val/labels \
                            --epochs 5 \
                            --output checkpoints/rl_policy.pt

  # Or with Roboflow auto-download:
  python pretrain_policy.py --roboflow --rf-key YOUR_KEY --epochs 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from policy import (
    RLScheduler, ModelAwareComplexityScorer,
    UncertaintyEstimate, HardwareCost,
    compute_pixel_entropy,
)
from telemetry import GPUProfiler, FrameMetrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DEFAULT_YOLO   = os.getenv("YOLO_WEIGHTS",  "data/models/yolo_cracks.pt")
_DEFAULT_RTDETR = os.getenv("RTDETR_PATH",   "data/models/rtdetr")
if not Path(_DEFAULT_YOLO).exists():
    _DEFAULT_YOLO = "yolov8n.pt"
if not Path(_DEFAULT_RTDETR).exists():
    _DEFAULT_RTDETR = "data/models/rt-detr"


# ── inline imports of wrappers (same as baseline_compare.py) ─────────────────

def _make_yolo(weights, conf):
    from ultralytics import YOLO
    m = YOLO(weights)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(2):
        m(dummy, verbose=False, device=DEVICE)
    return m, conf


def _make_rtdetr(path, conf):
    if not path:
        from ultralytics import RTDETR
        m = RTDETR("rtdetr-l.pt")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(2):
            m(dummy, verbose=False, device=DEVICE)
        return ("ul", m, None, conf)
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    import zipfile, tempfile
    if path.endswith(".zip"):
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmp)
        for dp, _, f in os.walk(tmp):
            if "config.json" in f:
                path = dp
                break
    proc = AutoImageProcessor.from_pretrained(path)
    mdl  = AutoModelForObjectDetection.from_pretrained(path).to(DEVICE).eval()
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    return ("hf", mdl, proc, conf)


def _yolo_infer(model_tuple, frame):
    model, conf = model_tuple
    t0 = time.perf_counter()
    results = model(frame, verbose=False, conf=conf, device=DEVICE)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms  = (time.perf_counter() - t0) * 1000
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        dets = {"boxes": np.empty((0, 4)), "scores": np.empty(0), "n": 0}
    else:
        dets = {"boxes":  res.boxes.xyxy.cpu().numpy(),
                "scores": res.boxes.conf.cpu().numpy(),
                "n":      len(res.boxes)}
    m = FrameMetrics("yolo", gpu_ms=ms, wall_ms=ms, device=DEVICE)
    m.n_detections   = dets["n"]
    m.avg_confidence = float(dets["scores"].mean()) if dets["n"] > 0 else 0.0
    return dets, m


def _rtdetr_infer(model_tuple, frame):
    mode, mdl, proc, conf = model_tuple
    t0 = time.perf_counter()
    if mode == "ul":
        res = mdl(frame, verbose=False, conf=conf, device=DEVICE)[0]
        if res.boxes is None or len(res.boxes) == 0:
            dets = {"boxes": np.empty((0, 4)), "scores": np.empty(0), "n": 0}
        else:
            dets = {"boxes":  res.boxes.xyxy.cpu().numpy(),
                    "scores": res.boxes.conf.cpu().numpy(),
                    "n":      len(res.boxes)}
    else:
        from PIL import Image as PILImage
        pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = frame.shape[:2]
        inp = proc(pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = mdl(**inp)
        r = proc.post_process_object_detection(
            out, target_sizes=[(h, w)], threshold=conf)[0]
        dets = {"boxes":  r["boxes"].cpu().numpy(),
                "scores": r["scores"].cpu().numpy(),
                "n":      len(r["boxes"])}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    m  = FrameMetrics("rtdetr", gpu_ms=ms, wall_ms=ms, device=DEVICE)
    m.n_detections   = dets["n"]
    m.avg_confidence = float(dets["scores"].mean()) if dets["n"] > 0 else 0.0
    return dets, m


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih   = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter    = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1e-9)


def _f1(dets, gt_boxes, iou_thr=0.30):
    n_pred, n_gt = dets["n"], len(gt_boxes)
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
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
    return float(2 * pr * rc / (pr + rc + 1e-9))


def _load_gt(lbl_path, w, h):
    boxes = []
    if not lbl_path or not os.path.exists(lbl_path):
        return np.empty((0, 4))
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            boxes.append([
                (cx - bw/2)*w, (cy - bh/2)*h,
                (cx + bw/2)*w, (cy + bh/2)*h,
            ])
    return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4))


def _make_unc(dets):
    n, c = dets["n"], dets["scores"]
    if n == 0:
        return UncertaintyEstimate(0, 0, 1.0, 0, 1.0, 1.0, 0, 0, 0)
    mu  = float(c.mean())
    var = float(c.var()) if len(c) > 1 else 0.0
    hist, _ = np.histogram(c, bins=10, range=(0, 1))
    p   = hist / (hist.sum() + 1e-9)
    ent = float(-np.sum(p * np.log2(p + 1e-9))) / np.log2(10)
    return UncertaintyEstimate(mu, var, ent, n, float((c < 0.5).mean()),
                               ent, var, var, mu)


# ── main pre-training loop ─────────────────────────────────────────────────────

def pretrain(
    images_dir: str,
    labels_dir: str,
    yolo_weights: str,
    rtdetr_path: str,
    epochs: int,
    lam: float,
    iou_thr: float,
    output: str,
    max_frames: int,
    batch_size: int,
):
    print("=" * 60)
    print("  AdaptAlloc Policy Pre-trainer")
    print("=" * 60)
    print(f"  Images : {images_dir}")
    print(f"  Labels : {labels_dir}")
    print(f"  Epochs : {epochs}")
    print(f"  Device : {DEVICE}")
    print()

    # Load images
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths    = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in img_exts
    )[:max_frames]
    if not paths:
        print("[ERROR] No images found in", images_dir)
        return

    frames   = [cv2.imread(str(p)) for p in paths]
    lbl_dir  = Path(labels_dir) if labels_dir else None
    gt_list  = []
    for p, fr in zip(paths, frames):
        if fr is None:
            gt_list.append(np.empty((0, 4)))
            continue
        if lbl_dir:
            lp = lbl_dir / (p.stem + ".txt")
            gt_list.append(_load_gt(str(lp), fr.shape[1], fr.shape[0]))
        else:
            gt_list.append(np.empty((0, 4)))

    # Filter out None frames
    pairs  = [(f, g) for f, g in zip(frames, gt_list) if f is not None]
    frames = [x[0] for x in pairs]
    gt_list = [x[1] for x in pairs]
    n_gt   = sum(len(g) > 0 for g in gt_list)
    print(f"[Data] {len(frames)} images, {n_gt} with labels.\n")

    print("[1/2] Loading YOLOv8 …")
    yolo_model = _make_yolo(yolo_weights, 0.25)
    print("[2/2] Loading RT-DETR …")
    rtdetr_model = _make_rtdetr(rtdetr_path, 0.25)
    print()

    scorer  = ModelAwareComplexityScorer()
    rl      = RLScheduler(lam=lam, epsilon=0.15,
                           epsilon_min=0.03, epsilon_decay=0.998)

    # Pre-compute YOLO results for all frames (cache for speed)
    print("Pre-computing YOLO detections …")
    yolo_cache = [_yolo_infer(yolo_model, fr) for fr in frames]
    print("Done.\n")

    total_updates = 0
    for epoch in range(1, epochs + 1):
        idx_order = np.random.permutation(len(frames))
        ep_rewards = []

        for i, idx in enumerate(idx_order):
            fr  = frames[idx]
            gt  = gt_list[idx]
            dy, my  = yolo_cache[idx]
            has_gt  = len(gt) > 0

            pixel_ent = compute_pixel_entropy(fr)
            unc       = _make_unc(dy)
            max_c     = float(dy["scores"].max()) if dy["n"] > 0 else 0.0
            cplx      = scorer.score(unc, dy["n"], max_c, pixel_ent)
            state     = rl.encode_state(cplx, unc, pixel_ent)
            action, log_prob, entropy = rl.select_action(state, training=True)

            if action == 1:
                dr, mr = _rtdetr_infer(rtdetr_model, fr)
                d_final, m_final, active = dr, mr, "rtdetr"
            else:
                d_final, m_final, active = dy, my, "yolo"

            f1_adapt = _f1(d_final, gt, iou_thr) if has_gt else m_final.avg_confidence
            f1_yolo  = _f1(dy, gt, iou_thr) if has_gt else my.avg_confidence

            cost = HardwareCost(
                m_final.gpu_ms, m_final.wall_ms, 0, 0, m_final.energy_mj,
                active, composite=m_final.gpu_ms + m_final.energy_mj)

            reward = rl.compute_reward(
                f1_adapt, f1_yolo, cost, action == 1,
                d_final["n"] > 0, has_gt=has_gt)

            rl.record(log_prob, reward, entropy)
            ep_rewards.append(reward)

            if (i + 1) % batch_size == 0:
                loss = rl.update()
                total_updates += 1

        # flush remaining buffer at end of epoch
        if rl.rewards:
            rl.update()
            total_updates += 1

        mean_r   = float(np.mean(ep_rewards))
        rt_usage = rl.rtdetr_ratio * 100
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"mean_reward={mean_r:+.4f}  "
              f"RT-DETR%={rt_usage:.1f}%  "
              f"ε={rl.epsilon:.4f}  "
              f"updates={total_updates}")

    rl.save(output)
    print(f"\n[Done] Policy saved to {output}")
    print(f"       RT-DETR ratio: {rl.rtdetr_ratio*100:.1f}%")
    print(f"       Final ε:       {rl.epsilon:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Pre-train AdaptAlloc policy")
    parser.add_argument("--images",   default="data/crack_val/images",
                        help="Path to validation image folder")
    parser.add_argument("--labels",   default="",
                        help="Path to YOLO-format labels folder")
    parser.add_argument("--roboflow", action="store_true")
    parser.add_argument("--rf-key",   default=os.getenv("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--yolo",     default=_DEFAULT_YOLO)
    parser.add_argument("--rtdetr",   default=_DEFAULT_RTDETR)
    parser.add_argument("--epochs",   type=int,   default=5)
    parser.add_argument("--lam",      type=float, default=0.40)
    parser.add_argument("--iou",      type=float, default=0.30)
    parser.add_argument("--batch",    type=int,   default=16,
                        help="REINFORCE update every N steps")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--output",   default="checkpoints/rl_policy.pt")
    args = parser.parse_args()

    if args.roboflow:
        if not args.rf_key:
            print("[ERROR] --rf-key required")
            return
        from baseline_compare import download_roboflow_dataset
        img_dir     = download_roboflow_dataset(args.rf_key)
        args.images = img_dir
        args.labels = str(Path(img_dir).parent / "labels")

    pretrain(
        images_dir   = args.images,
        labels_dir   = args.labels,
        yolo_weights = args.yolo,
        rtdetr_path  = args.rtdetr,
        epochs       = args.epochs,
        lam          = args.lam,
        iou_thr      = args.iou,
        output       = args.output,
        max_frames   = args.max_frames,
        batch_size   = args.batch,
    )


if __name__ == "__main__":
    main()
