"""
main.py — AdaptAlloc Demo: Adaptive vs Baseline Detection Dashboard
====================================================================

  1. YOLO_WEIGHTS / RTDETR_PATH default to fine-tuned crack models.
  2. Pixel-level Shannon entropy computed per frame → fed into state.
  3. Baseline stream B now shows Always-YOLO vs Always-RT-DETR separately,
     not always-both, so the energy comparison is honest.
  4. Reward uses confidence-improvement delta, not raw confidence.
  5. Policy ε decays per frame (see policy.py).

Run:  streamlit run main.py
      docker compose up
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
from turtle import pd
import warnings
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("ULTRALYTICS_VERBOSE", "False")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from policy import (
    RLScheduler, ModelAwareComplexityScorer,
    UncertaintyEstimate, SceneComplexityScore, HardwareCost,
    compute_pixel_entropy,
)
from telemetry import GPUProfiler, FrameMetrics, TelemetryAccumulator

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

# ── env / config ──────────────────────────────────────────────────────────────
# Point these at your fine-tuned models.
# Override with:  export YOLO_WEIGHTS=data/models/yolo_cracks.pt
#                 export RTDETR_PATH=data/models/rtdetr_folder
CHECKPOINT_PATH = os.getenv("POLICY_CHECKPOINT", "checkpoints/rl_policy.pt")
YOLO_WEIGHTS    = os.getenv("YOLO_WEIGHTS",  "data/models/yolo_cracks.pt")
RTDETR_PATH     = os.getenv("RTDETR_PATH",   "data/models/rtdetr_folder")
ENERGY_ALPHA    = float(os.getenv("ENERGY_ALPHA", "0.40"))
CONF_THRESH     = float(os.getenv("CONF_THRESH",  "0.25"))
IOU_THRESH      = float(os.getenv("IOU_THRESH",   "0.30"))
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Fall back to generic weights if fine-tuned files aren't present
if not Path(YOLO_WEIGHTS).exists():
    YOLO_WEIGHTS = "yolov8n.pt"
if not Path(RTDETR_PATH).exists():
    RTDETR_PATH = ""


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

class YOLOWrapper:
    def __init__(self, weights: str = YOLO_WEIGHTS, conf: float = CONF_THRESH):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf  = conf
        self.device = DEVICE
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self.model(dummy, verbose=False, device=self.device)

    def infer(self, frame: np.ndarray,
              profiler: Optional[GPUProfiler] = None
              ) -> Tuple[dict, FrameMetrics]:
        if profiler:
            with profiler.measure("yolo") as ref:
                results = self.model(frame, verbose=False,
                                     conf=self.conf, device=self.device)
            m = ref.value
        else:
            t0 = time.perf_counter()
            results = self.model(frame, verbose=False,
                                 conf=self.conf, device=self.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            m = FrameMetrics("yolo", gpu_ms=ms, wall_ms=ms, device=self.device)

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            dets = {"boxes": np.empty((0, 4)), "scores": np.empty(0),
                    "labels": np.empty(0, dtype=int), "n": 0}
        else:
            dets = {
                "boxes":  res.boxes.xyxy.cpu().numpy(),
                "scores": res.boxes.conf.cpu().numpy(),
                "labels": res.boxes.cls.cpu().numpy().astype(int),
                "n":      len(res.boxes),
            }
        m.n_detections   = dets["n"]
        m.avg_confidence = float(dets["scores"].mean()) if dets["n"] > 0 else 0.0
        return dets, m

    def infer_with_uncertainty(self, frame: np.ndarray,
                               profiler: Optional[GPUProfiler] = None
                               ) -> Tuple[dict, UncertaintyEstimate, FrameMetrics]:
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
            unc = UncertaintyEstimate(
                conf_mean=mu, conf_var=var, conf_entropy=ent,
                n_det=n, low_conf_ratio=float((c < 0.5).mean()),
                mutual_information=ent,
                feature_variance=var, variance=var, mean_confidence=mu)
        return dets, unc, m


class RTDETRWrapper:
    def __init__(self, model_path: str = RTDETR_PATH, conf: float = CONF_THRESH):
        if not model_path:
            from ultralytics import RTDETR
            self._mode = "ultralytics"
            self.model = RTDETR("rtdetr-l.pt")
        else:
            self._mode = "hf"
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            path = self._extract_if_zip(model_path)
            self.proc  = AutoImageProcessor.from_pretrained(path)
            self.model = AutoModelForObjectDetection.from_pretrained(path)
            self.model.to(DEVICE).eval()
        self.conf   = conf
        self.device = DEVICE
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self.infer(dummy)

    def _extract_if_zip(self, path: str) -> str:
        import zipfile
        if path.endswith(".zip"):
            tmp = tempfile.mkdtemp()
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(tmp)
            for dp, _, files in os.walk(tmp):
                if "config.json" in files:
                    return dp
            return tmp
        return path

    def infer(self, frame: np.ndarray,
              profiler: Optional[GPUProfiler] = None
              ) -> Tuple[dict, FrameMetrics]:
        if profiler:
            with profiler.measure("rtdetr") as ref:
                dets = self._forward(frame)
            m = ref.value
        else:
            t0 = time.perf_counter()
            dets = self._forward(frame)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            m  = FrameMetrics("rtdetr", gpu_ms=ms, wall_ms=ms, device=self.device)
        m.n_detections   = dets["n"]
        m.avg_confidence = float(dets["scores"].mean()) if dets["n"] > 0 else 0.0
        return dets, m

    def _forward(self, frame: np.ndarray) -> dict:
        if self._mode == "ultralytics":
            results = self.model(frame, verbose=False,
                                 conf=self.conf, device=self.device)
            res = results[0]
            if res.boxes is None or len(res.boxes) == 0:
                return {"boxes": np.empty((0, 4)), "scores": np.empty(0),
                        "labels": np.empty(0, dtype=int), "n": 0}
            return {
                "boxes":  res.boxes.xyxy.cpu().numpy(),
                "scores": res.boxes.conf.cpu().numpy(),
                "labels": res.boxes.cls.cpu().numpy().astype(int),
                "n":      len(res.boxes),
            }
        else:
            from PIL import Image as PILImage
            pil   = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h, w  = frame.shape[:2]
            inputs = self.proc(pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            res = self.proc.post_process_object_detection(
                outputs, target_sizes=[(h, w)], threshold=self.conf)[0]
            return {
                "boxes":  res["boxes"].cpu().numpy(),
                "scores": res["scores"].cpu().numpy(),
                "labels": res["labels"].cpu().numpy().astype(int),
                "n":      len(res["boxes"]),
            }


# ══════════════════════════════════════════════════════════════════════════════
# 2. DRAW UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

_COLOURS = {
    "yolo":     (0,  200, 100),
    "rtdetr":   (0,  100, 255),
    "adaptive": (255, 140,  0),
    "baseline": (180,  0,  180),
}


def draw_detections(frame: np.ndarray, dets: dict,
                    label: str, colour: tuple) -> np.ndarray:
    out = frame.copy()
    for i in range(dets["n"]):
        x1, y1, x2, y2 = map(int, dets["boxes"][i])
        conf = float(dets["scores"][i])
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(out, f"{label}:{conf:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return out


def overlay_hud(frame: np.ndarray, active_model: str,
                gpu_ms: float, energy_mj: float,
                vram_mb: float, reward: float,
                pixel_ent: float = 0.0, complexity: float = 0.0) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    colour = _COLOURS.get(active_model.lower(), (255, 255, 255))
    cv2.putText(frame,f"DECISION: {active_model.upper()}", (10, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, colour, 1, cv2.LINE_AA)
    stats = (f"GPU: {gpu_ms:.1f}ms | VRAM: {vram_mb:.0f}MB | "
             f"Energy: {energy_mj:.2f}mJ | R: {reward:+.3f}")
    cv2.putText(frame, stats, (10, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    ent_txt = f"SceneEntropy: {pixel_ent:.3f}"
    cv2.putText(frame,f"Scene Complexity:{complexity:.3f}", (10, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, ent_txt, (10, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 200, 255), 1, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE PROCESSORS
# ══════════════════════════════════════════════════════════════════════════════

class AdaptivePipeline:
    """Stream A: YOLO → Shannon entropy + complexity → REINFORCE → maybe RT-DETR."""

    def __init__(self, yolo: YOLOWrapper, rtdetr: RTDETRWrapper,
                 profiler: GPUProfiler, scheduler: RLScheduler,
                 scorer: ModelAwareComplexityScorer):
        self.yolo      = yolo
        self.rtdetr    = rtdetr
        self.profiler  = profiler
        self.scheduler = scheduler
        self.scorer    = scorer
        self._frame_count  = 0
        self._update_every = 8

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        self._frame_count += 1

        # ── Step 1: pixel Shannon entropy ─────────────────────────────────────
        pixel_ent = compute_pixel_entropy(frame)

        # ── Step 2: YOLO + detection uncertainty ─────────────────────────────
        dets_y, unc, m_y = self.yolo.infer_with_uncertainty(frame, self.profiler)

        # ── Step 3: complexity score (now includes pixel entropy) ────────────
        max_conf = float(dets_y["scores"].max()) if dets_y["n"] > 0 else 0.0
        cplx = self.scorer.score(unc, dets_y["n"], max_conf, pixel_ent)

        # ── Step 4: RL decision ───────────────────────────────────────────────
        state_t = self.scheduler.encode_state(cplx, unc, pixel_ent)
        action, log_prob, entropy = self.scheduler.select_action(state_t,
                                                                  training=True)
        run_rtdetr = (action == 1)

        # ── Step 5: run RT-DETR if selected ───────────────────────────────────
        if run_rtdetr:
            dets_final, m_final = self.rtdetr.infer(frame, self.profiler)
            active_model = "rtdetr"
        else:
            dets_final, m_final = dets_y, m_y
            active_model = "yolo"

        # ── Step 6: reward — improvement over YOLO as proxy ──────────────────
        proxy_f1 = m_final.avg_confidence
        yolo_f1  = m_y.avg_confidence
        cost_obj = HardwareCost(
            m_final.gpu_ms, m_final.wall_ms,
            m_final.vram_delta_mb, m_final.gflops,
            m_final.energy_mj, active_model,
            composite=m_final.gpu_ms + m_final.energy_mj,
        )
        reward = self.scheduler.compute_reward(
            proxy_f1, yolo_f1, cost_obj,
            run_rtdetr, dets_final["n"] > 0,
            has_gt=False,
        )
        self.scheduler.record(log_prob, reward, entropy)

        if self._frame_count % self._update_every == 0:
            self.scheduler.update()
        if self._frame_count % 200 == 0:
            self.scheduler.save(CHECKPOINT_PATH)

        # ── Draw ──────────────────────────────────────────────────────────────
        colour = _COLOURS["rtdetr"] if run_rtdetr else _COLOURS["yolo"]
        vis = draw_detections(frame, dets_final, active_model, colour)
        vis = overlay_hud(vis, active_model,
                          m_final.gpu_ms, m_final.energy_mj,
                          m_final.vram_used_mb, reward, pixel_ent,complexity=cplx.raw_score)
        

        return vis, {
            "active_model":  active_model,
            "gpu_ms":        m_final.gpu_ms,
            "energy_mj":     m_final.energy_mj,
            "vram_mb":       m_final.vram_used_mb,
            "n_det":         dets_final["n"],
            "confidence":    m_final.avg_confidence,
            "reward":        reward,
            "complexity":    cplx.raw_score,
            "pixel_entropy": pixel_ent,
            "epsilon":       self.scheduler.epsilon,
            "rtdetr_ratio":  self.scheduler.rtdetr_ratio,
            "metrics":       m_final,
        }


class BaselinePipeline:
    """
    Stream B: run Always-YOLO and Always-RT-DETR separately.
    This gives honest energy/latency numbers per model, not their sum.
    """

    def __init__(self, yolo: YOLOWrapper, rtdetr: RTDETRWrapper,
                 profiler: GPUProfiler):
        self.yolo     = yolo
        self.rtdetr   = rtdetr
        self.profiler = profiler

    def process(self, frame: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        dets_y, m_y = self.yolo.infer(frame, self.profiler)
        dets_r, m_r = self.rtdetr.infer(frame, self.profiler)

        vis_y = draw_detections(frame.copy(), dets_y, "YOLO",   _COLOURS["yolo"])
        vis_r = draw_detections(frame.copy(), dets_r, "RTDETR", _COLOURS["rtdetr"])
        vis_y = overlay_hud(vis_y, "YOLO",   m_y.gpu_ms,
                            m_y.energy_mj, m_y.vram_used_mb, 0.0)
        vis_r = overlay_hud(vis_r, "RTDETR", m_r.gpu_ms,
                            m_r.energy_mj, m_r.vram_used_mb, 0.0)
        side_by_side = np.concatenate([vis_y, vis_r], axis=1)

        return vis_y, vis_r, side_by_side, {
            "yolo_ms":       m_y.gpu_ms,
            "rtdetr_ms":     m_r.gpu_ms,
            "yolo_energy":   m_y.energy_mj,
            "rtdetr_energy": m_r.energy_mj,
            "yolo_n":        dets_y["n"],
            "rtdetr_n":      dets_r["n"],
            "yolo_conf":     m_y.avg_confidence,
            "rtdetr_conf":   m_r.avg_confidence,
            "combined_ms":   m_y.gpu_ms + m_r.gpu_ms,
            "combined_energy": m_y.energy_mj + m_r.energy_mj,
            "m_yolo":        m_y,
            "m_rtdetr":      m_r,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AdaptAlloc — REINFORCE Object Detection",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
h1,h2,h3{font-family:'Space Mono',monospace!important;}
.main{background:#0d0f14;}
.block-container{padding-top:1.2rem;padding-bottom:0.5rem;}
.metric-card{background:linear-gradient(135deg,#1a1f2e 0%,#12161f 100%);
  border:1px solid #2d3449;border-radius:10px;padding:14px 18px;
  text-align:center;margin-bottom:8px;}
.metric-val{font-size:1.7rem;font-weight:700;color:#e8f4fd;
  font-family:'Space Mono',monospace;}
.metric-lbl{font-size:0.72rem;color:#7a8aaa;text-transform:uppercase;
  letter-spacing:1px;margin-top:3px;}
.model-badge-yolo{background:#0d4f28;color:#4ade80;border-radius:6px;
  padding:4px 12px;font-family:'Space Mono',monospace;font-size:0.85rem;}
.model-badge-rtdetr{background:#1a2f5c;color:#60a5fa;border-radius:6px;
  padding:4px 12px;font-family:'Space Mono',monospace;font-size:0.85rem;}
.stProgress>div>div{background:linear-gradient(90deg,#3b82f6,#06b6d4);}
div[data-testid="stMetricValue"]{color:#e8f4fd!important;
  font-family:'Space Mono',monospace;}
div[data-testid="stMetricDelta"]{color:#4ade80!important;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='background:linear-gradient(90deg,#1e2540,#0d1525);
padding:18px 24px;border-radius:12px;margin-bottom:20px;
border:1px solid #2d3a5a'>
<h1 style='margin:0;font-size:1.6rem;color:#e8f4fd;letter-spacing:2px'>
🎯 AdaptAlloc — Contextual Bandit Object Detection
</h1>
<p style='margin:6px 0 0 0;color:#7a8aaa;font-size:0.85rem'>
REINFORCE policy routes frames between YOLOv8 and RT-DETR via
Shannon entropy scene complexity &nbsp;|&nbsp; λ={ENERGY_ALPHA} &nbsp;|&nbsp;
Device: {DEVICE.upper()} &nbsp;|&nbsp;
YOLO: {Path(YOLO_WEIGHTS).name}
</p>
</div>
""", unsafe_allow_html=True)


def _init_models():
    with st.spinner("Loading models (first run may download weights)…"):
        yolo   = YOLOWrapper(YOLO_WEIGHTS, CONF_THRESH)
        rtdetr = RTDETRWrapper(RTDETR_PATH, CONF_THRESH)
        prof   = GPUProfiler(device=DEVICE)
        sched  = RLScheduler(lam=ENERGY_ALPHA, epsilon=0.20)
        scorer = ModelAwareComplexityScorer()
        sched.load(CHECKPOINT_PATH)
        adap  = AdaptivePipeline(yolo, rtdetr, prof, sched, scorer)
        base  = BaselinePipeline(yolo, rtdetr, prof)
        accum = TelemetryAccumulator()
    return adap, base, accum


if "models_loaded" not in st.session_state:
    adap, base, accum = _init_models()
    st.session_state.update({
        "models_loaded": True,
        "adap":  adap,
        "base":  base,
        "accum": accum,
        "running":     False,
        "frame_count": 0,
        "last_reward": 0.0,
        "last_active": "—",
    })

adap:  AdaptivePipeline    = st.session_state["adap"]
base:  BaselinePipeline    = st.session_state["base"]
accum: TelemetryAccumulator = st.session_state["accum"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    source_type = st.radio("Input source",
                           ["📷 Webcam", "🎥 Upload video", "🖼️ Upload image"],
                           label_visibility="collapsed")
    uploaded = None
    if "Upload" in source_type:
        ext = (["mp4","avi","mov","mkv"] if "video" in source_type
               else ["jpg","jpeg","png","bmp"])
        uploaded = st.file_uploader("Choose file", type=ext)

    st.divider()
    st.markdown("### 🔧 Parameters")
    conf_slider  = st.slider("Confidence threshold", 0.05, 0.95, CONF_THRESH, 0.05)
    alpha_slider = st.slider("Energy penalty λ",     0.0,  1.0,  ENERGY_ALPHA, 0.05)
    fps_cap      = st.slider("Target FPS", 1, 30, 10)
    show_base    = st.checkbox("Show baseline stream", value=True)

    st.divider()
    st.markdown("### 🏋️ Policy")
    if st.button("💾 Save checkpoint"):
        adap.scheduler.save(CHECKPOINT_PATH)
        st.success("Checkpoint saved!")
    if st.button("🔄 Reset policy"):
        adap.scheduler = RLScheduler(lam=ENERGY_ALPHA)
        st.warning("Policy reset.")
    st.caption(f"ε = {adap.scheduler.epsilon:.3f}")
    st.caption(f"RT-DETR usage: {adap.scheduler.rtdetr_ratio*100:.1f}%")

# ── Layout ────────────────────────────────────────────────────────────────────
if show_base:
    col_adap, col_base = st.columns(2)
else:
    col_adap = st.columns(1)[0]

with col_adap:
    st.markdown("#### 🅰 Adaptive Pipeline")
    frame_placeholder_a = st.empty()

if show_base:
    with col_base:
        st.markdown("#### 🅱 Baseline (YOLO | RT-DETR)")
        frame_placeholder_b = st.empty()

st.divider()
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
metric_active  = m1.empty()
metric_gpu     = m2.empty()
metric_energy  = m3.empty()
metric_reward  = m4.empty()
metric_ratio   = m5.empty()
metric_epsilon = m6.empty()
metric_entropy = m7.empty()

st.markdown("#### 📊 Adaptive vs Baselines")
cc1, cc2, cc3 = st.columns(3)
with cc1:
    st.caption("Cumulative Energy (mJ) — lower is better")
    energy_chart = st.empty()
with cc2:
    st.caption("Avg GPU Latency (ms)")
    latency_chart = st.empty()
with cc3:
    st.caption("Avg Confidence / Proxy F1")
    accuracy_chart = st.empty()

st.markdown("#### 🕐 Model Decision Timeline (last 60 frames)")
timeline_ph = st.empty()

st.markdown("### Live Comparison Table")
table_ph= st.empty()


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def _cap_from_upload(f) -> "cv2.VideoCapture":
    suffix = Path(f.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(f.read()); tmp.flush()
    return cv2.VideoCapture(tmp.name)


def process_frame(frame: np.ndarray):
    fc = st.session_state["frame_count"] + 1
    st.session_state["frame_count"] = fc

    vis_a, info_a = adap.process(frame)
    frame_placeholder_a.image(cv2.cvtColor(vis_a, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)

    if show_base:
        _, _, vis_b, info_b = base.process(frame)
        frame_placeholder_b.image(cv2.cvtColor(vis_b, cv2.COLOR_BGR2RGB),
                                   channels="RGB", use_container_width=True)
        accum.record_baseline(info_b["m_yolo"], info_b["m_rtdetr"])
    else:
        info_b = {"combined_ms": 0, "combined_energy": 0,
                  "yolo_conf": 0, "rtdetr_conf": 0}

    accum.record_adaptive(info_a["metrics"])
    st.session_state["last_reward"] = info_a["reward"]
    st.session_state["last_active"] = info_a["active_model"].upper()

    badge_cls  = "rtdetr" if info_a["active_model"] == "rtdetr" else "yolo"
    badge_html = (f'<div class="model-badge-{badge_cls}">'
                  f'{info_a["active_model"].upper()}</div>')

    for ph, val, lbl in [
        (metric_active,  badge_html,                                    "Active Model"),
        (metric_gpu,     f'<div class="metric-val">{info_a["gpu_ms"]:.1f}</div>', "GPU ms"),
        (metric_energy,  f'<div class="metric-val">{info_a["energy_mj"]:.2f}</div>', "Energy mJ"),
        (metric_reward,  f'<div class="metric-val">{info_a["reward"]:+.3f}</div>', "RL Reward"),
        (metric_ratio,   f'<div class="metric-val">{info_a["rtdetr_ratio"]*100:.0f}%</div>', "RT-DETR Rate"),
        (metric_epsilon, f'<div class="metric-val">{info_a["epsilon"]:.3f}</div>', "Explore ε"),
        (metric_entropy, f'<div class="metric-val">{info_a["pixel_entropy"]:.3f}</div>', "Scene Entropy"),
    ]:
        ph.markdown(
            f'<div class="metric-card">{val}'
            f'<div class="metric-lbl">{lbl}</div></div>',
            unsafe_allow_html=True)

    import pandas as pd
    a_sum = accum.adaptive_summary()
    y_sum = accum.yolo_only_summary() if show_base else {"energy_mj": 0, "f1": 0, "gpu_ms": 0}
    r_sum = accum.rtdetr_only_summary() if show_base else {"energy_mj": 0, "f1": 0, "gpu_ms": 0}

    # ── Energy bar chart — 3 bars ─────────────────────────────────────────
    energy_chart.bar_chart(pd.DataFrame({
        "Always-YOLO":    [y_sum["energy_mj"]],
        "Always-RT-DETR": [r_sum["energy_mj"]],
        "AdaptAlloc-RL":  [a_sum["energy_mj"]],
    }), height=180)

    # ── Latency line chart — 3 lines over last 30 frames ──────────────────
    adap_ms  = accum.last_n_gpu_ms(30, "adaptive")
    yolo_ms  = [m.gpu_ms for m in list(accum._bl_yolo)[-30:]]  if show_base else []
    rtdet_ms = [m.gpu_ms for m in list(accum._bl_rtdetr)[-30:]] if show_base else []
    n = max(len(adap_ms), len(yolo_ms), len(rtdet_ms))
    if n > 0:
        lat_data: dict = {}
        if adap_ms:
            lat_data["AdaptAlloc-RL"]  = adap_ms  + [None] * (n - len(adap_ms))
        if yolo_ms:
            lat_data["Always-YOLO"]    = yolo_ms  + [None] * (n - len(yolo_ms))
        if rtdet_ms:
            lat_data["Always-RT-DETR"] = rtdet_ms + [None] * (n - len(rtdet_ms))
        latency_chart.line_chart(pd.DataFrame(lat_data), height=180)

    # ── Confidence/F1 bar chart — 3 bars ─────────────────────────────────
    accuracy_chart.bar_chart(pd.DataFrame({
        "Always-YOLO":    [y_sum["f1"]],
        "Always-RT-DETR": [r_sum["f1"]],
        "AdaptAlloc-RL":  [a_sum["f1"]],
    }), height=180)
 

    actions = adap.scheduler.episode_actions[-60:]
    timeline_html = "<div style='display:flex;gap:3px;flex-wrap:wrap'>"
    for a in actions:
        col = "#60a5fa" if a == 1 else "#4ade80"
        lbl = "RT-DETR" if a == 1 else "YOLO"
        timeline_html += (
            f"<div style='background:{col};color:#0d0f14;"
            f"padding:3px 7px;border-radius:4px;font-size:0.7rem;"
            f"font-family:monospace'>{lbl}</div>")
    timeline_html += "</div>"
    timeline_ph.markdown(timeline_html, unsafe_allow_html=True)

    #b_sum= accum.baseline_summary() if show_base else None
    rows = [
    {
        "Mode":         "Always-YOLO",
        "Avg GPU (ms)": f"{y_sum['gpu_ms']:.1f}",
        "Energy (mJ)":  f"{y_sum['energy_mj']:.2f}",
        "Avg Conf":     f"{y_sum['f1']:.3f}",
        "RT-DETR %":    "0%",
    },
    {
        "Mode":         "Always-RT-DETR",
        "Avg GPU (ms)": f"{r_sum['gpu_ms']:.1f}",
        "Energy (mJ)":  f"{r_sum['energy_mj']:.2f}",
        "Avg Conf":     f"{r_sum['f1']:.3f}",
        "RT-DETR %":    "100%",
    },
    {
        "Mode":         "AdaptAlloc-RL ",
        "Avg GPU (ms)": f"{info_a['gpu_ms']:.1f}",
        "Energy (mJ)":  f"{info_a['energy_mj']:.2f}",
        "Avg Conf":     f"{info_a['confidence']:.3f}",
        "RT-DETR %":    f"{info_a['rtdetr_ratio']*100:.0f}%",
    },]
    table_ph.table(pd.DataFrame(rows).set_index("Mode"))
    



start_btn = st.button(
    "▶ Start" if not st.session_state["running"] else "⏹ Stop",
    type="primary", use_container_width=True)

if start_btn:
    st.session_state["running"] = not st.session_state["running"]

if st.session_state["running"]:
    frame_delay = 1.0 / fps_cap

    if "Webcam" in source_type:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam not accessible.")
            st.session_state["running"] = False
        else:
            st.info("Streaming from webcam… click **Stop** to halt.")
            while st.session_state["running"]:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                process_frame(frame)
                time.sleep(max(0.0, frame_delay - (time.time() - t0)))
            cap.release()

    elif "video" in source_type and uploaded is not None:
        cap   = _cap_from_upload(uploaded)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        prog  = st.progress(0)
        fn    = 0
        while st.session_state["running"]:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                st.session_state["running"] = False
                break
            fn += 1
            prog.progress(min(fn / total, 1.0))
            process_frame(frame)
            time.sleep(max(0.0, frame_delay - (time.time() - t0)))
        cap.release()
        st.success("✅ Video complete.")

    elif "image" in source_type and uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        process_frame(frame)
        st.session_state["running"] = False
        st.success("✅ Image processed.")

    else:
        st.warning("Please select an input source and upload a file.")
        st.session_state["running"] = False

elif st.session_state["frame_count"] == 0:
    for ph in ([frame_placeholder_a, frame_placeholder_b]
               if show_base else [frame_placeholder_a]):
        ph.markdown("""
<div style='background:#1a1f2e;border:2px dashed #2d3449;
border-radius:12px;padding:60px;text-align:center;color:#5a6a8a'>
Press <b>▶ Start</b> to begin
</div>""", unsafe_allow_html=True)
