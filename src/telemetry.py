"""
src/telemetry.py — GPU profiling, per-frame metrics, telemetry accumulator
===========================================================================
Works on CPU-only machines (pynvml optional).
"""

from __future__ import annotations

import time
import contextlib
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Deque

import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


# ─────────────────────────────────────────────────────────────────────────────
# FrameMetrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameMetrics:
    model_name:      str   = "unknown"
    gpu_ms:          float = 0.0
    wall_ms:         float = 0.0
    vram_used_mb:    float = 0.0
    vram_delta_mb:   float = 0.0
    gflops:          float = 0.0
    energy_mj:       float = 0.0
    n_detections:    int   = 0
    avg_confidence:  float = 0.0
    device:          str   = "cpu"  # Track which device was used

    def __post_init__(self):
        # Energy estimate based on device type
        # E (mJ) = P (W) * t (s) * 1000
        if self.energy_mj == 0.0 and self.gpu_ms > 0:
            tdp_watts = _get_tdp_for_device(self.device)
            self.energy_mj = tdp_watts * (self.gpu_ms / 1000.0)


def _get_tdp_for_device(device: str) -> float:
    """Return realistic power consumption for the device.
    
    Args:
        device: "cuda" for GPU or "cpu" for CPU
    
    Returns:
        Power in watts
    """
    if device == "cuda":
        if _HAS_NVML and torch.cuda.is_available():
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except Exception:
                pass
        return 150.0  # Default GPU TDP
    else:
        # CPU power consumption: ~10-30W for typical inference on modern CPU
        # Use conservative estimate of 15W for CPU inference
        return 15.0


# ─────────────────────────────────────────────────────────────────────────────
# GPU Profiler
# ─────────────────────────────────────────────────────────────────────────────

class _MeasureRef:
    """Mutable container so context manager can write back the result."""
    value: FrameMetrics = None


class GPUProfiler:
    """Thin wrapper for timing GPU/CPU inference."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._vram_baseline = self._vram_used_mb()

    def _vram_used_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        if _HAS_NVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return info.used / (1024 ** 2)
            except Exception:
                pass
        return 0.0

    @contextlib.contextmanager
    def measure(self, model_name: str):
        ref = _MeasureRef()
        vram_before = self._vram_used_mb()

        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            start_event.record()
            t0 = time.perf_counter()
            yield ref
            end_event.record()
            torch.cuda.synchronize()
            gpu_ms  = start_event.elapsed_time(end_event)
            wall_ms = (time.perf_counter() - t0) * 1000.0
        else:
            t0 = time.perf_counter()
            yield ref
            wall_ms = gpu_ms = (time.perf_counter() - t0) * 1000.0

        vram_after = self._vram_used_mb()
        m = FrameMetrics(
            model_name    = model_name,
            gpu_ms        = gpu_ms,
            wall_ms       = wall_ms,
            vram_used_mb  = vram_after,
            vram_delta_mb = max(0.0, vram_after - vram_before),
            device        = self.device,
        )
        ref.value = m


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Accumulator
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryAccumulator:
    """
    Accumulates per-frame FrameMetrics for both adaptive and baseline streams.
    Provides rolling summaries for the dashboard charts.
    """

    def __init__(self, window: int = 300):
        self._window = window
        self._adaptive: Deque[FrameMetrics]  = deque(maxlen=window)
        self._bl_yolo:  Deque[FrameMetrics]  = deque(maxlen=window)
        self._bl_rtdetr: Deque[FrameMetrics] = deque(maxlen=window)

    def record_adaptive(self, m: FrameMetrics) -> None:
        self._adaptive.append(m)

    def record_baseline(self, m_yolo: FrameMetrics,
                        m_rtdetr: FrameMetrics) -> None:
        self._bl_yolo.append(m_yolo)
        self._bl_rtdetr.append(m_rtdetr)

    def _summarise(self, items: Deque[FrameMetrics]) -> dict:
        if not items:
            return {"energy_mj": 0.0, "cost": 0.0, "f1": 0.0,
                    "gpu_ms": 0.0, "n": 0}
        return {
            "energy_mj": sum(m.energy_mj for m in items),
            "gpu_ms":    sum(m.gpu_ms    for m in items) / len(items),
            "f1":        sum(m.avg_confidence for m in items) / len(items),
            "cost":      sum(m.gpu_ms + m.energy_mj for m in items),
            "n":         len(items),
        }

    def adaptive_summary(self) -> dict:
        return self._summarise(self._adaptive)

    def baseline_summary(self) -> dict:
        """Combined YOLO + RT-DETR baseline cost."""
        if not self._bl_yolo:
            return {"energy_mj": 0.0, "cost": 0.0, "f1": 0.0,
                    "gpu_ms": 0.0, "n": 0}
        n = len(self._bl_yolo)
    
        
        return {
            "energy_mj": sum(y.energy_mj + r.energy_mj
                             for y, r in zip(self._bl_yolo, self._bl_rtdetr)),
            "gpu_ms":    sum(y.gpu_ms + r.gpu_ms
                             for y, r in zip(self._bl_yolo, self._bl_rtdetr)) / n,
            "f1":        sum(r.avg_confidence
                             for r in self._bl_rtdetr) / n,
            "cost":      sum((y.gpu_ms + r.gpu_ms + y.energy_mj + r.energy_mj)
                             for y, r in zip(self._bl_yolo, self._bl_rtdetr)),
            "n":         n,
        }
    def yolo_only_summary(self) -> dict:
        return self._summarise(self._bl_yolo)
    
    def rtdetr_only_summary(self) -> dict:
        return self._summarise(self._bl_rtdetr)
        

    def last_n_gpu_ms(self, n: int, stream: str = "adaptive") -> List[float]:
        src = self._adaptive if stream == "adaptive" else self._bl_yolo
        items = list(src)[-n:]
        if stream == "baseline":
            rtd = list(self._bl_rtdetr)[-n:]
            items_rt = rtd[:len(items)]
            return [y.gpu_ms + r.gpu_ms
                    for y, r in zip(items, items_rt)]
        return [m.gpu_ms for m in items]
