# AdaptAlloc — Adaptive Object Detection with Reinforcement Learning

AdaptAlloc is a Streamlit dashboard that uses a REINFORCE policy to dynamically route each video frame to either **YOLOv8** (fast, lower energy) or **RT-DETR** (accurate, higher energy) based on scene complexity. The goal is to minimise compute cost while preserving detection quality — only invoking the heavier model when the scene actually warrants it.

---

## How It Works

Every incoming frame goes through this pipeline:

```
Frame
  │
  ├─► compute_pixel_entropy()         ← normalised Shannon entropy of pixel histogram
  │
  ├─► YOLOv8 inference                ← always runs (fast, cheap)
  │        │
  │        └─► UncertaintyEstimate    ← confidence spread, variance, low-conf ratio
  │
  ├─► ModelAwareComplexityScorer      ← produces 9-dim state vector
  │
  ├─► PolicyNetwork (REINFORCE)       ← action 0 = keep YOLO / action 1 = run RT-DETR
  │
  └─► Reward = ΔF1 − λ·cost          ← policy updates every 8 frames
```

A second "baseline" stream runs Always-YOLO and Always-RT-DETR in parallel so you can compare energy, latency, and accuracy against the adaptive policy in real time.

---

## Repository Structure

```
ADAPTIVE_ALLOC/
├── main.py                 # Streamlit dashboard (entry point)
├── pretrain_policy.py      # Offline policy pre-trainer
├── baseline_compare.py     # 3-way comparison script (saves PNG + CSV)
├── src/
│   ├── policy.py           # PolicyNetwork, RLScheduler, complexity scorer
│   └── telemetry.py        # GPU profiling and metrics accumulation
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Model Setup

The two fine-tuned models are **not included** in the repo and must be provided by you. Place them at these paths before running:

```
data/models/yolo_cracks.pt        ← YOLOv8 weights fine-tuned on crack detection
data/models/rt-detr/              ← RT-DETR model directory in Hugging Face format
```

**What the RT-DETR directory must contain:**

```
data/models/rt-detr/
├── config.json
├── preprocessor_config.json
├── pytorch_model.bin   (or model.safetensors)
└── ...
```

This is a standard Hugging Face `AutoModelForObjectDetection` checkpoint. You can export one from a fine-tuned RT-DETR by saving with `model.save_pretrained("data/models/rt-detr")` and `processor.save_pretrained("data/models/rt-detr")`.

**Fallback behaviour:** If neither model file is found, the code falls back to generic `yolov8n.pt` (auto-downloaded by Ultralytics) and `rtdetr-l.pt` (auto-downloaded) so the app still runs without your fine-tuned weights. Detection quality will be lower on crack imagery.

You can also override model paths with environment variables:

```bash
export YOLO_WEIGHTS=data/models/yolo_cracks.pt
export RTDETR_PATH=data/models/rt-detr
```

---

## Installation

**Requirements:** Python 3.9+, pip

```bash
git clone https://github.com/chaturyaganne/ADAPTIVE_ALLOC.git
cd ADAPTIVE_ALLOC

pip install -r requirements.txt
```

---

## Running the App

### Option 1 — Local (Streamlit)

```bash
streamlit run main.py
# Open http://localhost:8501
```

### Option 2 — Docker

```bash
docker compose up --build
# Open http://localhost:8501
```

The Docker image is based on `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`. The `checkpoints/`, `outputs/`, and `data/` directories are bind-mounted so your models and saved checkpoints persist across container restarts.

---

## Pre-Training the Policy (Recommended)

Without pre-training, the REINFORCE policy starts essentially random (ε=0.20) and spends the first few hundred live frames exploring rather than making smart decisions. Pre-training on a labelled validation set gives the policy a head start.

### What `pretrain_policy.py` does

1. Loads all images from a folder (up to `--max-frames`, default 500).
2. Loads YOLO-format ground-truth labels if provided.
3. Runs YOLOv8 on all frames once and caches the results (fast).
4. For each frame, it computes scene complexity, encodes the 9-dim state, samples an action (YOLO or RT-DETR), and computes a reward:
   - If ground-truth labels exist: reward uses real IoU-based F1 (`has_gt=True`).
   - Otherwise: reward uses confidence improvement as a proxy.
5. Every `--batch` frames (default 16), it runs a REINFORCE gradient update.
6. Repeats for `--epochs` passes over the data.
7. Saves the trained policy to `checkpoints/rl_policy.pt`.

The live dashboard (`main.py`) automatically loads this checkpoint at startup and continues online learning from there.

### Usage

```bash
# With local images and YOLO-format labels (best results):
python pretrain_policy.py \
    --images data/crack_val/images \
    --labels data/crack_val/labels \
    --epochs 5 \
    --output checkpoints/rl_policy.pt

# With local images only (no labels — uses confidence proxy):
python pretrain_policy.py \
    --images data/crack_val/images \
    --epochs 5

# Auto-download from Roboflow:
python pretrain_policy.py \
    --roboflow \
    --rf-key YOUR_ROBOFLOW_API_KEY \
    --epochs 5
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--images` | `data/crack_val/images` | Path to validation image folder |
| `--labels` | *(empty)* | Path to YOLO-format `.txt` label folder |
| `--yolo` | `data/models/yolo_cracks.pt` | YOLO weights to use during pre-training |
| `--rtdetr` | `data/models/rt-detr` | RT-DETR model path |
| `--epochs` | `5` | Number of passes over the image set |
| `--batch` | `16` | REINFORCE update every N frames |
| `--lam` | `0.40` | λ — energy penalty weight in the reward |
| `--iou` | `0.30` | IoU threshold for F1 matching |
| `--max-frames` | `500` | Maximum images to load |
| `--output` | `checkpoints/rl_policy.pt` | Where to save the checkpoint |
| `--roboflow` | *(flag)* | Auto-download the Roboflow crack dataset |
| `--rf-key` | `$ROBOFLOW_API_KEY` | Roboflow API key |

---

## Running the Baseline Comparison

`baseline_compare.py` runs all three modes (Always-YOLO, Always-RT-DETR, AdaptAlloc-RL) on the same set of frames and saves a chart and CSV.

```bash
# With labelled images (real F1 scores):
python baseline_compare.py \
    --source data/crack_val/images \
    --labels data/crack_val/labels \
    --output outputs/

# Video file:
python baseline_compare.py --source video.mp4 --output outputs/

# Webcam:
python baseline_compare.py --webcam

# Roboflow dataset:
python baseline_compare.py --roboflow --rf-key YOUR_KEY
```

Outputs saved to `outputs/`:
- `baseline_comparison.png` — side-by-side charts of energy, latency, F1, and entropy
- `results.csv` — per-frame metrics for all three modes

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_WEIGHTS` | `data/models/yolo_cracks.pt` | Path to YOLOv8 weights |
| `RTDETR_PATH` | `data/models/rt-detr` | Path to RT-DETR model directory |
| `POLICY_CHECKPOINT` | `checkpoints/rl_policy.pt` | Path to the saved REINFORCE policy |
| `ENERGY_ALPHA` | `0.40` | λ — weight of the energy penalty in the reward |
| `CONF_THRESH` | `0.25` | Confidence threshold for both models |
| `IOU_THRESH` | `0.30` | IoU threshold for F1 evaluation |
| `ROBOFLOW_API_KEY` | — | Roboflow API key for dataset download |

---

## File Reference

### `main.py`
The Streamlit entry point. Handles the full UI and processing loop.

- **`YOLOWrapper`** — wraps Ultralytics YOLO with warm-up inference, GPU timing, and an `infer_with_uncertainty()` method that computes confidence entropy, variance, and low-confidence ratio from the detection scores.
- **`RTDETRWrapper`** — supports two backends: Ultralytics `RTDETR` (when no custom path is set) or a Hugging Face `AutoModelForObjectDetection` checkpoint loaded from `RTDETR_PATH`. Handles `.zip` extraction automatically.
- **`AdaptivePipeline`** — Stream A. Per frame: computes pixel entropy → runs YOLO → scores complexity → queries the REINFORCE policy → conditionally runs RT-DETR → computes reward → calls `scheduler.update()` every 8 frames → autosaves checkpoint every 200 frames.
- **`BaselinePipeline`** — Stream B. Runs both YOLO and RT-DETR independently on the same frame to produce honest per-model energy/latency numbers (not their combined cost).
- **UI** — sidebar controls for input source (webcam / video / image), FPS cap, and baseline toggle. Live metrics cards (active model, GPU ms, energy mJ, RL reward, RT-DETR rate, epsilon, scene entropy), energy/latency/confidence charts, a colour-coded decision timeline, and a summary comparison table.

### `pretrain_policy.py`
Offline pre-training script. Described in detail in the [Pre-Training the Policy](#pre-training-the-policy-recommended) section above.

### `baseline_compare.py`
Offline evaluation script. Iterates over an image folder (or video / webcam) and runs all three modes on every frame. Computes real IoU-F1 when ground-truth labels are present. Saves a multi-panel Matplotlib figure and a results CSV. Also exposes `download_roboflow_dataset()` which is imported by `pretrain_policy.py` when `--roboflow` is used.

### `src/policy.py`
Core ML logic. Contains:

- **`compute_pixel_entropy(frame)`** — converts a BGR frame to grayscale, builds a 256-bin pixel intensity histogram, and returns normalised Shannon entropy in `[0, 1]` (1 = maximally random image, 0 = flat image). This is the primary scene-complexity signal.
- **`UncertaintyEstimate`** — dataclass holding statistics derived from YOLO's confidence scores: mean, variance, entropy of the confidence histogram, detection count, low-confidence ratio, and mutual information proxy.
- **`SceneComplexityScore`** — dataclass output of the scorer: raw composite score, sub-scores for uncertainty / density / confidence gap, and pixel entropy.
- **`HardwareCost`** — dataclass holding GPU ms, wall ms, VRAM delta, GFLOPs, energy mJ, model name, and a composite cost scalar used in the reward.
- **`ModelAwareComplexityScorer`** — combines YOLO uncertainty and pixel entropy into a `SceneComplexityScore`. Pixel entropy is weighted at 35%, YOLO uncertainty at 35%, detection density at 20%, and confidence gap at 10%.
- **`PolicyNetwork`** — a small MLP (`9 → 64 → 32 → 2` with LayerNorm) that outputs a softmax over `{YOLO, RT-DETR}`. Initialised near-uniform so the policy starts unbiased.
- **`RLScheduler`** — manages everything around the policy: state encoding (building the 9-dim float32 vector), ε-greedy action selection with exponential decay (`ε` starts at 0.20, floors at 0.03), reward computation, reward normalisation (online Welford mean/variance), the REINFORCE gradient update with entropy regularisation and gradient clipping, and checkpoint save/load.

### `src/telemetry.py`
Hardware measurement utilities.

- **`FrameMetrics`** — dataclass storing per-inference measurements (GPU ms, wall ms, VRAM used/delta, GFLOPs, energy mJ, detection count, average confidence). Energy is estimated from GPU TDP: `energy_mj = TDP_watts × (gpu_ms / 1000)`. If `pynvml` is available, actual GPU TDP is read; otherwise 150 W is assumed.
- **`GPUProfiler`** — context-manager-based profiler. On CUDA it uses `torch.cuda.Event` for accurate GPU timing; on CPU it falls back to `perf_counter`.
- **`TelemetryAccumulator`** — rolling window (default 300 frames) of `FrameMetrics` for the adaptive stream and both baseline streams. Provides `adaptive_summary()`, `yolo_only_summary()`, `rtdetr_only_summary()`, and `last_n_gpu_ms()` for the dashboard charts.

### `requirements.txt`
Python dependencies. Key packages:

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 and RT-DETR (fallback) |
| `torch` / `torchvision` | Model inference and REINFORCE training |
| `transformers` | Hugging Face RT-DETR checkpoint loading |
| `streamlit` | Dashboard UI |
| `opencv-python-headless` | Frame I/O and pixel entropy |
| `pynvml` | GPU TDP and VRAM telemetry |
| `roboflow` | Optional dataset download |
| `thop` | Optional GFLOPs profiling |

### `Dockerfile`
Builds from `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`. Installs system OpenGL/display libraries needed by OpenCV, copies the repo, creates `checkpoints/`, `outputs/`, `data/models/`, and `data/sample_frames/` directories, and starts `streamlit run main.py` on port 8501.

### `docker-compose.yml`
Defines two services:
- **`adaptalloc`** — the Streamlit dashboard, always started. Bind-mounts `./checkpoints`, `./outputs`, and `./data` so models and checkpoints persist.
- **`baseline`** — runs `baseline_compare.py` on startup. Only activated with `docker compose --profile baseline up`.

---

## Policy State Vector

The 9 features fed to the policy network on every frame:

| Index | Feature | Source |
|-------|---------|--------|
| 0 | `conf_entropy` | Shannon entropy of YOLO's confidence histogram |
| 1 | `low_conf_ratio` | Fraction of detections below confidence 0.5 |
| 2 | `detection_density` | Detection count normalised by 10 |
| 3 | `mean_confidence` | Average YOLO confidence across detections |
| 4 | `conf_var_scaled` | Confidence variance × 10 |
| 5 | `max_conf_gap` | 1 − mean confidence (when detections exist) |
| 6 | `raw_complexity` | Composite score from `ModelAwareComplexityScorer` |
| 7 | `has_any_detection` | 1.0 if YOLO found anything, else 0.0 |
| 8 | `pixel_entropy` | Normalised pixel-level Shannon entropy |

---

## Reward Function

```
λ = ENERGY_ALPHA (default 0.40)
cost_norm = min(composite_cost / 200, 1.0)

When ground-truth labels are available (--labels provided):
  RT-DETR chosen:   R = F1_adapt + 0.5 × max(F1_adapt − F1_yolo, 0) − λ × 0.3 × cost_norm
  YOLO chosen (F1 ≥ 0.5):  R = F1_adapt + λ × 0.5     ← saved compute and good quality
  YOLO chosen (F1 < 0.5):  R = F1_adapt − λ × 0.5     ← missed detections, penalised

Live mode (no labels — confidence used as proxy for F1):
  RT-DETR chosen:          R = (adapt_conf − yolo_conf) − λ × 0.3 × cost_norm
  YOLO chosen, conf ≥ 0.4: R = conf × 0.5 + λ × 0.4   ← YOLO was good enough
  YOLO chosen, no dets:    R = +λ × 0.6                ← correct cheap skip
  YOLO chosen, conf < 0.4: R = conf × 0.5 − λ × 0.3   ← low quality, should have escalated

All rewards clipped to [−2, 2].
```
