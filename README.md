# AdaptAlloc — Contextual Bandit Object Detection

> REINFORCE policy dynamically routes each frame to **YOLOv8** (fast) or
> **RT-DETR** (accurate) based on **pixel-level Shannon entropy** scene
> complexity, minimising `Cost = Energy − α·Accuracy`.

---

## Architecture

```
Input (webcam / video / image)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Stream A — Adaptive Pipeline                                 │
│                                                               │
│  Frame ──► compute_pixel_entropy()  (Shannon H, dim 8)       │
│         │                                                     │
│         └─► YOLOv8 (finetuned) → UncertaintyEstimate         │
│                    │                                          │
│             ModelAwareComplexityScorer (9-dim state)          │
│                    │                                          │
│             PolicyNetwork (REINFORCE, ε-greedy decay)         │
│                    │                                          │
│             action=0 → YOLO-only                              │
│             action=1 → run RT-DETR (finetuned)                │
│                    │                                          │
│  Reward  R = ΔF1 − λ·cost  (real F1 when gt available)       │
│  Update every 8 frames                                        │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│  Stream B — Baseline (side-by-side)                           │
│  Frame → YOLO-only   │  Frame → RT-DETR-only                 │
└───────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Place your fine-tuned models

```
data/models/yolo_cracks.pt          ← your YOLOv8 crack weights
data/models/rtdetr_folder/          ← your RT-DETR HF dir (or zip)
```

### 2. (Recommended) Pre-train the policy

```bash
# With Roboflow dataset (auto-download):
python pretrain_policy.py --roboflow --rf-key $ROBOFLOW_API_KEY --epochs 5

# Or with local images + labels:
python pretrain_policy.py \
    --images data/crack_val/images \
    --labels data/crack_val/labels \
    --epochs 5
```

### 3. Run the dashboard

```bash
# Local
pip install -r requirements.txt
streamlit run main.py

# Docker
docker compose up --build
# Open http://localhost:8501
```

### 4. Run the 3-way comparison report

```bash
# With Roboflow (recommended — gives real F1):
python baseline_compare.py --roboflow --rf-key $ROBOFLOW_API_KEY

# With local labelled images:
python baseline_compare.py \
    --source data/crack_val/images \
    --labels data/crack_val/labels \
    --output outputs/

# No labels (confidence proxy only):
python baseline_compare.py --source /path/to/images
```

Outputs: `outputs/baseline_comparison.png` + `outputs/results.csv`

---

## Policy State Vector (9-dim)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `conf_entropy` | Spread of YOLO confidence scores |
| 1 | `low_conf_ratio` | Fraction of detections below 0.5 |
| 2 | `detection_density` | Normalised detection count |
| 3 | `mean_confidence` | Average confidence |
| 4 | `conf_var_scaled` | Confidence variance ×10 |
| 5 | `max_conf_gap` | 1 − max_confidence |
| 6 | `raw_complexity` | Composite complexity score |
| 7 | `has_any_detection` | Binary: YOLO found anything |
| 8 | **`pixel_entropy`** | **Normalised Shannon entropy of pixel histogram** ← NEW |

---

## Reward Function

```
If has_gt (real labels available):
    if ran_rtdetr:  R = F1_adapt + 0.5·max(ΔF1, 0) − λ·0.3·cost_norm
    else (F1≥0.5):  R = F1_adapt + λ·0.5     # saved compute + good F1
         (F1<0.5):  R = F1_adapt − λ·0.5     # missed detections, penalise

Else (no labels — live mode):
    ΔF1 = adapt_conf − yolo_conf   # improvement proxy
    if ran_rtdetr:  R = ΔF1 − λ·0.3·cost_norm
    if skipped and has_det and conf≥0.4:  R = conf·0.5 + λ·0.4
    if skipped and no_det:                R = +λ·0.6   # correct cheap skip
    if skipped and conf<0.4:              R = conf·0.5 − λ·0.3
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_WEIGHTS` | `data/models/yolo_cracks.pt` | Crack-finetuned YOLO weights |
| `RTDETR_PATH` | `data/models/rtdetr_folder` | Crack-finetuned RT-DETR dir |
| `POLICY_CHECKPOINT` | `checkpoints/rl_policy.pt` | REINFORCE checkpoint |
| `ENERGY_ALPHA` | `0.40` | λ in reward R = F1 − λ·cost |
| `CONF_THRESH` | `0.25` | Detection confidence threshold |
| `IOU_THRESH` | `0.30` | IoU threshold for F1 matching |
| `ROBOFLOW_API_KEY` | — | Roboflow API key |
