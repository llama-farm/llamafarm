---
title: Vision Model Evaluation
sidebar_position: 6
---

# Vision Model Evaluation

Evaluate, compare, and auto-promote vision models with proper COCO metrics. Closes the train→eval→serve loop with zero orchestration.

## Overview

The evaluation pipeline provides:

- **Model evaluation** — run ultralytics `.val()` against any dataset YAML
- **Head-to-head comparison** — compare two models on the same dataset
- **Leaderboard** — SQLite-backed ranking of all evaluated models
- **Auto-eval/promote** — automatically evaluate after training and promote the best model

All evaluation jobs run asynchronously and are polled by job ID.

## Quick Start

### Evaluate a Model

```bash
# Through LlamaFarm server (:14345, Form data)
curl -X POST http://localhost:14345/v1/vision/eval \
  -F "model=my-detector" \
  -F "dataset=/path/to/dataset.yaml"

# Through runtime directly (:11540, JSON)
curl -X POST http://localhost:11540/v1/vision/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-detector",
    "dataset": "/path/to/dataset.yaml"
  }'
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "queued",
  "created_at": "2026-02-18T14:30:00"
}
```

### Poll for Results

```bash
curl http://localhost:14345/v1/vision/eval/a1b2c3d4
```

**Response (completed):**
```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "result": {
    "model_name": "my-detector",
    "mAP50": 0.213,
    "mAP50_95": 0.108,
    "precision": 0.267,
    "recall": 0.247,
    "f1": 0.256,
    "per_class": {
      "person": 0.31,
      "car": 0.18,
      "bicycle": 0.12
    },
    "inference_ms": 45.2,
    "small_object_recall": 0.15,
    "medium_object_recall": 0.38,
    "large_object_recall": 0.72,
    "score": 0.1842,
    "model_size_mb": 22.5
  }
}
```

## API Reference

### POST /v1/vision/eval

Start a model evaluation job.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Model name or path to `.pt` file |
| `dataset` | string | required | Path to dataset YAML (ultralytics format) |
| `imgsz` | int | 640 | Image size for evaluation (320–2560) |
| `batch_size` | int | 16 | Batch size (1–256) |
| `weights` | object | null | Custom scoring weight overrides |

The `model` parameter accepts:
- A **model name** — resolves to `~/.llamafarm/models/vision/\{name\}/current.pt`
- A **direct path** to a `.pt` file (e.g., `/path/to/best.pt`)
- A **versioned model** — if no `current.pt`, uses the latest `v\{n\}.pt`

### GET /v1/vision/eval/\{job_id\}

Poll evaluation job status. Returns the same `EvalJobResponse` with `status` being one of: `queued`, `running`, `completed`, `failed`.

### POST /v1/vision/eval/compare

Head-to-head comparison of two models on the same dataset.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_a` | string | required | First model name or path |
| `model_b` | string | required | Second model name or path |
| `dataset` | string | required | Dataset YAML |
| `imgsz` | int | 640 | Image size |
| `batch_size` | int | 16 | Batch size |

Returns an async job. Result includes per-metric comparison:

```json
{
  "model_a": "aerial-5epoch",
  "model_b": "aerial-50epoch",
  "winner": "aerial-5epoch",
  "score_diff": 0.072,
  "metrics": {
    "mAP50": {"a": 0.213, "b": 0.112, "diff": 0.101, "better": "a"},
    "mAP50_95": {"a": 0.108, "b": 0.056, "diff": 0.052, "better": "a"},
    "inference_ms": {"a": 45, "b": 48, "diff": -3, "better": "a"},
    "small_object_recall": {"a": 0.15, "b": 0.08, "diff": 0.07, "better": "a"},
    "score": {"a": 0.184, "b": 0.112, "diff": 0.072, "better": "a"}
  }
}
```

For `inference_ms`, lower is better. For all other metrics, higher is better.

### GET /v1/vision/eval/leaderboard

Ranked model leaderboard sorted by composite score.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | string | null | Filter by dataset path |
| `limit` | int | 20 | Max results to return |

```bash
# All models
curl http://localhost:14345/v1/vision/eval/leaderboard

# Filter by dataset
curl "http://localhost:14345/v1/vision/eval/leaderboard?dataset=/path/to/visdrone.yaml&limit=10"
```

**Response:**
```json
[
  {
    "rank": 1,
    "model_name": "aerial-5epoch",
    "score": 0.1842,
    "mAP50": 0.213,
    "mAP50_95": 0.108,
    "precision": 0.267,
    "recall": 0.247,
    "f1": 0.256,
    "inference_ms": 45.2,
    "small_object_recall": 0.15,
    "dataset": "/data/visdrone.yaml",
    "timestamp": "2026-02-18T14:30:00"
  }
]
```

### GET /v1/vision/eval/leaderboard/\{model_name\}

Evaluation history for a specific model across all runs.

```bash
curl http://localhost:14345/v1/vision/eval/leaderboard/aerial-5epoch
```

## Auto-Eval and Auto-Promote

The most powerful feature: close the train→eval→serve loop with a single request.

### Training with Auto-Eval

Add `auto_eval=true` and `auto_promote=true` to any training request:

```bash
# Through :14345 (Form data)
curl -X POST http://localhost:14345/v1/vision/train \
  -F "model=aerial-v3" \
  -F "dataset=/path/to/dataset.yaml" \
  -F "task=detection" \
  -F "epochs=10" \
  -F "auto_eval=true" \
  -F "auto_promote=true"
```

```bash
# Through :11540 (JSON)
curl -X POST http://localhost:11540/v1/vision/train \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aerial-v3",
    "dataset": "/path/to/dataset.yaml",
    "task": "detection",
    "config": {"epochs": 10},
    "auto_eval": true,
    "auto_promote": true
  }'
```

### What Happens

1. **Train** — model trains for the specified epochs
2. **Save** — versioned checkpoint saved as `v\{n\}.pt` + ONNX export
3. **Eval** — automatic `.val()` against the training dataset
4. **Compare** — score compared against leaderboard best for that dataset
5. **Promote** — if score is higher, `current.pt` is updated to the new model

The caller only needs to poll the training job — evaluation and promotion happen automatically in the background.

### Custom Scoring Weights

Override the default composite scoring formula per-request:

```json
{
  "model": "security-cam-v2",
  "dataset": "/data/security.yaml",
  "task": "detection",
  "config": {"epochs": 20},
  "auto_eval": true,
  "auto_promote": true,
  "eval_weights": {
    "mAP50_95": 0.20,
    "mAP50": 0.15,
    "small_object_recall": 0.10,
    "f1": 0.15,
    "speed": 0.40
  }
}
```

Use cases for different weight profiles:

| Use Case | Prioritize | Example Weights |
|----------|-----------|-----------------|
| **Aerial/drone** | Small objects | `small_object_recall: 0.40` |
| **Security camera** | Speed + accuracy | `speed: 0.40, mAP50: 0.30` |
| **Quality inspection** | Precision | `mAP50_95: 0.50, f1: 0.30` |
| **Real-time edge** | Speed | `speed: 0.60` |

## Metrics Reference

### Core COCO Metrics

| Metric | Description |
|--------|-------------|
| `mAP50` | Mean Average Precision at IoU 0.50 |
| `mAP50_95` | Mean Average Precision at IoU 0.50:0.95 (stricter) |
| `precision` | Mean precision across classes |
| `recall` | Mean recall across classes |
| `f1` | Harmonic mean of precision and recall |
| `per_class` | Per-class mAP50 breakdown |

### Size-Based Recall

Computed from actual prediction matching against ground truth, not approximations.

| Metric | Object Size | Use Case |
|--------|------------|----------|
| `small_object_recall` | < 32×32 px | Aerial, drone, satellite |
| `medium_object_recall` | 32–96 px | Street-level, dashcam |
| `large_object_recall` | > 96 px | Indoor, close-range |

### Speed Metrics

| Metric | Description |
|--------|-------------|
| `inference_ms` | Per-image inference time |
| `preprocess_ms` | Per-image preprocessing time |
| `postprocess_ms` | Per-image NMS/postprocessing time |

### Composite Score

Default formula:

```
score = 0.35 × mAP50-95
      + 0.20 × mAP50
      + 0.20 × small_object_recall
      + 0.15 × F1
      + 0.10 × speed_score
```

`speed_score` is 1.0 at ≤50ms inference, 0.0 at ≥500ms, linearly interpolated.

## Storage

### Leaderboard Database

All evaluation results are stored in SQLite:

```
~/.llamafarm/models/vision/eval.db
```

Tables:
- `eval_results` — all metrics from every evaluation run
- `promotions` — audit trail of auto-promoted models (eval_id, timestamp, reason)

The database uses WAL mode for concurrent read/write safety.

### Model Resolution

When you pass a model name (not a path), the evaluator looks up:

```
~/.llamafarm/models/vision/{model_name}/
├── current.pt          ← preferred (active version)
├── v1.pt               ← fallback: latest versioned
├── v2.pt
├── v1.onnx
└── pipeline.json
```

## Python Client Example

```python
import httpx
import time

LLAMAFARM = "http://localhost:14345"

# Start evaluation
resp = httpx.post(f"{LLAMAFARM}/v1/vision/eval", data={
    "model": "my-detector",
    "dataset": "/data/coco-subset.yaml",
    "imgsz": 640,
})
job_id = resp.json()["job_id"]

# Poll until complete
while True:
    status = httpx.get(f"{LLAMAFARM}/v1/vision/eval/{job_id}").json()
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(5)

if status["status"] == "completed":
    result = status["result"]
    print(f"mAP50: {result['mAP50']:.3f}")
    print(f"Score: {result['score']:.4f}")
    print(f"Small obj recall: {result['small_object_recall']:.3f}")

# Check leaderboard
lb = httpx.get(f"{LLAMAFARM}/v1/vision/eval/leaderboard").json()
for entry in lb:
    print(f"#{entry['rank']} {entry['model_name']}: {entry['score']:.4f}")
```

## Train→Eval→Promote Example

```python
import httpx
import time

LLAMAFARM = "http://localhost:14345"

# Single request: train + auto-eval + auto-promote
resp = httpx.post(f"{LLAMAFARM}/v1/vision/train", data={
    "model": "aerial-v4",
    "dataset": "/data/visdrone.yaml",
    "task": "detection",
    "epochs": 20,
    "batch_size": 8,
    "auto_eval": "true",
    "auto_promote": "true",
})
job_id = resp.json()["job_id"]

# Just wait for training — eval + promote happen automatically
while True:
    status = httpx.get(f"{LLAMAFARM}/v1/vision/train/{job_id}").json()
    print(f"Status: {status['status']} Progress: {status.get('progress', 0):.0%}")
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(10)

# Check what got promoted
lb = httpx.get(f"{LLAMAFARM}/v1/vision/eval/leaderboard").json()
print(f"Current best: {lb[0]['model_name']} (score: {lb[0]['score']:.4f})")
```

## Dataset Format

Evaluation expects standard ultralytics dataset YAML:

```yaml
path: /data/my-dataset
train: images/train
val: images/val

nc: 3
names:
  0: person
  1: car
  2: bicycle
```

The evaluator runs against the `val` split. For size-based recall, it also reads YOLO-format labels from the corresponding `labels/` directory.

## Dependencies

No additional pip packages required. Uses:
- `ultralytics` (already required for vision training/detection)
- `sqlite3` (Python standard library)
