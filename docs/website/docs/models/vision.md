---
title: Vision Pipeline Guide
sidebar_position: 5
---

# Vision Pipeline Guide

The Universal Runtime provides a complete computer vision pipeline with YOLO object detection, CLIP zero-shot classification, cascade streaming inference, model training, and ONNX export.

## Overview

The vision pipeline supports:

- **YOLO Object Detection**: Detect and locate objects in images with bounding boxes
- **CLIP Classification**: Zero-shot image classification with custom class labels
- **Cascade Streaming**: Multi-model chains that escalate when confidence is low
- **Training**: Fine-tune detection and classification models on custom datasets
- **Model Management**: Save, load, list, and export models (ONNX, CoreML, TensorRT)

## Quick Start

### Detect Objects

```bash
# Base64-encode an image and detect objects
IMAGE=$(base64 -w0 photo.jpg)

curl -X POST http://localhost:11540/v1/vision/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$IMAGE'",
    "model": "yolov8n",
    "confidence_threshold": 0.5
  }'
```

Response:
```json
{
  "detections": [
    {
      "box": {"x1": 120.5, "y1": 80.2, "x2": 350.8, "y2": 420.1},
      "class_name": "person",
      "class_id": 0,
      "confidence": 0.92
    },
    {
      "box": {"x1": 400.0, "y1": 200.0, "x2": 550.0, "y2": 380.0},
      "class_name": "dog",
      "class_id": 16,
      "confidence": 0.87
    }
  ],
  "model": "yolov8n",
  "inference_time_ms": 45.2
}
```

### Classify Images

```bash
curl -X POST http://localhost:11540/v1/vision/classify \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$IMAGE'",
    "model": "clip-vit-base",
    "classes": ["cat", "dog", "bird", "car", "person"],
    "top_k": 3
  }'
```

Response:
```json
{
  "class_name": "dog",
  "class_id": 1,
  "confidence": 0.89,
  "all_scores": {
    "cat": 0.05,
    "dog": 0.89,
    "bird": 0.02,
    "car": 0.01,
    "person": 0.03
  },
  "model": "clip-vit-base",
  "inference_time_ms": 32.1
}
```

---

## Object Detection (YOLO)

### `POST /v1/vision/detect`

Detect objects in an image using YOLO models. Returns bounding boxes with class labels and confidence scores.

**Request:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image` | string | Yes | — | Base64-encoded image |
| `model` | string | No | `yolov8n` | YOLO model variant |
| `confidence_threshold` | float | No | `0.5` | Minimum confidence (0.0–1.0) |
| `classes` | string[] | No | all | Filter to specific class names |

**Available Models:**

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `yolov8n` | Fastest | Good | Real-time, edge devices |
| `yolov8s` | Fast | Better | Balanced performance |
| `yolov8m` | Medium | High | General purpose |
| `yolov8l` | Slow | Higher | High accuracy needs |
| `yolov8x` | Slowest | Highest | Maximum accuracy |

### Python Example

```python
import base64
import requests

with open("photo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:11540/v1/vision/detect", json={
    "image": image_b64,
    "model": "yolov8n",
    "confidence_threshold": 0.5,
    "classes": ["person", "car"]
})

for det in resp.json()["detections"]:
    print(f"{det['class_name']}: {det['confidence']:.2f} at ({det['box']['x1']:.0f},{det['box']['y1']:.0f})")
```

---

## Zero-Shot Classification (CLIP)

### `POST /v1/vision/classify`

Classify images into arbitrary categories without training using CLIP. You provide the class labels at inference time.

**Request:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image` | string | Yes | — | Base64-encoded image |
| `model` | string | No | `clip-vit-base` | CLIP model variant |
| `classes` | string[] | Yes | — | Class labels for zero-shot classification |
| `top_k` | int | No | `5` | Number of top results (1–100) |

### Python Example

```python
resp = requests.post("http://localhost:11540/v1/vision/classify", json={
    "image": image_b64,
    "model": "clip-vit-base",
    "classes": ["defective product", "good product", "packaging damage"],
    "top_k": 3
})

result = resp.json()
print(f"Classification: {result['class_name']} ({result['confidence']:.2%})")
```

---

## Cascade Streaming

Cascade streaming processes frames through a chain of models, escalating to more powerful (or remote) models when confidence is low. This is ideal for real-time monitoring where you want fast inference most of the time but accuracy on difficult frames.

### Start a Session

```bash
curl -X POST http://localhost:11540/v1/vision/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "chain": ["yolov8n", "yolov8m"],
      "confidence_threshold": 0.7
    },
    "target_fps": 1.0,
    "action_classes": ["person", "vehicle"],
    "cooldown_seconds": 5.0
  }'
```

Response:
```json
{"session_id": "a1b2c3d4"}
```

### Process Frames

```bash
curl -X POST http://localhost:11540/v1/vision/stream/frame \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "a1b2c3d4",
    "image": "'$IMAGE'"
  }'
```

Response (confident detection):
```json
{
  "status": "action",
  "detections": [
    {"x1": 100, "y1": 50, "x2": 300, "y2": 400, "class_name": "person", "class_id": 0, "confidence": 0.85}
  ],
  "confidence": 0.85,
  "resolved_by": "yolov8n"
}
```

Response (escalated to larger model):
```json
{
  "status": "escalated",
  "detections": [...],
  "confidence": 0.78,
  "resolved_by": "yolov8m"
}
```

Response (no confident detection):
```json
{"status": "ok"}
```

### Stop Session

```bash
curl -X POST http://localhost:11540/v1/vision/stream/stop \
  -H "Content-Type: application/json" \
  -d '{"session_id": "a1b2c3d4"}'
```

Response:
```json
{
  "session_id": "a1b2c3d4",
  "frames_processed": 150,
  "actions_triggered": 12,
  "escalations": 3,
  "duration_seconds": 152.4
}
```

### Cascade Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chain` | string[] | `["yolov8n"]` | Models to try in order. Can include `remote:http://...` for remote models |
| `confidence_threshold` | float | `0.7` | Minimum confidence before escalating to next model |
| `target_fps` | float | `1.0` | Target frame processing rate |
| `action_classes` | string[] | all | Filter detections to specific classes |
| `cooldown_seconds` | float | `5.0` | Minimum seconds between action triggers |

:::tip Remote Cascade
You can include remote models in the chain for Atmosphere mesh integration:
```json
{"chain": ["yolov8n", "remote:http://gpu-server:11540/v1/vision/detect"]}
```
Remote hosts must be in the configured allowlist (SSRF protection).
:::

### Python Streaming Example

```python
import base64
import time
import requests

BASE = "http://localhost:11540"

# Start session
session = requests.post(f"{BASE}/v1/vision/stream/start", json={
    "config": {"chain": ["yolov8n", "yolov8m"], "confidence_threshold": 0.7},
    "action_classes": ["person"],
    "cooldown_seconds": 2.0
}).json()

sid = session["session_id"]

# Process frames (e.g., from a camera)
for frame_bytes in camera_frames():
    image_b64 = base64.b64encode(frame_bytes).decode()
    result = requests.post(f"{BASE}/v1/vision/stream/frame", json={
        "session_id": sid,
        "image": image_b64
    }).json()

    if result["status"] in ("action", "escalated"):
        print(f"Detected: {result['detections']} (by {result['resolved_by']})")

# Stop session
stats = requests.post(f"{BASE}/v1/vision/stream/stop", json={"session_id": sid}).json()
print(f"Processed {stats['frames_processed']} frames, {stats['actions_triggered']} actions")
```

---

## Training

Fine-tune detection or classification models on your own datasets.

### Start Training

```bash
curl -X POST http://localhost:11540/v1/vision/train \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-detector",
    "dataset": "/path/to/dataset",
    "task": "detection",
    "config": {
      "epochs": 50,
      "batch_size": 16,
      "learning_rate": 0.001
    },
    "base_model": "yolov8n"
  }'
```

Response:
```json
{
  "job_id": "train-abc123",
  "status": "running",
  "progress": 0.0,
  "metrics": null
}
```

### Check Training Status

```bash
curl http://localhost:11540/v1/vision/train/train-abc123
```

Response:
```json
{
  "job_id": "train-abc123",
  "status": "running",
  "progress": 0.65,
  "current_epoch": 33,
  "total_epochs": 50,
  "metrics": {
    "mAP50": 0.82,
    "mAP50-95": 0.61,
    "loss": 0.034
  },
  "error": null
}
```

### Cancel Training

```bash
curl -X DELETE http://localhost:11540/v1/vision/train/train-abc123
```

### Training Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | Yes | — | Name for the trained model |
| `dataset` | string | Yes | — | Path to training dataset |
| `task` | string | Yes | — | `detection` or `classification` |
| `config.epochs` | int | No | `10` | Training epochs (1–1000) |
| `config.batch_size` | int | No | `16` | Batch size (1–256) |
| `config.learning_rate` | float | No | `0.001` | Learning rate |
| `base_model` | string | No | — | Pre-trained model to fine-tune from |

---

## Model Management

### List Models

```bash
curl http://localhost:11540/v1/vision/models
```

Response:
```json
{
  "models": [
    {
      "name": "my-detector",
      "source_model_id": "yolov8n",
      "versions": 3,
      "has_current": true,
      "size_mb": 12.4
    }
  ],
  "total": 1
}
```

### Save a Model

```bash
curl -X POST "http://localhost:11540/v1/vision/models/save?model_id=yolov8n&name=production-detector"
```

### Load a Model

```bash
curl -X POST "http://localhost:11540/v1/vision/models/load?name=production-detector"
```

### Export Model

Export models to optimized formats for deployment:

```bash
curl -X POST http://localhost:11540/v1/vision/models/export \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "my-detector",
    "format": "onnx",
    "quantization": "fp16"
  }'
```

Response:
```json
{
  "export_path": "/models/exports/my-detector.onnx",
  "format": "onnx",
  "size_mb": 6.2,
  "export_time_seconds": 3.45
}
```

**Export Formats:**

| Format | Description | Best For |
|--------|-------------|----------|
| `onnx` | Open Neural Network Exchange | Cross-platform deployment |
| `coreml` | Apple Core ML | iOS/macOS apps |
| `tensorrt` | NVIDIA TensorRT | NVIDIA GPU inference |
| `tflite` | TensorFlow Lite | Mobile/edge devices |
| `openvino` | Intel OpenVINO | Intel hardware |

**Quantization Options:**

| Option | Description |
|--------|-------------|
| `fp32` | Full precision (largest, highest quality) |
| `fp16` | Half precision (good balance) |
| `int8` | 8-bit integer (smallest, fastest) |

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/vision/detect` | POST | Detect objects with YOLO |
| `/v1/vision/classify` | POST | Classify images with CLIP |
| `/v1/vision/stream/start` | POST | Start cascade streaming session |
| `/v1/vision/stream/frame` | POST | Process a frame in a session |
| `/v1/vision/stream/stop` | POST | Stop a streaming session |
| `/v1/vision/train` | POST | Start a training job |
| `/v1/vision/train/{job_id}` | GET | Get training job status |
| `/v1/vision/train/{job_id}` | DELETE | Cancel a training job |
| `/v1/vision/models` | GET | List saved models |
| `/v1/vision/models/save` | POST | Save a model |
| `/v1/vision/models/load` | POST | Load a saved model |
| `/v1/vision/models/export` | POST | Export to ONNX/CoreML/etc. |

---

## Next Steps

- [Specialized ML Models](./specialized-ml.md) — OCR, document extraction, and more
- [ML Addons](./ml-addons.md) — Time-series forecasting, drift detection, CatBoost
- [Anomaly Detection Guide](./anomaly-detection.md) — Outlier detection for monitoring
