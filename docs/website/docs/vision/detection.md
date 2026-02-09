---
title: Detection, Classification & Embeddings
sidebar_position: 1
sidebar_label: Detection & Classification
---

# Detection, Classification & Embeddings

Single-frame inference endpoints for object detection, image classification, segmentation, and CLIP embeddings. All endpoints run on the Universal Runtime (port 11540).

## Object Detection

Detect objects in images using YOLO models.

**Endpoint:** `POST /v1/vision/detect`

```bash
curl -X POST http://localhost:11540/v1/vision/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n",
    "images": ["data:image/png;base64,..."],
    "confidence_threshold": 0.5,
    "classes": ["person", "car"]
  }'
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | Yes | - | YOLO model ID (e.g., `yolov8n`, `yolov8m`, `yolov8x`) |
| `images` | list[string] | Yes | - | Base64-encoded images |
| `confidence_threshold` | float | No | `0.5` | Minimum detection confidence |
| `classes` | list[string] | No | all | Filter to specific class names |

**Response:**

```json
{
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.92,
      "bbox": [120.5, 80.3, 340.2, 410.7]
    }
  ],
  "model": "yolov8n",
  "inference_time_ms": 12.4
}
```

### Supported Models

Any YOLO-compatible model works. Common choices:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolov8n` | 6 MB | ~5ms | Good | Real-time, edge devices |
| `yolov8s` | 22 MB | ~10ms | Better | Balanced |
| `yolov8m` | 50 MB | ~25ms | High | General purpose |
| `yolov8l` | 83 MB | ~50ms | Higher | When accuracy matters |
| `yolov8x` | 131 MB | ~80ms | Highest | Audit, validation |

## Image Classification

Zero-shot image classification using CLIP models. Classify images into arbitrary categories without training.

**Endpoint:** `POST /v1/vision/classify`

```bash
curl -X POST http://localhost:11540/v1/vision/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/clip-vit-base-patch32",
    "images": ["data:image/png;base64,..."],
    "labels": ["cat", "dog", "bird", "fish"]
  }'
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | Yes | - | CLIP model ID from HuggingFace |
| `images` | list[string] | Yes | - | Base64-encoded images |
| `labels` | list[string] | Yes | - | Candidate class labels |

## Segmentation

Instance and semantic segmentation to get pixel-level object boundaries.

**Endpoint:** `POST /v1/vision/segment`

```bash
curl -X POST http://localhost:11540/v1/vision/segment \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n-seg",
    "images": ["data:image/png;base64,..."],
    "confidence_threshold": 0.5
  }'
```

Segmentation masks are returned as polygon coordinates or run-length encoded (RLE) format. In the streaming pipeline, segmentation masks automatically attach to detections when `enrich_on_escalation` is enabled.

## CLIP Embeddings

Generate embeddings for images and/or text using CLIP models. Useful for multimodal RAG, image search, and similarity matching.

**Endpoint:** `POST /v1/vision/embed`

```bash
# Embed images
curl -X POST http://localhost:11540/v1/vision/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/clip-vit-base-patch32",
    "images": ["data:image/png;base64,..."]
  }'

# Embed text
curl -X POST http://localhost:11540/v1/vision/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/clip-vit-base-patch32",
    "texts": ["a photo of a cat", "a painting of a dog"]
  }'
```

Image and text embeddings live in the same vector space, so you can search images by text description and vice versa.

## Model Management

List, load, save, and delete vision models.

```bash
# List available models
curl http://localhost:11540/v1/vision/models

# Save a model to disk
curl -X POST http://localhost:11540/v1/vision/models/save \
  -H "Content-Type: application/json" \
  -d '{"model": "yolov8n", "task": "detection", "name": "my-detector"}'

# Load a saved model
curl -X POST http://localhost:11540/v1/vision/models/load \
  -H "Content-Type: application/json" \
  -d '{"task": "detection", "name": "my-detector"}'

# Delete a saved model
curl -X DELETE http://localhost:11540/v1/vision/models/detection/my-detector
```
