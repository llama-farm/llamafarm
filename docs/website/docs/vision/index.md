---
title: Vision Pipeline
sidebar_position: 8
---

# Vision Pipeline

LlamaFarm includes a complete vision pipeline for object detection, image classification, segmentation, CLIP embeddings, and OCR. The pipeline supports real-time streaming with an automatic learning loop that improves models over time with minimal human intervention.

## Capabilities

| Capability | Endpoint | Description |
|-----------|----------|-------------|
| [Object Detection](./detection) | `POST /v1/vision/detect` | Detect objects with YOLO models |
| [Image Classification](./detection#image-classification) | `POST /v1/vision/classify` | Classify images with CLIP zero-shot |
| [Segmentation](./detection#segmentation) | `POST /v1/vision/segment` | Instance/semantic segmentation |
| [CLIP Embeddings](./detection#clip-embeddings) | `POST /v1/vision/embed` | Image and text embeddings for multimodal RAG |
| [OCR](../models/specialized-ml#ocr-text-extraction) | `POST /v1/vision/ocr` | Extract text from images and PDFs |
| [Streaming](./streaming) | `POST /v1/vision/stream/*` | Real-time frame processing with cascade |
| [Training](./training) | `POST /v1/vision/train` | Fine-tune detection models |
| [Federation](./federation) | `/v1/vision/federation/*` | Multi-node model distribution |

## Architecture Overview

The vision system is organized into layers:

```
                        ┌─────────────────────────┐
                        │   Vision API Endpoints   │
                        │  detect / classify / ocr │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │   Streaming Pipeline     │
                        │  cascade chain + enrich  │
                        └────────────┬────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼─────────┐ ┌─────────▼─────────┐ ┌─────────▼─────────┐
    │   Local Models     │ │   Remote Models    │ │   Review Queue     │
    │  YOLO / CLIP / SAM │ │  RemoteModelProxy  │ │  Human feedback    │
    └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │    Replay Buffer         │
                        │  SQLite-backed samples   │
                        └────────────┬────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼─────────┐ ┌─────────▼─────────┐ ┌─────────▼─────────┐
    │   Auto Trainer     │ │   Audit Pipeline   │ │  Validation Gate   │
    │  periodic retrain  │ │  model-checks-model│ │  blue/green swap   │
    └───────────────────┘ └───────────────────┘ └───────────────────┘
```

## Deployment Modes

The vision pipeline works identically in two deployment modes:

**Standalone (laptop/desktop):** All models run locally. The cascade chain tries progressively larger models (e.g., yolov8n -> yolov8m -> yolov8x). Training happens on your GPU/CPU.

**Mesh/Edge (Atmosphere):** A tiny model runs on the edge device. Uncertain detections escalate to GPU peers discovered via mesh service discovery. The `RemoteModelProxy` wraps remote LlamaFarm instances with the same `DetectionModel` interface -- the cascade loop doesn't know or care whether a model is local or remote.

## The Learning Loop

The system is designed to be 99% automatic. The only human task is occasionally reviewing uncertain detections in the review queue.

1. **Detect** -- Fast model runs on every frame
2. **Escalate** -- Uncertain detections cascade to bigger models with full context (bounding boxes, segmentation masks, prior opinions)
3. **Feedback** -- Resolved detections automatically become training samples in the replay buffer
4. **Train** -- Auto-trainer fires when the replay buffer hits a threshold (default: 50 samples)
5. **Validate** -- Candidate model must beat current model on a held-out validation set
6. **Promote** -- Blue/green swap with automatic rollback if needed
7. **Audit** -- A bigger model periodically spot-checks the fast model's predictions, catching systematic errors

## Quick Start

```bash
# Start the Universal Runtime
nx start universal-runtime

# Detect objects in an image
curl -X POST http://localhost:14345/v1/vision/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n",
    "images": ["data:image/png;base64,..."],
    "confidence_threshold": 0.5
  }'

# Start a streaming session with cascade
curl -X POST http://localhost:14345/v1/vision/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n",
    "cascade": {
      "enabled": true,
      "cascade_chain": ["yolov8n", "yolov8m"],
      "confidence_threshold": 0.7
    }
  }'
```

## Next Steps

- [Detection, Classification & Embeddings](./detection) -- Single-frame inference endpoints
- [Streaming & Cascade](./streaming) -- Real-time processing with automatic learning
- [Training & Validation](./training) -- Auto-training, validation gate, model management
- [Federation & Distribution](./federation) -- Multi-node deployment and model packaging
