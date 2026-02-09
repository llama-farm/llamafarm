---
title: Streaming & Cascade
sidebar_position: 2
---

# Streaming & Cascade

The streaming pipeline processes video frames in real-time with an automatic multi-hop cascade. When a model is uncertain, the detection escalates to bigger models with full context -- bounding boxes, segmentation masks, and every prior model's opinion.

## Streaming Sessions

### Start a Session

**Endpoint:** `POST /v1/vision/stream/start`

```bash
curl -X POST http://localhost:14345/v1/vision/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n",
    "cascade": {
      "enabled": true,
      "cascade_chain": ["yolov8n", "yolov8m"],
      "confidence_threshold": 0.7,
      "enrich_on_escalation": true,
      "segmentation_model_id": "yolov8n-seg",
      "classification_model_id": "openai/clip-vit-base-patch32"
    }
  }'
```

**Cascade Configuration:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable multi-hop cascade |
| `cascade_chain` | list[string] | `[]` | Ordered list of model IDs to try |
| `confidence_threshold` | float | `0.7` | Minimum confidence to accept a detection |
| `enrich_on_escalation` | bool | `true` | Attach segmentation masks and CLIP labels before escalating |
| `segmentation_model_id` | string | `null` | Model for bbox segmentation enrichment |
| `classification_model_id` | string | `null` | CLIP model for classification enrichment |

### Process Frames

**Endpoint:** `POST /v1/vision/stream/frame`

```bash
curl -X POST http://localhost:14345/v1/vision/stream/frame \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "image": "data:image/png;base64,..."
  }'
```

**Response:**

```json
{
  "detections": [
    {
      "class_name": "bird",
      "confidence": 0.85,
      "bbox": [100, 50, 300, 250],
      "model": "yolov8m"
    }
  ],
  "hop_count": 2,
  "cascade_resolved_by": "yolov8m",
  "inference_time_ms": 45.2
}
```

### Stop a Session

**Endpoint:** `POST /v1/vision/stream/stop`

```bash
curl -X POST http://localhost:14345/v1/vision/stream/stop \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123"}'
```

### List Sessions and Stats

```bash
# List active sessions
curl http://localhost:14345/v1/vision/stream/sessions

# Get session statistics
curl http://localhost:14345/v1/vision/stream/sessions/abc123/stats
```

## How the Cascade Works

When a detection's confidence falls below the threshold, the pipeline escalates through the cascade chain:

```
Frame arrives
  │
  ├── Hop 0: yolov8n (fast, ~5ms)
  │   confidence >= 0.7 → DONE (return result)
  │   confidence < 0.7  → enrich bbox with seg + CLIP → Hop 1
  │
  ├── Hop 1: yolov8m (medium, ~25ms)
  │   confidence >= 0.7 → DONE + auto-feedback to replay buffer
  │   confidence < 0.7  → Hop 2
  │
  ├── Hop 2: yolov8x or remote model (~80ms+)
  │   confidence >= 0.5 → DONE + feedback to hop 0 AND hop 1
  │   confidence < 0.5  → REVIEW QUEUE (needs human eyes)
  │
  └── Max 3 hops (circuit breaker)
```

At each hop, the pipeline builds a `ModelOpinion` recording what that model saw. All opinions flow forward so the next model has full context.

### Cross-Modal Enrichment

When `enrich_on_escalation` is enabled, before escalating to the next model the pipeline:

1. Crops the bounding box region from the full image
2. Runs segmentation to get a pixel-level mask of the object
3. Runs CLIP classification to get candidate class labels
4. Packages everything into the escalation so the next model gets precise visual context

### Escalation Envelope

The data structure flowing through the cascade:

```
EscalationEnvelope
├── image_bytes          # Original full frame
├── image_hash           # For dedup
├── source_id            # Camera/source identifier
├── opinions[]           # What each model predicted
│   ├── model_id         # "yolov8n" or "remote:gpu-server/yolov8x"
│   ├── node_id          # "local" or Atmosphere node ID
│   ├── class_name       # What it thinks this is
│   ├── confidence       # How sure it is
│   ├── bbox             # Bounding box coordinates
│   └── mask_polygon     # Segmentation mask (if available)
├── detections[]         # Bounding boxes with crops and masks
├── hops                 # How many models have seen this
└── max_hops             # Circuit breaker (default: 3)
```

## Corrections and Replay Buffer

Submit human corrections for detections and inspect the replay buffer.

```bash
# Submit a correction
curl -X POST http://localhost:14345/v1/vision/corrections \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "img_abc123",
    "correct_class": "eagle",
    "bbox": [100, 50, 300, 250]
  }'

# Check replay buffer status
curl http://localhost:14345/v1/vision/replay-buffer

# Clear replay buffer
curl -X POST http://localhost:14345/v1/vision/replay-buffer/clear
```

The replay buffer stores training samples with priority weighting:

| Source | Priority | Description |
|--------|----------|-------------|
| Human correction | 2.0 | Manual review queue feedback |
| Audit disagreement | 2.0 | Bigger model disagrees with small model |
| Cascade resolved (hop 2+) | 1.8 | Multiple models needed to resolve |
| Escalation resolved (hop 1) | 1.5 | Single escalation resolved it |
| Low confidence | ~0.4 | Detection below threshold |

Samples persist to SQLite so they survive restarts. When the buffer hits the training threshold (default: 50 samples), the auto-trainer fires automatically.
