---
title: Training & Validation
sidebar_position: 3
---

# Training & Validation

The vision pipeline includes automatic model retraining with a validation gate that prevents regressions. Models must prove they're better before going live.

## Auto-Training

When the replay buffer accumulates enough training samples (default: 50), the auto-trainer fires automatically. No manual intervention needed.

### How It Works

1. **Collect** -- Cascade resolutions, human corrections, and audit disagreements fill the replay buffer
2. **Dataset** -- Auto-trainer builds a YOLO-format dataset with proper class maps and 80/20 train/val split
3. **Train** -- Fine-tunes a candidate model for a few epochs with EWC (Elastic Weight Consolidation) to prevent catastrophic forgetting
4. **Validate** -- Candidate runs against a held-out validation set of human-verified and audit-verified images
5. **Promote or Reject** -- If the candidate beats the current model, blue/green swap. Otherwise, reject and keep current.

### Check Auto-Train Status

```bash
curl http://localhost:14345/v1/vision/auto-train/status
```

### Trigger Training Manually

```bash
curl -X POST http://localhost:14345/v1/vision/auto-train/trigger
```

## Training API

For manual training control:

```bash
# Start a training job
curl -X POST http://localhost:14345/v1/vision/train \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n",
    "dataset_path": "/path/to/dataset",
    "epochs": 10,
    "batch_size": 16
  }'

# Check training status
curl http://localhost:14345/v1/vision/train/job_abc123

# List all training jobs
curl http://localhost:14345/v1/vision/train

# Cancel a training job
curl -X DELETE http://localhost:14345/v1/vision/train/job_abc123
```

## Validation Gate

The validation gate is the safety net between training and deployment. A candidate model must demonstrate improvement on known-good images before it replaces the current model.

### Validation Set Sources

The gate automatically builds a validation set from two sources:

- **Human-verified images** -- Images where a human confirmed the correct label via the review queue (highest trust)
- **Audit-verified images** -- Images where the audit pipeline's larger model agreed with the primary model's prediction

### Blue/Green Promotion

When validation passes:

1. The current model is backed up as `{model_id}_v{N}.pt`
2. The candidate is installed as the new current model
3. A few live frames run through both models as a final check
4. The old model stays available for rollback

### Rollback

If a promoted model misbehaves, the system can roll back to the previous version:

- Automatic rollback if live-frame validation fails during promotion
- Manual rollback via the model management API

## Audit Pipeline

A bigger model periodically reviews what the fast model has been classifying with high confidence. This catches systematic errors the cascade missed.

### How It Works

1. Sample N recent high-confidence predictions (the ones that were NOT escalated)
2. Re-run through a larger model (local or remote via `RemoteModelProxy`)
3. Compare results:
   - **Agreement** -- Mark as verified (becomes validation data)
   - **Disagreement** -- Add to replay buffer as training sample (priority 2.0)

The audit model can be any model -- a larger local YOLO model, or a `RemoteModelProxy` pointing at a GPU server. The pipeline doesn't distinguish between local and remote.

## Dataset Format

The auto-trainer produces YOLO-format datasets:

```
dataset/
├── data.yaml          # Class names, paths
├── class_map.json     # {class_name: index}
├── train/
│   ├── images/        # Training images
│   └── labels/        # YOLO-format .txt labels
└── val/
    ├── images/        # Validation images (20%)
    └── labels/        # YOLO-format .txt labels
```

Each label file contains one line per detection:

```
# class_index center_x center_y width height (all normalized 0-1)
0 0.45 0.52 0.30 0.60
1 0.72 0.31 0.15 0.25
```
