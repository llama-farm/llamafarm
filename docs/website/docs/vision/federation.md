---
title: Federation & Distribution
sidebar_position: 4
---

# Federation & Distribution

Federation connects multiple LlamaFarm instances so they can share model inference, distribute trained models, and route vision work to the best available GPU.

## Concepts

**Peers** -- Other LlamaFarm instances that can run vision models. On standalone, you configure peers manually. On mesh (Atmosphere), peers are discovered automatically.

**Remote Model Proxy** -- A `DetectionModel` that calls `/v1/vision/federation/escalate` on a remote LlamaFarm instance. The cascade chain doesn't know or care whether a model is local or remote.

**Model Packages** -- Portable `.tar.gz` bundles containing model weights, metadata, class maps, and validation metrics. Used to distribute trained models across nodes.

## Peer Management

```bash
# List federation peers
curl http://localhost:11540/v1/vision/federation/peers

# Register a peer
curl -X POST http://localhost:11540/v1/vision/federation/peers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-server",
    "url": "http://192.168.1.100:11540",
    "models": ["yolov8x"],
    "gpu_vram_gb": 24,
    "priority": 0,
    "timeout": 30.0
  }'

# Remove a peer
curl -X DELETE http://localhost:11540/v1/vision/federation/peers/gpu-server

# Check federation health
curl http://localhost:11540/v1/vision/federation/status
```

## Inbound Escalation

When another LlamaFarm instance is uncertain about a detection, it sends the full escalation envelope to this endpoint. The local node runs its model and returns its opinion.

**Endpoint:** `POST /v1/vision/federation/escalate`

```bash
curl -X POST http://localhost:11540/v1/vision/federation/escalate \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64-encoded>",
    "model": "yolov8x",
    "confidence_threshold": 0.5,
    "opinions": [
      {
        "model_id": "yolov8n",
        "node_id": "edge-device-1",
        "class_name": "bird",
        "confidence": 0.45
      }
    ]
  }'
```

**Response:**

```json
{
  "model": "yolov8x",
  "detections": [
    {
      "class_name": "eagle",
      "confidence": 0.91,
      "bbox": [100, 50, 300, 250]
    }
  ],
  "inference_time_ms": 82.3,
  "node_id": "gpu-server"
}
```

## Using Remote Models in the Cascade

Add remote models to a streaming session's cascade chain by prefixing with `remote:`:

```bash
curl -X POST http://localhost:11540/v1/vision/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n",
    "cascade": {
      "enabled": true,
      "cascade_chain": [
        "yolov8n",
        "remote:gpu-server/yolov8x"
      ],
      "confidence_threshold": 0.7
    }
  }'
```

The `RemoteModelProxy` implements the same `DetectionModel` interface as local models. The cascade loop calls `.detect()` on each -- it doesn't distinguish local from remote.

## Model Packages

Package trained models for distribution to other nodes.

### List Packages

```bash
curl http://localhost:11540/v1/vision/federation/packages
```

### Create a Package

```bash
curl -X POST http://localhost:11540/v1/vision/federation/packages \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "bird-detector-v3",
    "model_path": "/path/to/best.pt",
    "description": "Fine-tuned bird detector"
  }'
```

### Import a Package

```bash
curl -X POST http://localhost:11540/v1/vision/federation/packages/import \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/path/to/bird-detector-v3.tar.gz"
  }'
```

### Package Format

Packages are `.tar.gz` archives containing:

```
package.tar.gz
├── model.pt                  # Model weights
├── metadata.json             # Model ID, version, base model, class names, etc.
├── class_map.json            # {class_name: index} mapping
└── validation_metrics.json   # Accuracy, mAP, per-class metrics
```

Packages are created automatically when the validation gate promotes a new model. On mesh deployments, packages are announced via Atmosphere gossip so peers can pull updated models.
