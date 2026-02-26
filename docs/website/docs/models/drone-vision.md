---
title: Drone Vision API
sidebar_position: 8
---

# Drone Vision API

Drone Vision adds real-time aerial inference workflows on top of the core vision pipeline. It is designed for live drone feeds, telemetry-aware georeferencing, and export to edge runtimes (Jetson, ONNX, TFLite, etc.).

## What This Covers

- **WebSocket streaming inference** for frame-by-frame detection
- **Telemetry fusion** (GPS/altitude/heading/gimbal)
- **Geo-referencing** detections into latitude/longitude
- **Edge export profiles** for common drone deployment targets
- **Operational endpoints** for active sessions and training presets

## Base URL

Drone Vision HTTP endpoints are exposed on both:

- `http://localhost:14345` (LlamaFarm server proxy)
- `http://localhost:11540` (Universal Runtime direct)

Streaming WebSocket is currently runtime-direct:

- `ws://localhost:11540/v1/vision/drone/stream`

---

## 1) Streaming Inference

### `WS /v1/vision/drone/stream`

Open a WebSocket session for continuous frame inference.

### Protocol

1. Connect to `/v1/vision/drone/stream`
2. Send one JSON config message
3. Send frame messages (binary image or JSON with base64 image + telemetry)
4. Receive per-frame detections and optional geo detections
5. Send `{"action":"stop"}` or disconnect

### Initial Config Message

```json
{
  "model": "yolov8n",
  "confidence": 0.25,
  "classes": ["person", "car"],
  "camera": {
    "focal_length_mm": 4.5,
    "sensor_width_mm": 6.17,
    "sensor_height_mm": 4.55,
    "image_width_px": 3840,
    "image_height_px": 2160
  }
}
```

### Frame Message (JSON)

```json
{
  "image": "<base64-image>",
  "telemetry": {
    "latitude": 30.2672,
    "longitude": -97.7431,
    "altitude_m": 72.4,
    "heading_deg": 184.0,
    "gimbal_pitch_deg": -90.0,
    "speed_mps": 11.2,
    "timestamp": "2026-02-24T19:10:01Z"
  }
}
```

You can also send raw binary frame bytes (JPEG/PNG) instead of JSON.

### Ready Response

```json
{
  "session_id": "a1b2c3d4",
  "status": "ready"
}
```

### Per-frame Response

```json
{
  "frame_id": 42,
  "detections": [
    {
      "x1": 102.2,
      "y1": 318.7,
      "x2": 144.9,
      "y2": 366.1,
      "class_name": "person",
      "class_id": 0,
      "confidence": 0.9132
    }
  ],
  "geo_detections": [
    {
      "x1": 102.2,
      "y1": 318.7,
      "x2": 144.9,
      "y2": 366.1,
      "class_name": "person",
      "class_id": 0,
      "confidence": 0.9132,
      "lat": 30.26719488,
      "lng": -97.74309121,
      "alt_m": 72.4
    }
  ],
  "latency_ms": 18.4,
  "telemetry_echo": {
    "latitude": 30.2672,
    "longitude": -97.7431,
    "altitude_m": 72.4,
    "heading_deg": 184.0,
    "gimbal_pitch_deg": -90.0
  }
}
```

### Session Limits

- Max **50 concurrent** WebSocket sessions
- Session list available via `GET /v1/vision/drone/sessions`

---

## 2) Geo-reference Detections

### `POST /v1/vision/drone/geo`

Projects detection boxes into approximate lat/lng using camera intrinsics and telemetry.

### Request

```json
{
  "detections": [
    {
      "x1": 100,
      "y1": 200,
      "x2": 220,
      "y2": 320,
      "class_name": "vehicle",
      "class_id": 2,
      "confidence": 0.88
    }
  ],
  "telemetry": {
    "latitude": 30.2672,
    "longitude": -97.7431,
    "altitude_m": 80,
    "heading_deg": 90,
    "gimbal_pitch_deg": -90
  },
  "camera": {
    "focal_length_mm": 4.5,
    "sensor_width_mm": 6.17,
    "sensor_height_mm": 4.55,
    "image_width_px": 3840,
    "image_height_px": 2160
  }
}
```

### Response

```json
{
  "geo_detections": [
    {
      "x1": 100,
      "y1": 200,
      "x2": 220,
      "y2": 320,
      "class_name": "vehicle",
      "class_id": 2,
      "confidence": 0.88,
      "lat": 30.26720312,
      "lng": -97.74308791,
      "alt_m": 80.0
    }
  ]
}
```

### Validation Rules

- `telemetry.latitude` and `telemetry.longitude` are required
- `telemetry.altitude_m` is required
- Returns HTTP `422` when required telemetry fields are missing

---

## 3) Export for Edge Deployment

### `GET /v1/vision/drone/export/profiles`

Returns built-in export presets.

Current preset names:

- `drone-nano`
- `drone-standard`
- `drone-hd`
- `edge-onnx`
- `edge-tflite`

### `POST /v1/vision/drone/export`

Export a trained model for edge deployment.

### Request

```json
{
  "model_id": "drone-aerial-general",
  "target": "onnx",
  "precision": "fp16",
  "imgsz": 640,
  "profile": "edge-onnx",
  "calibration_dataset": null
}
```

### Field Notes

| Field | Type | Description |
|---|---|---|
| `model_id` | string | Vision model ID under `~/.llamafarm/models/vision/{model_id}` |
| `target` | enum | `onnx`, `tensorrt`, `coreml`, `tflite`, `openvino` |
| `precision` | enum | `fp32`, `fp16`, `int8` |
| `imgsz` | int | Input size, range `128..2048` |
| `profile` | string \| null | Optional profile override |
| `calibration_dataset` | string \| null | Optional INT8 calibration data path |

### Response

```json
{
  "model_id": "drone-aerial-general",
  "export_path": "/Users/me/.llamafarm/models/vision/drone-aerial-general/exports/best.onnx",
  "target": "onnx",
  "precision": "fp16",
  "imgsz": 640,
  "size_mb": 24.8,
  "export_time_seconds": 6.41,
  "profile": "edge-onnx"
}
```

### Errors

- `400` invalid model ID or unknown profile
- `404` model not found
- `500` export runtime failure (for TensorRT, verify CUDA/TensorRT availability)

---

## 4) Operational Endpoints

### `GET /v1/vision/drone/train/presets`

Returns preset configurations intended for aerial/orthomosaic workflows:

- `aerial`
- `aerial-nano`
- `orthomosaic`

These are reference presets (image size, augmentation, epochs, batch size) to help build training requests.

### `GET /v1/vision/drone/sessions`

Lists active streaming sessions with:

- `session_id`
- `model`
- `frames_processed`
- `total_detections`
- `avg_latency_ms`
- `duration_seconds`

---

## Minimal Python Streaming Example

```python
import asyncio
import base64
import json
import websockets

RUNTIME_WS = "ws://localhost:11540/v1/vision/drone/stream"

async def main():
    async with websockets.connect(RUNTIME_WS, max_size=8_000_000) as ws:
        await ws.send(json.dumps({
            "model": "yolov8n",
            "confidence": 0.3,
            "classes": ["person", "car"]
        }))

        ready = json.loads(await ws.recv())
        print("session:", ready)

        with open("frame.jpg", "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        await ws.send(json.dumps({
            "image": image_b64,
            "telemetry": {
                "latitude": 30.2672,
                "longitude": -97.7431,
                "altitude_m": 75.0,
                "heading_deg": 175,
                "gimbal_pitch_deg": -90
            }
        }))

        result = json.loads(await ws.recv())
        print(result)

        await ws.send(json.dumps({"action": "stop"}))

asyncio.run(main())
```
