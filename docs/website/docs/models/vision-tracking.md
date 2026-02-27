---
title: Vision Object Tracking
sidebar_position: 7
---

# Vision Object Tracking

LlamaFarm provides server-side multi-object tracking with persistent IDs across frames. Built on ultralytics tracking (ByteTrack, BoT-SORT, OC-SORT), each detection gets a stable `track_id` that persists as long as the object is visible.

## Quick Start

```python
import httpx
import base64

LLAMAFARM = "http://localhost:14345"

# Start a tracking session (optionally include first frame)
resp = httpx.post(f"{LLAMAFARM}/v1/vision/track/start", data={
    "model": "drone-aerial-general",
    "tracker": "bytetrack",
})
session_id = resp.json()["session_id"]

# Send frames — each detection has a persistent track_id
with open("frame001.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

resp = httpx.post(f"{LLAMAFARM}/v1/vision/track/frame", data={
    "session_id": session_id,
    "image": image_b64,
})
for det in resp.json()["detections"]:
    print(f"Track {det['track_id']}: {det['class_name']} ({det['confidence']:.2f})")

# Stop when done
httpx.post(f"{LLAMAFARM}/v1/vision/track/stop", data={
    "session_id": session_id,
})
```

## Tracker Algorithms

| Tracker | Description | Best For |
|---------|-------------|----------|
| `bytetrack` | IoU-based, handles low-confidence detections | General purpose, fast |
| `botsort` | ByteTrack + camera motion compensation | Moving cameras, drones |
| `ocsort` | Observation-centric, handles occlusion | Crowded scenes |

All three are provided by ultralytics and use Kalman filters for state prediction. No GPU required for tracking — it runs on detection bounding boxes.

## API Reference

### POST /v1/vision/track/start

Start a tracking session. Each session loads its own YOLO model with independent tracker state.

**Parameters (Form data):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | Model name or path to `.pt` file |
| `tracker` | string | `bytetrack` | Algorithm: `bytetrack`, `botsort`, `ocsort` |
| `confidence_threshold` | float | `0.25` | Minimum detection confidence |
| `target_fps` | float | `10.0` | Target frame rate hint |
| `image` | string | null | Optional base64 first frame (returns detections immediately) |

**Response:**

```json
{
  "session_id": "67a924e1",
  "tracker": "bytetrack",
  "model": "drone-aerial-general",
  "detections": null,
  "tracks_summary": null,
  "inference_time_ms": null,
  "tracking_time_ms": null
}
```

If `image` is provided, `detections` and `tracks_summary` are populated with first-frame results.

### POST /v1/vision/track/frame

Process a frame through the tracker. Returns detections with persistent track IDs.

**Parameters (Form data):**

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session ID from start |
| `image` | string | Base64-encoded image |

**Response:**

```json
{
  "detections": [
    {
      "x1": 225.6, "y1": 406.9, "x2": 345.0, "y2": 767.1,
      "class_name": "pedestrian",
      "class_id": 0,
      "confidence": 0.87,
      "track_id": 1,
      "track_state": "tracked"
    },
    {
      "x1": 622.8, "y1": 235.6, "x2": 634.8, "y2": 256.8,
      "class_name": "car",
      "class_id": 2,
      "confidence": 0.72,
      "track_id": 3,
      "track_state": "tracked"
    }
  ],
  "tracks_summary": {
    "active": 2,
    "total_created": 5
  },
  "inference_time_ms": 71.2,
  "tracking_time_ms": 7.9,
  "frame_number": 42
}
```

**Detection fields:**

| Field | Description |
|-------|-------------|
| `track_id` | Persistent ID across frames (same object = same ID) |
| `track_state` | `tracked` (active), `new` (first appearance), or `lost` |
| `x1, y1, x2, y2` | Bounding box coordinates |
| `class_name` | Detection class |
| `confidence` | Detection confidence |

### GET /v1/vision/track/\{session_id\}

Get tracking session status.

**Response:**

```json
{
  "session_id": "67a924e1",
  "model": "drone-aerial-general",
  "tracker": "bytetrack",
  "frames_processed": 847,
  "total_tracks_created": 45,
  "idle_seconds": 0.1,
  "duration_seconds": 84.7
}
```

### POST /v1/vision/track/stop

Stop a tracking session and release the model.

**Parameters (Form data):**

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session ID to stop |

**Response:**

```json
{
  "session_id": "67a924e1",
  "frames_processed": 847,
  "total_tracks_created": 45,
  "duration_seconds": 84.7
}
```

## Session Management

- **Max 50 concurrent sessions** — returns 429 if exceeded
- **120-second idle TTL** — sessions auto-expire if no frames received
- **Each session loads its own model** — independent tracker state, no cross-session interference
- **Background cleanup** runs every 15 seconds

## Model Resolution

The `model` parameter accepts:
- A **model name** — resolves to `~/.llamafarm/models/vision/\{name\}/current.pt`
- A **direct path** to a `.pt` file (must be within `~/.llamafarm` or cwd)
- A **versioned model** — if no `current.pt`, uses the latest `v\{n\}.pt`

## Runtime vs Server

| Port | Format | Use |
|------|--------|-----|
| `:11540` (runtime) | JSON body | Direct runtime access |
| `:14345` (server) | Form data (multipart) | Production proxy chain |

Both support all 4 endpoints. The server proxies to the runtime transparently.

## Dependencies

Tracking requires `lapx` for the linear assignment solver used by ByteTrack/BoT-SORT:

```bash
uv add lapx
```

This is included in the default runtime dependencies.
