"""Drone vision router — streaming inference + telemetry fusion + edge export.

Endpoints:
- WS  /v1/vision/drone/stream      — WebSocket streaming detection with telemetry
- POST /v1/vision/drone/export      — Edge model export (TensorRT/INT8/ONNX profiles)
- GET  /v1/vision/drone/export/profiles — List available export profiles
- POST /v1/vision/drone/geo         — Geo-reference detections given telemetry
"""

import asyncio
import base64
import json
import logging
import math
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from services.error_handler import handle_endpoint_errors

logger = logging.getLogger(__name__)
router = APIRouter(tags=["drone-vision"])

# Dependency injection
_load_detection_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None
_load_export_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None
_VISION_MODELS_DIR: Path | None = None


def set_drone_detection_loader(fn: Callable[..., Coroutine[Any, Any, Any]] | None) -> None:
    global _load_detection_fn
    _load_detection_fn = fn


def set_drone_export_loader(fn: Callable[..., Coroutine[Any, Any, Any]] | None) -> None:
    global _load_export_fn
    _load_export_fn = fn


def set_drone_models_dir(d: Path) -> None:
    global _VISION_MODELS_DIR
    _VISION_MODELS_DIR = d


# =============================================================================
# Telemetry Types
# =============================================================================

class Telemetry(BaseModel):
    """Per-frame drone telemetry."""
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None  # meters AGL (above ground level)
    heading_deg: float | None = None  # 0-360 degrees
    gimbal_pitch_deg: float | None = None  # negative = looking down (-90 = nadir)
    speed_mps: float | None = None  # meters per second
    timestamp: str | None = None  # ISO 8601


class GeoBox(BaseModel):
    """A geo-referenced bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    class_id: int
    confidence: float
    # Geo fields (None if telemetry insufficient)
    lat: float | None = None
    lng: float | None = None
    alt_m: float | None = None


class CameraIntrinsics(BaseModel):
    """Camera intrinsics for pixel → world projection."""
    focal_length_mm: float = 4.5  # typical drone camera
    sensor_width_mm: float = 6.17  # 1/2.3" sensor (DJI Mini etc.)
    sensor_height_mm: float = 4.55
    image_width_px: int = 3840  # 4K
    image_height_px: int = 2160


# =============================================================================
# Streaming Session
# =============================================================================

@dataclass
class DroneStreamSession:
    session_id: str
    model: str
    confidence_threshold: float = 0.25
    classes: list[str] | None = None
    camera: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    frames_processed: int = 0
    total_detections: int = 0
    total_latency_ms: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_frame_at: float = field(default_factory=time.time)


_ws_sessions: dict[str, DroneStreamSession] = {}
MAX_WS_SESSIONS = 50


# =============================================================================
# WebSocket Streaming Endpoint
# =============================================================================

@router.websocket("/v1/vision/drone/stream")
async def drone_stream_ws(ws: WebSocket):
    """WebSocket endpoint for real-time drone vision streaming.

    Protocol:
    1. Client connects and sends a JSON config message:
       {"model": "yolov8n", "confidence": 0.25, "classes": ["person","car"],
        "camera": {"focal_length_mm": 4.5, ...}}
    2. Client sends binary frames (JPEG/PNG) or JSON with base64:
       {"image": "<base64>", "telemetry": {"latitude": ..., "longitude": ..., ...}}
    3. Server responds with JSON per frame:
       {"frame_id": 1, "detections": [...], "latency_ms": 12.3, "geo_detections": [...]}
    4. Client sends {"action": "stop"} or disconnects to end.
    """
    if len(_ws_sessions) >= MAX_WS_SESSIONS:
        await ws.close(code=1013, reason="Too many concurrent sessions")
        return

    if _load_detection_fn is None:
        await ws.close(code=1011, reason="Detection loader not initialized")
        return

    await ws.accept()

    # Wait for config message
    try:
        config_raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        config = json.loads(config_raw)
    except (asyncio.TimeoutError, json.JSONDecodeError, WebSocketDisconnect) as e:
        logger.warning(f"Drone WS: bad config: {e}")
        await ws.close(code=1002, reason="Expected JSON config message")
        return

    sid = str(uuid.uuid4())[:8]
    camera = CameraIntrinsics(**config.get("camera", {})) if "camera" in config else CameraIntrinsics()
    session = DroneStreamSession(
        session_id=sid,
        model=config.get("model", "yolov8n"),
        confidence_threshold=config.get("confidence", 0.25),
        classes=config.get("classes"),
        camera=camera,
    )
    _ws_sessions[sid] = session

    # Send session ID back
    await ws.send_json({"session_id": sid, "status": "ready"})

    # Pre-load the model
    try:
        model = await _load_detection_fn(session.model)
    except Exception as e:
        await ws.send_json({"error": f"Failed to load model: {e}"})
        await ws.close(code=1011)
        _ws_sessions.pop(sid, None)
        return

    frame_id = 0
    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            telemetry = None
            image_bytes = None

            # Handle binary frame (raw JPEG/PNG)
            if "bytes" in msg and msg["bytes"]:
                image_bytes = msg["bytes"]

            # Handle JSON message
            elif "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"error": "Invalid JSON"})
                    continue

                if data.get("action") == "stop":
                    break

                if "image" in data:
                    image_bytes = base64.b64decode(data["image"])

                if "telemetry" in data:
                    telemetry = Telemetry(**data["telemetry"])

            if image_bytes is None:
                await ws.send_json({"error": "No image data"})
                continue

            # Run detection
            frame_id += 1
            t0 = time.perf_counter()
            try:
                det_result = await model.detect(
                    image=image_bytes,
                    confidence_threshold=session.confidence_threshold,
                    classes=session.classes,
                )
            except Exception as e:
                await ws.send_json({"frame_id": frame_id, "error": str(e)})
                continue

            latency_ms = (time.perf_counter() - t0) * 1000
            session.frames_processed += 1
            session.last_frame_at = time.time()
            session.total_latency_ms += latency_ms

            # Build detections list
            detections = []
            if hasattr(det_result, "boxes"):
                for b in det_result.boxes:
                    det = {
                        "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                        "class_name": b.class_name, "class_id": b.class_id,
                        "confidence": round(b.confidence, 4),
                    }
                    detections.append(det)

            session.total_detections += len(detections)

            # Geo-reference if telemetry available
            geo_detections = None
            if telemetry and telemetry.latitude is not None and telemetry.altitude_m is not None:
                geo_detections = _geo_reference_detections(
                    detections, telemetry, session.camera,
                )

            response = {
                "frame_id": frame_id,
                "detections": detections,
                "detection_count": len(detections),
                "latency_ms": round(latency_ms, 2),
            }
            if geo_detections:
                response["geo_detections"] = geo_detections
            if telemetry:
                response["telemetry_echo"] = telemetry.model_dump(exclude_none=True)

            await ws.send_json(response)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Drone WS error [{sid}]: {e}")
    finally:
        session = _ws_sessions.pop(sid, None)
        if session:
            avg_latency = (
                session.total_latency_ms / session.frames_processed
                if session.frames_processed > 0 else 0
            )
            logger.info(
                f"Drone stream [{sid}] ended: "
                f"{session.frames_processed} frames, "
                f"{session.total_detections} detections, "
                f"avg {avg_latency:.1f}ms/frame, "
                f"{time.time() - session.created_at:.1f}s total"
            )


# =============================================================================
# Geo-referencing
# =============================================================================

class GeoReferenceRequest(BaseModel):
    """Geo-reference detections given telemetry."""
    detections: list[dict[str, Any]]
    telemetry: Telemetry
    camera: CameraIntrinsics = Field(default_factory=CameraIntrinsics)


class GeoReferenceResponse(BaseModel):
    geo_detections: list[GeoBox]


@router.post("/v1/vision/drone/geo", response_model=GeoReferenceResponse)
@handle_endpoint_errors("drone_geo_reference")
async def geo_reference(request: GeoReferenceRequest) -> GeoReferenceResponse:
    """Geo-reference pixel bounding boxes to lat/lng coordinates.

    Uses camera intrinsics + drone telemetry (altitude, GPS, gimbal angle)
    to project pixel center of each detection to world coordinates.
    """
    if request.telemetry.latitude is None or request.telemetry.longitude is None:
        raise HTTPException(status_code=422, detail="Telemetry must include latitude and longitude")
    if request.telemetry.altitude_m is None:
        raise HTTPException(status_code=422, detail="Telemetry must include altitude_m")

    geo_dets = _geo_reference_detections(
        request.detections, request.telemetry, request.camera,
    )
    return GeoReferenceResponse(geo_detections=geo_dets)


def _geo_reference_detections(
    detections: list[dict[str, Any]],
    telemetry: Telemetry,
    camera: CameraIntrinsics,
) -> list[GeoBox]:
    """Project pixel detections to geo coordinates.

    Uses simple pinhole camera model:
    1. Pixel offset from image center → angular offset
    2. Angular offset + altitude → ground offset in meters
    3. Ground offset + GPS → lat/lng
    """
    if telemetry.latitude is None or telemetry.longitude is None or telemetry.altitude_m is None:
        return []

    alt = telemetry.altitude_m
    lat0 = telemetry.latitude
    lng0 = telemetry.longitude
    heading = math.radians(telemetry.heading_deg or 0)
    gimbal = math.radians(telemetry.gimbal_pitch_deg or -90)  # default nadir

    # Meters per degree at this latitude
    m_per_deg_lat = 111_320.0
    m_per_deg_lng = 111_320.0 * math.cos(math.radians(lat0))

    # Ground footprint per pixel (at nadir, simplified)
    # GSD = (altitude * sensor_width) / (focal_length * image_width)
    gsd_x = (alt * camera.sensor_width_mm) / (camera.focal_length_mm * camera.image_width_px)
    gsd_y = (alt * camera.sensor_height_mm) / (camera.focal_length_mm * camera.image_height_px)

    # If gimbal is not nadir, scale by cos(gimbal_pitch) for oblique view
    cos_gimbal = abs(math.cos(gimbal)) if abs(gimbal) < math.radians(85) else 0.087  # floor at ~5°
    gsd_x /= cos_gimbal
    gsd_y /= cos_gimbal

    cx = camera.image_width_px / 2
    cy = camera.image_height_px / 2

    geo_boxes: list[GeoBox] = []
    for det in detections:
        # Detection center in pixels
        det_cx = (det.get("x1", 0) + det.get("x2", 0)) / 2
        det_cy = (det.get("y1", 0) + det.get("y2", 0)) / 2

        # Offset from image center in meters
        dx_m = (det_cx - cx) * gsd_x
        dy_m = (cy - det_cy) * gsd_y  # y is inverted in image coords

        # Rotate by heading
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        north_m = dx_m * sin_h + dy_m * cos_h
        east_m = dx_m * cos_h - dy_m * sin_h

        # Convert to lat/lng offset
        dlat = north_m / m_per_deg_lat
        dlng = east_m / m_per_deg_lng

        geo_boxes.append(GeoBox(
            x1=det.get("x1", 0),
            y1=det.get("y1", 0),
            x2=det.get("x2", 0),
            y2=det.get("y2", 0),
            class_name=det.get("class_name", "unknown"),
            class_id=det.get("class_id", 0),
            confidence=det.get("confidence", 0),
            lat=round(lat0 + dlat, 8),
            lng=round(lng0 + dlng, 8),
            alt_m=round(alt, 2),
        ))

    return geo_boxes


# =============================================================================
# Edge Export Pipeline
# =============================================================================

ExportTarget = Literal["onnx", "tensorrt", "coreml", "tflite", "openvino"]
ExportPrecision = Literal["fp32", "fp16", "int8"]


class ExportProfile(BaseModel):
    """Pre-defined export profile for common drone configurations."""
    name: str
    description: str
    target: ExportTarget
    precision: ExportPrecision
    imgsz: int
    base_model: str  # Recommended YOLO variant


EXPORT_PROFILES: dict[str, ExportProfile] = {
    "drone-nano": ExportProfile(
        name="drone-nano",
        description="Minimal latency for resource-constrained edge (Jetson Nano, RPi5)",
        target="tensorrt",
        precision="int8",
        imgsz=320,
        base_model="yolov8n",
    ),
    "drone-standard": ExportProfile(
        name="drone-standard",
        description="Balanced speed/accuracy for Jetson Orin NX",
        target="tensorrt",
        precision="fp16",
        imgsz=640,
        base_model="yolov8s",
    ),
    "drone-hd": ExportProfile(
        name="drone-hd",
        description="High resolution for Jetson AGX Orin, optimized for small objects",
        target="tensorrt",
        precision="fp16",
        imgsz=1280,
        base_model="yolov8m",
    ),
    "edge-onnx": ExportProfile(
        name="edge-onnx",
        description="Portable ONNX for cross-platform deployment",
        target="onnx",
        precision="fp16",
        imgsz=640,
        base_model="yolov8s",
    ),
    "edge-tflite": ExportProfile(
        name="edge-tflite",
        description="TFLite for Coral Edge TPU or Android",
        target="tflite",
        precision="int8",
        imgsz=320,
        base_model="yolov8n",
    ),
}


class EdgeExportRequest(BaseModel):
    model_id: str = Field(..., description="Vision model ID to export")
    target: ExportTarget = Field("onnx", description="Export format")
    precision: ExportPrecision = Field("fp16", description="Quantization precision")
    imgsz: int = Field(640, description="Input image size", ge=128, le=2048)
    profile: str | None = Field(None, description="Use a preset profile (overrides target/precision/imgsz)")
    calibration_dataset: str | None = Field(None, description="Path to calibration images for INT8 quantization")


class EdgeExportResponse(BaseModel):
    model_id: str
    export_path: str
    target: str
    precision: str
    imgsz: int
    size_mb: float
    export_time_seconds: float
    profile: str | None = None


@router.get("/v1/vision/drone/export/profiles")
@handle_endpoint_errors("drone_export_profiles")
async def list_export_profiles() -> dict[str, Any]:
    """List available drone/edge export profiles."""
    return {
        "profiles": [p.model_dump() for p in EXPORT_PROFILES.values()],
        "total": len(EXPORT_PROFILES),
    }


@router.post("/v1/vision/drone/export", response_model=EdgeExportResponse)
@handle_endpoint_errors("drone_export_model")
async def export_for_edge(request: EdgeExportRequest) -> EdgeExportResponse:
    """Export a vision model for drone/edge deployment.

    Supports TensorRT (Jetson), ONNX (cross-platform), CoreML (Apple),
    TFLite (Coral/Android), and OpenVINO (Intel).

    Use `profile` to apply a preset (e.g., "drone-nano" for INT8 320px TensorRT).
    For INT8, provide `calibration_dataset` for best accuracy.
    """
    # Validate inputs before checking loader state
    safe_id = Path(request.model_id).name
    if safe_id != request.model_id or ".." in request.model_id:
        raise HTTPException(status_code=400, detail="Invalid model_id")

    # Apply profile overrides
    target = request.target
    precision = request.precision
    imgsz = request.imgsz
    profile_name = request.profile

    if profile_name:
        profile = EXPORT_PROFILES.get(profile_name)
        if not profile:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown profile '{profile_name}'. "
                f"Available: {list(EXPORT_PROFILES.keys())}",
            )
        target = profile.target
        precision = profile.precision
        imgsz = profile.imgsz

    if _VISION_MODELS_DIR is None:
        raise HTTPException(status_code=500, detail="Vision models dir not configured")
    model_dir = _VISION_MODELS_DIR / safe_id
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
    if _load_export_fn is None:
        raise HTTPException(status_code=500, detail="Export loader not initialized")

    # Load model
    model = await _load_export_fn(request.model_id)

    # Build export kwargs
    kwargs: dict[str, Any] = {"imgsz": imgsz, "simplify": True}
    if precision == "fp16":
        kwargs["half"] = True
    elif precision == "int8":
        kwargs["int8"] = True
        if request.calibration_dataset:
            kwargs["data"] = request.calibration_dataset

    # Export
    export_dir = model_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        export_path = await asyncio.to_thread(
            model.export, format=target, **kwargs,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {e}. "
            f"Note: TensorRT export requires a Jetson/CUDA device with TensorRT installed.",
        ) from e

    elapsed = time.perf_counter() - t0

    p = Path(export_path)
    size = (
        sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        if p.is_dir()
        else p.stat().st_size
    )

    return EdgeExportResponse(
        model_id=request.model_id,
        export_path=str(p),
        target=target,
        precision=precision,
        imgsz=imgsz,
        size_mb=round(size / (1024 * 1024), 2),
        export_time_seconds=round(elapsed, 2),
        profile=profile_name,
    )


# =============================================================================
# Drone Training Presets
# =============================================================================

class DroneTrainPreset(BaseModel):
    """Training preset for drone/aerial imagery."""
    name: str
    description: str
    imgsz: int
    augmentation: dict[str, Any]
    epochs: int
    batch_size: int


TRAIN_PRESETS: dict[str, DroneTrainPreset] = {
    "aerial": DroneTrainPreset(
        name="aerial",
        description="General aerial imagery — optimized for small objects at altitude",
        imgsz=1280,
        augmentation={
            "mosaic": 1.0,
            "mixup": 0.15,
            "scale": 0.9,
            "flipud": 0.5,  # vertical flip (common in aerial)
            "degrees": 180.0,  # full rotation (nadir invariant)
            "copy_paste": 0.3,  # synthetic small object augmentation
        },
        epochs=100,
        batch_size=4,
    ),
    "aerial-nano": DroneTrainPreset(
        name="aerial-nano",
        description="Fast aerial training for edge deployment (lower res, fewer epochs)",
        imgsz=640,
        augmentation={
            "mosaic": 1.0,
            "scale": 0.5,
            "flipud": 0.5,
            "degrees": 180.0,
        },
        epochs=50,
        batch_size=8,
    ),
    "orthomosaic": DroneTrainPreset(
        name="orthomosaic",
        description="High-res orthomosaic tiled training for mapping surveys",
        imgsz=1280,
        augmentation={
            "mosaic": 0.0,  # no mosaic — tiles already provide context
            "scale": 0.3,
            "flipud": 0.5,
            "fliplr": 0.5,
            "degrees": 90.0,
        },
        epochs=150,
        batch_size=2,
    ),
}


@router.get("/v1/vision/drone/train/presets")
@handle_endpoint_errors("drone_train_presets")
async def list_train_presets() -> dict[str, Any]:
    """List available drone training presets."""
    return {
        "presets": [p.model_dump() for p in TRAIN_PRESETS.values()],
        "total": len(TRAIN_PRESETS),
    }


# =============================================================================
# Session Info Endpoint
# =============================================================================

class DroneSessionInfo(BaseModel):
    session_id: str
    model: str
    frames_processed: int
    total_detections: int
    avg_latency_ms: float
    duration_seconds: float


@router.get("/v1/vision/drone/sessions")
@handle_endpoint_errors("drone_sessions")
async def list_drone_sessions() -> dict[str, Any]:
    """List active drone streaming sessions."""
    now = time.time()
    sessions = []
    for s in _ws_sessions.values():
        avg = s.total_latency_ms / s.frames_processed if s.frames_processed > 0 else 0
        sessions.append(DroneSessionInfo(
            session_id=s.session_id,
            model=s.model,
            frames_processed=s.frames_processed,
            total_detections=s.total_detections,
            avg_latency_ms=round(avg, 2),
            duration_seconds=round(now - s.created_at, 1),
        ))
    return {"sessions": [s.model_dump() for s in sessions], "count": len(sessions)}
