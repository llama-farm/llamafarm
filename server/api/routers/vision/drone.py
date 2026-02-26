"""Drone router — proxy HTTP drone vision endpoints to Universal Runtime.

Note: Streaming endpoint is currently runtime-only WebSocket:
- ws://<runtime>:11540/v1/vision/drone/stream
"""

import json
from typing import Any

from fastapi import APIRouter, Form, HTTPException
from server.services.vision import VisionDroneService

router = APIRouter(prefix="/vision", tags=["vision-drone"])


@router.post("/drone/geo")
async def geo_reference(
    detections: str = Form(..., description="JSON array of detection objects"),
    telemetry: str = Form(..., description="JSON object with telemetry fields"),
    camera: str | None = Form(default=None, description="Optional JSON camera intrinsics object"),
) -> dict[str, Any]:
    """Geo-reference detections from form payload."""
    try:
        parsed_detections = json.loads(detections)
        parsed_telemetry = json.loads(telemetry)
        payload: dict[str, Any] = {
            "detections": parsed_detections,
            "telemetry": parsed_telemetry,
        }
        if camera:
            payload["camera"] = json.loads(camera)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in form fields: {e}") from e

    return await VisionDroneService.geo_reference(payload)


@router.get("/drone/export/profiles")
async def export_profiles() -> dict[str, Any]:
    """List available drone edge export profiles."""
    return await VisionDroneService.list_export_profiles()


@router.post("/drone/export")
async def export_for_edge(
    model_id: str = Form(...),
    target: str = Form(default="onnx"),
    precision: str = Form(default="fp16"),
    imgsz: int = Form(default=640),
    profile: str | None = Form(default=None),
    calibration_dataset: str | None = Form(default=None),
) -> dict[str, Any]:
    """Export model for edge deployment."""
    payload: dict[str, Any] = {
        "model_id": model_id,
        "target": target,
        "precision": precision,
        "imgsz": imgsz,
    }
    if profile:
        payload["profile"] = profile
    if calibration_dataset:
        payload["calibration_dataset"] = calibration_dataset

    return await VisionDroneService.export_for_edge(payload)


@router.get("/drone/train/presets")
async def train_presets() -> dict[str, Any]:
    """List drone training presets."""
    return await VisionDroneService.list_train_presets()


@router.get("/drone/sessions")
async def sessions() -> dict[str, Any]:
    """List active drone streaming sessions from runtime."""
    return await VisionDroneService.list_sessions()
