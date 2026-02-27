"""Tracking router — proxy to Universal Runtime tracking endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Form
from server.services.vision import VisionTrackingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision", tags=["vision-tracking"])


@router.post("/track/start")
async def start_tracking(
    model: str = Form(..., description="Model name or path to .pt file"),
    tracker: str = Form(default="bytetrack", description="Tracker: bytetrack, botsort, ocsort"),
    confidence_threshold: float = Form(default=0.25),
    target_fps: float = Form(default=10.0),
    image: str | None = Form(default=None, description="Optional base64 first frame"),
) -> dict[str, Any]:
    """Start a tracking session."""
    payload: dict[str, Any] = {
        "model": model,
        "tracker": tracker,
        "confidence_threshold": confidence_threshold,
        "target_fps": target_fps,
    }
    if image:
        payload["image"] = image
    return await VisionTrackingService.start_tracking(payload)


@router.post("/track/frame")
async def track_frame(
    session_id: str = Form(..., description="Tracking session ID"),
    image: str = Form(..., description="Base64-encoded image"),
) -> dict[str, Any]:
    """Process a frame through the tracker."""
    return await VisionTrackingService.track_frame({
        "session_id": session_id,
        "image": image,
    })


# Specific paths before parameterized
@router.post("/track/stop")
async def stop_tracking(
    session_id: str = Form(..., description="Tracking session ID"),
) -> dict[str, Any]:
    """Stop a tracking session."""
    return await VisionTrackingService.stop_tracking({
        "session_id": session_id,
    })


@router.get("/track/{session_id}")
async def track_status(session_id: str) -> dict[str, Any]:
    """Get tracking session status."""
    return await VisionTrackingService.get_status(session_id)
