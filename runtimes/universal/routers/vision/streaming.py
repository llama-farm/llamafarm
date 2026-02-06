"""Streaming vision router for real-time detection sessions.

Provides endpoints for:
- Starting/stopping streaming sessions
- Processing individual frames
- Session statistics
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from api_types.vision import (
    BoundingBox,
    Detection,
    StreamingConfig,
    StreamStartRequest,
    StreamStartResponse,
    StreamFrameRequest,
    StreamFrameResponse,
    StreamStopRequest,
    StreamStopResponse,
)
from models.streaming_vision import (
    StreamingConfig as ModelStreamingConfig,
    get_streaming_detector,
)
from services.error_handler import handle_endpoint_errors

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision-streaming"])


@router.post("/v1/vision/stream/start", response_model=StreamStartResponse)
@handle_endpoint_errors("vision_stream_start")
async def start_streaming_session(request: StreamStartRequest) -> StreamStartResponse:
    """Start a new streaming detection session.
    
    Creates a session that can process frames and trigger actions
    when objects of interest are detected.
    
    Example:
    ```json
    {
        "model": "yolov8n",
        "config": {
            "target_fps": 1.0,
            "confidence_threshold": 0.7,
            "action_classes": ["person", "car"],
            "cooldown_seconds": 5.0
        }
    }
    ```
    
    Returns:
    ```json
    {
        "session_id": "abc12345",
        "config": {...}
    }
    ```
    """
    detector = get_streaming_detector()
    
    # Convert API config to model config
    config = ModelStreamingConfig(
        target_fps=request.config.target_fps,
        confidence_threshold=request.config.confidence_threshold,
        escalation_threshold=request.config.escalation_threshold,
        action_classes=request.config.action_classes,
        cooldown_seconds=request.config.cooldown_seconds,
    )
    
    session = await detector.start_session(
        model_id=request.model,
        config=config,
    )
    
    return StreamStartResponse(
        session_id=session.session_id,
        config=request.config,
    )


@router.post("/v1/vision/stream/frame", response_model=StreamFrameResponse)
@handle_endpoint_errors("vision_stream_frame")
async def process_frame(request: StreamFrameRequest) -> StreamFrameResponse:
    """Process a single frame in a streaming session.
    
    Returns status:
    - "ok": No action needed
    - "action": Detection triggered (with detections)
    - "review": Uncertain detection flagged for review
    
    Example request:
    ```json
    {
        "session_id": "abc12345",
        "image": "data:image/jpeg;base64,..."
    }
    ```
    """
    detector = get_streaming_detector()
    
    session = detector.get_session(request.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found"
        )
    
    # Decode image
    image_bytes = _decode_base64_image(request.image)
    
    # Process frame
    result = await detector.process_frame(
        session_id=request.session_id,
        image=image_bytes,
    )
    
    # Convert detections to API format
    detections = None
    if result.detections:
        detections = [
            Detection(
                box=BoundingBox(
                    x1=d.x1,
                    y1=d.y1,
                    x2=d.x2,
                    y2=d.y2,
                ),
                class_name=d.class_name,
                class_id=d.class_id,
                confidence=d.confidence,
            )
            for d in result.detections
        ]
    
    return StreamFrameResponse(
        status=result.status,
        detections=detections,
        confidence=result.confidence if result.confidence > 0 else None,
        image_id=result.image_id,
    )


@router.post("/v1/vision/stream/stop", response_model=StreamStopResponse)
@handle_endpoint_errors("vision_stream_stop")
async def stop_streaming_session(request: StreamStopRequest) -> StreamStopResponse:
    """Stop a streaming session and get statistics.
    
    Returns session stats including frames processed, actions triggered,
    and average FPS.
    """
    detector = get_streaming_detector()
    
    try:
        stats = await detector.stop_session(request.session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return StreamStopResponse(
        session_id=stats["session_id"],
        frames_processed=stats["frames_processed"],
        actions_triggered=stats["actions_triggered"],
        duration_seconds=stats["duration_seconds"],
    )


@router.get("/v1/vision/stream/sessions")
@handle_endpoint_errors("vision_stream_list")
async def list_sessions() -> dict[str, Any]:
    """List active streaming sessions."""
    detector = get_streaming_detector()
    sessions = detector.list_sessions()
    
    return {
        "sessions": sessions,
        "count": len(sessions),
    }


def _decode_base64_image(image_str: str) -> bytes:
    """Decode base64 image string to bytes."""
    import base64

    if image_str.startswith("data:"):
        _, base64_data = image_str.split(",", 1)
    else:
        base64_data = image_str

    return base64.b64decode(base64_data)
