"""Streaming vision router for real-time detection with cascade and auto-learning.

Provides endpoints for:
- Starting/stopping streaming sessions with cascade configuration
- Processing individual frames with automatic escalation
- Submitting corrections for auto-learning
- Replay buffer status and management
- Session statistics including cascade metrics
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from api_types.vision import (
    BoundingBox,
    CascadeConfig as APICascadeConfig,
    CorrectionRequest,
    CorrectionResponse,
    Detection,
    ReplayBufferResponse,
    ReplayBufferStats,
    StreamingConfig,
    StreamStartRequest,
    StreamStartResponse,
    StreamFrameRequest,
    StreamFrameResponse,
    StreamStopRequest,
    StreamStopResponse,
)
from models.streaming_vision import (
    CascadeConfig,
    StreamingConfig as ModelStreamingConfig,
    get_streaming_detector,
)
from services.error_handler import handle_endpoint_errors
from .utils import decode_base64_image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision-streaming"])


@router.post("/v1/vision/stream/start", response_model=StreamStartResponse)
@handle_endpoint_errors("vision_stream_start")
async def start_streaming_session(request: StreamStartRequest) -> StreamStartResponse:
    """Start a new streaming detection session with optional cascade.
    
    Creates a session that can process frames and trigger actions
    when objects of interest are detected. Configure cascade for
    automatic escalation to a larger model for uncertain detections.
    
    Example:
    ```json
    {
        "model": "yolov8n",
        "config": {
            "target_fps": 1.0,
            "confidence_threshold": 0.7,
            "escalation_threshold": 0.5,
            "action_classes": ["person", "car"],
            "cooldown_seconds": 5.0,
            "cascade": {
                "secondary_model_id": "yolov8m",
                "feedback_to_primary": true,
                "save_uncertain_images": true
            }
        }
    }
    ```
    
    The cascade flow:
    1. Primary model (yolov8n) processes frame
    2. If confidence >= 0.7 → return result
    3. If 0.5 <= confidence < 0.7 → escalate to yolov8m
       - If yolov8m confident → return result + add to replay buffer
       - If yolov8m uncertain → send to review queue
    4. If confidence < 0.5 → send to review queue
    
    Returns:
    ```json
    {
        "session_id": "abc12345",
        "config": {...}
    }
    ```
    """
    detector = get_streaming_detector()
    
    # Convert API cascade config to model cascade config
    cascade_config = None
    if request.config.cascade:
        cascade_config = CascadeConfig(
            secondary_model_id=request.config.cascade.secondary_model_id,
            cascade_chain=request.config.cascade.cascade_chain,
            feedback_to_primary=request.config.cascade.feedback_to_primary,
            save_uncertain_images=request.config.cascade.save_uncertain_images,
            segmentation_model_id=request.config.cascade.segmentation_model_id,
            classification_model_id=request.config.cascade.classification_model_id,
            enrich_on_escalation=request.config.cascade.enrich_on_escalation,
            max_hops=request.config.cascade.max_hops,
        )
    
    # Convert API config to model config
    config = ModelStreamingConfig(
        target_fps=request.config.target_fps,
        confidence_threshold=request.config.confidence_threshold,
        escalation_threshold=request.config.escalation_threshold,
        action_classes=request.config.action_classes,
        cooldown_seconds=request.config.cooldown_seconds,
        cascade=cascade_config,
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
    """Process a single frame with automatic cascade and learning.
    
    Returns status:
    - "ok": No action needed
    - "action": Detection triggered (with detections)
    - "review": Uncertain detection flagged for review
    - "escalated": Detection came from secondary model
    
    When cascade is configured:
    - Uncertain detections are automatically sent to the secondary model
    - Successful secondary results are added to the replay buffer
    - The primary model will be retrained on these examples
    
    Example request:
    ```json
    {
        "session_id": "abc12345",
        "image": "data:image/jpeg;base64,..."
    }
    ```
    
    Example response (escalated):
    ```json
    {
        "status": "action",
        "detections": [...],
        "confidence": 0.85,
        "escalated_to": "yolov8m",
        "added_to_replay": true
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
    image_bytes = decode_base64_image(request.image)
    
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
        suppressed=result.suppressed,
        escalated_to=result.escalated_to,
        added_to_replay=result.added_to_replay,
    )


@router.post("/v1/vision/stream/stop", response_model=StreamStopResponse)
@handle_endpoint_errors("vision_stream_stop")
async def stop_streaming_session(request: StreamStopRequest) -> StreamStopResponse:
    """Stop a streaming session and get statistics.
    
    Returns session stats including:
    - frames_processed: Total frames analyzed
    - actions_triggered: Number of alert triggers
    - escalations: Times secondary model was called
    - secondary_success_rate: How often secondary model succeeded
    - review_queue_count: Images sent for human review
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


@router.get("/v1/vision/stream/sessions/{session_id}/stats")
@handle_endpoint_errors("vision_stream_stats")
async def get_session_stats(session_id: str) -> dict[str, Any]:
    """Get detailed statistics for an active session."""
    detector = get_streaming_detector()
    session = detector.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    import time
    duration = time.time() - session.created_at
    
    return {
        "session_id": session_id,
        "model_id": session.model_id,
        "secondary_model_id": session.secondary_model_id,
        "frames_processed": session.frames_processed,
        "actions_triggered": session.actions_triggered,
        "escalations_to_secondary": session.escalations_to_secondary,
        "secondary_successes": session.secondary_successes,
        "secondary_success_rate": (
            session.secondary_successes / session.escalations_to_secondary
            if session.escalations_to_secondary > 0 else 0.0
        ),
        "review_queue_count": session.review_queue_count,
        "duration_seconds": duration,
        "avg_fps": session.frames_processed / duration if duration > 0 else 0,
    }


# =============================================================================
# Correction Feedback Endpoints
# =============================================================================


@router.post("/v1/vision/corrections", response_model=CorrectionResponse)
@handle_endpoint_errors("vision_correction")
async def submit_correction(request: CorrectionRequest) -> CorrectionResponse:
    """Submit a correction for a detection.
    
    Use this endpoint when a detection was wrong and you have the correct label.
    This feeds into the replay buffer for training the primary model.
    
    Example:
    ```json
    {
        "image_id": "abc12345_1",
        "corrected_class": "truck",
        "original_confidence": 0.45,
        "session_id": "abc12345"
    }
    ```
    
    Response:
    ```json
    {
        "image_id": "abc12345_1",
        "added_to_replay": true,
        "replay_buffer_size": 42,
        "training_triggered": false
    }
    ```
    """
    detector = get_streaming_detector()
    
    # Convert box if provided
    from models.vision_base import DetectionBox
    box = None
    if request.box:
        box = DetectionBox(
            x1=request.box.x1,
            y1=request.box.y1,
            x2=request.box.x2,
            y2=request.box.y2,
            class_name=request.corrected_class,
            class_id=0,
            confidence=1.0,
        )
    
    added = await detector.submit_correction(
        session_id=request.session_id or "manual",
        image_id=request.image_id,
        corrected_class=request.corrected_class,
        box=box,
        original_confidence=request.original_confidence,
    )
    
    # Get buffer stats
    buffer_stats = detector.get_replay_buffer_stats()
    buffer_size = buffer_stats["size"] if buffer_stats else 0
    
    return CorrectionResponse(
        image_id=request.image_id,
        added_to_replay=added,
        replay_buffer_size=buffer_size,
        training_triggered=False,  # TODO: Check if training was triggered
    )


# =============================================================================
# Replay Buffer Endpoints
# =============================================================================


@router.get("/v1/vision/replay-buffer", response_model=ReplayBufferResponse)
@handle_endpoint_errors("vision_replay_buffer")
async def get_replay_buffer_status() -> ReplayBufferResponse:
    """Get replay buffer status and statistics.
    
    The replay buffer stores:
    - Successful escalation results (secondary model verified)
    - Human/AI corrections from the review queue
    
    When the buffer hits the threshold, auto-training is triggered
    to improve the primary model.
    
    Response:
    ```json
    {
        "stats": {
            "size": 42,
            "max_size": 1000,
            "by_source": {"escalation": 30, "correction": 12},
            "avg_priority": 1.2
        },
        "auto_train_threshold": 50,
        "training_eligible": false
    }
    ```
    """
    detector = get_streaming_detector()
    
    stats = detector.get_replay_buffer_stats()
    if stats is None:
        return ReplayBufferResponse(
            stats=ReplayBufferStats(
                size=0,
                max_size=1000,
                by_source={},
                avg_priority=0.0,
            ),
            auto_train_threshold=50,
            training_eligible=False,
        )
    
    return ReplayBufferResponse(
        stats=ReplayBufferStats(
            size=stats["size"],
            max_size=stats["max_size"],
            by_source=stats["by_source"],
            avg_priority=stats["avg_priority"],
        ),
        auto_train_threshold=50,  # TODO: Get from config
        training_eligible=stats["size"] >= 50,
    )


@router.post("/v1/vision/replay-buffer/clear")
@handle_endpoint_errors("vision_replay_buffer_clear")
async def clear_replay_buffer() -> dict[str, Any]:
    """Clear the replay buffer.
    
    Use with caution - this removes all pending training samples.
    """
    detector = get_streaming_detector()
    
    stats = detector.get_replay_buffer_stats()
    if stats and stats["size"] > 0:
        old_size = stats["size"]
        detector.clear_replay_buffer()
        return {"cleared": True, "samples_removed": old_size}
    
    return {"cleared": False, "samples_removed": 0}


