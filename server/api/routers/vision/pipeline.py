"""Pipeline router — streaming, training, model management endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Form, HTTPException
from server.services.vision import VisionPipelineService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision", tags=["vision-pipeline"])


# =============================================================================
# Detect + Classify (combo endpoint)
# =============================================================================

@router.post("/detect_classify")
async def detect_classify(
    image: str = Form(..., description="Base64-encoded image"),
    detection_model: str = Form(default="yolov8n"),
    classification_model: str = Form(default="clip-vit-base"),
    classes: str = Form(..., description="Comma-separated classification classes"),
    confidence_threshold: float = Form(default=0.5),
    detection_classes: str | None = Form(default=None),
    top_k: int = Form(default=3),
) -> dict[str, Any]:
    """Detect objects then classify each crop — single round-trip."""
    if not detection_model or not detection_model.strip():
        raise HTTPException(status_code=422, detail="detection_model must not be empty")
    if not classification_model or not classification_model.strip():
        raise HTTPException(status_code=422, detail="classification_model must not be empty")
    cls_list = [c.strip() for c in classes.split(",") if c.strip()]
    if not cls_list:
        raise HTTPException(status_code=422, detail="classes must contain at least one non-empty value")
    det_cls = [c.strip() for c in detection_classes.split(",") if c.strip()] if detection_classes else None
    return await VisionPipelineService.detect_classify({
        "image": image,
        "detection_model": detection_model,
        "classification_model": classification_model,
        "classes": cls_list,
        "confidence_threshold": confidence_threshold,
        "detection_classes": det_cls,
        "top_k": top_k,
    })


# =============================================================================
# Streaming
# =============================================================================

@router.get("/stream/sessions")
async def stream_sessions() -> dict[str, Any]:
    """List active streaming sessions."""
    return await VisionPipelineService.stream_sessions()


@router.post("/stream/start")
async def stream_start(
    chain: str = Form(default="yolov8n", description="Comma-separated model chain"),
    confidence_threshold: float = Form(default=0.7),
    target_fps: float = Form(default=1.0),
    action_classes: str | None = Form(default=None),
) -> dict[str, Any]:
    """Start a streaming detection session."""
    chain_list = [c.strip() for c in chain.split(",")]
    classes = [c.strip() for c in action_classes.split(",") if c.strip()] if action_classes else None
    return await VisionPipelineService.stream_start({
        "config": {"chain": chain_list, "confidence_threshold": confidence_threshold},
        "target_fps": target_fps,
        "action_classes": classes,
    })


@router.post("/stream/frame")
async def stream_frame(
    session_id: str = Form(...),
    image: str = Form(..., description="Base64-encoded image"),
) -> dict[str, Any]:
    """Process a frame through the cascade."""
    return await VisionPipelineService.stream_frame(session_id, image)


@router.post("/stream/stop")
async def stream_stop(session_id: str = Form(...)) -> dict[str, Any]:
    """Stop a streaming session."""
    return await VisionPipelineService.stream_stop(session_id)


# =============================================================================
# Training
# =============================================================================

@router.post("/train")
async def start_training(
    model: str = Form(...),
    dataset: str = Form(...),
    task: str = Form(default="detection"),
    epochs: int = Form(default=10),
    batch_size: int = Form(default=16),
) -> dict[str, Any]:
    """Start a training job."""
    return await VisionPipelineService.train(
        model=model, dataset=dataset, task=task,
        config={"epochs": epochs, "batch_size": batch_size},
    )


@router.get("/train/{job_id}")
async def training_status(job_id: str) -> dict[str, Any]:
    """Get training job status."""
    return await VisionPipelineService.train_status(job_id)


@router.delete("/train/{job_id}")
async def cancel_training(job_id: str) -> dict[str, Any]:
    """Cancel a training job."""
    return await VisionPipelineService.train_cancel(job_id)


# =============================================================================
# Model Management
# =============================================================================

@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List saved vision models."""
    return await VisionPipelineService.list_models()


@router.post("/models/export")
async def export_model(
    model_id: str = Form(...),
    format: str = Form(default="onnx"),
    quantization: str = Form(default="fp16"),
) -> dict[str, Any]:
    """Export a model."""
    return await VisionPipelineService.export_model(model_id, format, quantization)
