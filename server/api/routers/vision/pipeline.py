"""Pipeline router — streaming, training, model management endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Form
from server.services.vision import VisionPipelineService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision", tags=["vision-pipeline"])


# =============================================================================
# Streaming
# =============================================================================

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
