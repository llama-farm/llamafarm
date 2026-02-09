"""Vision Pipeline Router - Streaming, training, models, segmentation, federation.

Proxies requests from the LlamaFarm server (14345) to the
Universal Runtime (11540) for vision pipeline operations.
"""

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.vision import (
    VisionFederationService,
    VisionModelService,
    VisionSegmentationService,
    VisionStreamingService,
    VisionTrainingService,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision-pipeline"])


# =============================================================================
# Request models
# =============================================================================


class CascadeConfig(BaseModel):
    secondary_model_id: str | None = None
    cascade_chain: list[str] | None = None
    feedback_to_primary: bool = True
    save_uncertain_images: bool = True
    segmentation_model_id: str | None = None
    classification_model_id: str | None = None
    enrich_on_escalation: bool = True
    max_hops: int = 3


class StreamingSessionConfig(BaseModel):
    target_fps: float = 1.0
    confidence_threshold: float = 0.7
    escalation_threshold: float = 0.5
    action_classes: list[str] | None = None
    cooldown_seconds: float = 5.0
    cascade: CascadeConfig | None = None


class StartSessionRequest(BaseModel):
    model: str = "yolov8n"
    config: StreamingSessionConfig = Field(default_factory=StreamingSessionConfig)


class ProcessFrameRequest(BaseModel):
    session_id: str
    image: str = Field(..., description="Base64-encoded image")


class StopSessionRequest(BaseModel):
    session_id: str


class CorrectionRequest(BaseModel):
    image_id: str
    correct_class: str
    bbox: list[float] | None = None
    reviewed_by: str = "anonymous"


class TrainRequest(BaseModel):
    model: str = "yolov8n"
    dataset: str = ""
    task: str = "detection"
    epochs: int = 5
    batch_size: int = 16


class SaveModelRequest(BaseModel):
    model: str
    task: str = "detection"
    name: str = ""


class LoadModelRequest(BaseModel):
    task: str = "detection"
    name: str = ""


class ExportModelRequest(BaseModel):
    model_id: str
    format: str = "onnx"
    quantization: str = "fp16"


class SegmentRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    model: str = "yolov8n-seg"
    confidence_threshold: float = 0.5
    return_masks: bool = True


class PeerRequest(BaseModel):
    name: str
    url: str
    models: list[str] = []
    gpu_vram_gb: float = 0
    priority: int = 0
    timeout: float = 30.0


class EscalateRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    model: str = ""
    confidence_threshold: float = 0.5
    opinions: list[dict[str, Any]] = []


class CreatePackageRequest(BaseModel):
    model_id: str
    model_path: str = ""
    description: str = ""


class ImportPackageRequest(BaseModel):
    source: str = Field(..., description="File path, URL, or peer name")
    model_id: str = ""


# =============================================================================
# Streaming endpoints
# =============================================================================


@router.post("/stream/start")
async def start_streaming_session(request: StartSessionRequest) -> Any:
    """Start a vision streaming session with optional cascade."""
    return await VisionStreamingService.start_session(request.model_dump())


@router.post("/stream/frame")
async def process_frame(request: ProcessFrameRequest) -> Any:
    """Process a video frame in an active streaming session."""
    return await VisionStreamingService.process_frame(request.model_dump())


@router.post("/stream/stop")
async def stop_streaming_session(request: StopSessionRequest) -> Any:
    """Stop a streaming session."""
    return await VisionStreamingService.stop_session(request.model_dump())


@router.get("/stream/sessions")
async def list_sessions() -> Any:
    """List active streaming sessions."""
    return await VisionStreamingService.list_sessions()


@router.get("/stream/sessions/{session_id}/stats")
async def get_session_stats(session_id: str) -> Any:
    """Get statistics for a streaming session."""
    return await VisionStreamingService.get_session_stats(session_id)


@router.post("/corrections")
async def submit_correction(request: CorrectionRequest) -> Any:
    """Submit a human correction for a detection."""
    return await VisionStreamingService.submit_correction(request.model_dump())


@router.get("/replay-buffer")
async def get_replay_buffer_status() -> Any:
    """Get replay buffer status and sample counts."""
    return await VisionStreamingService.get_replay_buffer_status()


@router.post("/replay-buffer/clear")
async def clear_replay_buffer() -> Any:
    """Clear all samples from the replay buffer."""
    return await VisionStreamingService.clear_replay_buffer()


# =============================================================================
# Training endpoints
# =============================================================================


@router.post("/train")
async def start_training(request: TrainRequest) -> Any:
    """Start a vision model training job."""
    return await VisionTrainingService.start_training(request.model_dump())


@router.get("/train")
async def list_training_jobs() -> Any:
    """List all training jobs."""
    return await VisionTrainingService.list_training_jobs()


@router.get("/train/{job_id}")
async def get_training_status(job_id: str) -> Any:
    """Get status of a training job."""
    return await VisionTrainingService.get_training_status(job_id)


@router.delete("/train/{job_id}")
async def cancel_training(job_id: str) -> Any:
    """Cancel a training job."""
    return await VisionTrainingService.cancel_training(job_id)


@router.get("/auto-train/status")
async def get_auto_train_status() -> Any:
    """Get auto-training pipeline status."""
    return await VisionTrainingService.get_auto_train_status()


@router.post("/auto-train/trigger")
async def trigger_auto_training() -> Any:
    """Manually trigger auto-training."""
    return await VisionTrainingService.trigger_auto_training()


# =============================================================================
# Model management endpoints
# =============================================================================


@router.get("/models")
async def list_vision_models() -> Any:
    """List available vision models."""
    return await VisionModelService.list_models()


@router.post("/models/save")
async def save_vision_model(request: SaveModelRequest) -> Any:
    """Save a vision model to disk."""
    return await VisionModelService.save_model(request.model_dump())


@router.post("/models/load")
async def load_vision_model(request: LoadModelRequest) -> Any:
    """Load a saved vision model."""
    return await VisionModelService.load_model(request.model_dump())


@router.delete("/models/{task}/{name}")
async def delete_vision_model(task: str, name: str) -> Any:
    """Delete a saved vision model."""
    return await VisionModelService.delete_model(task, name)


@router.post("/models/export")
async def export_vision_model(request: ExportModelRequest) -> Any:
    """Export a model to ONNX/CoreML/TensorRT/TFLite/OpenVINO with quantization."""
    return await VisionModelService.export_model(request.model_dump())


# =============================================================================
# Segmentation endpoint
# =============================================================================


@router.post("/segment")
async def segment_image(request: SegmentRequest) -> Any:
    """Segment objects in an image."""
    return await VisionSegmentationService.segment(request.model_dump())


# =============================================================================
# Federation endpoints
# =============================================================================


@router.get("/federation/peers")
async def list_peers() -> Any:
    """List federation peers."""
    return await VisionFederationService.list_peers()


@router.post("/federation/peers")
async def register_peer(request: PeerRequest) -> Any:
    """Register a federation peer."""
    return await VisionFederationService.register_peer(request.model_dump())


@router.delete("/federation/peers/{name}")
async def remove_peer(name: str) -> Any:
    """Remove a federation peer."""
    return await VisionFederationService.remove_peer(name)


@router.get("/federation/status")
async def federation_status() -> Any:
    """Get federation health status."""
    return await VisionFederationService.federation_status()


@router.post("/federation/escalate")
async def handle_escalation(request: EscalateRequest) -> Any:
    """Handle inbound escalation from a peer node."""
    return await VisionFederationService.escalate(request.model_dump())


@router.get("/federation/packages")
async def list_packages() -> Any:
    """List available model packages."""
    return await VisionFederationService.list_packages()


@router.post("/federation/packages")
async def create_package(request: CreatePackageRequest) -> Any:
    """Create a model package from a trained model."""
    return await VisionFederationService.create_package(request.model_dump())


@router.post("/federation/packages/import")
async def import_package(request: ImportPackageRequest) -> Any:
    """Import a model package."""
    return await VisionFederationService.import_package(request.model_dump())
