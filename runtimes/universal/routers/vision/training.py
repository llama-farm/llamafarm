"""Training router — /v1/vision/train endpoints"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.error_handler import handle_endpoint_errors
from vision_training.trainer import TrainingConfig, get_trainer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-training"])


class TrainConfigRequest(BaseModel):
    epochs: int = Field(default=10, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=0.001, gt=0.0)

class TrainRequest(BaseModel):
    model: str
    dataset: str
    task: Literal["detection", "classification"]
    config: TrainConfigRequest = Field(default_factory=TrainConfigRequest)
    base_model: str | None = None

class TrainResponse(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    metrics: dict | None = None

class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    current_epoch: int | None = None
    total_epochs: int | None = None
    metrics: dict | None = None
    error: str | None = None


@router.post("/v1/vision/train", response_model=TrainResponse)
@handle_endpoint_errors("vision_train")
async def start_training(request: TrainRequest) -> TrainResponse:
    """Start a training job."""
    trainer = get_trainer()
    config = TrainingConfig(
        epochs=request.config.epochs,
        batch_size=request.config.batch_size,
        learning_rate=request.config.learning_rate,
    )
    job = await trainer.start_training(
        model_id=request.model, dataset_path=request.dataset,
        task=request.task, config=config, base_model=request.base_model,
    )
    return TrainResponse(job_id=job.job_id, status=job.status.value,
                         progress=job.progress, metrics=job.metrics or None)


@router.get("/v1/vision/train/{job_id}", response_model=TrainStatusResponse)
@handle_endpoint_errors("vision_train_status")
async def get_training_status(job_id: str) -> TrainStatusResponse:
    """Get training job status."""
    job = get_trainer().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return TrainStatusResponse(
        job_id=job.job_id, status=job.status.value, progress=job.progress,
        current_epoch=job.current_epoch, total_epochs=job.config.epochs,
        metrics=job.metrics or None, error=job.error,
    )


@router.delete("/v1/vision/train/{job_id}")
@handle_endpoint_errors("vision_train_cancel")
async def cancel_training(job_id: str) -> dict[str, Any]:
    """Cancel a training job."""
    trainer = get_trainer()
    if not await trainer.cancel_job(job_id):
        job = trainer.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        raise HTTPException(status_code=400, detail=f"Cannot cancel (status: {job.status.value})")
    return {"job_id": job_id, "cancelled": True}
