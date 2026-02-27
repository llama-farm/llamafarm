"""Training router — /v1/vision/train endpoints"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.error_handler import handle_endpoint_errors
from vision_training.trainer import TrainingConfig, TrainingStatus, get_trainer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-training"])


class TrainConfigRequest(BaseModel):
    epochs: int = Field(default=10, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=0.001, gt=0.0)
    imgsz: int = Field(default=640, ge=320, le=2560, description="Training image size (px)")
    patience: int = Field(default=50, ge=0, le=500, description="Early stopping patience (0=disabled)")

class TrainRequest(BaseModel):
    model: str
    dataset: str
    task: Literal["detection", "classification"]
    config: TrainConfigRequest = Field(default_factory=TrainConfigRequest)
    base_model: str | None = None
    auto_eval: bool = Field(default=False, description="Run evaluation after training completes")
    auto_promote: bool = Field(default=False, description="Auto-promote if eval score beats leaderboard best")
    eval_weights: dict[str, float] | None = Field(
        None, description="Custom scoring weights for auto-eval (keys: mAP50_95, mAP50, small_object_recall, f1, speed)"
    )

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
    started_at: str | None = None
    completed_at: str | None = None


@router.post("/v1/vision/train", response_model=TrainResponse)
@handle_endpoint_errors("vision_train")
async def start_training(request: TrainRequest) -> TrainResponse:
    """Start a training job. Set auto_eval=true to evaluate on completion,
    auto_promote=true to auto-promote if score beats leaderboard best."""
    trainer = get_trainer()
    config = TrainingConfig(
        epochs=request.config.epochs,
        batch_size=request.config.batch_size,
        learning_rate=request.config.learning_rate,
        imgsz=request.config.imgsz,
        patience=request.config.patience,
    )

    # auto_promote implies auto_eval
    auto_eval = request.auto_eval or request.auto_promote

    # Build completion callback for auto-eval
    on_complete = None
    if auto_eval:
        from .evaluation import auto_eval_after_training

        async def on_complete(model_id: str, model_path: str) -> None:
            await auto_eval_after_training(
                model_id=model_id,
                model_path=model_path,
                dataset=request.dataset,
                auto_promote=request.auto_promote,
                weights=request.eval_weights,
            )

    job = await trainer.start_training(
        model_id=request.model, dataset_path=request.dataset,
        task=request.task, config=config, base_model=request.base_model,
        on_complete=on_complete,
    )
    return TrainResponse(job_id=job.job_id, status=job.status.value,
                         progress=job.progress, metrics=job.metrics or None)


@router.get("/v1/vision/train/{job_id}", response_model=TrainStatusResponse)
@handle_endpoint_errors("vision_train_status")
async def get_training_status(job_id: str) -> TrainStatusResponse:
    """Get the status of a training job."""
    trainer = get_trainer()
    job = trainer.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return TrainStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        current_epoch=job.current_epoch,
        total_epochs=job.config.epochs,
        metrics=job.metrics or None,
        error=job.error,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.get("/v1/vision/train")
@handle_endpoint_errors("vision_train_list")
async def list_training_jobs(
    status: str | None = None,
) -> dict[str, Any]:
    """List training jobs."""
    trainer = get_trainer()

    status_filter = None
    if status:
        try:
            status_filter = TrainingStatus(status)
        except ValueError as err:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}"
            ) from err

    jobs = trainer.list_jobs(status=status_filter)

    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "model_id": j.model_id,
                "task": j.task,
                "status": j.status.value,
                "progress": j.progress,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            }
            for j in jobs
        ],
        "total": len(jobs),
    }


@router.delete("/v1/vision/train/{job_id}")
@handle_endpoint_errors("vision_train_cancel")
async def cancel_training(job_id: str) -> dict[str, Any]:
    """Cancel a running training job."""
    trainer = get_trainer()

    cancelled = await trainer.cancel_job(job_id)

    if not cancelled:
        job = trainer.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} cannot be cancelled (status: {job.status.value})"
        )

    return {"job_id": job_id, "cancelled": True}
