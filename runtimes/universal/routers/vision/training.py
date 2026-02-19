"""Training router for vision model fine-tuning.

Provides endpoints for:
- Starting training jobs
- Checking training status
- Listing jobs
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from api_types.vision import (
    TrainRequest,
    TrainResponse,
    TrainStatusResponse,
)
from services.error_handler import handle_endpoint_errors
from vision_training.trainer import (
    TrainingConfig,
    TrainingStatus,
    get_trainer,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision-training"])


@router.post("/v1/vision/train", response_model=TrainResponse)
@handle_endpoint_errors("vision_train")
async def start_training(request: TrainRequest) -> TrainResponse:
    """Start a training job for a vision model.
    
    Trains or fine-tunes a model on a custom dataset.
    Training runs asynchronously - use the job_id to check status.
    
    Example:
    ```json
    {
        "model": "yolov8n",
        "dataset": "/path/to/dataset",
        "task": "detection",
        "config": {
            "epochs": 10,
            "batch_size": 16
        }
    }
    ```
    """
    trainer = get_trainer()
    
    # Convert API config to training config
    config = TrainingConfig(
        epochs=request.config.epochs,
        batch_size=request.config.batch_size,
        learning_rate=request.config.learning_rate,
        use_ewc=request.config.use_ewc,
        ewc_lambda=request.config.ewc_lambda,
        use_replay=request.config.use_replay,
        replay_ratio=request.config.replay_ratio,
        validation_split=request.config.validation_split,
    )
    
    job = await trainer.start_training(
        model_id=request.model,
        dataset_path=request.dataset,
        task=request.task,
        config=config,
        base_model=request.base_model,
    )
    
    return TrainResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        metrics=job.metrics or None,
    )


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
        started_at=job.started_at,
        completed_at=job.completed_at,
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
