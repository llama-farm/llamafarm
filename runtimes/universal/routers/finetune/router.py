"""FastAPI router for fine-tuning endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException

from finetune.data_prep import validate_cpt_dataset, validate_sft_dataset
from finetune.helpers import (
    check_template_compatibility,
    get_supported_templates,
    validate_model_for_finetune,
)
from finetune.trainer import (
    CPTJobConfig,
    FineTuneTrainer,
    SFTJobConfig,
)
from routers.finetune.types import (
    CPTRequest,
    JobStatus,
    SFTRequest,
    ValidateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/finetune", tags=["finetune"])

# Global trainer instance (set by server.py)
_trainer: FineTuneTrainer | None = None


def set_trainer(trainer: FineTuneTrainer):
    """Set the global trainer instance."""
    global _trainer
    _trainer = trainer


def get_trainer() -> FineTuneTrainer:
    """Get the global trainer instance."""
    if _trainer is None:
        raise HTTPException(status_code=503, detail="Fine-tuning service not available")
    return _trainer


@router.post("/sft")
async def create_sft_job(request: SFTRequest) -> JobStatus:
    """Create a supervised fine-tuning job.
    
    Args:
        request: SFT job configuration
    
    Returns:
        Job status with job_id
    """
    trainer = get_trainer()
    
    # Validate model
    validation = validate_model_for_finetune(request.model)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model validation failed: {validation['errors']}"
        )
    
    if validation["warnings"]:
        logger.warning(f"Model warnings: {validation['warnings']}")
    
    # Validate dataset
    dataset_validation = validate_sft_dataset(
        request.dataset,
        request.dataset_format
    )
    
    if not dataset_validation.valid:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset validation failed: {dataset_validation.errors}"
        )
    
    # Create job config
    job_id = str(uuid.uuid4())[:8]
    
    config = SFTJobConfig(
        job_id=job_id,
        model=request.model,
        dataset=request.dataset,
        dataset_format=request.dataset_format,
        chat_template=request.chat_template,
        train_on_responses_only=request.train_on_responses_only,
        output_dir=request.output_dir,
        output_gguf=request.output_gguf,
        quantization=request.quantization,
        lora_rank=request.lora_rank,
        lora_alpha=request.lora_alpha,
        target_modules=request.target_modules,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        max_seq_length=request.max_seq_length,
        max_steps=request.max_steps,
        warmup_steps=request.warmup_steps,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
    )
    
    # Queue job
    job = await trainer.queue_sft_job(config)
    
    return JobStatus(
        job_id=job.job_id,
        status=job.status.value,
        type=job.type,
        model=job.model,
        progress=job.progress,
        metrics=job.metrics if job.metrics else None,
        output_dir=job.output_dir,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post("/cpt")
async def create_cpt_job(request: CPTRequest) -> JobStatus:
    """Create a continued pre-training job.
    
    Args:
        request: CPT job configuration
    
    Returns:
        Job status with job_id
    """
    trainer = get_trainer()
    
    # Validate model
    validation = validate_model_for_finetune(request.model)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model validation failed: {validation['errors']}"
        )
    
    # Validate dataset
    dataset_validation = validate_cpt_dataset(request.dataset)
    
    if not dataset_validation.valid:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset validation failed: {dataset_validation.errors}"
        )
    
    # Create job config
    job_id = str(uuid.uuid4())[:8]
    
    config = CPTJobConfig(
        job_id=job_id,
        model=request.model,
        dataset=request.dataset,
        output_dir=request.output_dir,
        output_gguf=request.output_gguf,
        quantization=request.quantization,
        lora_rank=request.lora_rank,
        lora_alpha=request.lora_alpha,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        embedding_learning_rate=request.embedding_learning_rate,
        max_seq_length=request.max_seq_length,
        max_steps=request.max_steps,
    )
    
    # Queue job
    job = await trainer.queue_cpt_job(config)
    
    return JobStatus(
        job_id=job.job_id,
        status=job.status.value,
        type=job.type,
        model=job.model,
        progress=job.progress,
        metrics=job.metrics if job.metrics else None,
        output_dir=job.output_dir,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.get("/jobs")
async def list_jobs() -> list[JobStatus]:
    """List all training jobs.
    
    Returns:
        List of job statuses
    """
    trainer = get_trainer()
    jobs = trainer.list_jobs()
    
    return [
        JobStatus(
            job_id=job.job_id,
            status=job.status.value,
            type=job.type,
            model=job.model,
            progress=job.progress,
            metrics=job.metrics if job.metrics else None,
            output_dir=job.output_dir,
            error=job.error,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
        for job in jobs
    ]


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> JobStatus:
    """Get status of a specific job.
    
    Args:
        job_id: Job ID
    
    Returns:
        Job status
    """
    trainer = get_trainer()
    job = trainer.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobStatus(
        job_id=job.job_id,
        status=job.status.value,
        type=job.type,
        model=job.model,
        progress=job.progress,
        metrics=job.metrics if job.metrics else None,
        output_dir=job.output_dir,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict:
    """Cancel a training job.
    
    Args:
        job_id: Job ID to cancel
    
    Returns:
        Cancellation result
    """
    trainer = get_trainer()
    success = await trainer.cancel_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Could not cancel job {job_id} (not found or already completed)"
        )
    
    return {"job_id": job_id, "cancelled": True}


@router.post("/validate")
async def validate_dataset(request: ValidateRequest) -> dict:
    """Validate a dataset for fine-tuning.
    
    Args:
        request: Dataset validation request
    
    Returns:
        Validation result
    """
    result = validate_sft_dataset(
        request.dataset,
        request.dataset_format
    )
    
    return {
        "valid": result.valid,
        "format": result.format,
        "num_examples": result.num_examples,
        "errors": result.errors,
        "warnings": result.warnings,
    }


@router.get("/templates")
async def list_templates() -> dict:
    """Get list of supported chat templates.
    
    Returns:
        Dict with templates list
    """
    templates = get_supported_templates()
    
    return {
        "templates": templates,
        "count": len(templates)
    }


@router.get("/templates/{model_name:path}")
async def get_model_template(model_name: str) -> dict:
    """Get recommended template for a model.
    
    Args:
        model_name: Model name or path
    
    Returns:
        Template recommendation
    """
    result = check_template_compatibility(model_name)
    
    return {
        "model": model_name,
        "recommended_template": result["template"],
        "confidence": result["confidence"],
    }
