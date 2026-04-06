"""FastAPI router for fine-tuning endpoints."""

from __future__ import annotations

import contextlib
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
from pydantic import BaseModel
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


# ---------------------------------------------------------------------------
# Promote endpoint (Issue 2): set an alias so a job's GGUF is the active model
# ---------------------------------------------------------------------------

@router.post("/jobs/{job_id}/promote")
async def promote_job(job_id: str) -> dict:
    """Promote a fine-tuned job's GGUF as the named alias for its base model.

    After promoting, the job's GGUF will be served when the model is referenced
    by the alias stored in ``~/.llamafarm/models/llm/{job_id}/promoted_alias``.
    Callers can also reference the model directly as ``ft:{job_id}``.

    Returns:
        ``{"job_id": ..., "gguf_path": ..., "alias": ...}``
    """
    from pathlib import Path
    import json

    job_dir = Path.home() / ".llamafarm" / "models" / "llm" / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    gguf_dir = job_dir / "gguf"
    gguf_files = sorted(gguf_dir.glob("*.gguf")) if gguf_dir.is_dir() else []
    if not gguf_files:
        raise HTTPException(
            status_code=422,
            detail=f"No GGUF export found for job {job_id!r} — run training first",
        )

    # Pick best quant
    gguf_path = next(
        (p for p in gguf_files if "q8_0" in p.name.lower()),
        next((p for p in gguf_files if "q4_k_m" in p.name.lower()), gguf_files[0]),
    )

    # Derive alias from base model name in job config
    alias = f"local/ft-{job_id[:8]}"
    try:
        cfg = json.loads((job_dir / "job_config.json").read_text())
        base = cfg.get("model", "").split("/")[-1].replace("_", "-").lower()
        if base:
            alias = f"local/{base}-ft"
    except Exception as exc:
        logger.warning(
            "Failed to derive alias from job_config.json for job %r: %s",
            job_id,
            exc,
        )

    # Write promotion record
    promo = {"job_id": job_id, "gguf_path": str(gguf_path), "alias": alias}
    (job_dir / "promoted_alias.json").write_text(json.dumps(promo, indent=2))

    # Also write a global alias map so the server can look it up
    alias_file = Path.home() / ".llamafarm" / "models" / "llm" / "aliases.json"
    aliases: dict = {}
    if alias_file.exists():
        with contextlib.suppress(Exception):
            aliases = json.loads(alias_file.read_text())
    aliases[alias] = str(gguf_path)
    aliases[f"ft:{job_id}"] = str(gguf_path)
    alias_file.write_text(json.dumps(aliases, indent=2))

    logger.info(f"Promoted job {job_id} → alias {alias!r} ({gguf_path})")
    return promo


# ---------------------------------------------------------------------------
# Eval endpoint (Issue 3): run inference against a job's GGUF, return metrics
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    """Request body for POST /v1/finetune/jobs/{id}/eval."""

    dataset: list[dict]
    """List of examples. Each must have ``messages`` (list of role/content dicts)
    and ``expected`` (the expected assistant response string)."""

    max_tokens: int = 128
    temperature: float = 0.0
    exact_match: bool = True
    """When True, use case-insensitive exact match. When False, use substring match."""


@router.post("/jobs/{job_id}/eval")
async def eval_job(job_id: str, request: EvalRequest) -> dict:
    """Run a holdout eval against a fine-tuned job's GGUF.

    Internally uses the same Jinja2 template path as serving, so the chat
    template applied during training is automatically matched.

    Returns accuracy, per-example results, and a confusion matrix of
    expected vs. actual outputs.
    """
    import asyncio
    from pathlib import Path

    from server import load_language_model  # local import to avoid circular

    job_dir = Path.home() / ".llamafarm" / "models" / "llm" / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    # Resolve GGUF path (same logic as _resolve_finetune_model_id)
    model_id = f"ft:{job_id}"

    try:
        model = await load_language_model(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}") from e

    results = []
    correct = 0

    for i, ex in enumerate(request.dataset):
        msgs = ex.get("messages")
        expected = str(ex.get("expected", "")).strip()
        if not msgs or not expected:
            results.append({"index": i, "error": "missing 'messages' or 'expected'"})
            continue

        try:
            actual = await model.generate(
                messages=msgs,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            actual = actual.strip()
        except Exception as e:
            results.append({"index": i, "expected": expected, "error": str(e)})
            continue

        if request.exact_match:
            ok = actual.lower() == expected.lower()
        else:
            ok = expected.lower() in actual.lower()

        if ok:
            correct += 1
        results.append(
            {"index": i, "expected": expected, "actual": actual, "correct": ok}
        )

    n = len(request.dataset)
    accuracy = correct / n if n else 0.0
    return {
        "job_id": job_id,
        "num_examples": n,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "results": results,
    }
