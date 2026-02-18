"""Evaluation router — /v1/vision/eval endpoints"""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.eval_model import DEFAULT_WEIGHTS, EvalDB, EvalResult, ModelEvaluator
from services.error_handler import handle_endpoint_errors

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-eval"])

# ── Shared state ────────────────────────────────────────────────────────────

_eval_jobs: dict[str, dict[str, Any]] = {}
_eval_lock = asyncio.Lock()
_evaluator: ModelEvaluator | None = None

_VISION_MODELS_DIR: Path | None = None


def set_eval_models_dir(d: Path) -> None:
    global _VISION_MODELS_DIR
    _VISION_MODELS_DIR = d


def _get_evaluator() -> ModelEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = ModelEvaluator()
    return _evaluator


# ── Request/Response models ─────────────────────────────────────────────────


class EvalRequest(BaseModel):
    """Evaluate a model against a dataset."""
    model: str = Field(..., description="Model name or path to .pt file")
    dataset: str = Field(..., description="Path to dataset YAML")
    imgsz: int = Field(640, ge=320, le=2560, description="Image size for evaluation")
    batch_size: int = Field(16, ge=1, le=256, description="Batch size")
    weights: dict[str, float] | None = Field(
        None, description="Scoring weights override (keys: mAP50_95, mAP50, small_object_recall, f1, speed)"
    )


class EvalCompareRequest(BaseModel):
    """Head-to-head model comparison."""
    model_a: str = Field(..., description="First model name or path")
    model_b: str = Field(..., description="Second model name or path")
    dataset: str = Field(..., description="Dataset YAML for evaluation")
    imgsz: int = Field(640, ge=320, le=2560)
    batch_size: int = Field(16, ge=1, le=256)


class EvalJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class LeaderboardEntry(BaseModel):
    rank: int
    model_name: str
    score: float
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1: float
    inference_ms: float
    small_object_recall: float
    dataset: str
    timestamp: str


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_model_path(model: str) -> str:
    """Resolve model name to .pt path. Accepts paths or model IDs."""
    p = Path(model)
    if p.exists() and p.suffix == ".pt":
        return str(p)

    # Check vision models directory
    if _VISION_MODELS_DIR:
        # Sanitize: basename only, reject traversal
        safe_name = Path(model).name
        if safe_name != model or ".." in model or ":" in model or "\\" in model:
            raise HTTPException(400, "Invalid model name")

        current = _VISION_MODELS_DIR / safe_name / "current.pt"
        if current.exists():
            return str(current)

        # Check versioned — find latest
        model_dir = _VISION_MODELS_DIR / safe_name
        if model_dir.is_dir():
            versions = sorted(model_dir.glob("v*.pt"), reverse=True)
            if versions:
                return str(versions[0])

    raise HTTPException(404, f"Model not found: {model}")


def _create_job(job_type: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    _eval_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "type": job_type,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    }
    return job_id


# ── Background runners ──────────────────────────────────────────────────────


async def _run_eval(job_id: str, model_path: str, request: EvalRequest) -> None:
    try:
        _eval_jobs[job_id]["status"] = "running"
        evaluator = _get_evaluator()
        result = await evaluator.evaluate(
            model_path=model_path,
            data_yaml=request.dataset,
            model_name=request.model,
            imgsz=request.imgsz,
            batch=request.batch_size,
            weights=request.weights,
        )
        _eval_jobs[job_id]["status"] = "completed"
        _eval_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        _eval_jobs[job_id]["result"] = result.to_dict()
    except Exception as e:
        logger.error(f"Eval job {job_id} failed: {e}")
        _eval_jobs[job_id]["status"] = "failed"
        _eval_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        _eval_jobs[job_id]["error"] = "Evaluation failed"


async def _run_compare(
    job_id: str, path_a: str, path_b: str, request: EvalCompareRequest
) -> None:
    try:
        _eval_jobs[job_id]["status"] = "running"
        evaluator = _get_evaluator()
        result_a = await evaluator.evaluate(
            model_path=path_a, data_yaml=request.dataset,
            model_name=request.model_a, imgsz=request.imgsz, batch=request.batch_size,
        )
        result_b = await evaluator.evaluate(
            model_path=path_b, data_yaml=request.dataset,
            model_name=request.model_b, imgsz=request.imgsz, batch=request.batch_size,
        )
        comparison = evaluator.compare(result_a, result_b)
        _eval_jobs[job_id]["status"] = "completed"
        _eval_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        _eval_jobs[job_id]["result"] = comparison
    except Exception as e:
        logger.error(f"Compare job {job_id} failed: {e}")
        _eval_jobs[job_id]["status"] = "failed"
        _eval_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        _eval_jobs[job_id]["error"] = "Comparison failed"


# ── Auto-eval callback (called from training completion) ────────────────────


async def auto_eval_after_training(
    model_id: str,
    model_path: str,
    dataset: str,
    auto_promote: bool = False,
    weights: dict[str, float] | None = None,
) -> EvalResult | None:
    """Run evaluation after training completes. Optionally auto-promote.

    Called from the training pipeline when auto_eval=true.
    Returns the EvalResult or None on failure.
    """
    try:
        evaluator = _get_evaluator()
        result = await evaluator.evaluate(
            model_path=model_path,
            data_yaml=dataset,
            model_name=model_id,
            weights=weights,
        )
        logger.info(f"Auto-eval for {model_id}: score={result.score:.4f} mAP50={result.mAP50:.4f}")

        if auto_promote:
            db = evaluator.db
            best = db.best_model(dataset)

            if not best or best["model_name"] == model_id:
                # First model or this IS the best → promote
                _promote_model(model_id, model_path)
                db.record_promotion(model_id, result.eval_id or 0, "auto-promoted as best model")
                logger.info(f"Auto-promoted {model_id} (first or best)")
            elif result.score > best["score"]:
                # New model beats current best → promote
                _promote_model(model_id, model_path)
                db.record_promotion(
                    model_id, result.eval_id or 0,
                    f"auto-promoted: score {result.score:.4f} > {best['score']:.4f} ({best['model_name']})",
                )
                logger.info(
                    f"Auto-promoted {model_id} (score {result.score:.4f} > {best['score']:.4f})"
                )
            else:
                logger.info(
                    f"Not promoting {model_id}: score {result.score:.4f} <= "
                    f"best {best['score']:.4f} ({best['model_name']})"
                )

        return result
    except Exception as e:
        logger.error(f"Auto-eval failed for {model_id}: {e}")
        return None


def _promote_model(model_id: str, model_path: str) -> None:
    """Update current.pt symlink to point to the evaluated model."""
    if not _VISION_MODELS_DIR:
        return
    import shutil

    model_dir = _VISION_MODELS_DIR / Path(model_id).name
    model_dir.mkdir(parents=True, exist_ok=True)
    current = model_dir / "current.pt"
    src = Path(model_path)
    if src.exists():
        shutil.copy2(str(src), str(current))
        logger.info(f"Promoted {model_id}: {current}")


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/v1/vision/eval", response_model=EvalJobResponse)
@handle_endpoint_errors("vision_eval")
async def start_eval(request: EvalRequest) -> EvalJobResponse:
    """Evaluate a model against a dataset. Returns a job ID to poll."""
    model_path = _resolve_model_path(request.model)
    job_id = _create_job("eval")
    asyncio.create_task(_run_eval(job_id, model_path, request))
    return EvalJobResponse(**_eval_jobs[job_id])


@router.get("/v1/vision/eval/{job_id}", response_model=EvalJobResponse)
@handle_endpoint_errors("vision_eval_status")
async def get_eval_status(job_id: str) -> EvalJobResponse:
    """Poll eval job status and results."""
    if job_id not in _eval_jobs:
        raise HTTPException(404, f"Eval job not found: {job_id}")
    return EvalJobResponse(**_eval_jobs[job_id])


@router.post("/v1/vision/eval/compare", response_model=EvalJobResponse)
@handle_endpoint_errors("vision_eval_compare")
async def start_compare(request: EvalCompareRequest) -> EvalJobResponse:
    """Compare two models on the same dataset."""
    path_a = _resolve_model_path(request.model_a)
    path_b = _resolve_model_path(request.model_b)
    job_id = _create_job("compare")
    asyncio.create_task(_run_compare(job_id, path_a, path_b, request))
    return EvalJobResponse(**_eval_jobs[job_id])


@router.get("/v1/vision/eval/leaderboard", response_model=list[LeaderboardEntry])
@handle_endpoint_errors("vision_eval_leaderboard")
async def leaderboard(dataset: str | None = None, limit: int = 20) -> list[LeaderboardEntry]:
    """Ranked model leaderboard by composite score."""
    db = _get_evaluator().db
    results = db.leaderboard(dataset, limit)
    return [
        LeaderboardEntry(
            rank=i + 1,
            model_name=r["model_name"],
            score=r.get("score", 0.0),
            mAP50=r.get("mAP50", 0.0),
            mAP50_95=r.get("mAP50_95", 0.0),
            precision=r.get("precision", 0.0),
            recall=r.get("recall", 0.0),
            f1=r.get("f1", 0.0),
            inference_ms=r.get("inference_ms", 0.0),
            small_object_recall=r.get("small_object_recall", 0.0),
            dataset=r.get("dataset", ""),
            timestamp=r.get("timestamp", ""),
        )
        for i, r in enumerate(results)
    ]


@router.get("/v1/vision/eval/leaderboard/{model_name}")
@handle_endpoint_errors("vision_eval_model_history")
async def model_history(model_name: str, limit: int = 50) -> list[dict[str, Any]]:
    """Evaluation history for a specific model."""
    db = _get_evaluator().db
    history = db.model_history(model_name, limit)
    if not history:
        raise HTTPException(404, f"No evaluations found for: {model_name}")
    return history
