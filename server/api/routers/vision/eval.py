"""Eval router — vision model evaluation proxy endpoints.

All endpoints proxy to the Universal Runtime for actual computation.
Uses Form data (multipart) to match the server-side pattern.

NOTE: Specific paths (/eval/compare, /eval/leaderboard) MUST be defined
before parameterized paths (/eval/{job_id}) to avoid FastAPI route shadowing.
"""

import logging
from typing import Any

from fastapi import APIRouter, Form
from server.services.vision import VisionEvalService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision", tags=["vision-eval"])


@router.post("/eval")
async def start_eval(
    model: str = Form(..., description="Model name or path to .pt file"),
    dataset: str = Form(..., description="Path to dataset YAML"),
    imgsz: int = Form(default=640, description="Image size for evaluation"),
    batch_size: int = Form(default=16, description="Batch size"),
    auto_promote: bool = Form(default=False, description="Auto-promote if best"),
) -> dict[str, Any]:
    """Start a model evaluation job. Returns job_id to poll."""
    payload: dict[str, Any] = {
        "model": model, "dataset": dataset,
        "imgsz": imgsz, "batch_size": batch_size,
    }
    if auto_promote:
        payload["auto_promote"] = True
    return await VisionEvalService.start_eval(payload)


@router.post("/eval/compare")
async def compare_models(
    model_a: str = Form(..., description="First model name or path"),
    model_b: str = Form(..., description="Second model name or path"),
    dataset: str = Form(..., description="Dataset YAML"),
    imgsz: int = Form(default=640),
    batch_size: int = Form(default=16),
) -> dict[str, Any]:
    """Head-to-head model comparison."""
    return await VisionEvalService.start_compare({
        "model_a": model_a, "model_b": model_b,
        "dataset": dataset, "imgsz": imgsz, "batch_size": batch_size,
    })


@router.get("/eval/leaderboard")
async def leaderboard(
    dataset: str | None = None,
    limit: int = 20,
) -> Any:
    """Ranked model leaderboard by composite score."""
    return await VisionEvalService.leaderboard(dataset, limit)


@router.get("/eval/leaderboard/{model_name}")
async def model_history(model_name: str, limit: int = 50) -> Any:
    """Evaluation history for a specific model."""
    return await VisionEvalService.model_history(model_name, limit)


# Parameterized path MUST come after specific paths to avoid shadowing
@router.get("/eval/{job_id}")
async def eval_status(job_id: str) -> dict[str, Any]:
    """Poll evaluation job status and results."""
    return await VisionEvalService.get_eval_status(job_id)
