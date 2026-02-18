"""Review queue router."""

import logging
from typing import Any

from fastapi import APIRouter, Form
from server.services.vision import VisionReviewService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision/review", tags=["vision-review"])


@router.get("/pending")
async def get_pending(limit: int = 50, source: str | None = None) -> dict[str, Any]:
    """Get images pending review."""
    return await VisionReviewService.get_pending(limit=limit, source=source)


@router.post("/decide")
async def submit_decision(
    image_id: str = Form(...),
    decision: str = Form(..., description="correct, wrong, or adjusted"),
    corrected_class: str | None = Form(default=None),
) -> dict[str, Any]:
    """Submit a review decision."""
    corrections = None
    if corrected_class:
        corrections = [{"class_name": corrected_class}]
    return await VisionReviewService.submit_decision(
        image_id=image_id, decision=decision, corrections=corrections,
    )
