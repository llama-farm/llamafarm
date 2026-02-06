"""Review Queue Router for human-in-the-loop verification.

Server-only endpoints (not mirrored in runtime).
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Form

from services.vision import VisionReviewService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/review", tags=["vision-review"])


@router.get("")
async def list_review_queue(
    status: str = "pending",
    limit: int = 50,
    offset: int = 0,
    source: str | None = None,
) -> dict[str, Any]:
    """List items in the review queue.
    
    Query params:
    - status: pending, approved, rejected, corrected, all
    - limit: Max items to return
    - offset: Pagination offset
    - source: Filter by source (e.g., "stream:camera1")
    
    Returns list of images pending human review with thumbnails
    and initial predictions.
    """
    return await VisionReviewService.list_queue(
        status=status,
        limit=limit,
        offset=offset,
        source=source,
    )


@router.post("")
async def submit_review(
    image_id: str = Form(..., description="Image ID to review"),
    decision: str = Form(..., description="correct, wrong, or adjusted"),
    corrections: str = Form(default="", description="JSON corrections if adjusted"),
    reviewed_by: str = Form(default="anonymous", description="Reviewer ID"),
) -> dict[str, Any]:
    """Submit a human review decision.
    
    Decisions:
    - correct: Prediction was correct
    - wrong: Prediction was wrong (false positive)
    - adjusted: Prediction needed correction (returns corrected boxes)
    
    Corrections are added to replay buffer for incremental training.
    """
    import json
    
    corrections_list = None
    if corrections:
        try:
            corrections_list = json.loads(corrections)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid corrections JSON")
    
    return await VisionReviewService.submit_review(
        image_id=image_id,
        decision=decision,
        corrections=corrections_list,
        reviewed_by=reviewed_by,
    )


@router.get("/stats")
async def get_review_stats() -> dict[str, Any]:
    """Get review queue statistics.
    
    Returns counts of pending, reviewed, corrections,
    and estimated model accuracy.
    """
    return await VisionReviewService.get_stats()


@router.post("/batch")
async def batch_review(
    image_ids: str = Form(..., description="Comma-separated image IDs"),
    decision: str = Form(..., description="correct or wrong"),
    reviewed_by: str = Form(default="anonymous"),
) -> dict[str, Any]:
    """Batch approve or reject multiple images.
    
    For quickly processing a set of similar predictions.
    """
    ids = [id.strip() for id in image_ids.split(",") if id.strip()]
    
    if not ids:
        raise HTTPException(400, "No image IDs provided")
    
    results = []
    for image_id in ids:
        result = await VisionReviewService.submit_review(
            image_id=image_id,
            decision=decision,
            reviewed_by=reviewed_by,
        )
        results.append(result)
    
    return {
        "processed": len(results),
        "results": results,
    }
