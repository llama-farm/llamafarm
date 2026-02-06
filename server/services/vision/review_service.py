"""Review Queue Service for human-in-the-loop verification.

Manages the review queue for uncertain detections and corrections.
"""

import logging
from datetime import datetime
from typing import Any

from core.settings import settings
from services.universal_runtime_service import get_runtime_client

logger = logging.getLogger(__name__)


class VisionReviewService:
    """Service for managing the vision review queue.
    
    The review queue contains images that need human verification:
    - Low-confidence detections
    - Flagged for review by streaming detector
    - User-submitted for labeling
    """
    
    @classmethod
    async def list_queue(
        cls,
        status: str = "pending",
        limit: int = 50,
        offset: int = 0,
        source: str | None = None,
    ) -> dict[str, Any]:
        """List items in the review queue.
        
        Server-only endpoint - not proxied to runtime.
        Uses local SQLite storage.
        """
        # For now, return empty list since storage is in runtime
        # In production, this would query the image store
        return {
            "items": [],
            "total": 0,
            "pending": 0,
            "status_filter": status,
        }
    
    @classmethod
    async def submit_review(
        cls,
        image_id: str,
        decision: str,
        corrections: list[dict] | None = None,
        reviewed_by: str = "anonymous",
    ) -> dict[str, Any]:
        """Submit a human review decision.
        
        Args:
            image_id: Image to review
            decision: correct, wrong, or adjusted
            corrections: Corrected detections (for adjusted)
            reviewed_by: Reviewer identifier
        """
        # In production, this would:
        # 1. Update image store
        # 2. Add to replay buffer if corrected
        # 3. Return next image
        
        return {
            "image_id": image_id,
            "decision": decision,
            "processed": True,
            "next_image_id": None,
        }
    
    @classmethod
    async def get_stats(cls) -> dict[str, Any]:
        """Get review queue statistics."""
        return {
            "pending": 0,
            "reviewed_today": 0,
            "corrections": 0,
            "accuracy_estimate": 0.0,
        }
