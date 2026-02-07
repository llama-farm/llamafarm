"""Review Queue Service for human-in-the-loop verification.

Manages the review queue for uncertain detections and corrections.
Wires into the replay buffer for auto-learning.
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_image_store = None
_replay_buffer = None


def get_image_store():
    """Get or create the image store."""
    global _image_store
    if _image_store is None:
        try:
            from storage.image_store import ImageStore
            _image_store = ImageStore()
        except ImportError:
            logger.warning("ImageStore not available")
    return _image_store


def get_replay_buffer():
    """Get or create the replay buffer."""
    global _replay_buffer
    if _replay_buffer is None:
        try:
            from vision_training.replay_buffer import get_replay_buffer as get_buffer
            _replay_buffer = get_buffer()
        except ImportError:
            logger.warning("ReplayBuffer not available")
    return _replay_buffer


class VisionReviewService:
    """Service for managing the vision review queue.
    
    The review queue contains images that need human verification:
    - Low-confidence detections
    - Flagged for review by streaming detector
    - User-submitted for labeling
    
    Corrections are automatically added to the replay buffer for
    incremental training of the primary model.
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
        
        Args:
            status: Filter by status (pending, approved, rejected, corrected, all)
            limit: Max items to return
            offset: Pagination offset
            source: Filter by source (e.g., "stream:camera1")
            
        Returns:
            Dict with items list and counts
        """
        store = get_image_store()
        if store is None:
            return {
                "items": [],
                "total": 0,
                "pending": 0,
                "status_filter": status,
            }
        
        try:
            # Query image store
            items = await store.list_for_review(
                status=status if status != "all" else None,
                limit=limit,
                offset=offset,
                source=source,
            )
            
            # Get counts
            total = await store.count_reviews(status=status if status != "all" else None)
            pending = await store.count_reviews(status="pending")
            
            return {
                "items": [cls._record_to_item(item) for item in items],
                "total": total,
                "pending": pending,
                "status_filter": status,
            }
        except Exception as e:
            logger.error(f"Failed to list review queue: {e}")
            return {
                "items": [],
                "total": 0,
                "pending": 0,
                "status_filter": status,
                "error": str(e),
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
            
        Returns:
            Dict with processing result
        """
        store = get_image_store()
        buffer = get_replay_buffer()
        
        try:
            # Get image record
            record = None
            if store:
                record = await store.get_image(image_id)
            
            # Handle different decisions
            if decision == "correct":
                # Mark as approved, no training needed
                if store:
                    await store.mark_reviewed(
                        image_id=image_id,
                        decision="approved",
                        reviewed_by=reviewed_by,
                    )
                    
            elif decision == "wrong":
                # Mark as rejected (false positive), no training value
                if store:
                    await store.mark_reviewed(
                        image_id=image_id,
                        decision="rejected",
                        reviewed_by=reviewed_by,
                    )
                    
            elif decision == "adjusted" and corrections:
                # Add corrections to replay buffer for training
                if store:
                    await store.mark_reviewed(
                        image_id=image_id,
                        decision="corrected",
                        reviewed_by=reviewed_by,
                    )
                
                # Add to replay buffer
                if buffer and record:
                    for correction in corrections:
                        buffer.add_correction(
                            image_id=f"{image_id}_{correction.get('class_name', 'unknown')}",
                            image_path=record.file_path if record else "",
                            corrected_label=correction.get("class_name", "unknown"),
                            original_confidence=correction.get("original_confidence", 0.0),
                        )
                    logger.info(f"Added {len(corrections)} corrections to replay buffer")
            
            # Get next pending image
            next_image_id = None
            if store:
                next_items = await store.list_for_review(
                    status="pending",
                    limit=1,
                    offset=0,
                )
                if next_items:
                    next_image_id = next_items[0].id
            
            return {
                "image_id": image_id,
                "decision": decision,
                "processed": True,
                "next_image_id": next_image_id,
                "replay_buffer_size": len(buffer) if buffer else 0,
            }
            
        except Exception as e:
            logger.error(f"Failed to process review: {e}")
            return {
                "image_id": image_id,
                "decision": decision,
                "processed": False,
                "error": str(e),
            }
    
    @classmethod
    async def get_stats(cls) -> dict[str, Any]:
        """Get review queue statistics."""
        store = get_image_store()
        buffer = get_replay_buffer()
        
        try:
            pending = 0
            reviewed_today = 0
            corrections = 0
            
            if store:
                pending = await store.count_reviews(status="pending")
                reviewed_today = await store.count_reviews_today()
                corrections = await store.count_reviews(status="corrected")
            
            # Calculate estimated accuracy from corrections ratio
            total_reviewed = reviewed_today if reviewed_today > 0 else 1
            accuracy_estimate = 1.0 - (corrections / total_reviewed) if total_reviewed > 0 else 0.0
            
            return {
                "pending": pending,
                "reviewed_today": reviewed_today,
                "corrections": corrections,
                "accuracy_estimate": accuracy_estimate,
                "replay_buffer_size": len(buffer) if buffer else 0,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "pending": 0,
                "reviewed_today": 0,
                "corrections": 0,
                "accuracy_estimate": 0.0,
                "error": str(e),
            }
    
    @classmethod
    async def get_image(cls, image_id: str) -> dict[str, Any] | None:
        """Get a specific image for review."""
        store = get_image_store()
        if store is None:
            return None
        
        try:
            record = await store.get_image(image_id)
            if record:
                return cls._record_to_item(record)
            return None
        except Exception as e:
            logger.error(f"Failed to get image: {e}")
            return None
    
    @classmethod
    def _record_to_item(cls, record: Any) -> dict[str, Any]:
        """Convert storage record to review item."""
        return {
            "image_id": record.id,
            "image_url": f"/v1/vision/images/{record.id}",
            "thumbnail_url": f"/v1/vision/images/{record.id}/thumbnail",
            "timestamp": record.created_at.isoformat() if record.created_at else None,
            "prediction": None,  # TODO: Include detection info
            "confidence": 0.0,
            "model": "",
            "source": record.source,
            "status": "reviewed" if record.reviewed else "pending",
        }
