"""Review Queue Service for human-in-the-loop verification.

Manages the review queue for uncertain detections and corrections.
Wires into the replay buffer for auto-learning.

Review items include:
- Bounding boxes from all models that examined the image
- Each model's opinion (class, confidence, bbox)
- Human reviewer can see exactly what each model thought
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_image_store = None
_metadata_store = None
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


def get_metadata_store():
    """Get or create the metadata store (for detection history)."""
    global _metadata_store
    if _metadata_store is None:
        try:
            from storage.image_store import ImageMetadataStore
            _metadata_store = ImageMetadataStore()
        except ImportError:
            logger.warning("ImageMetadataStore not available")
    return _metadata_store


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
        reviewer_type: str = "human",
        reviewer_model_id: str | None = None,
        reviewer_confidence: float | None = None,
    ) -> dict[str, Any]:
        """Submit a review decision (human or model).

        Args:
            image_id: Image to review
            decision: correct, wrong, or adjusted
            corrections: Corrected detections (for adjusted)
            reviewed_by: Reviewer identifier
            reviewer_type: "human" or "model" (audit pipeline)
            reviewer_model_id: Model ID if reviewer_type is "model"
            reviewer_confidence: Model confidence if reviewer_type is "model"

        Returns:
            Dict with processing result
        """
        store = get_image_store()
        buffer = get_replay_buffer()

        # Model reviews get higher training priority than human reviews
        priority = 2.0 if reviewer_type == "human" else 1.8

        try:
            # Get image record
            record = None
            if store:
                record = await store.get_image(image_id)

            # Handle different decisions
            if decision == "correct":
                if store:
                    await store.mark_reviewed(
                        image_id=image_id,
                        decision="approved",
                        reviewed_by=reviewed_by,
                    )

            elif decision == "wrong":
                if store:
                    await store.mark_reviewed(
                        image_id=image_id,
                        decision="rejected",
                        reviewed_by=reviewed_by,
                    )

            elif decision == "adjusted" and corrections:
                if store:
                    await store.mark_reviewed(
                        image_id=image_id,
                        decision="corrected",
                        reviewed_by=reviewed_by,
                    )

                # Add to replay buffer
                if buffer is not None and record:
                    for correction in corrections:
                        buffer.add_correction(
                            image_id=f"{image_id}_{correction.get('class_name', 'unknown')}",
                            image_path=record.file_path if record else "",
                            corrected_label=correction.get("class_name", "unknown"),
                            original_confidence=correction.get("original_confidence", 0.0),
                        )
                    logger.info(
                        f"Added {len(corrections)} corrections to replay buffer "
                        f"(reviewer={reviewer_type})"
                    )

            # Record in metadata store if model reviewer
            if reviewer_type == "model" and reviewer_model_id:
                metadata = get_metadata_store()
                if metadata is not None:
                    try:
                        for correction in (corrections or []):
                            box = correction.get("box", {})
                            metadata.add_detection_history(
                                image_id=image_id,
                                model_id=reviewer_model_id,
                                class_name=correction.get("class_name", ""),
                                confidence=reviewer_confidence or 0.0,
                                box=(
                                    box.get("x1", 0), box.get("y1", 0),
                                    box.get("x2", 0), box.get("y2", 0),
                                ),
                                stage="review",
                                node_id="local",
                            )
                    except Exception as e:
                        logger.warning(f"Failed to record model review: {e}")

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
                "reviewer_type": reviewer_type,
                "next_image_id": next_image_id,
                "replay_buffer_size": len(buffer) if buffer is not None else 0,
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
            
            # Calculate estimated accuracy: corrections / total non-pending reviews
            total_reviewed = await store.count_reviews() if store else 0
            non_pending = total_reviewed - pending if total_reviewed > pending else 0
            accuracy_estimate = 1.0 - (corrections / non_pending) if non_pending > 0 else 0.0
            
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
        """Convert storage record to review item.

        Includes bounding boxes and all model opinions so humans
        can see exactly what each model thought about the image.
        """
        item: dict[str, Any] = {
            "image_id": record.id,
            "image_url": f"/v1/vision/images/{record.id}",
            "thumbnail_url": f"/v1/vision/images/{record.id}/thumbnail",
            "timestamp": record.created_at.isoformat() if record.created_at else None,
            "prediction": None,
            "confidence": 0.0,
            "model": "",
            "source": record.source,
            "status": "reviewed" if record.reviewed else "pending",
            "all_opinions": [],
        }

        # Fetch detection history for this image
        metadata = get_metadata_store()
        if metadata is not None:
            try:
                history = metadata.get_detection_history(record.id)
                if history:
                    # Use the primary detection as the main prediction
                    primary = next(
                        (h for h in history if h.stage == "primary"),
                        history[0],
                    )
                    item["prediction"] = {
                        "box": {
                            "x1": primary.x1,
                            "y1": primary.y1,
                            "x2": primary.x2,
                            "y2": primary.y2,
                        },
                        "class_name": primary.class_name,
                        "confidence": primary.confidence,
                    }
                    item["confidence"] = primary.confidence
                    item["model"] = primary.model_id

                    # All model opinions for context
                    item["all_opinions"] = [
                        {
                            "model": h.model_id,
                            "node": h.node_id,
                            "class_name": h.class_name,
                            "confidence": h.confidence,
                            "box": {
                                "x1": h.x1, "y1": h.y1,
                                "x2": h.x2, "y2": h.y2,
                            },
                            "stage": h.stage,
                            "hop": h.hop_number,
                        }
                        for h in history
                    ]
            except Exception as e:
                logger.warning(f"Failed to get detection history for {record.id}: {e}")

        return item
