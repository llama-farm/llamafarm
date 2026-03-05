"""Review queue service — queries runtime image store, submits corrections."""

from typing import Any


class VisionReviewService:
    """Review queue operations via runtime proxy."""

    # Note: The runtime doesn't expose review endpoints directly yet.
    # This service is ready for when they're added. For now it's a stub
    # that can be expanded.

    @staticmethod
    async def get_pending(limit: int = 50, source: str | None = None) -> dict[str, Any]:
        """Get pending review items. Placeholder for future runtime endpoint."""
        return {"items": [], "total": 0, "pending": 0}

    @staticmethod
    async def submit_decision(image_id: str, decision: str,
                              corrections: list[dict] | None = None) -> dict[str, Any]:
        """Submit a review decision. Placeholder."""
        return {"image_id": image_id, "decision": decision, "processed": False}
