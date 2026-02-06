"""Vision Detection Service - Proxies detection requests to Universal Runtime."""

import logging
from typing import Any

from core.settings import settings
from services.universal_runtime_service import get_runtime_client

logger = logging.getLogger(__name__)


class VisionDetectionService:
    """Service for object detection operations.
    
    Communicates with Universal Runtime's vision detection endpoint.
    """
    
    @classmethod
    def _get_base_url(cls) -> str:
        """Get the base URL for Universal Runtime."""
        return f"http://{settings.universal_host}:{settings.universal_port}"
    
    @classmethod
    async def detect(
        cls,
        image: str,
        model: str = "yolov8n",
        confidence_threshold: float = 0.5,
        classes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detect objects in an image.
        
        Args:
            image: Base64-encoded image
            model: YOLO model variant (yolov8n, yolov8s, etc.)
            confidence_threshold: Minimum confidence for detections
            classes: Optional list of classes to filter
            
        Returns:
            Detection response with bounding boxes
        """
        client = await get_runtime_client()
        
        payload = {
            "image": image,
            "model": model,
            "confidence_threshold": confidence_threshold,
        }
        if classes:
            payload["classes"] = classes
        
        response = await client.post(
            "/v1/vision/detect",
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        
        return response.json()
