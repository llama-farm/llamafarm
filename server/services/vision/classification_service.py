"""Vision Classification Service - Proxies classification requests to Universal Runtime."""

import logging
from typing import Any

from core.settings import settings
from services.universal_runtime_service import get_runtime_client

logger = logging.getLogger(__name__)


class VisionClassificationService:
    """Service for image classification operations.
    
    Communicates with Universal Runtime's vision classification endpoint.
    """
    
    @classmethod
    async def classify(
        cls,
        image: str,
        model: str = "clip-vit-base",
        classes: list[str] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Classify an image.
        
        Args:
            image: Base64-encoded image
            model: CLIP model variant
            classes: Classes for zero-shot classification
            top_k: Number of top predictions to return
            
        Returns:
            Classification response with class scores
        """
        client = await get_runtime_client()
        
        payload = {
            "image": image,
            "model": model,
            "top_k": top_k,
        }
        if classes:
            payload["classes"] = classes
        
        response = await client.post(
            "/v1/vision/classify",
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        
        return response.json()
