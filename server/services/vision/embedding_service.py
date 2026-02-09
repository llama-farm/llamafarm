"""Vision Embedding Service - Proxies embedding requests to Universal Runtime."""

import logging
from typing import Any

from services.universal_runtime_service import get_runtime_client

logger = logging.getLogger(__name__)


class VisionEmbeddingService:
    """Service for image/text embedding operations.
    
    Communicates with Universal Runtime's vision embedding endpoint.
    CLIP embeddings enable cross-modal similarity search.
    """
    
    @classmethod
    async def embed(
        cls,
        model: str = "clip-vit-base",
        images: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate CLIP embeddings for images and/or texts.
        
        Args:
            model: CLIP model variant
            images: Base64-encoded images
            texts: Text strings to embed
            
        Returns:
            Embedding response with vectors
        """
        client = await get_runtime_client()
        
        payload = {"model": model}
        if images:
            payload["images"] = images
        if texts:
            payload["texts"] = texts
        
        response = await client.post(
            "/v1/vision/embed",
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        
        return response.json()
