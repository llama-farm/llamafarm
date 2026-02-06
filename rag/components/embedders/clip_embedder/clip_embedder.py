"""CLIP-based image embedder for multimodal RAG.

Generates embeddings for images using CLIP via the Universal Runtime.
Enables text-to-image and image-to-image similarity search.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """Image embedder using CLIP via Universal Runtime.
    
    CLIP embeddings exist in a shared vector space with text,
    enabling cross-modal similarity search.
    
    Example:
        ```python
        embedder = CLIPEmbedder()
        
        # Embed images
        embeddings = await embedder.embed_images([image_bytes])
        
        # Embed text queries (for text-to-image search)
        text_embeddings = await embedder.embed_texts(["a photo of a cat"])
        
        # Compute similarity
        similarity = np.dot(embeddings[0], text_embeddings[0])
        ```
    """
    
    def __init__(
        self,
        name: str = "CLIPEmbedder",
        config: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ):
        """Initialize CLIP embedder.
        
        Args:
            name: Embedder name for logging
            config: Configuration dict with:
                - model: CLIP model ID (default: clip-vit-base)
                - base_url: Universal Runtime URL
                - batch_size: Batch size for processing
            project_dir: Project directory (unused, for API compatibility)
        """
        self.name = name
        config = config or {}
        
        self.model = config.get("model", "clip-vit-base")
        self.base_url = config.get("base_url", "http://127.0.0.1:11540")
        self.batch_size = config.get("batch_size", 8)
        self.timeout = config.get("timeout", 60)
        self.embedding_dim = 512  # CLIP ViT-B/32 dimension
    
    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        """Generate embeddings for a batch of images.
        
        Args:
            images: List of image bytes (JPEG/PNG)
            
        Returns:
            List of embedding vectors (512 dimensions for CLIP)
        """
        embeddings = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            batch_b64 = [
                f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"
                for img in batch
            ]
            
            response = requests.post(
                f"{self.base_url}/v1/vision/embed",
                json={
                    "model": self.model,
                    "images": batch_b64,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings.extend(result["embeddings"])
        
        return embeddings
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for text (for cross-modal search).
        
        CLIP embeds text and images into the same vector space,
        enabling text-to-image search.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        response = requests.post(
            f"{self.base_url}/v1/vision/embed",
            json={
                "model": self.model,
                "texts": texts,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        
        return response.json()["embeddings"]
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts (alias for embed_texts for API compatibility)."""
        return self.embed_texts(texts)
    
    def get_info(self) -> dict[str, Any]:
        """Get embedder information."""
        return {
            "name": self.name,
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "base_url": self.base_url,
            "type": "clip",
        }
