"""Universal Runtime-based embedder for shared RAG library."""

import logging
from typing import Any
from urllib.parse import urlparse

import requests

from .base import Embedder

logger = logging.getLogger(__name__)


class UniversalEmbedder(Embedder):
    """Embedder using Universal Runtime API for embeddings."""

    def __init__(
        self,
        name: str = "UniversalEmbedder",
        config: dict[str, Any] | None = None,
    ):
        super().__init__(name, config)
        config = config or {}

        # Model configuration
        self.model = config.get("model", "sentence-transformers/all-MiniLM-L6-v2")

        # API configuration
        self.api_base = config.get("api_base", "http://127.0.0.1:11540")
        # Ensure base_url includes the /v1 path
        if not self.api_base.endswith("/v1"):
            parsed = urlparse(self.api_base)
            if parsed.port is not None:
                self.api_base = f"{self.api_base}/v1"
            else:
                if parsed.scheme:
                    self.api_base = (
                        f"{parsed.scheme}://{parsed.netloc}:11540{parsed.path}/v1"
                    )
                else:
                    self.api_base = f"{self.api_base}:11540/v1"

        self.api_key = config.get("api_key", "universal")
        self.batch_size = max(config.get("batch_size", 32), 1)
        self.timeout = config.get("timeout", 120)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts using Universal Runtime."""
        if not texts:
            return []

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # Default dimensions for common models
        dimension_map = {
            "all-MiniLM-L6-v2": 384,
            "all-minilm": 384,
            "all-mpnet-base-v2": 768,
            "all-mpnet": 768,
            "bge-base": 768,
            "bge-large": 1024,
            "nomic-embed": 768,
            "e5-base": 768,
            "e5-large": 1024,
        }

        # Check if model name contains any known model
        for model_key, dim in dimension_map.items():
            if model_key.lower() in self.model.lower():
                return dim

        # Default to 768 (most common)
        return 768

    def _call_embedding_api(self, text: str) -> list[float]:
        """Call Universal Runtime API for a single text."""
        url = f"{self.api_base}/embeddings"

        # Prepare request in OpenAI format
        payload = {
            "model": self.model,
            "input": text,
        }

        headers = {
            "Content-Type": "application/json",
        }

        # Add API key if provided
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                data = result.get("data", [])
                if data and len(data) > 0:
                    return data[0].get("embedding", [])
                return []
            else:
                error_msg = (
                    f"Universal Runtime API error {response.status_code}: {response.text}"
                )
                logger.error(error_msg)
                raise Exception(error_msg)

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise ConnectionError(f"Cannot connect to Universal Runtime: {e}") from e
