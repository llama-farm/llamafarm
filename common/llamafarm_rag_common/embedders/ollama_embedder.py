"""Ollama-based embedder for shared RAG library."""

import logging
from typing import Any

import requests

from .base import Embedder

logger = logging.getLogger(__name__)


class OllamaEmbedder(Embedder):
    """Embedder using Ollama API for local embeddings."""

    def __init__(
        self,
        name: str = "OllamaEmbedder",
        config: dict[str, Any] | None = None,
    ):
        super().__init__(name, config)
        config = config or {}
        self.model = config.get("model", "nomic-embed-text")
        self.api_base = config.get("api_base", "http://localhost:11434")
        self.dimension = config.get("dimension", 768)
        self.batch_size = max(config.get("batch_size", 32), 1)
        self.timeout = config.get("timeout", 60)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts using Ollama."""
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
        return self.dimension

    def _call_embedding_api(self, text: str) -> list[float]:
        """Call Ollama API for a single text."""
        url = f"{self.api_base}/api/embeddings"

        payload = {
            "model": self.model,
            "prompt": text,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                error_msg = f"Ollama API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise ConnectionError(f"Cannot connect to Ollama: {e}") from e
