"""Ollama-based embedding generator."""

from pathlib import Path
from core.settings import settings
import requests
import json
from typing import List, Dict, Any, Optional

from core.base import Embedder

from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.embedders.ollama_embedder.ollama_embedder")


class OllamaEmbedder(Embedder):
    """Embedder using Ollama API for local embeddings."""

    def __init__(
        self,
        name: str = "OllamaEmbedder",
        config: Optional[Dict[str, Any]] = None,
        project_dir: Path | None = None,
    ):
        # Ensure name is always a string
        if not isinstance(name, str):
            name = "OllamaEmbedder"
        super().__init__(name, config, project_dir)
        config = config or {}
        self.model = config.get("model", "nomic-embed-text")
        self.api_base = config.get("api_base") or config.get(
            "base_url", settings.OLLAMA_HOST
        )
        self.base_url = self.api_base  # Alias for compatibility
        self.batch_size = max(
            config.get("batch_size", 32), 1
        )  # Ensure positive batch size
        self.timeout = config.get("timeout", 60)

    def validate_config(self) -> bool:
        """Validate configuration and check Ollama availability."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.warning(f"Ollama not available at {self.base_url}")
                return False

            # Check if response has proper JSON structure
            response_data = response.json()
            if response_data is None:
                logger.warning("Invalid response from Ollama API")
                return False

            # Check if model is available
            models = response_data.get("models", [])
            model_names = [m.get("name", "") for m in models if isinstance(m, dict)]
            # Check for exact match or partial match (e.g., "nomic-embed-text" matches "nomic-embed-text:latest")
            model_available = any(
                self.model == name or name.startswith(f"{self.model}:")
                for name in model_names
            )
            if not model_available:
                logger.warning(
                    f"Model {self.model} not found. Available models: {model_names}"
                )
                logger.info(f"Will attempt to pull {self.model} when first used")

            return True
        except Exception as e:
            logger.warning(f"Failed to validate Ollama embedder config: {e}")
            return False

    def embed(self, texts: List[str]) -> List[List[float]]:
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

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        embeddings = []

        for text in texts:
            try:
                result = self._call_ollama_api(text)
                embedding = result.get("embedding", [])
                if embedding:
                    embeddings.append(embedding)
                else:
                    logger.warning(f"No embedding returned for text: {text[:50]}...")
                    embeddings.append([0.0] * self.get_embedding_dimension())
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                embeddings.append([0.0] * self.get_embedding_dimension())

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # Default dimensions for common models (avoid test embedding during tests)
        dimension_map = {
            "nomic-embed-text": 768,
            "all-minilm": 384,
            "sentence-transformers": 384,
        }

        for model_name, dim in dimension_map.items():
            if model_name in self.model:
                return dim

        return 768  # Default

    def _call_ollama_api(self, text: str) -> Dict[str, Any]:
        """Call Ollama API for a single text."""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Ollama API error {response.status_code}: {response.text}"
            ) from response.raise_for_status()

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        if not text or not text.strip():
            return [0.0] * self.get_embedding_dimension()

        try:
            result = self._call_ollama_api(text)
            return result.get("embedding", [0.0] * self.get_embedding_dimension())
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return [0.0] * self.get_embedding_dimension()

    def _check_model_availability(self) -> bool:
        """Check if the model is available."""
        return self.validate_config()

    @classmethod
    def get_description(cls) -> str:
        """Get embedder description."""
        return "Ollama-based embedder for local text embedding generation using various models."
