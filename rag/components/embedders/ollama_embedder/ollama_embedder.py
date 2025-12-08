"""Ollama-based embedding generator with circuit breaker protection."""

from pathlib import Path
from typing import Any

import requests

from core.base import Embedder
from core.logging import RAGStructLogger
from core.settings import settings
from utils.embedding_safety import (
    EmbedderUnavailableError,
    is_zero_vector,
)

logger = RAGStructLogger("rag.components.embedders.ollama_embedder.ollama_embedder")


class OllamaEmbedder(Embedder):
    """Embedder using Ollama API for local embeddings with circuit breaker protection."""

    def __init__(
        self,
        name: str = "OllamaEmbedder",
        config: dict[str, Any] | None = None,
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
        self.dimension = config.get("dimension", 768)  # Read from config
        self.batch_size = max(
            config.get("batch_size", 32), 1
        )  # Ensure positive batch size
        self.timeout = config.get("timeout", 60)

        # Track consecutive failures for logging
        self._consecutive_failures = 0

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

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts using Ollama.

        Raises:
            CircuitBreakerOpenError: If too many consecutive failures have occurred
            EmbedderUnavailableError: If Ollama is unavailable and fail_fast is enabled
        """
        if not texts:
            return []

        # Check circuit breaker before starting
        self.check_circuit_breaker()

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Raises:
            EmbedderUnavailableError: If Ollama is unavailable and fail_fast is enabled
            CircuitBreakerOpenError: If circuit breaker trips during batch processing
        """
        embeddings = []

        for text in texts:
            try:
                # Check circuit breaker before each request
                self.check_circuit_breaker()

                result = self._call_ollama_api(text)
                embedding = result.get("embedding", [])

                if embedding and not is_zero_vector(embedding):
                    embeddings.append(embedding)
                    self.record_success()
                    self._consecutive_failures = 0
                else:
                    # Empty or zero embedding returned - treat as failure
                    self._consecutive_failures += 1
                    error_msg = f"No valid embedding returned for text: {text[:50]}..."
                    logger.warning(error_msg)
                    self.record_failure(Exception(error_msg))

                    if self._fail_fast:
                        raise EmbedderUnavailableError(
                            f"Ollama returned empty/invalid embedding. "
                            f"Consecutive failures: {self._consecutive_failures}"
                        )
                    else:
                        # Legacy behavior: append zero vector (not recommended)
                        embeddings.append([0.0] * self.get_embedding_dimension())

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                self._consecutive_failures += 1
                logger.error(
                    f"Error generating embedding (failure {self._consecutive_failures}): {e}"
                )
                self.record_failure(e)

                if self._fail_fast:
                    raise EmbedderUnavailableError(
                        f"Ollama is unavailable at {self.base_url}: {e}. "
                        f"Consecutive failures: {self._consecutive_failures}"
                    ) from e
                else:
                    # Legacy behavior: append zero vector (not recommended)
                    embeddings.append([0.0] * self.get_embedding_dimension())

            except Exception as e:
                self._consecutive_failures += 1
                logger.error(
                    f"Error generating embedding (failure {self._consecutive_failures}): {e}"
                )
                self.record_failure(e)

                if self._fail_fast:
                    raise EmbedderUnavailableError(
                        f"Failed to generate embedding: {e}. "
                        f"Consecutive failures: {self._consecutive_failures}"
                    ) from e
                else:
                    # Legacy behavior: append zero vector (not recommended)
                    embeddings.append([0.0] * self.get_embedding_dimension())

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # Return the configured dimension from llamafarm.yaml
        return self.dimension

    def _call_ollama_api(self, text: str) -> dict[str, Any]:
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

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Raises:
            EmbedderUnavailableError: If Ollama is unavailable and fail_fast is enabled
            CircuitBreakerOpenError: If circuit breaker is open
        """
        if not text or not text.strip():
            if self._fail_fast:
                raise EmbedderUnavailableError("Cannot embed empty text")
            return [0.0] * self.get_embedding_dimension()

        # Check circuit breaker
        self.check_circuit_breaker()

        try:
            result = self._call_ollama_api(text)
            embedding = result.get("embedding", [])

            if embedding and not is_zero_vector(embedding):
                self.record_success()
                self._consecutive_failures = 0
                return embedding
            else:
                self._consecutive_failures += 1
                self.record_failure(Exception("Empty embedding returned"))

                if self._fail_fast:
                    raise EmbedderUnavailableError(
                        f"Ollama returned empty/invalid embedding for text. "
                        f"Consecutive failures: {self._consecutive_failures}"
                    )
                return [0.0] * self.get_embedding_dimension()

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            self._consecutive_failures += 1
            logger.error(f"Error embedding text: {e}")
            self.record_failure(e)

            if self._fail_fast:
                raise EmbedderUnavailableError(
                    f"Ollama is unavailable at {self.base_url}: {e}"
                ) from e
            return [0.0] * self.get_embedding_dimension()

        except Exception as e:
            self._consecutive_failures += 1
            logger.error(f"Error embedding text: {e}")
            self.record_failure(e)

            if self._fail_fast:
                raise EmbedderUnavailableError(f"Failed to embed text: {e}") from e
            return [0.0] * self.get_embedding_dimension()

    def _check_model_availability(self) -> bool:
        """Check if the model is available."""
        return self.validate_config()

    @classmethod
    def get_description(cls) -> str:
        """Get embedder description."""
        return "Ollama-based embedder for local text embedding generation using various models."
