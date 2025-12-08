"""Universal Runtime-based embedding generator with circuit breaker protection."""

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests  # type: ignore

from core.base import Embedder
from core.logging import RAGStructLogger
from core.settings import settings
from utils.embedding_safety import (
    EmbedderUnavailableError,
    is_zero_vector,
)

logger = RAGStructLogger(
    "rag.components.embedders.universal_embedder.universal_embedder"
)


class UniversalEmbedder(Embedder):
    """Embedder using Universal Runtime API for embeddings with circuit breaker protection."""

    def __init__(
        self,
        name: str = "UniversalEmbedder",
        config: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ):
        # Ensure name is always a string
        if not isinstance(name, str):
            name = "UniversalEmbedder"
        super().__init__(name, config, project_dir)
        config = config or {}

        # Model configuration
        self.model = config.get("model", "sentence-transformers/all-MiniLM-L6-v2")

        # API configuration
        self.api_base = config.get("api_base") or config.get(
            "base_url", getattr(settings, "UNIVERSAL_HOST", "http://127.0.0.1:11540")
        )
        # Ensure base_url includes the /v1 path for OpenAI compatibility
        if not self.api_base.endswith("/v1"):
            # Parse URL to check if port is explicitly specified
            parsed = urlparse(self.api_base)
            # If there's a port in netloc (e.g., "localhost:8080"), it will be in parsed.port
            # Otherwise parsed.port will be None
            if parsed.port is not None:
                # Port is explicitly specified, just add /v1
                self.api_base = f"{self.api_base}/v1"
            else:
                # No port specified, add default port and /v1
                # Preserve the scheme if present, otherwise default to http
                if parsed.scheme:
                    self.api_base = (
                        f"{parsed.scheme}://{parsed.netloc}:11540{parsed.path}/v1"
                    )
                else:
                    self.api_base = f"{self.api_base}:11540/v1"

        self.base_url = self.api_base  # Alias for compatibility
        self.api_key = config.get("api_key", "universal")

        # Processing configuration
        self.batch_size = max(
            config.get("batch_size", 32), 1
        )  # Ensure positive batch size
        self.timeout = config.get(
            "timeout", 120
        )  # Longer timeout for initial model loading

        # Normalization (Universal Runtime doesn't normalize by default)
        self.normalize = config.get("normalize", True)

        # Track consecutive failures for logging
        self._consecutive_failures = 0

    def validate_config(self) -> bool:
        """Validate configuration and check Universal Runtime availability."""
        try:
            # Check if server is available
            health_url = self.base_url.replace("/v1", "/health")
            response = requests.get(health_url, timeout=5)
            if response.status_code != 200:
                logger.warning(f"Universal Runtime not available at {health_url}")
                return False

            # Optionally check if embeddings endpoint is available
            # by listing models
            models_url = f"{self.base_url}/models"
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Universal Runtime available at {self.base_url}")
                return True
            else:
                logger.warning("Could not list models from Universal Runtime")
                return False

        except Exception as e:
            logger.warning(f"Failed to validate Universal Runtime embedder config: {e}")
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts using Universal Runtime.

        Raises:
            CircuitBreakerOpenError: If too many consecutive failures have occurred
            EmbedderUnavailableError: If Universal Runtime is unavailable and fail_fast is enabled
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
        """Embed a batch of texts using Universal Runtime.

        Raises:
            EmbedderUnavailableError: If Universal Runtime is unavailable and fail_fast is enabled
            CircuitBreakerOpenError: If circuit breaker trips during batch processing
        """
        # Check circuit breaker before batch
        self.check_circuit_breaker()

        try:
            result = self._call_universal_api(texts)

            # Parse OpenAI-compatible response format
            data = result.get("data", [])
            if not data:
                self._consecutive_failures += 1
                error_msg = f"No embeddings returned for batch of {len(texts)} texts"
                logger.warning(error_msg)
                self.record_failure(Exception(error_msg))

                if self._fail_fast:
                    raise EmbedderUnavailableError(
                        f"Universal Runtime returned no embeddings. "
                        f"Consecutive failures: {self._consecutive_failures}"
                    )
                return [
                    [0.0] * self.get_embedding_dimension() for _ in range(len(texts))
                ]

            # Extract embeddings from data array
            embeddings = []
            zero_count = 0
            for item in data:
                embedding = item.get("embedding", [])
                if embedding and not is_zero_vector(embedding):
                    embeddings.append(embedding)
                else:
                    zero_count += 1
                    logger.warning("Empty or zero embedding in response")
                    if self._fail_fast:
                        self._consecutive_failures += 1
                        self.record_failure(Exception("Empty embedding in response"))
                        raise EmbedderUnavailableError(
                            f"Universal Runtime returned empty/invalid embedding. "
                            f"Consecutive failures: {self._consecutive_failures}"
                        )
                    embeddings.append([0.0] * self.get_embedding_dimension())

            # If all embeddings are valid, record success
            if zero_count == 0:
                self.record_success()
                self._consecutive_failures = 0

            # Ensure we have the right number of embeddings
            while len(embeddings) < len(texts):
                if self._fail_fast:
                    self._consecutive_failures += 1
                    self.record_failure(Exception("Incomplete embedding response"))
                    raise EmbedderUnavailableError(
                        f"Universal Runtime returned fewer embeddings than expected. "
                        f"Expected {len(texts)}, got {len(embeddings)}"
                    )
                embeddings.append([0.0] * self.get_embedding_dimension())

            return embeddings[: len(texts)]

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            self._consecutive_failures += 1
            logger.error(
                f"Error generating embeddings (failure {self._consecutive_failures}): {e}"
            )
            self.record_failure(e)

            if self._fail_fast:
                raise EmbedderUnavailableError(
                    f"Universal Runtime is unavailable at {self.base_url}: {e}. "
                    f"Consecutive failures: {self._consecutive_failures}"
                ) from e
            return [[0.0] * self.get_embedding_dimension() for _ in range(len(texts))]

        except EmbedderUnavailableError:
            # Re-raise our own exceptions
            raise

        except Exception as e:
            self._consecutive_failures += 1
            logger.error(
                f"Error generating embeddings (failure {self._consecutive_failures}): {e}"
            )
            self.record_failure(e)

            if self._fail_fast:
                raise EmbedderUnavailableError(
                    f"Failed to generate embeddings: {e}. "
                    f"Consecutive failures: {self._consecutive_failures}"
                ) from e
            return [[0.0] * self.get_embedding_dimension() for _ in range(len(texts))]

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

    def _call_universal_api(self, texts: list[str]) -> dict[str, Any]:
        """Call Universal Runtime API for embedding generation."""
        url = f"{self.base_url}/embeddings"

        # Prepare request in OpenAI format
        payload = {
            "model": self.model,
            "input": texts if len(texts) > 1 else texts[0],
        }

        headers = {
            "Content-Type": "application/json",
        }

        # Add API key if provided
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

        if response.status_code == 200:
            return response.json()
        else:
            error_msg = (
                f"Universal Runtime API error {response.status_code}: {response.text}"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Raises:
            EmbedderUnavailableError: If Universal Runtime is unavailable and fail_fast is enabled
            CircuitBreakerOpenError: If circuit breaker is open
        """
        if not text or not text.strip():
            if self._fail_fast:
                raise EmbedderUnavailableError("Cannot embed empty text")
            return [0.0] * self.get_embedding_dimension()

        # Check circuit breaker
        self.check_circuit_breaker()

        try:
            result = self._call_universal_api([text])
            data = result.get("data", [])
            if data and len(data) > 0:
                embedding = data[0].get("embedding", [])
                if embedding and not is_zero_vector(embedding):
                    self.record_success()
                    self._consecutive_failures = 0
                    return embedding
                else:
                    self._consecutive_failures += 1
                    self.record_failure(Exception("Empty embedding returned"))

                    if self._fail_fast:
                        raise EmbedderUnavailableError(
                            f"Universal Runtime returned empty/invalid embedding. "
                            f"Consecutive failures: {self._consecutive_failures}"
                        )
                    return [0.0] * self.get_embedding_dimension()
            else:
                self._consecutive_failures += 1
                self.record_failure(Exception("No data in response"))

                if self._fail_fast:
                    raise EmbedderUnavailableError(
                        f"Universal Runtime returned no data. "
                        f"Consecutive failures: {self._consecutive_failures}"
                    )
                return [0.0] * self.get_embedding_dimension()

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            self._consecutive_failures += 1
            logger.error(f"Error embedding text: {e}")
            self.record_failure(e)

            if self._fail_fast:
                raise EmbedderUnavailableError(
                    f"Universal Runtime is unavailable at {self.base_url}: {e}"
                ) from e
            return [0.0] * self.get_embedding_dimension()

        except EmbedderUnavailableError:
            raise

        except Exception as e:
            self._consecutive_failures += 1
            logger.error(f"Error embedding text: {e}")
            self.record_failure(e)

            if self._fail_fast:
                raise EmbedderUnavailableError(f"Failed to embed text: {e}") from e
            return [0.0] * self.get_embedding_dimension()

    def _check_model_availability(self) -> bool:
        """Check if the Universal Runtime is available."""
        return self.validate_config()

    @classmethod
    def get_description(cls) -> str:
        """Get embedder description."""
        return "Universal Runtime-based embedder for text embedding generation using any HuggingFace model."
