"""Base embedder class for shared RAG library."""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Simple circuit breaker for embedders."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.is_open = False
        self.last_failure_time = 0

    def can_execute(self) -> bool:
        """Check if circuit allows execution."""
        if not self.is_open:
            return True

        import time
        if time.time() - self.last_failure_time > self.reset_timeout:
            self.is_open = False
            self.failure_count = 0
            return True

        return False

    def record_success(self) -> None:
        """Record successful operation."""
        self.failure_count = 0
        self.is_open = False

    def record_failure(self, error: Exception | None = None) -> None:
        """Record failed operation."""
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.is_open = True


class Embedder(ABC):
    """Base class for embedding generators with circuit breaker protection."""

    DEFAULT_FAILURE_THRESHOLD = 5
    DEFAULT_RESET_TIMEOUT = 60.0

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.config = config or {}

        # Initialize circuit breaker
        circuit_config = self.config.get("circuit_breaker", {})
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_config.get(
                "failure_threshold", self.DEFAULT_FAILURE_THRESHOLD
            ),
            reset_timeout=circuit_config.get(
                "reset_timeout", self.DEFAULT_RESET_TIMEOUT
            ),
        )

        self._fail_fast = self.config.get("fail_fast", True)
        self._consecutive_failures = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass

    @abstractmethod
    def _call_embedding_api(self, text: str) -> list[float]:
        """Make the actual API call to generate an embedding for a single text."""
        pass

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string with error handling."""
        # Handle empty text
        if not text or not text.strip():
            if self._fail_fast:
                raise ValueError("Cannot embed empty text")
            return [0.0] * self.get_embedding_dimension()

        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            raise RuntimeError(
                f"Circuit breaker is open for {self.name}. "
                f"Too many consecutive failures ({self._circuit_breaker.failure_count})"
            )

        try:
            embedding = self._call_embedding_api(text)

            # Validate embedding
            if embedding and not self._is_zero_vector(embedding):
                self._circuit_breaker.record_success()
                self._consecutive_failures = 0
                return embedding
            else:
                self._consecutive_failures += 1
                self._circuit_breaker.record_failure()

                if self._fail_fast:
                    raise ValueError(
                        f"{self.name} returned empty/invalid embedding. "
                        f"Consecutive failures: {self._consecutive_failures}"
                    )
                return [0.0] * self.get_embedding_dimension()

        except (ConnectionError, TimeoutError) as e:
            self._consecutive_failures += 1
            logger.error(f"Connection error embedding text: {e}")
            self._circuit_breaker.record_failure(e)

            if self._fail_fast:
                raise RuntimeError(
                    f"{self.name} is unavailable: {e}. "
                    f"Consecutive failures: {self._consecutive_failures}"
                ) from e
            return [0.0] * self.get_embedding_dimension()

        except Exception as e:
            self._consecutive_failures += 1
            logger.error(f"Error embedding text: {e}")
            self._circuit_breaker.record_failure(e)

            if self._fail_fast:
                raise RuntimeError(
                    f"Failed to embed text: {e}. "
                    f"Consecutive failures: {self._consecutive_failures}"
                ) from e
            return [0.0] * self.get_embedding_dimension()

    @staticmethod
    def _is_zero_vector(vec: list[float]) -> bool:
        """Check if vector is all zeros."""
        return all(v == 0.0 for v in vec)
