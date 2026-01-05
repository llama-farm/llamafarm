"""
Embedding safety utilities to prevent runaway data growth.

This module provides:
- Circuit breaker pattern for embedder failures
- Embedding validation to detect zero/invalid vectors
- Health check utilities for embedders
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.utils.embedding_safety")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failures exceeded threshold, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class EmbeddingError(Exception):
    """Base exception for embedding errors."""

    pass


class EmbedderUnavailableError(EmbeddingError):
    """Raised when the embedder service is unavailable."""

    pass


class CircuitBreakerOpenError(EmbeddingError):
    """Raised when circuit breaker is open and blocking requests."""

    def __init__(self, message: str, failures: int, reset_time: float):
        super().__init__(message)
        self.failures = failures
        self.reset_time = reset_time


class InvalidEmbeddingError(EmbeddingError):
    """Raised when an embedding is invalid (e.g., all zeros)."""

    pass


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern implementation for embedder protection.

    Prevents cascading failures by stopping requests after too many failures,
    and periodically testing if the service has recovered.
    """

    failure_threshold: int = 5  # Open circuit after this many consecutive failures
    reset_timeout: float = 60.0  # Seconds to wait before trying again
    half_open_max_calls: int = 1  # Max calls to allow in half-open state

    # State tracking
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    half_open_calls: int = field(default=0)

    def __post_init__(self):
        self._name = "CircuitBreaker"

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if reset timeout has passed
            if time.time() - self.last_failure_time >= self.reset_timeout:
                logger.info(
                    "Circuit breaker transitioning to half-open state",
                    extra={"reset_timeout": self.reset_timeout},
                )
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0  # Reset counter, will be incremented below
                # Fall through to HALF_OPEN check
            else:
                return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                logger.info(
                    "Circuit breaker closing after successful recovery",
                    extra={"success_count": self.success_count},
                )
                self._reset()
        else:
            # Reset failure count on success in closed state
            self.failure_count = 0

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state reopens the circuit
            logger.warning(
                "Circuit breaker reopening after failure in half-open state",
                extra={"error": str(error) if error else "Unknown"},
            )
            self.state = CircuitState.OPEN
            self.half_open_calls = 0
            self.success_count = 0
        elif self.failure_count >= self.failure_threshold:
            logger.error(
                "Circuit breaker opening due to consecutive failures",
                extra={
                    "failure_count": self.failure_count,
                    "threshold": self.failure_threshold,
                    "error": str(error) if error else "Unknown",
                },
            )
            self.state = CircuitState.OPEN

    def _reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0

    def get_state_info(self) -> dict[str, Any]:
        """Get current state information."""
        info = {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
        }

        if self.state == CircuitState.OPEN:
            time_until_reset = max(
                0, self.reset_timeout - (time.time() - self.last_failure_time)
            )
            info["time_until_reset"] = round(time_until_reset, 1)

        return info

    def force_reset(self) -> None:
        """Force reset the circuit breaker (for manual recovery)."""
        logger.info("Circuit breaker manually reset")
        self._reset()


def is_zero_vector(embedding: list[float], tolerance: float = 1e-10) -> bool:
    """
    Check if an embedding is effectively a zero vector.

    Args:
        embedding: The embedding vector to check
        tolerance: Threshold for considering values as zero

    Returns:
        True if the embedding is all zeros (or near-zeros)
    """
    if not embedding:
        return True

    return all(abs(v) < tolerance for v in embedding)


def is_valid_embedding(
    embedding: list[float],
    expected_dimension: int | None = None,
    allow_zero: bool = False,
) -> tuple[bool, str | None]:
    """
    Validate an embedding vector.

    Args:
        embedding: The embedding vector to validate
        expected_dimension: Expected dimension (if known)
        allow_zero: Whether to allow zero vectors

    Returns:
        Tuple of (is_valid, error_message)
    """
    if embedding is None:
        return False, "Embedding is None"

    if not isinstance(embedding, list):
        return False, f"Embedding is not a list: {type(embedding)}"

    if len(embedding) == 0:
        return False, "Embedding is empty"

    if expected_dimension is not None and len(embedding) != expected_dimension:
        return (
            False,
            f"Embedding dimension mismatch: expected {expected_dimension}, got {len(embedding)}",
        )

    if not allow_zero and is_zero_vector(embedding):
        return False, "Embedding is a zero vector (likely from failed embedding generation)"

    # Check for NaN or Inf values
    for i, v in enumerate(embedding):
        if not isinstance(v, (int, float)):
            return False, f"Embedding contains non-numeric value at index {i}: {type(v)}"
        if v != v:  # NaN check
            return False, f"Embedding contains NaN at index {i}"
        if abs(v) == float("inf"):
            return False, f"Embedding contains Inf at index {i}"

    return True, None


def validate_embeddings_batch(
    embeddings: list[list[float]],
    expected_dimension: int | None = None,
    allow_zero: bool = False,
) -> tuple[bool, list[int], list[str]]:
    """
    Validate a batch of embeddings.

    Args:
        embeddings: List of embedding vectors
        expected_dimension: Expected dimension for all embeddings
        allow_zero: Whether to allow zero vectors

    Returns:
        Tuple of (all_valid, invalid_indices, error_messages)
    """
    invalid_indices = []
    error_messages = []

    for i, embedding in enumerate(embeddings):
        is_valid, error = is_valid_embedding(
            embedding, expected_dimension=expected_dimension, allow_zero=allow_zero
        )
        if not is_valid:
            invalid_indices.append(i)
            error_messages.append(f"Index {i}: {error}")

    return len(invalid_indices) == 0, invalid_indices, error_messages


def count_zero_embeddings(embeddings: list[list[float]]) -> int:
    """Count the number of zero vectors in a batch of embeddings."""
    return sum(1 for e in embeddings if is_zero_vector(e))
