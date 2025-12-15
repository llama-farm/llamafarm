"""
Tests for embedding safety utilities.

These tests verify the circuit breaker pattern and embedding validation
that prevent runaway data growth (issue #514).
"""

import time
from unittest.mock import patch

import pytest

from utils.embedding_safety import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    EmbedderUnavailableError,
    InvalidEmbeddingError,
    count_zero_embeddings,
    is_valid_embedding,
    is_zero_vector,
    validate_embeddings_batch,
)


class TestZeroVectorDetection:
    """Tests for zero vector detection."""

    def test_zero_vector_all_zeros(self):
        """All zeros should be detected as zero vector."""
        zero_vec = [0.0] * 768
        assert is_zero_vector(zero_vec) is True

    def test_zero_vector_near_zeros(self):
        """Near-zero values should be detected as zero vector."""
        near_zero = [1e-15] * 768
        assert is_zero_vector(near_zero) is True

    def test_non_zero_vector(self):
        """Non-zero vectors should not be detected as zero."""
        valid_vec = [0.1, 0.2, 0.3] + [0.0] * 765
        assert is_zero_vector(valid_vec) is False

    def test_empty_vector(self):
        """Empty vector should be considered zero."""
        assert is_zero_vector([]) is True

    def test_single_non_zero(self):
        """Single non-zero value should not be zero vector."""
        vec = [0.0] * 767 + [0.5]
        assert is_zero_vector(vec) is False


class TestEmbeddingValidation:
    """Tests for embedding validation."""

    def test_valid_embedding(self):
        """Valid embedding should pass validation."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        is_valid, error = is_valid_embedding(embedding)
        assert is_valid is True
        assert error is None

    def test_none_embedding(self):
        """None embedding should fail validation."""
        is_valid, error = is_valid_embedding(None)
        assert is_valid is False
        assert "None" in error

    def test_empty_embedding(self):
        """Empty embedding should fail validation."""
        is_valid, error = is_valid_embedding([])
        assert is_valid is False
        assert "empty" in error.lower()

    def test_zero_embedding_rejected(self):
        """Zero embedding should be rejected by default."""
        zero_vec = [0.0] * 768
        is_valid, error = is_valid_embedding(zero_vec, allow_zero=False)
        assert is_valid is False
        assert "zero vector" in error.lower()

    def test_zero_embedding_allowed(self):
        """Zero embedding should pass when allowed."""
        zero_vec = [0.0] * 768
        is_valid, error = is_valid_embedding(zero_vec, allow_zero=True)
        assert is_valid is True
        assert error is None

    def test_dimension_mismatch(self):
        """Wrong dimension should fail validation."""
        embedding = [0.1] * 384
        is_valid, error = is_valid_embedding(embedding, expected_dimension=768)
        assert is_valid is False
        assert "dimension mismatch" in error.lower()

    def test_dimension_match(self):
        """Correct dimension should pass validation."""
        embedding = [0.1] * 768
        is_valid, error = is_valid_embedding(embedding, expected_dimension=768)
        assert is_valid is True

    def test_nan_values_rejected(self):
        """NaN values should be rejected."""
        embedding = [0.1, 0.2, float("nan"), 0.4]
        is_valid, error = is_valid_embedding(embedding)
        assert is_valid is False
        assert "NaN" in error

    def test_inf_values_rejected(self):
        """Infinity values should be rejected."""
        embedding = [0.1, 0.2, float("inf"), 0.4]
        is_valid, error = is_valid_embedding(embedding)
        assert is_valid is False
        assert "Inf" in error


class TestBatchValidation:
    """Tests for batch embedding validation."""

    def test_all_valid_batch(self):
        """All valid embeddings should pass."""
        embeddings = [[0.1, 0.2, 0.3] for _ in range(10)]
        all_valid, invalid_indices, errors = validate_embeddings_batch(embeddings)
        assert all_valid is True
        assert len(invalid_indices) == 0
        assert len(errors) == 0

    def test_some_invalid_batch(self):
        """Batch with some invalid embeddings should report them."""
        embeddings = [
            [0.1, 0.2, 0.3],  # Valid
            [0.0, 0.0, 0.0],  # Zero - invalid
            [0.4, 0.5, 0.6],  # Valid
            [0.0, 0.0, 0.0],  # Zero - invalid
        ]
        all_valid, invalid_indices, errors = validate_embeddings_batch(
            embeddings, allow_zero=False
        )
        assert all_valid is False
        assert invalid_indices == [1, 3]
        assert len(errors) == 2

    def test_count_zero_embeddings(self):
        """Should correctly count zero embeddings."""
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0],
            [0.4, 0.5, 0.6],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        assert count_zero_embeddings(embeddings) == 3


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    def test_initial_state_closed(self):
        """Circuit breaker should start in closed state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_opens_after_threshold(self):
        """Circuit should open after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures
        for _ in range(3):
            cb.record_failure(Exception("Test error"))

        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_stays_closed_below_threshold(self):
        """Circuit should stay closed below threshold."""
        cb = CircuitBreaker(failure_threshold=5)

        for _ in range(4):
            cb.record_failure(Exception("Test error"))

        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_success_resets_failure_count(self):
        """Success should reset failure count."""
        cb = CircuitBreaker(failure_threshold=5)

        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        cb.record_success()

        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_timeout(self):
        """Circuit should go half-open after reset timeout."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        # Open the circuit
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Should be able to execute (transitions to half-open)
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self):
        """Half-open circuit should close on success."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1, half_open_max_calls=1)

        # Open the circuit
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))

        # Wait for timeout and transition to half-open
        time.sleep(0.15)
        cb.can_execute()  # Triggers transition

        # Success in half-open state should close
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """Half-open circuit should reopen on failure."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        # Open the circuit
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))

        # Wait for timeout and transition to half-open
        time.sleep(0.15)
        cb.can_execute()  # Triggers transition
        assert cb.state == CircuitState.HALF_OPEN

        # Failure in half-open state should reopen
        cb.record_failure(Exception("Error 3"))
        assert cb.state == CircuitState.OPEN

    def test_half_open_limits_calls(self):
        """Half-open state should limit number of calls."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1, half_open_max_calls=2)

        # Open the circuit
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))

        # Wait for timeout
        time.sleep(0.15)

        # First call should be allowed (transitions to half-open)
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.half_open_calls == 1

        # Second call should be allowed (within limit)
        assert cb.can_execute() is True
        assert cb.half_open_calls == 2

        # Third call should be blocked (limit reached)
        assert cb.can_execute() is False
        assert cb.half_open_calls == 2  # Counter shouldn't increase

    def test_force_reset(self):
        """Force reset should close circuit."""
        cb = CircuitBreaker(failure_threshold=2)

        # Open the circuit
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        assert cb.state == CircuitState.OPEN

        # Force reset
        cb.force_reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_get_state_info(self):
        """Should return useful state information."""
        cb = CircuitBreaker(failure_threshold=5)

        cb.record_failure(Exception("Test"))
        cb.record_failure(Exception("Test"))

        info = cb.get_state_info()
        assert info["state"] == "closed"
        assert info["failure_count"] == 2
        assert info["failure_threshold"] == 5


class TestEmbedderIntegration:
    """Integration tests for embedder with circuit breaker."""

    def test_ollama_embedder_circuit_breaker(self):
        """OllamaEmbedder should use circuit breaker."""
        from components.embedders.ollama_embedder.ollama_embedder import OllamaEmbedder

        embedder = OllamaEmbedder(config={"fail_fast": True})

        # Should have circuit breaker
        assert hasattr(embedder, "_circuit_breaker")
        assert embedder._circuit_breaker.state == CircuitState.CLOSED

    def test_universal_embedder_circuit_breaker(self):
        """UniversalEmbedder should use circuit breaker."""
        from components.embedders.universal_embedder.universal_embedder import (
            UniversalEmbedder,
        )

        embedder = UniversalEmbedder(config={"fail_fast": True})

        # Should have circuit breaker
        assert hasattr(embedder, "_circuit_breaker")
        assert embedder._circuit_breaker.state == CircuitState.CLOSED

    def test_embedder_raises_on_circuit_open(self):
        """Embedder should raise when circuit is open."""
        from components.embedders.ollama_embedder.ollama_embedder import OllamaEmbedder

        embedder = OllamaEmbedder(
            config={
                "fail_fast": True,
                "circuit_breaker": {"failure_threshold": 2},
            }
        )

        # Manually open the circuit
        embedder._circuit_breaker.record_failure(Exception("Test 1"))
        embedder._circuit_breaker.record_failure(Exception("Test 2"))

        # Should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            embedder.embed(["test text"])

    def test_embedder_fail_fast_disabled(self):
        """Embedder with fail_fast=False should return zero vectors."""
        from components.embedders.ollama_embedder.ollama_embedder import OllamaEmbedder

        embedder = OllamaEmbedder(
            config={
                "fail_fast": False,  # Legacy behavior
            }
        )

        # Mock the API call to fail (now uses _call_embedding_api)
        with patch.object(embedder, "_call_embedding_api") as mock_api:
            mock_api.side_effect = Exception("Connection refused")

            # Should return zero vector instead of raising
            result = embedder.embed(["test text"])
            assert len(result) == 1
            assert is_zero_vector(result[0])


class TestExceptionTypes:
    """Tests for custom exception types."""

    def test_circuit_breaker_open_error(self):
        """CircuitBreakerOpenError should have useful attributes."""
        error = CircuitBreakerOpenError("Circuit is open", failures=5, reset_time=30.0)
        assert error.failures == 5
        assert error.reset_time == 30.0
        assert "Circuit is open" in str(error)

    def test_embedder_unavailable_error(self):
        """EmbedderUnavailableError should be an EmbeddingError."""
        from utils.embedding_safety import EmbeddingError

        error = EmbedderUnavailableError("Service unavailable")
        assert isinstance(error, EmbeddingError)

    def test_invalid_embedding_error(self):
        """InvalidEmbeddingError should be an EmbeddingError."""
        from utils.embedding_safety import EmbeddingError

        error = InvalidEmbeddingError("Invalid embedding")
        assert isinstance(error, EmbeddingError)
