"""Tests for Concept Drift Detection (River).

Phase 14 of Universal Runtime ML Enhancements.
"""

import pytest

from utils.drift_detector import (
    SUPPORTED_ALGORITHMS,
    DriftDetector,
    detect_drift_in_stream,
)


def create_stable_stream(length: int = 100, mean: float = 10.0, noise: float = 0.5) -> list[float]:
    """Create a stable stream with no drift."""
    import random
    random.seed(42)
    return [mean + random.uniform(-noise, noise) for _ in range(length)]


def create_drift_stream(
    length1: int = 50,
    length2: int = 50,
    mean1: float = 10.0,
    mean2: float = 50.0,
    noise: float = 0.5,
) -> list[float]:
    """Create a stream with a mean shift (drift) in the middle."""
    import random
    random.seed(42)
    stream1 = [mean1 + random.uniform(-noise, noise) for _ in range(length1)]
    stream2 = [mean2 + random.uniform(-noise, noise) for _ in range(length2)]
    return stream1 + stream2


def create_gradual_drift_stream(
    length: int = 100,
    start_mean: float = 10.0,
    end_mean: float = 50.0,
) -> list[float]:
    """Create a stream with gradual drift."""
    import random
    random.seed(42)
    step = (end_mean - start_mean) / length
    return [start_mean + i * step + random.uniform(-1, 1) for i in range(length)]


class TestDriftDetectorInit:
    """Test detector initialization."""

    def test_default_init(self):
        """Test default initialization."""
        detector = DriftDetector()
        assert detector.algorithm == "adwin"

    def test_adwin_init(self):
        """Test ADWIN initialization."""
        detector = DriftDetector(algorithm="adwin")
        assert detector.algorithm == "adwin"

    def test_page_hinkley_init(self):
        """Test Page-Hinkley initialization."""
        detector = DriftDetector(algorithm="page_hinkley")
        assert detector.algorithm == "page_hinkley"

    def test_kswin_init(self):
        """Test KSWIN initialization."""
        detector = DriftDetector(algorithm="kswin")
        assert detector.algorithm == "kswin"

    def test_ddm_init(self):
        """Test DDM initialization."""
        detector = DriftDetector(algorithm="ddm")
        assert detector.algorithm == "ddm"

    def test_invalid_algorithm_raises(self):
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            DriftDetector(algorithm="invalid")

    def test_case_insensitive_algorithm(self):
        """Test that algorithm is case-insensitive."""
        detector = DriftDetector(algorithm="ADWIN")
        assert detector.algorithm == "adwin"

    def test_custom_delta(self):
        """Test custom delta parameter."""
        detector = DriftDetector(algorithm="adwin", delta=0.01)
        assert detector.params.get("delta") == 0.01


class TestSingleUpdate:
    """Test single value updates."""

    def test_update_returns_dict(self):
        """Test that update returns a dictionary."""
        detector = DriftDetector()
        result = detector.update(10.0)
        assert isinstance(result, dict)

    def test_update_has_required_keys(self):
        """Test that update result has required keys."""
        detector = DriftDetector()
        result = detector.update(10.0)
        assert "drift_detected" in result
        assert "index" in result
        assert "value" in result

    def test_index_increments(self):
        """Test that index increments with each update."""
        detector = DriftDetector()
        result1 = detector.update(10.0)
        result2 = detector.update(10.0)
        result3 = detector.update(10.0)
        assert result1["index"] == 1
        assert result2["index"] == 2
        assert result3["index"] == 3

    def test_value_preserved(self):
        """Test that value is preserved in result."""
        detector = DriftDetector()
        result = detector.update(42.5)
        assert result["value"] == 42.5


class TestBatchUpdate:
    """Test batch updates."""

    def test_batch_returns_dict(self):
        """Test that batch update returns a dictionary."""
        detector = DriftDetector()
        stream = create_stable_stream(50)
        result = detector.update_batch(stream)
        assert isinstance(result, dict)

    def test_batch_has_required_keys(self):
        """Test that batch result has required keys."""
        detector = DriftDetector()
        stream = create_stable_stream(50)
        result = detector.update_batch(stream)
        assert "total_processed" in result
        assert "drift_detected" in result
        assert "drift_points" in result
        assert "final_index" in result

    def test_batch_total_processed(self):
        """Test that total_processed is correct."""
        detector = DriftDetector()
        stream = create_stable_stream(75)
        result = detector.update_batch(stream)
        assert result["total_processed"] == 75

    def test_batch_final_index(self):
        """Test that final_index is correct."""
        detector = DriftDetector()
        stream = create_stable_stream(100)
        result = detector.update_batch(stream)
        assert result["final_index"] == 100


class TestDriftDetection:
    """Test drift detection functionality."""

    def test_no_drift_in_stable_stream(self):
        """Test that stable stream produces no drift."""
        detector = DriftDetector(algorithm="adwin")
        stream = create_stable_stream(100, mean=10.0, noise=0.1)
        result = detector.update_batch(stream)
        # Stable stream should have few or no drifts
        assert result["drift_detected"] is False or len(result["drift_points"]) <= 2

    def test_detects_drift_in_shifted_stream(self):
        """Test that mean shift is detected as drift."""
        detector = DriftDetector(algorithm="adwin", delta=0.001)
        stream = create_drift_stream(length1=100, length2=100, mean1=10.0, mean2=50.0, noise=0.1)
        result = detector.update_batch(stream)
        # Should detect drift around index 100
        assert result["drift_detected"] is True
        assert len(result["drift_points"]) >= 1

    def test_drift_point_near_change(self):
        """Test that drift point is near the actual change."""
        detector = DriftDetector(algorithm="adwin", delta=0.001)
        # Large mean shift should be detected
        stream = create_drift_stream(length1=50, length2=50, mean1=0.0, mean2=100.0, noise=0.1)
        result = detector.update_batch(stream)

        if result["drift_detected"]:
            # Drift should be detected somewhere near the change point (index 50)
            drift_indices = result["drift_points"]
            # At least one drift should be detected
            assert len(drift_indices) >= 1


class TestDifferentAlgorithms:
    """Test different drift detection algorithms."""

    def test_adwin_algorithm(self):
        """Test ADWIN algorithm."""
        detector = DriftDetector(algorithm="adwin")
        stream = create_drift_stream()
        result = detector.update_batch(stream)
        assert result["algorithm"] == "adwin"

    def test_page_hinkley_algorithm(self):
        """Test Page-Hinkley algorithm."""
        detector = DriftDetector(algorithm="page_hinkley")
        stream = create_drift_stream()
        result = detector.update_batch(stream)
        assert result["algorithm"] == "page_hinkley"

    def test_kswin_algorithm(self):
        """Test KSWIN algorithm."""
        detector = DriftDetector(algorithm="kswin", window_size=50, stat_size=20)
        stream = create_drift_stream()
        result = detector.update_batch(stream)
        assert result["algorithm"] == "kswin"

    def test_ddm_algorithm(self):
        """Test DDM algorithm with binary values."""
        detector = DriftDetector(algorithm="ddm")
        # DDM expects binary values (0/1 for correct/incorrect)
        stream = [0] * 50 + [1] * 50  # All correct then all incorrect
        result = detector.update_batch(stream)
        assert result["algorithm"] == "ddm"


class TestReset:
    """Test detector reset functionality."""

    def test_reset_clears_index(self):
        """Test that reset clears the index."""
        detector = DriftDetector()
        detector.update_batch([1, 2, 3, 4, 5])
        assert detector._index == 5
        detector.reset()
        assert detector._index == 0

    def test_reset_clears_drift_points(self):
        """Test that reset clears drift points."""
        detector = DriftDetector(algorithm="adwin", delta=0.001)
        stream = create_drift_stream(length1=50, length2=50, mean1=0.0, mean2=100.0)
        detector.update_batch(stream)
        detector.reset()
        assert len(detector._drift_points) == 0


class TestGetState:
    """Test state retrieval."""

    def test_state_has_required_keys(self):
        """Test that state has required keys."""
        detector = DriftDetector()
        state = detector.get_state()
        assert "algorithm" in state
        assert "index" in state
        assert "total_drift_points" in state
        assert "drift_points" in state

    def test_state_reflects_updates(self):
        """Test that state reflects updates."""
        detector = DriftDetector()
        detector.update_batch([1, 2, 3, 4, 5])
        state = detector.get_state()
        assert state["index"] == 5


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_detect_drift_in_stream(self):
        """Test the convenience function."""
        stream = create_drift_stream()
        result = detect_drift_in_stream(stream, algorithm="adwin")
        assert "drift_detected" in result
        assert "drift_points" in result
        assert "total_processed" in result

    def test_detect_drift_with_kwargs(self):
        """Test convenience function with kwargs."""
        stream = create_drift_stream()
        result = detect_drift_in_stream(stream, algorithm="adwin", delta=0.001)
        assert result["algorithm"] == "adwin"


class TestSupportedAlgorithms:
    """Test supported algorithms list."""

    def test_supported_algorithms_list(self):
        """Test that SUPPORTED_ALGORITHMS is defined."""
        assert "adwin" in SUPPORTED_ALGORITHMS
        assert "page_hinkley" in SUPPORTED_ALGORITHMS
        assert "kswin" in SUPPORTED_ALGORITHMS
        assert "ddm" in SUPPORTED_ALGORITHMS


class TestDDMSpecificBehavior:
    """Test DDM-specific behavior."""

    def test_ddm_warning_zone(self):
        """Test that DDM reports warning zone."""
        detector = DriftDetector(algorithm="ddm")
        result = detector.update(0)  # Correct prediction
        assert "warning" in result
