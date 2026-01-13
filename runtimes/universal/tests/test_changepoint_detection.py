"""Tests for Change Point Detection (Ruptures).

Phase 12 of Universal Runtime ML Enhancements.
"""

import pytest

from utils.changepoint_detector import ChangePointDetector, SUPPORTED_ALGORITHMS, SUPPORTED_MODELS


def create_step_signal(lengths: list[int], values: list[float]) -> list[float]:
    """Create a signal with step changes."""
    signal = []
    for length, value in zip(lengths, values):
        signal.extend([value] * length)
    return signal


def create_linear_signal(length: int, start: float, step: float) -> list[float]:
    """Create a linear trend signal."""
    return [start + i * step for i in range(length)]


class TestChangePointDetectorInit:
    """Test detector initialization."""

    def test_default_init(self):
        """Test default initialization."""
        detector = ChangePointDetector()
        assert detector.algorithm == "pelt"
        assert detector.model == "rbf"
        assert detector.min_size == 2

    def test_custom_algorithm(self):
        """Test custom algorithm."""
        detector = ChangePointDetector(algorithm="binseg")
        assert detector.algorithm == "binseg"

    def test_custom_model(self):
        """Test custom cost model."""
        detector = ChangePointDetector(model="l2")
        assert detector.model == "l2"

    def test_invalid_algorithm_raises(self):
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            ChangePointDetector(algorithm="invalid")

    def test_invalid_model_raises(self):
        """Test that invalid model raises error."""
        with pytest.raises(ValueError, match="Unsupported model"):
            ChangePointDetector(model="invalid")

    def test_case_insensitive_algorithm(self):
        """Test that algorithm is case-insensitive."""
        detector = ChangePointDetector(algorithm="PELT")
        assert detector.algorithm == "pelt"

    def test_case_insensitive_model(self):
        """Test that model is case-insensitive."""
        detector = ChangePointDetector(model="RBF")
        assert detector.model == "rbf"


class TestDetection:
    """Test change point detection."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        return ChangePointDetector(algorithm="pelt", model="rbf")

    def test_detect_returns_dict(self, detector):
        """Test that detect returns a dictionary."""
        signal = create_step_signal([10, 10, 10], [1.0, 5.0, 2.0])
        result = detector.detect(signal)
        assert isinstance(result, dict)

    def test_detect_has_change_points(self, detector):
        """Test that result has change_points key."""
        signal = create_step_signal([10, 10, 10], [1.0, 5.0, 2.0])
        result = detector.detect(signal)
        assert "change_points" in result

    def test_detect_has_n_segments(self, detector):
        """Test that result has n_segments key."""
        signal = create_step_signal([10, 10, 10], [1.0, 5.0, 2.0])
        result = detector.detect(signal)
        assert "n_segments" in result

    def test_detect_has_segment_boundaries(self, detector):
        """Test that result has segment_boundaries key."""
        signal = create_step_signal([10, 10, 10], [1.0, 5.0, 2.0])
        result = detector.detect(signal)
        assert "segment_boundaries" in result

    def test_detect_step_signal(self):
        """Test detection on a clear step signal."""
        # Use binseg which supports n_changepoints
        detector = ChangePointDetector(algorithm="binseg", model="l2")
        # Create signal: 10 values at 1.0, 10 at 5.0, 10 at 2.0
        signal = create_step_signal([10, 10, 10], [1.0, 5.0, 2.0])
        # Use explicit n_changepoints for deterministic test
        result = detector.detect(signal, n_changepoints=2)

        # Should find 2 change points (at indices ~10 and ~20)
        assert len(result["change_points"]) == 2
        assert result["n_segments"] == 3

    def test_detect_no_change(self, detector):
        """Test detection on a constant signal."""
        signal = [1.0] * 30
        result = detector.detect(signal)

        # May find 0 or few change points for constant signal
        # Exact number depends on penalty/algorithm
        assert "change_points" in result

    def test_segment_boundaries_format(self, detector):
        """Test segment boundaries have correct format."""
        signal = create_step_signal([10, 10, 10], [1.0, 5.0, 2.0])
        result = detector.detect(signal)

        for segment in result["segment_boundaries"]:
            assert "start" in segment
            assert "end" in segment
            assert segment["start"] < segment["end"]

    def test_segments_cover_signal(self, detector):
        """Test that segments cover entire signal."""
        signal = create_step_signal([10, 10, 10], [1.0, 5.0, 2.0])
        result = detector.detect(signal)

        boundaries = result["segment_boundaries"]
        assert boundaries[0]["start"] == 0
        assert boundaries[-1]["end"] == len(signal)


class TestNChangepoints:
    """Test specifying exact number of change points."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        return ChangePointDetector(algorithm="binseg", model="l2")

    def test_exact_n_changepoints(self, detector):
        """Test requesting exact number of change points."""
        signal = create_step_signal([10, 10, 10, 10], [1.0, 5.0, 2.0, 8.0])
        result = detector.detect(signal, n_changepoints=3)

        # Should return exactly 3 change points
        assert len(result["change_points"]) == 3
        assert result["n_segments"] == 4

    def test_zero_changepoints(self, detector):
        """Test requesting zero change points."""
        signal = create_step_signal([10, 10], [1.0, 5.0])
        result = detector.detect(signal, n_changepoints=0)

        assert len(result["change_points"]) == 0
        assert result["n_segments"] == 1


class TestPenalty:
    """Test penalty parameter."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        return ChangePointDetector(algorithm="pelt", model="l2")

    def test_high_penalty_fewer_points(self, detector):
        """Test that high penalty results in fewer change points."""
        signal = create_step_signal([10, 10, 10, 10], [1.0, 2.0, 3.0, 4.0])

        result_low = detector.detect(signal, penalty=1.0)
        result_high = detector.detect(signal, penalty=100.0)

        # Higher penalty should give fewer or equal change points
        assert len(result_high["change_points"]) <= len(result_low["change_points"])


class TestAlgorithms:
    """Test different algorithms."""

    def test_pelt_algorithm(self):
        """Test PELT algorithm."""
        detector = ChangePointDetector(algorithm="pelt")
        signal = create_step_signal([10, 10], [1.0, 5.0])
        result = detector.detect(signal)
        assert result["algorithm"] == "pelt"

    def test_binseg_algorithm(self):
        """Test Binary Segmentation algorithm."""
        detector = ChangePointDetector(algorithm="binseg")
        signal = create_step_signal([10, 10], [1.0, 5.0])
        result = detector.detect(signal)
        assert result["algorithm"] == "binseg"

    def test_window_algorithm(self):
        """Test Window algorithm."""
        detector = ChangePointDetector(algorithm="window")
        signal = create_step_signal([10, 10], [1.0, 5.0])
        result = detector.detect(signal)
        assert result["algorithm"] == "window"

    def test_bottomup_algorithm(self):
        """Test Bottom-Up algorithm."""
        detector = ChangePointDetector(algorithm="bottomup")
        signal = create_step_signal([10, 10], [1.0, 5.0])
        result = detector.detect(signal)
        assert result["algorithm"] == "bottomup"


class TestModels:
    """Test different cost models."""

    def test_rbf_model(self):
        """Test RBF cost model."""
        detector = ChangePointDetector(model="rbf")
        signal = create_step_signal([10, 10], [1.0, 5.0])
        result = detector.detect(signal)
        assert result["model"] == "rbf"

    def test_l1_model(self):
        """Test L1 cost model."""
        detector = ChangePointDetector(model="l1")
        signal = create_step_signal([10, 10], [1.0, 5.0])
        result = detector.detect(signal)
        assert result["model"] == "l1"

    def test_l2_model(self):
        """Test L2 cost model."""
        detector = ChangePointDetector(model="l2")
        signal = create_step_signal([10, 10], [1.0, 5.0])
        result = detector.detect(signal)
        assert result["model"] == "l2"

    def test_normal_model(self):
        """Test Normal distribution cost model."""
        detector = ChangePointDetector(model="normal")
        signal = create_step_signal([10, 10], [1.0, 5.0])
        result = detector.detect(signal)
        assert result["model"] == "normal"

    def test_ar_model(self):
        """Test Autoregressive cost model."""
        detector = ChangePointDetector(model="ar")
        signal = create_step_signal([15, 15], [1.0, 5.0])  # AR needs more points
        result = detector.detect(signal)
        assert result["model"] == "ar"


class TestBatchDetection:
    """Test batch change point detection."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        return ChangePointDetector(algorithm="pelt", model="rbf")

    def test_batch_returns_list(self, detector):
        """Test that batch detection returns a list."""
        series = [
            create_step_signal([10, 10], [1.0, 5.0]),
            create_step_signal([10, 10], [2.0, 8.0]),
        ]
        results = detector.detect_batch(series)
        assert isinstance(results, list)

    def test_batch_correct_count(self, detector):
        """Test that batch returns correct number of results."""
        series = [
            create_step_signal([10, 10], [1.0, 5.0]),
            create_step_signal([10, 10], [2.0, 8.0]),
            create_step_signal([10, 10], [3.0, 9.0]),
        ]
        results = detector.detect_batch(series)
        assert len(results) == 3

    def test_batch_each_result_complete(self, detector):
        """Test that each batch result has all fields."""
        series = [
            create_step_signal([10, 10], [1.0, 5.0]),
            create_step_signal([10, 10], [2.0, 8.0]),
        ]
        results = detector.detect_batch(series)

        for result in results:
            assert "change_points" in result
            assert "n_segments" in result
            assert "segment_boundaries" in result


class TestValidation:
    """Test input validation."""

    def test_too_short_signal_raises(self):
        """Test that too short signal raises error."""
        detector = ChangePointDetector(min_size=2)
        with pytest.raises(ValueError, match="too short"):
            detector.detect([1.0, 2.0, 3.0])  # Need at least 4 points

    def test_min_size_respected(self):
        """Test that min_size is respected."""
        detector = ChangePointDetector(min_size=5)
        # Need at least 10 points
        with pytest.raises(ValueError, match="at least 10"):
            detector.detect([1.0] * 8)


class TestMetadata:
    """Test result metadata."""

    def test_signal_length_in_result(self):
        """Test that signal length is in result."""
        detector = ChangePointDetector()
        signal = create_step_signal([10, 10], [1.0, 5.0])
        result = detector.detect(signal)
        assert result["signal_length"] == 20

    def test_algorithm_in_result(self):
        """Test that algorithm is in result."""
        detector = ChangePointDetector(algorithm="binseg")
        signal = create_step_signal([10, 10], [1.0, 5.0])
        result = detector.detect(signal)
        assert result["algorithm"] == "binseg"

    def test_model_in_result(self):
        """Test that model is in result."""
        detector = ChangePointDetector(model="l1")
        signal = create_step_signal([10, 10], [1.0, 5.0])
        result = detector.detect(signal)
        assert result["model"] == "l1"


class TestSupportedOptions:
    """Test supported algorithms and models."""

    def test_supported_algorithms_list(self):
        """Test that SUPPORTED_ALGORITHMS is defined."""
        assert "pelt" in SUPPORTED_ALGORITHMS
        assert "binseg" in SUPPORTED_ALGORITHMS
        assert "window" in SUPPORTED_ALGORITHMS
        assert "bottomup" in SUPPORTED_ALGORITHMS

    def test_supported_models_list(self):
        """Test that SUPPORTED_MODELS is defined."""
        assert "l1" in SUPPORTED_MODELS
        assert "l2" in SUPPORTED_MODELS
        assert "rbf" in SUPPORTED_MODELS
        assert "normal" in SUPPORTED_MODELS
        assert "ar" in SUPPORTED_MODELS
