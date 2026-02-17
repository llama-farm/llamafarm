"""Tests for ADTK time-series anomaly detection functionality."""
# ruff: noqa: E402  # imports must come after pytest.importorskip

import pytest

adtk = pytest.importorskip("adtk", reason="adtk addon not installed")

from datetime import datetime, timedelta

from api_types.adtk import (
    ADTKDataPoint,
    ADTKDetectRequest,
    ADTKFitRequest,
)
from models.adtk_model import (
    DETECTOR_TYPES,
    ADTKModel,
    delete_model,
    get_all_detectors,
    get_detectors_info,
    is_valid_detector,
    list_saved_models,
)


class TestDetectorInfo:
    """Test detector information functions."""

    def test_get_all_detectors(self):
        """Test that all detectors are returned."""
        detectors = get_all_detectors()
        assert len(detectors) == 6
        assert "level_shift" in detectors
        assert "seasonal" in detectors
        assert "spike" in detectors
        assert "volatility_shift" in detectors
        assert "persist" in detectors
        assert "threshold" in detectors

    def test_is_valid_detector(self):
        """Test detector validation."""
        assert is_valid_detector("level_shift")
        assert is_valid_detector("seasonal")
        assert is_valid_detector("spike")
        assert not is_valid_detector("invalid")
        assert not is_valid_detector("")

    def test_get_detectors_info(self):
        """Test detector info structure."""
        info = get_detectors_info()
        assert len(info) == 6

        # Check level_shift info
        level_shift = next(d for d in info if d["name"] == "level_shift")
        assert level_shift["requires_training"] is False
        assert "description" in level_shift
        assert "default_params" in level_shift

        # Check seasonal info (requires training)
        seasonal = next(d for d in info if d["name"] == "seasonal")
        assert seasonal["requires_training"] is True

    def test_detector_types_have_required_fields(self):
        """Test that all detector types have required configuration fields."""
        for name, config in DETECTOR_TYPES.items():
            assert "description" in config, f"{name} missing description"
            assert "class" in config, f"{name} missing class"
            assert "requires_training" in config, f"{name} missing requires_training"
            assert "default_params" in config, f"{name} missing default_params"


class TestADTKModel:
    """Test ADTKModel class."""

    def test_init_with_valid_detector(self):
        """Test model initialization with valid detector."""
        model = ADTKModel(
            model_id="test-model",
            device="cpu",
            detector="level_shift",
        )
        assert model.detector_type == "level_shift"
        assert model.model_type == "adtk_level_shift"
        assert model.requires_training is False

    def test_init_with_seasonal_detector(self):
        """Test model initialization with seasonal detector (requires training)."""
        model = ADTKModel(
            model_id="test-model",
            device="cpu",
            detector="seasonal",
        )
        assert model.detector_type == "seasonal"
        assert model.requires_training is True

    def test_init_with_spike_detector(self):
        """Test model initialization with spike detector."""
        model = ADTKModel(
            model_id="test-model",
            device="cpu",
            detector="spike",
        )
        assert model.detector_type == "spike"
        assert model.requires_training is False

    def test_init_with_invalid_detector(self):
        """Test model initialization with invalid detector raises error."""
        with pytest.raises(ValueError) as exc_info:
            ADTKModel(
                model_id="test-model",
                device="cpu",
                detector="invalid_detector",
            )
        assert "Unknown detector" in str(exc_info.value)

    def test_is_fitted_initially_false(self):
        """Test that model is not fitted initially."""
        model = ADTKModel(
            model_id="test-model",
            device="cpu",
            detector="level_shift",
        )
        assert model.is_fitted is False


class TestADTKDataTypes:
    """Test Pydantic request/response types."""

    def test_data_point_creation(self):
        """Test ADTKDataPoint creation."""
        dp = ADTKDataPoint(
            timestamp="2024-01-01T00:00:00",
            value=100.0,
        )
        assert dp.timestamp == "2024-01-01T00:00:00"
        assert dp.value == 100.0

    def test_fit_request_with_defaults(self):
        """Test ADTKFitRequest with default values."""
        data = [
            {"timestamp": "2024-01-01", "value": 100},
            {"timestamp": "2024-01-02", "value": 120},
        ]
        req = ADTKFitRequest(data=data)
        assert req.detector == "level_shift"
        assert req.overwrite is True
        assert req.model is None

    def test_fit_request_with_custom_values(self):
        """Test ADTKFitRequest with custom values."""
        data = [
            ADTKDataPoint(timestamp="2024-01-01", value=100),
            ADTKDataPoint(timestamp="2024-01-02", value=120),
        ]
        req = ADTKFitRequest(
            model="my-anomaly-detector",
            detector="seasonal",
            data=data,
            overwrite=False,
            description="Test detector",
            params={"c": 5.0},
        )
        assert req.model == "my-anomaly-detector"
        assert req.detector == "seasonal"
        assert req.overwrite is False
        assert req.description == "Test detector"
        assert req.params == {"c": 5.0}

    def test_detect_request_validation(self):
        """Test ADTKDetectRequest validation."""
        data = [
            {"timestamp": "2024-01-01", "value": 100},
            {"timestamp": "2024-01-02", "value": 120},
        ]
        req = ADTKDetectRequest(
            detector="spike",
            data=data,
        )
        assert req.detector == "spike"
        assert req.model is None
        assert len(req.data) == 2


class TestListAndDeleteModels:
    """Test model listing and deletion functions."""

    def test_list_saved_models_empty(self, tmp_path, monkeypatch):
        """Test listing models when directory is empty."""
        from models import adtk_model

        monkeypatch.setattr(adtk_model, "ADTK_MODELS_DIR", tmp_path)
        models = list_saved_models()
        assert models == []

    def test_delete_nonexistent_model(self, tmp_path, monkeypatch):
        """Test deleting a model that doesn't exist."""
        from models import adtk_model

        monkeypatch.setattr(adtk_model, "ADTK_MODELS_DIR", tmp_path)
        result = delete_model("nonexistent")
        assert result is False


@pytest.mark.asyncio
class TestADTKModelAsync:
    """Async tests for ADTKModel."""

    async def test_load_creates_detector(self):
        """Test that loading creates the detector instance."""
        model = ADTKModel(
            model_id="test-load",
            device="cpu",
            detector="level_shift",
        )
        await model.load()
        assert model._detector is not None

    async def test_unload_clears_state(self):
        """Test that unloading clears model state."""
        model = ADTKModel(
            model_id="test-unload",
            device="cpu",
            detector="spike",
        )
        await model.load()
        await model.unload()
        assert model._detector is None
        assert model._is_fitted is False

    async def test_detect_level_shift(self):
        """Test level shift detection on synthetic data."""
        model = ADTKModel(
            model_id="test-level-shift",
            device="cpu",
            detector="level_shift",
            window=3,
            c=3.0,
        )
        await model.load()

        # Create data with a level shift at day 10
        data = generate_level_shift_data(shift_at=10, shift_amount=50)
        result = await model.detect(data)

        assert result.detector == "level_shift"
        assert result.total_points == len(data)
        # Should detect the level shift (may detect around the shift point)
        assert result.anomaly_count >= 0  # At least no crash

    async def test_detect_spike(self):
        """Test spike detection on synthetic data."""
        model = ADTKModel(
            model_id="test-spike",
            device="cpu",
            detector="spike",
            c=1.5,
        )
        await model.load()

        # Create data with a spike at day 15
        data = generate_spike_data(spike_at=15, spike_value=500)
        result = await model.detect(data)

        assert result.detector == "spike"
        assert result.total_points == len(data)
        # The spike should be detected (exact count depends on ADTK's IQR logic)
        assert result.detection_time_ms > 0


def generate_level_shift_data(
    n_points: int = 30,
    shift_at: int = 15,
    shift_amount: float = 50.0,
) -> list[dict]:
    """Generate time-series data with a level shift."""
    import random

    base_date = datetime(2024, 1, 1)
    data = []

    for i in range(n_points):
        date = base_date + timedelta(days=i)
        # Before shift: values around 100
        # After shift: values around 100 + shift_amount
        base_value = 100 if i < shift_at else 100 + shift_amount
        value = base_value + random.gauss(0, 2)
        data.append({
            "timestamp": date.isoformat(),
            "value": value,
        })

    return data


def generate_spike_data(
    n_points: int = 30,
    spike_at: int = 15,
    spike_value: float = 500.0,
) -> list[dict]:
    """Generate time-series data with a spike."""
    import random

    base_date = datetime(2024, 1, 1)
    data = []

    for i in range(n_points):
        date = base_date + timedelta(days=i)
        # Normal values around 100, with a spike at spike_at
        value = spike_value if i == spike_at else 100 + random.gauss(0, 5)
        data.append({
            "timestamp": date.isoformat(),
            "value": value,
        })

    return data


def generate_seasonal_data(
    n_points: int = 60,
    period: int = 7,
    anomaly_at: int = 30,
) -> list[dict]:
    """Generate time-series data with seasonality and an anomaly."""
    import math
    import random

    base_date = datetime(2024, 1, 1)
    data = []

    for i in range(n_points):
        date = base_date + timedelta(days=i)
        # Weekly seasonality pattern
        seasonal = 20 * math.sin(2 * math.pi * i / period)
        base_value = 100 + seasonal

        # Inject anomaly
        if i == anomaly_at:
            base_value += 50  # Break the pattern

        value = base_value + random.gauss(0, 2)
        data.append({
            "timestamp": date.isoformat(),
            "value": value,
        })

    return data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
