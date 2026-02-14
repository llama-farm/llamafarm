"""Tests for timeseries forecasting functionality."""
# ruff: noqa: E402  # imports must come after pytest.importorskip

import pytest

darts = pytest.importorskip("darts", reason="timeseries addon not installed")

from datetime import datetime, timedelta

from api_types.timeseries import (
    TimeseriesDataPoint,
    TimeseriesFitRequest,
    TimeseriesPredictRequest,
)
from models.timeseries_model import (
    TRAINABLE_BACKENDS,
    ZERO_SHOT_BACKENDS,
    TimeseriesModel,
    delete_model,
    get_all_backends,
    get_backends_info,
    is_valid_backend,
    list_saved_models,
)


class TestBackendInfo:
    """Test backend information functions."""

    def test_get_all_backends(self):
        """Test that all backends are returned."""
        backends = get_all_backends()
        assert len(backends) == 5
        assert "arima" in backends
        assert "exponential_smoothing" in backends
        assert "theta" in backends
        assert "chronos" in backends
        assert "chronos-bolt" in backends

    def test_is_valid_backend(self):
        """Test backend validation."""
        assert is_valid_backend("arima")
        assert is_valid_backend("chronos")
        assert not is_valid_backend("invalid")
        assert not is_valid_backend("")

    def test_get_backends_info(self):
        """Test backend info structure."""
        info = get_backends_info()
        assert len(info) == 5

        # Check ARIMA info
        arima = next(b for b in info if b.name == "arima")
        assert arima.requires_training is True
        assert arima.supports_confidence_intervals is True

        # Check Chronos info
        chronos = next(b for b in info if b.name == "chronos")
        assert chronos.requires_training is False

    def test_trainable_vs_zero_shot(self):
        """Test backend categorization."""
        assert "arima" in TRAINABLE_BACKENDS
        assert "exponential_smoothing" in TRAINABLE_BACKENDS
        assert "theta" in TRAINABLE_BACKENDS

        assert "chronos" in ZERO_SHOT_BACKENDS
        assert "chronos-bolt" in ZERO_SHOT_BACKENDS


class TestTimeseriesModel:
    """Test TimeseriesModel class."""

    def test_init_with_valid_backend(self):
        """Test model initialization with valid backend."""
        model = TimeseriesModel(
            model_id="test-model",
            device="cpu",
            backend="arima",
        )
        assert model.backend == "arima"
        assert model.model_type == "timeseries_arima"
        assert model.requires_training is True

    def test_init_with_zero_shot_backend(self):
        """Test model initialization with zero-shot backend."""
        model = TimeseriesModel(
            model_id="test-model",
            device="cpu",
            backend="chronos",
        )
        assert model.backend == "chronos"
        assert model.requires_training is False

    def test_init_with_invalid_backend(self):
        """Test model initialization with invalid backend raises error."""
        with pytest.raises(ValueError) as exc_info:
            TimeseriesModel(
                model_id="test-model",
                device="cpu",
                backend="invalid_backend",
            )
        assert "Unknown backend" in str(exc_info.value)

    def test_is_fitted_initially_false(self):
        """Test that model is not fitted initially."""
        model = TimeseriesModel(
            model_id="test-model",
            device="cpu",
            backend="arima",
        )
        assert model.is_fitted is False


class TestTimeseriesDataTypes:
    """Test Pydantic request/response types."""

    def test_data_point_creation(self):
        """Test TimeseriesDataPoint creation."""
        dp = TimeseriesDataPoint(
            timestamp="2024-01-01T00:00:00",
            value=100.0,
        )
        assert dp.timestamp == "2024-01-01T00:00:00"
        assert dp.value == 100.0

    def test_fit_request_with_defaults(self):
        """Test TimeseriesFitRequest with default values."""
        data = [
            {"timestamp": "2024-01-01", "value": 100},
            {"timestamp": "2024-01-02", "value": 120},
        ]
        req = TimeseriesFitRequest(data=data)
        assert req.backend == "arima"
        assert req.overwrite is True
        assert req.model is None
        assert req.frequency is None

    def test_fit_request_with_custom_values(self):
        """Test TimeseriesFitRequest with custom values."""
        data = [
            TimeseriesDataPoint(timestamp="2024-01-01", value=100),
            TimeseriesDataPoint(timestamp="2024-01-02", value=120),
        ]
        req = TimeseriesFitRequest(
            model="sales-forecast",
            backend="exponential_smoothing",
            data=data,
            frequency="D",
            overwrite=False,
            description="Test model",
        )
        assert req.model == "sales-forecast"
        assert req.backend == "exponential_smoothing"
        assert req.frequency == "D"
        assert req.overwrite is False
        assert req.description == "Test model"

    def test_predict_request_validation(self):
        """Test TimeseriesPredictRequest validation."""
        req = TimeseriesPredictRequest(
            model="my-model",
            horizon=30,
        )
        assert req.horizon == 30
        assert req.confidence_level == 0.95
        assert req.data is None

    def test_predict_request_with_data(self):
        """Test TimeseriesPredictRequest with zero-shot data."""
        data = [
            {"timestamp": "2024-01-01", "value": 100},
            {"timestamp": "2024-01-02", "value": 120},
        ]
        req = TimeseriesPredictRequest(
            model="chronos-model",
            horizon=7,
            data=data,
        )
        assert req.horizon == 7
        assert len(req.data) == 2


class TestListAndDeleteModels:
    """Test model listing and deletion functions."""

    def test_list_saved_models_empty(self, tmp_path, monkeypatch):
        """Test listing models when directory is empty."""
        from models import timeseries_model

        monkeypatch.setattr(timeseries_model, "TIMESERIES_MODELS_DIR", tmp_path)
        models = list_saved_models()
        assert models == []

    def test_delete_nonexistent_model(self, tmp_path, monkeypatch):
        """Test deleting a model that doesn't exist."""
        from models import timeseries_model

        monkeypatch.setattr(timeseries_model, "TIMESERIES_MODELS_DIR", tmp_path)
        result = delete_model("nonexistent")
        assert result is False


@pytest.mark.asyncio
class TestTimeseriesModelAsync:
    """Async tests for TimeseriesModel."""

    async def test_load_initializes_darts(self):
        """Test that loading a Darts backend initializes correctly."""
        model = TimeseriesModel(
            model_id="test-darts",
            device="cpu",
            backend="arima",
        )
        await model.load()
        # Darts models are created during fit, not load
        assert model._forecaster is None

    async def test_unload_clears_state(self):
        """Test that unloading clears model state."""
        model = TimeseriesModel(
            model_id="test-unload",
            device="cpu",
            backend="arima",
        )
        await model.load()
        await model.unload()
        assert model._forecaster is None
        assert model._training_series is None
        assert model.is_fitted is False


def generate_sample_data(n_points: int = 100, freq: str = "D") -> list[dict]:
    """Generate sample time-series data for testing."""
    import random

    base_date = datetime(2024, 1, 1)
    data = []

    for i in range(n_points):
        if freq == "D":
            date = base_date + timedelta(days=i)
        elif freq == "H":
            date = base_date + timedelta(hours=i)
        else:
            date = base_date + timedelta(days=i)

        # Generate with trend and some noise
        value = 100 + i * 0.5 + random.gauss(0, 5)
        data.append({
            "timestamp": date.isoformat(),
            "value": value,
        })

    return data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
