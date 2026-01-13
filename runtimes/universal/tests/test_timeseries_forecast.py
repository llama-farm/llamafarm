"""Tests for Time-Series Forecasting Model (Chronos-Bolt).

Phase 11 of Universal Runtime ML Enhancements.
"""

import pytest

from models.timeseries_model import TimeSeriesModel


def create_linear_series(length: int = 30, start: float = 1.0, step: float = 1.0) -> list[float]:
    """Create a simple linear time-series."""
    return [start + i * step for i in range(length)]


def create_seasonal_series(length: int = 30, period: int = 7) -> list[float]:
    """Create a seasonal time-series."""
    import math
    return [10.0 + 5.0 * math.sin(2 * math.pi * i / period) for i in range(length)]


class TestTimeSeriesModelLoading:
    """Test Chronos model loading."""

    @pytest.mark.asyncio
    async def test_model_loads_successfully(self):
        """Test that Chronos model loads."""
        model = TimeSeriesModel(model_id="test-chronos", device="cpu")
        await model.load()
        assert model.is_loaded
        await model.unload()

    @pytest.mark.asyncio
    async def test_model_unload(self):
        """Test that model can be unloaded."""
        model = TimeSeriesModel(model_id="test-chronos", device="cpu")
        await model.load()
        assert model.is_loaded
        await model.unload()
        assert not model.is_loaded

    @pytest.mark.asyncio
    async def test_model_reloads(self):
        """Test that model can be reloaded after unload."""
        model = TimeSeriesModel(model_id="test-chronos", device="cpu")
        await model.load()
        await model.unload()
        await model.load()
        assert model.is_loaded
        await model.unload()


class TestForecasting:
    """Test forecasting functionality."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = TimeSeriesModel(model_id="test-chronos", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_forecast_returns_dict(self, model):
        """Test that forecast returns a dictionary."""
        values = create_linear_series(length=20)
        result = await model.forecast(values, horizon=5)
        assert isinstance(result, dict)
        assert "forecasts" in result
        assert "horizon" in result
        assert "input_length" in result

    @pytest.mark.asyncio
    async def test_forecast_horizon_length(self, model):
        """Test that forecast returns correct number of steps."""
        values = create_linear_series(length=20)
        horizon = 7
        result = await model.forecast(values, horizon=horizon)
        assert len(result["forecasts"]) == horizon

    @pytest.mark.asyncio
    async def test_forecast_has_point_and_intervals(self, model):
        """Test that each forecast has point and intervals."""
        values = create_linear_series(length=20)
        result = await model.forecast(values, horizon=5)

        for forecast in result["forecasts"]:
            assert "step" in forecast
            assert "point" in forecast
            assert "lower" in forecast
            assert "upper" in forecast

    @pytest.mark.asyncio
    async def test_forecast_intervals_ordered(self, model):
        """Test that lower <= point <= upper."""
        values = create_linear_series(length=20)
        result = await model.forecast(values, horizon=5)

        for forecast in result["forecasts"]:
            assert forecast["lower"] <= forecast["point"]
            assert forecast["point"] <= forecast["upper"]

    @pytest.mark.asyncio
    async def test_forecast_linear_trend(self, model):
        """Test forecasting a linear trend."""
        values = create_linear_series(length=30, start=1.0, step=1.0)
        result = await model.forecast(values, horizon=5)

        # Forecasts should generally continue the trend
        # Last input value is 30, forecasts should be >= 30 on average
        first_point = result["forecasts"][0]["point"]
        assert first_point > values[-1] * 0.5  # Reasonable continuation


class TestQuantiles:
    """Test quantile functionality."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = TimeSeriesModel(model_id="test-chronos", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_default_quantiles(self, model):
        """Test default quantiles are returned."""
        values = create_linear_series(length=20)
        result = await model.forecast(values, horizon=5)
        assert "quantiles" in result
        assert result["quantiles"] == [0.1, 0.5, 0.9]

    @pytest.mark.asyncio
    async def test_custom_quantiles(self, model):
        """Test custom quantiles."""
        values = create_linear_series(length=20)
        custom_quantiles = [0.05, 0.5, 0.95]
        result = await model.forecast(values, horizon=5, quantiles=custom_quantiles)
        assert result["quantiles"] == custom_quantiles


class TestVariousSeriesLengths:
    """Test with various input lengths."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = TimeSeriesModel(model_id="test-chronos", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_minimum_length(self, model):
        """Test with minimum required length (3)."""
        values = [1.0, 2.0, 3.0]
        result = await model.forecast(values, horizon=3)
        assert len(result["forecasts"]) == 3

    @pytest.mark.asyncio
    async def test_longer_series(self, model):
        """Test with longer series."""
        values = create_linear_series(length=100)
        result = await model.forecast(values, horizon=10)
        assert len(result["forecasts"]) == 10

    @pytest.mark.asyncio
    async def test_input_length_reported(self, model):
        """Test that input length is correctly reported."""
        values = create_linear_series(length=25)
        result = await model.forecast(values, horizon=5)
        assert result["input_length"] == 25


class TestBatchForecasting:
    """Test batch forecasting."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = TimeSeriesModel(model_id="test-chronos", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_batch_forecast(self, model):
        """Test batch forecasting returns list."""
        series_list = [
            create_linear_series(length=20),
            create_seasonal_series(length=20),
        ]
        results = await model.forecast_batch(series_list, horizon=5)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_batch_forecast_each_result(self, model):
        """Test each batch result has correct format."""
        series_list = [
            create_linear_series(length=20),
            create_linear_series(length=15),
        ]
        results = await model.forecast_batch(series_list, horizon=5)

        for result in results:
            assert "forecasts" in result
            assert "horizon" in result
            assert len(result["forecasts"]) == 5


class TestValidation:
    """Test input validation."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = TimeSeriesModel(model_id="test-chronos", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_too_few_values_raises(self, model):
        """Test that too few values raises error."""
        with pytest.raises(ValueError, match="(?i)at least 3"):
            await model.forecast([1.0, 2.0], horizon=5)

    @pytest.mark.asyncio
    async def test_zero_horizon_raises(self, model):
        """Test that zero horizon raises error."""
        with pytest.raises(ValueError, match="at least 1"):
            await model.forecast([1.0, 2.0, 3.0], horizon=0)

    @pytest.mark.asyncio
    async def test_negative_horizon_raises(self, model):
        """Test that negative horizon raises error."""
        with pytest.raises(ValueError, match="at least 1"):
            await model.forecast([1.0, 2.0, 3.0], horizon=-1)


class TestModelNotLoaded:
    """Test behavior when model is not loaded."""

    @pytest.mark.asyncio
    async def test_forecast_without_load_raises(self):
        """Test that forecast without load raises error."""
        model = TimeSeriesModel(model_id="test-chronos", device="cpu")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await model.forecast([1.0, 2.0, 3.0, 4.0, 5.0], horizon=5)


class TestModelInfo:
    """Test model information retrieval."""

    @pytest.mark.asyncio
    async def test_model_info(self):
        """Test getting model info."""
        model = TimeSeriesModel(model_id="test-chronos", device="cpu")
        await model.load()

        info = model.get_model_info()
        assert "hf_model_name" in info
        assert "is_loaded" in info
        assert "default_quantiles" in info
        assert info["is_loaded"] is True
        assert info["default_quantiles"] == [0.1, 0.5, 0.9]

        await model.unload()
