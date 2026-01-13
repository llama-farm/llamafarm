"""Time-Series Forecasting Model using Chronos-Bolt.

Provides probabilistic time-series forecasting using Amazon's Chronos-Bolt
transformer model, which produces predictions with uncertainty quantification.

Usage:
    model = TimeSeriesModel(model_id="chronos")
    await model.load()
    result = await model.forecast([1.0, 2.0, 3.0, ...], horizon=7)
    # Returns: {"forecasts": [{"point": 4.0, "lower": 3.5, "upper": 4.5}, ...]}
"""

import asyncio
import logging
from typing import Any

import numpy as np
import torch

from models.base import BaseModel

logger = logging.getLogger(__name__)

# Default Chronos model - t5-small is fast and accurate
DEFAULT_CHRONOS_MODEL = "amazon/chronos-t5-small"


class TimeSeriesModel(BaseModel):
    """Time-series forecasting model using Chronos-Bolt.

    Chronos-Bolt is a transformer-based time-series forecasting model that
    produces probabilistic forecasts with confidence intervals.

    Attributes:
        model_id: Unique identifier for this model instance
        device: Device to run inference on (cpu, cuda, mps)
        hf_model_name: HuggingFace model name
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        hf_model_name: str = DEFAULT_CHRONOS_MODEL,
        token: str | None = None,
    ):
        """Initialize time-series forecasting model.

        Args:
            model_id: Unique identifier for this model instance
            device: Device to run on (cpu, cuda, mps)
            hf_model_name: HuggingFace model to load
            token: Optional HuggingFace token
        """
        super().__init__(model_id=model_id, device=device, token=token)
        self.hf_model_name = hf_model_name
        self.model_type = "timeseries_forecast"
        self.supports_streaming = False
        self._is_loaded = False
        self.pipeline = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._is_loaded and self.pipeline is not None

    async def load(self) -> None:
        """Load the Chronos model pipeline."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} already loaded")
            return

        logger.info(f"Loading Chronos time-series model: {self.hf_model_name}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

        self._is_loaded = True
        logger.info(f"Chronos model loaded: {self.model_id}")

    def _load_sync(self) -> None:
        """Synchronous model loading (run in executor)."""
        from chronos import ChronosPipeline

        # Map device for chronos
        device_map = self.device
        if self.device == "mps":
            # Chronos may have MPS issues, use CPU for stability
            device_map = "cpu"

        self.pipeline = ChronosPipeline.from_pretrained(
            self.hf_model_name,
            device_map=device_map,
            torch_dtype=self.get_dtype(force_float32=(self.device == "mps")),
        )

    async def unload(self) -> None:
        """Unload the model and free resources."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        await super().unload()
        self._is_loaded = False

    async def forecast(
        self,
        values: list[float],
        horizon: int = 7,
        quantiles: list[float] | None = None,
        num_samples: int = 20,
    ) -> dict[str, Any]:
        """Generate time-series forecasts.

        Args:
            values: Historical time-series values (at least 3 values recommended)
            horizon: Number of future steps to forecast
            quantiles: Quantile levels for prediction intervals (default: [0.1, 0.5, 0.9])
            num_samples: Number of sample paths for uncertainty estimation

        Returns:
            Dictionary with:
                - forecasts: List of point forecasts with uncertainty
                - horizon: Number of steps forecasted
                - input_length: Length of input time-series
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if len(values) < 3:
            raise ValueError("At least 3 historical values are required")

        if horizon < 1:
            raise ValueError("Horizon must be at least 1")

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._forecast_sync,
            values,
            horizon,
            quantiles,
            num_samples,
        )

    def _forecast_sync(
        self,
        values: list[float],
        horizon: int,
        quantiles: list[float],
        num_samples: int,
    ) -> dict[str, Any]:
        """Synchronous forecasting."""
        # Convert to tensor
        context = torch.tensor(values, dtype=torch.float32)

        # Generate forecasts
        # Chronos returns (num_series, num_samples, horizon)
        forecast_samples = self.pipeline.predict(
            context,
            prediction_length=horizon,
            num_samples=num_samples,
        )

        # Convert to numpy for quantile calculation
        # forecast_samples shape: (1, num_samples, horizon) for single series
        samples = forecast_samples[0].cpu().numpy()  # (num_samples, horizon)

        # Calculate quantiles for each horizon step
        forecasts = []
        for t in range(horizon):
            step_samples = samples[:, t]
            quantile_values = np.quantile(step_samples, quantiles)

            # Find lower, point (median), and upper
            lower = float(quantile_values[0])  # 0.1 quantile
            point = float(np.median(step_samples))  # median
            upper = float(quantile_values[-1])  # 0.9 quantile

            forecasts.append({
                "step": t + 1,
                "point": point,
                "lower": lower,
                "upper": upper,
            })

        return {
            "forecasts": forecasts,
            "horizon": horizon,
            "input_length": len(values),
            "quantiles": quantiles,
        }

    async def forecast_batch(
        self,
        series_list: list[list[float]],
        horizon: int = 7,
        quantiles: list[float] | None = None,
        num_samples: int = 20,
    ) -> list[dict[str, Any]]:
        """Generate forecasts for multiple time-series.

        Args:
            series_list: List of time-series (each is a list of floats)
            horizon: Number of future steps to forecast
            quantiles: Quantile levels for prediction intervals
            num_samples: Number of sample paths

        Returns:
            List of forecast results (one per series)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        results = []
        for values in series_list:
            result = await self.forecast(
                values,
                horizon=horizon,
                quantiles=quantiles,
                num_samples=num_samples,
            )
            results.append(result)
        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "hf_model_name": self.hf_model_name,
            "is_loaded": self.is_loaded,
            "default_quantiles": [0.1, 0.5, 0.9],
        })
        return info
