"""TimeSeries service for forecasting, changepoint detection, and drift detection.

Handles:
- Time-series forecasting (Chronos)
- Change point detection (ruptures)
- Concept drift detection (River)
"""

import logging
import uuid
from typing import TYPE_CHECKING, Any

from services.model_manager import ModelManager, ModelType, model_manager
from utils.device import get_optimal_device

if TYPE_CHECKING:
    from models import TimeSeriesModel

logger = logging.getLogger(__name__)

# Global device cache
_current_device: str | None = None


def get_device() -> str:
    """Get the optimal device for the current platform."""
    global _current_device
    if _current_device is None:
        _current_device = get_optimal_device()
        logger.info(f"Using device: {_current_device}")
    return _current_device


class TimeSeriesService:
    """Service for time-series operations."""

    def __init__(self, manager: ModelManager | None = None):
        """Initialize the TimeSeries service.

        Args:
            manager: Optional ModelManager instance. Uses global singleton if not provided.
        """
        self._manager = manager or model_manager
        self._drift_detectors: dict[str, Any] = {}

    # =========================================================================
    # Time-Series Forecasting
    # =========================================================================

    async def load_forecast_model(
        self, model_name: str = "amazon/chronos-t5-small"
    ) -> "TimeSeriesModel":
        """Load a time-series forecasting model.

        Args:
            model_name: HuggingFace model name

        Returns:
            Loaded TimeSeriesModel instance
        """
        from models import TimeSeriesModel

        cache_key = f"timeseries:{model_name}"

        async def loader():
            logger.info(f"Loading time-series model: {model_name}")
            device = get_device()
            model = TimeSeriesModel(
                model_id=cache_key,
                device=device,
                hf_model_name=model_name,
            )
            await model.load()
            return model

        return await self._manager.get_or_load(ModelType.TIMESERIES, cache_key, loader)

    async def forecast(
        self,
        values: list[float],
        horizon: int = 7,
        quantiles: list[float] | None = None,
        num_samples: int = 20,
        model_name: str = "amazon/chronos-t5-small",
    ) -> dict:
        """Generate time-series forecast.

        Args:
            values: Historical time-series values
            horizon: Number of future steps to forecast
            quantiles: Quantile levels for uncertainty
            num_samples: Number of samples for uncertainty estimation
            model_name: HuggingFace model name

        Returns:
            Forecast result with predictions and uncertainty
        """
        model = await self.load_forecast_model(model_name)
        result = await model.forecast(
            values,
            horizon=horizon,
            quantiles=quantiles,
            num_samples=num_samples,
        )
        return {
            "forecasts": result["forecasts"],
            "horizon": result["horizon"],
            "input_length": result["input_length"],
            "quantiles": result["quantiles"],
        }

    async def forecast_batch(
        self,
        series: list[list[float]],
        horizon: int = 7,
        quantiles: list[float] | None = None,
        num_samples: int = 20,
        model_name: str = "amazon/chronos-t5-small",
    ) -> list[dict]:
        """Generate forecasts for multiple time-series.

        Args:
            series: List of time-series
            horizon: Number of future steps to forecast
            quantiles: Quantile levels for uncertainty
            num_samples: Number of samples for uncertainty estimation
            model_name: HuggingFace model name

        Returns:
            List of forecast results
        """
        model = await self.load_forecast_model(model_name)
        return await model.forecast_batch(
            series,
            horizon=horizon,
            quantiles=quantiles,
            num_samples=num_samples,
        )

    # =========================================================================
    # Change Point Detection
    # =========================================================================

    def detect_changepoints(
        self,
        values: list[float],
        n_changepoints: int | None = None,
        penalty: float | None = None,
        algorithm: str = "pelt",
        model: str = "rbf",
        min_size: int = 2,
    ) -> dict:
        """Detect change points in a time-series.

        Args:
            values: Time-series values
            n_changepoints: Exact number of changepoints (if known)
            penalty: Penalty for regularization
            algorithm: Detection algorithm (pelt, binseg, window, bottomup)
            model: Cost function (l1, l2, rbf, normal, ar)
            min_size: Minimum segment size

        Returns:
            Change point detection result
        """
        from utils.changepoint_detector import ChangePointDetector

        detector = ChangePointDetector(
            algorithm=algorithm,
            model=model,
            min_size=min_size,
        )

        return detector.detect(
            values,
            n_changepoints=n_changepoints,
            penalty=penalty,
        )

    def detect_changepoints_batch(
        self,
        series: list[list[float]],
        n_changepoints: int | None = None,
        penalty: float | None = None,
        algorithm: str = "pelt",
        model: str = "rbf",
        min_size: int = 2,
    ) -> list[dict]:
        """Detect change points in multiple time-series.

        Args:
            series: List of time-series
            n_changepoints: Exact number of changepoints (if known)
            penalty: Penalty for regularization
            algorithm: Detection algorithm
            model: Cost function
            min_size: Minimum segment size

        Returns:
            List of change point detection results
        """
        from utils.changepoint_detector import ChangePointDetector

        detector = ChangePointDetector(
            algorithm=algorithm,
            model=model,
            min_size=min_size,
        )

        return detector.detect_batch(
            series,
            n_changepoints=n_changepoints,
            penalty=penalty,
        )

    # =========================================================================
    # Drift Detection
    # =========================================================================

    def detect_drift(
        self,
        values: list[float],
        algorithm: str = "adwin",
        delta: float | None = None,
        threshold: float | None = None,
        alpha: float | None = None,
        window_size: int | None = None,
    ) -> dict:
        """Detect concept drift in a data stream.

        Args:
            values: Data stream values
            algorithm: Drift detection algorithm (adwin, page_hinkley, kswin, ddm)
            delta: Sensitivity parameter
            threshold: Detection threshold
            alpha: Significance level
            window_size: Window size

        Returns:
            Drift detection result
        """
        from utils.drift_detector import DriftDetector

        kwargs: dict[str, Any] = {}
        if delta is not None:
            kwargs["delta"] = delta
        if threshold is not None:
            kwargs["threshold"] = threshold
        if alpha is not None:
            kwargs["alpha"] = alpha
        if window_size is not None:
            kwargs["window_size"] = window_size

        detector = DriftDetector(algorithm=algorithm, **kwargs)
        return detector.update_batch(values)

    def create_drift_detector(
        self,
        algorithm: str = "adwin",
        detector_id: str | None = None,
        delta: float | None = None,
        threshold: float | None = None,
        alpha: float | None = None,
        window_size: int | None = None,
    ) -> dict:
        """Create a stateful drift detector for streaming updates.

        Args:
            algorithm: Drift detection algorithm
            detector_id: Optional ID (generated if not provided)
            delta: Sensitivity parameter
            threshold: Detection threshold
            alpha: Significance level
            window_size: Window size

        Returns:
            Detector info with ID
        """
        from utils.drift_detector import DriftDetector

        if detector_id is None:
            detector_id = str(uuid.uuid4())[:8]

        kwargs: dict[str, Any] = {}
        if delta is not None:
            kwargs["delta"] = delta
        if threshold is not None:
            kwargs["threshold"] = threshold
        if alpha is not None:
            kwargs["alpha"] = alpha
        if window_size is not None:
            kwargs["window_size"] = window_size

        detector = DriftDetector(algorithm=algorithm, **kwargs)
        self._drift_detectors[detector_id] = detector

        return {
            "detector_id": detector_id,
            "algorithm": algorithm,
            "status": "created",
        }

    def update_drift_detector(self, detector_id: str, value: float) -> dict:
        """Update a drift detector with a new value.

        Args:
            detector_id: ID of the detector
            value: New data point

        Returns:
            Update result with drift detection status

        Raises:
            KeyError: If detector not found
        """
        if detector_id not in self._drift_detectors:
            raise KeyError(f"Detector '{detector_id}' not found")

        detector = self._drift_detectors[detector_id]
        result = detector.update(value)

        return {
            "detector_id": detector_id,
            "drift_detected": result["drift_detected"],
            "index": result["index"],
            "value": result["value"],
        }

    def get_drift_detector_state(self, detector_id: str) -> dict:
        """Get the current state of a drift detector.

        Args:
            detector_id: ID of the detector

        Returns:
            Detector state

        Raises:
            KeyError: If detector not found
        """
        if detector_id not in self._drift_detectors:
            raise KeyError(f"Detector '{detector_id}' not found")

        detector = self._drift_detectors[detector_id]
        state = detector.get_state()

        return {
            "detector_id": detector_id,
            **state,
        }

    def delete_drift_detector(self, detector_id: str) -> dict:
        """Delete a drift detector.

        Args:
            detector_id: ID of the detector

        Returns:
            Deletion confirmation

        Raises:
            KeyError: If detector not found
        """
        if detector_id not in self._drift_detectors:
            raise KeyError(f"Detector '{detector_id}' not found")

        del self._drift_detectors[detector_id]

        return {
            "detector_id": detector_id,
            "status": "deleted",
        }


# Global service instance
timeseries_service = TimeSeriesService()
