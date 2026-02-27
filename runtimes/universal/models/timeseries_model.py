"""
Time-series forecasting model wrapper using Darts and Chronos.

Supports multiple backends:
- Classical (Darts): ARIMA, ExponentialSmoothing, Theta
- Foundation Model (Chronos): Zero-shot forecasting

Backends:
---------
Classical (require training):
- arima: Auto-ARIMA for stationary series
- exponential_smoothing: Trend + seasonality decomposition
- theta: Simple and robust forecasting

Zero-shot (no training required):
- chronos: Amazon's Chronos foundation model (T5-based)
- chronos-bolt: Faster Chronos variant

Security Notes:
- Model loading is restricted to TIMESERIES_MODELS_DIR
- Path traversal attacks are prevented
- Serialization uses joblib for classical models
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from services.path_validator import (
    TIMESERIES_MODELS_DIR,
    sanitize_model_name,
    validate_model_path,
)

from .base import BaseModel

logger = logging.getLogger(__name__)

# Backend types
TimeseriesBackendType = Literal[
    "arima",
    "exponential_smoothing",
    "theta",
    "chronos",
    "chronos-bolt",
]

# Backends that require training
TRAINABLE_BACKENDS = {"arima", "exponential_smoothing", "theta"}

# Zero-shot backends (no training needed)
ZERO_SHOT_BACKENDS = {"chronos", "chronos-bolt"}

# All valid backends
ALL_BACKENDS = TRAINABLE_BACKENDS | ZERO_SHOT_BACKENDS


def is_valid_backend(backend: str) -> bool:
    """Check if a backend is valid."""
    return backend in ALL_BACKENDS


def get_all_backends() -> list[str]:
    """Get list of all available backends."""
    return sorted(ALL_BACKENDS)


@dataclass
class BackendInfo:
    """Information about a timeseries backend."""

    name: str
    description: str
    requires_training: bool
    supports_confidence_intervals: bool
    speed: str  # "fast", "medium", "slow"


def get_backends_info() -> list[BackendInfo]:
    """Get information about all available backends."""
    return [
        BackendInfo(
            name="arima",
            description="Auto-ARIMA for stationary time series, handles trend and seasonality",
            requires_training=True,
            supports_confidence_intervals=True,
            speed="medium",
        ),
        BackendInfo(
            name="exponential_smoothing",
            description="Exponential smoothing with trend and seasonality decomposition",
            requires_training=True,
            supports_confidence_intervals=True,
            speed="fast",
        ),
        BackendInfo(
            name="theta",
            description="Simple and robust forecasting method",
            requires_training=True,
            supports_confidence_intervals=True,
            speed="fast",
        ),
        BackendInfo(
            name="chronos",
            description="Amazon's Chronos T5-based foundation model for zero-shot forecasting",
            requires_training=False,
            supports_confidence_intervals=True,
            speed="medium",
        ),
        BackendInfo(
            name="chronos-bolt",
            description="Faster Chronos variant with slightly reduced accuracy",
            requires_training=False,
            supports_confidence_intervals=True,
            speed="fast",
        ),
    ]


@dataclass
class Prediction:
    """A single forecast prediction."""

    timestamp: str
    value: float
    lower: float | None = None
    upper: float | None = None


@dataclass
class FitResult:
    """Result from fitting a forecaster."""

    model: str
    backend: str
    samples_fitted: int
    training_time_ms: float
    saved_path: str
    description: str | None = None
    model_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictResult:
    """Result from forecasting."""

    model_id: str
    backend: str
    predictions: list[Prediction]
    fit_time_ms: float | None = None
    predict_time_ms: float = 0.0


class TimeseriesModel(BaseModel):
    """Wrapper for time-series forecasting models.

    Provides a unified interface to Darts classical models and
    Amazon Chronos foundation models.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        backend: TimeseriesBackendType = "arima",
        **backend_params: Any,
    ):
        """Initialize time-series forecasting model.

        Args:
            model_id: Model identifier for caching/saving
            device: Target device (cpu/cuda)
            backend: Forecasting algorithm
            **backend_params: Additional backend-specific parameters
        """
        super().__init__(model_id, device)

        if not is_valid_backend(backend):
            raise ValueError(
                f"Unknown backend: {backend}. Available: {get_all_backends()}"
            )

        self.backend = backend
        self._backend_params = backend_params
        self.model_type = f"timeseries_{backend}"
        self.supports_streaming = False

        # Forecaster instance
        self._forecaster = None
        self._is_fitted = False
        self._training_series = None

        # Chronos-specific
        self._chronos_pipeline = None

        # Metadata
        self._frequency = None
        self._description: str | None = None

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted or ready for prediction."""
        if self.backend in ZERO_SHOT_BACKENDS:
            return self._chronos_pipeline is not None
        return self._is_fitted

    @property
    def requires_training(self) -> bool:
        """Check if this backend requires training."""
        return self.backend in TRAINABLE_BACKENDS

    async def load(self) -> None:
        """Load or initialize the forecasting model."""
        logger.info(f"Loading timeseries model: {self.backend}")

        model_path = Path(self.model_id)
        if model_path.exists() and model_path.suffix == ".joblib":
            try:
                validated_path = validate_model_path(model_path, "timeseries")
                await self._load_pretrained(validated_path)
            except ValueError as e:
                logger.error(f"Security validation failed: {e}")
                raise
        else:
            # Try to find persisted model by name in models directory
            persisted = self._find_persisted_model()
            if persisted:
                await self._load_pretrained(persisted)
            else:
                await self._initialize_backend()

        logger.info(f"Timeseries model initialized: {self.backend}")

    def _find_persisted_model(self) -> Path | None:
        """Find a persisted model file by name in the models directory."""
        if not TIMESERIES_MODELS_DIR.exists():
            return None
        # Match {model_id}_{backend}.joblib or {model_id}_*.joblib
        safe_name = Path(self.model_id).name
        if safe_name != self.model_id or ".." in self.model_id:
            return None
        candidates = sorted(
            TIMESERIES_MODELS_DIR.glob(f"{safe_name}_{self.backend}.joblib"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
        # Fallback: any backend match
        candidates = sorted(
            TIMESERIES_MODELS_DIR.glob(f"{safe_name}_*.joblib"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    async def _load_pretrained(self, model_path: Path) -> None:
        """Load a pre-trained model from disk."""
        logger.info(f"Loading pretrained model from: {model_path}")

        import joblib
        data = joblib.load(model_path)

        self._forecaster = data.get("forecaster")
        self._training_series = data.get("training_series")
        self._frequency = data.get("frequency")
        self._description = data.get("description")
        self.backend = data.get("backend", self.backend)
        self._is_fitted = True

    async def _initialize_backend(self) -> None:
        """Initialize a fresh forecaster."""
        if self.backend in ZERO_SHOT_BACKENDS:
            await self._initialize_chronos()
        else:
            await self._initialize_darts()

    async def _initialize_chronos(self) -> None:
        """Initialize Chronos pipeline for zero-shot forecasting."""
        try:
            import torch
            from chronos import ChronosPipeline

            # Select model variant
            if self.backend == "chronos-bolt":
                model_name = "amazon/chronos-bolt-small"
            else:
                model_name = self._backend_params.get(
                    "model_name", "amazon/chronos-t5-small"
                )

            # Determine device
            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            logger.info(f"Loading Chronos model: {model_name} on {device}")
            self._chronos_pipeline = ChronosPipeline.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=dtype,
            )
            logger.info("Chronos pipeline loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"Chronos not available: {e}. Install with: pip install chronos-forecasting"
            ) from e

    async def _initialize_darts(self) -> None:
        """Initialize Darts forecaster."""
        # Darts models are created fresh for each fit
        self._forecaster = None

    def _create_darts_model(self):
        """Create a new Darts model instance."""
        if self.backend == "arima":
            from darts.models import AutoARIMA
            return AutoARIMA(**self._backend_params)
        elif self.backend == "exponential_smoothing":
            from darts.models import ExponentialSmoothing
            return ExponentialSmoothing(**self._backend_params)
        elif self.backend == "theta":
            from darts.models import Theta
            return Theta(**self._backend_params)
        else:
            raise ValueError(f"Unknown Darts backend: {self.backend}")

    def _to_darts_series(
        self, data: list[dict], frequency: str | None = None
    ):
        """Convert list of dicts to Darts TimeSeries.

        Args:
            data: List of {"timestamp": str, "value": float}
            frequency: Optional frequency hint (D, H, M, etc.)

        Returns:
            Darts TimeSeries object
        """
        import pandas as pd
        from darts import TimeSeries

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # Infer or set frequency
        if frequency:
            df = df.asfreq(frequency)
        else:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                df = df.asfreq(inferred_freq)
                self._frequency = inferred_freq

        # Fill any gaps
        df = df.ffill()

        return TimeSeries.from_dataframe(df, value_cols="value")

    async def fit(
        self,
        data: list[dict],
        frequency: str | None = None,
        model_name: str | None = None,
        overwrite: bool = True,
        description: str | None = None,
        use_executor: bool = True,
    ) -> FitResult:
        """Fit the forecaster on training data.

        Args:
            data: Training data as list of {"timestamp": str, "value": float}
            frequency: Time frequency (D, H, M, etc.), auto-detected if None
            model_name: Name for saving (auto-generated if None)
            overwrite: If True, overwrite existing model
            description: Optional model description
            use_executor: If True, run training in thread pool

        Returns:
            FitResult with training statistics
        """
        if self.backend in ZERO_SHOT_BACKENDS:
            # Zero-shot models don't train - just store the data reference
            return await self._fit_zero_shot(data, frequency, model_name, description)

        if use_executor:
            from services.training_executor import run_in_executor
            return await run_in_executor(
                self._fit_sync, data, frequency, model_name, overwrite, description
            )
        else:
            return self._fit_sync(data, frequency, model_name, overwrite, description)

    async def _fit_zero_shot(
        self,
        data: list[dict],
        frequency: str | None,
        model_name: str | None,
        description: str | None,
    ) -> FitResult:
        """Handle fit for zero-shot models (just validates data)."""
        if self._chronos_pipeline is None:
            await self._initialize_chronos()

        # Validate data format
        if not data:
            raise ValueError("Data cannot be empty")

        for item in data[:5]:
            if "timestamp" not in item or "value" not in item:
                raise ValueError("Each data point must have 'timestamp' and 'value'")

        # Generate model name if needed
        if not model_name:
            model_name = f"timeseries-{uuid.uuid4().hex[:8]}"

        return FitResult(
            model=model_name,
            backend=self.backend,
            samples_fitted=len(data),
            training_time_ms=0.0,
            saved_path="",  # Zero-shot models aren't saved
            description=description,
            model_params={"note": "Zero-shot model, no training required"},
        )

    def _fit_sync(
        self,
        data: list[dict],
        frequency: str | None,
        model_name: str | None,
        overwrite: bool,
        description: str | None,
    ) -> FitResult:
        """Synchronous fit for thread pool execution."""
        start_time = time.perf_counter()

        # Convert to Darts TimeSeries
        series = self._to_darts_series(data, frequency)
        self._training_series = series

        # Create and fit model
        self._forecaster = self._create_darts_model()
        self._forecaster.fit(series)
        self._is_fitted = True
        self._description = description

        training_time_ms = (time.perf_counter() - start_time) * 1000

        # Generate model name if needed
        if not model_name:
            model_name = f"timeseries-{uuid.uuid4().hex[:8]}"

        safe_name = sanitize_model_name(model_name)

        # Auto-save the model
        saved_path = self._save_sync(safe_name, overwrite, description)

        return FitResult(
            model=safe_name,
            backend=self.backend,
            samples_fitted=len(data),
            training_time_ms=training_time_ms,
            saved_path=str(saved_path),
            description=description,
            model_params=self._backend_params,
        )

    def _save_sync(
        self,
        model_name: str,
        overwrite: bool,
        description: str | None,
    ) -> Path:
        """Save model to disk synchronously."""
        from datetime import datetime

        import joblib

        # Determine filename
        if overwrite:
            filename = f"{model_name}_{self.backend}.joblib"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}_{self.backend}.joblib"

        # Ensure directory exists
        TIMESERIES_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        save_path = TIMESERIES_MODELS_DIR / filename

        # Save model state
        data = {
            "forecaster": self._forecaster,
            "training_series": self._training_series,
            "frequency": self._frequency,
            "backend": self.backend,
            "description": description,
            "saved_at": datetime.now().isoformat(),
        }

        joblib.dump(data, save_path)
        logger.info(f"Saved timeseries model to: {save_path}")

        # Save metadata
        self._save_metadata(save_path.stem, description)

        return save_path

    def _save_metadata(self, model_name: str, description: str | None) -> None:
        """Save model metadata to JSON file."""
        import json

        metadata_path = TIMESERIES_MODELS_DIR / f"{model_name}.metadata.json"
        metadata = {
            "model_name": model_name,
            "backend": self.backend,
            "description": description,
            "frequency": self._frequency,
            "created_at": datetime.now().isoformat(),
        }

        try:
            metadata_path.write_text(json.dumps(metadata, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    async def predict(
        self,
        horizon: int,
        data: list[dict] | None = None,
        confidence_level: float = 0.95,
        use_executor: bool = True,
    ) -> PredictResult:
        """Generate forecasts.

        Args:
            horizon: Number of periods to forecast
            data: Historical data (required for zero-shot, optional otherwise)
            confidence_level: Confidence level for intervals (0.80, 0.90, 0.95)
            use_executor: If True, run prediction in thread pool

        Returns:
            PredictResult with predictions
        """
        if self.backend in ZERO_SHOT_BACKENDS:
            return await self._predict_chronos(horizon, data, confidence_level)

        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if use_executor:
            from services.training_executor import run_in_executor
            return await run_in_executor(
                self._predict_sync, horizon, confidence_level
            )
        else:
            return self._predict_sync(horizon, confidence_level)

    async def _predict_chronos(
        self,
        horizon: int,
        data: list[dict] | None,
        confidence_level: float,
    ) -> PredictResult:
        """Generate predictions using Chronos (zero-shot)."""
        import torch

        if self._chronos_pipeline is None:
            await self._initialize_chronos()

        if data is None:
            raise ValueError("Data is required for zero-shot Chronos prediction")

        start_time = time.perf_counter()

        # Convert data to tensor
        import pandas as pd
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        values = torch.tensor(df["value"].values, dtype=torch.float32)

        # Generate predictions
        num_samples = 20  # For confidence intervals
        forecast = self._chronos_pipeline.predict(
            inputs=values,
            prediction_length=horizon,
            num_samples=num_samples,
        )

        # Calculate statistics
        # Output shape is (batch=1, num_samples, horizon) - squeeze the batch dimension
        forecast_np = forecast.numpy()
        if forecast_np.ndim == 3:
            forecast_np = forecast_np[0]  # Remove batch dimension -> (num_samples, horizon)

        # Now shape is (num_samples, horizon)
        median = np.median(forecast_np, axis=0)  # -> (horizon,)

        # Confidence intervals
        alpha = 1 - confidence_level
        lower = np.percentile(forecast_np, alpha * 100 / 2, axis=0)
        upper = np.percentile(forecast_np, 100 - alpha * 100 / 2, axis=0)

        # Generate timestamps for predictions
        last_timestamp = df.index[-1]
        freq = pd.infer_freq(df.index) or "D"
        offset = pd.tseries.frequencies.to_offset(freq)
        future_dates = pd.date_range(
            start=last_timestamp + offset,
            periods=horizon,
            freq=freq,
        )

        predict_time_ms = (time.perf_counter() - start_time) * 1000

        predictions = [
            Prediction(
                timestamp=ts.isoformat(),
                value=float(med),
                lower=float(lo),
                upper=float(up),
            )
            for ts, med, lo, up in zip(future_dates, median, lower, upper, strict=True)
        ]

        return PredictResult(
            model_id=self.model_id,
            backend=self.backend,
            predictions=predictions,
            predict_time_ms=predict_time_ms,
        )

    def _predict_sync(
        self,
        horizon: int,
        confidence_level: float,
    ) -> PredictResult:
        """Synchronous prediction for Darts models."""
        start_time = time.perf_counter()

        # Generate predictions
        forecast = self._forecaster.predict(n=horizon)

        # Get values
        values = forecast.values().flatten()
        timestamps = forecast.time_index

        # Try to get confidence intervals if supported
        lower_vals = None
        upper_vals = None

        try:
            if hasattr(self._forecaster, "predict"):
                # Some Darts models support historical_forecasts for intervals
                pred_int = self._forecaster.predict(
                    n=horizon,
                    num_samples=100,
                )
                if pred_int.n_samples > 1:
                    alpha = 1 - confidence_level
                    percentiles = pred_int.quantile_timeseries(
                        [alpha / 2, 1 - alpha / 2]
                    )
                    lower_vals = percentiles.values()[:, 0]
                    upper_vals = percentiles.values()[:, 1]
        except Exception as e:
            logger.debug(f"Could not generate confidence intervals: {e}")

        predict_time_ms = (time.perf_counter() - start_time) * 1000

        predictions = []
        for i, (ts, val) in enumerate(zip(timestamps, values, strict=True)):
            pred = Prediction(
                timestamp=ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                value=float(val),
                lower=float(lower_vals[i]) if lower_vals is not None else None,
                upper=float(upper_vals[i]) if upper_vals is not None else None,
            )
            predictions.append(pred)

        return PredictResult(
            model_id=self.model_id,
            backend=self.backend,
            predictions=predictions,
            predict_time_ms=predict_time_ms,
        )

    async def save(self, path: str | Path) -> None:
        """Save the model to a specific path.

        Args:
            path: Path to save to (must be within TIMESERIES_MODELS_DIR)
        """
        if self.backend in ZERO_SHOT_BACKENDS:
            logger.info("Zero-shot models do not need to be saved")
            return

        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")

        path = Path(path)
        validated_path = validate_model_path(path, "timeseries")

        import joblib

        data = {
            "forecaster": self._forecaster,
            "training_series": self._training_series,
            "frequency": self._frequency,
            "backend": self.backend,
            "description": self._description,
            "saved_at": datetime.now().isoformat(),
        }

        joblib.dump(data, validated_path)
        logger.info(f"Saved timeseries model to: {validated_path}")

    async def unload(self) -> None:
        """Free model resources."""
        self._forecaster = None
        self._training_series = None
        self._chronos_pipeline = None
        self._is_fitted = False
        logger.info(f"Unloaded timeseries model: {self.model_id}")


def list_saved_models() -> list[dict]:
    """List all saved timeseries models.

    Returns:
        List of model info dicts
    """
    import json

    models = []
    if not TIMESERIES_MODELS_DIR.exists():
        return models

    for item in TIMESERIES_MODELS_DIR.iterdir():
        if item.is_file() and item.suffix == ".joblib":
            name = item.stem

            # Parse backend from filename
            backend = "unknown"
            for known in ALL_BACKENDS:
                if name.endswith(f"_{known}"):
                    backend = known
                    name = name[: -len(f"_{known}")]
                    break

            # Try to load metadata
            description = None
            metadata_path = TIMESERIES_MODELS_DIR / f"{item.stem}.metadata.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text())
                    description = metadata.get("description")
                except Exception:
                    pass

            models.append({
                "name": name,
                "filename": item.name,
                "backend": backend,
                "path": str(item),
                "size_bytes": item.stat().st_size,
                "created": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                "description": description,
            })

    # Sort by creation time, newest first
    models.sort(key=lambda x: x["created"], reverse=True)
    return models


def delete_model(model_name: str, backend: str | None = None) -> bool:
    """Delete a saved model.

    Args:
        model_name: Name of the model
        backend: Optional backend hint for filename matching

    Returns:
        True if deleted, False if not found
    """
    if not TIMESERIES_MODELS_DIR.exists():
        return False

    safe_name = sanitize_model_name(model_name)

    # Try exact filename match first
    for item in TIMESERIES_MODELS_DIR.iterdir():
        if (
            item.is_file()
            and item.suffix == ".joblib"
            and (item.stem == safe_name or item.stem.startswith(f"{safe_name}_"))
        ):
                # Validate path
                try:
                    validate_model_path(item, "timeseries")
                except ValueError:
                    continue

                # Delete model file
                item.unlink()
                logger.info(f"Deleted timeseries model: {item.name}")

                # Delete metadata if exists
                metadata_path = TIMESERIES_MODELS_DIR / f"{item.stem}.metadata.json"
                if metadata_path.exists():
                    metadata_path.unlink()

                return True

    return False
