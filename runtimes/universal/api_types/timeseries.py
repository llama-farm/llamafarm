"""Time-series forecasting types for fit, predict, and load endpoints.

Supports multiple backends:
- Classical (Darts): ARIMA, ExponentialSmoothing, Theta
- Foundation Model (Chronos): Zero-shot forecasting
"""

from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# Backend Type
# =============================================================================

# All supported time-series forecasting backends
TimeseriesBackendType = Literal[
    # Classical (require training)
    "arima",                  # Auto-ARIMA for stationary series
    "exponential_smoothing",  # Trend + seasonality decomposition
    "theta",                  # Simple and robust forecasting
    # Zero-shot (no training required)
    "chronos",                # Amazon Chronos T5-based foundation model
    "chronos-bolt",           # Faster Chronos variant
]


# =============================================================================
# Common Types
# =============================================================================


class TimeseriesDataPoint(BaseModel):
    """A single time-series data point."""

    timestamp: str = Field(..., description="ISO 8601 timestamp")
    value: float = Field(..., description="Numeric value at this timestamp")


class TimeseriesPrediction(BaseModel):
    """A single forecast prediction with optional confidence intervals."""

    timestamp: str = Field(..., description="Predicted timestamp")
    value: float = Field(..., description="Point forecast value")
    lower: float | None = Field(None, description="Lower confidence bound")
    upper: float | None = Field(None, description="Upper confidence bound")


class TimeseriesBackendInfo(BaseModel):
    """Information about a timeseries backend."""

    name: str = Field(..., description="Backend identifier")
    description: str = Field(..., description="Human-readable description")
    requires_training: bool = Field(
        ..., description="Whether this backend requires training data"
    )
    supports_confidence_intervals: bool = Field(
        ..., description="Whether this backend can produce confidence intervals"
    )
    speed: Literal["fast", "medium", "slow"] = Field(
        ..., description="Relative execution speed"
    )


# =============================================================================
# Request Types
# =============================================================================


class TimeseriesFitRequest(BaseModel):
    """Request to train a time-series forecaster.

    Training is only required for classical backends (arima, exponential_smoothing, theta).
    Zero-shot backends (chronos, chronos-bolt) don't need training.

    The model is auto-saved after training. If model name is not provided,
    a unique name is auto-generated (e.g., "timeseries-a1b2c3d4").
    """

    model: str | None = Field(
        None,
        description="Model name (auto-generated if not provided)",
    )
    backend: TimeseriesBackendType = Field(
        "arima",
        description="Forecasting algorithm to use",
    )
    data: list[TimeseriesDataPoint] = Field(
        ...,
        description="Training data as list of {timestamp, value} objects",
    )
    frequency: str | None = Field(
        None,
        description="Time frequency (D, H, M, etc.). Auto-detected if not provided.",
    )
    overwrite: bool = Field(
        True,
        description="If True, overwrite existing model. If False, version with timestamp.",
    )
    description: str | None = Field(
        None,
        description="Optional model description (saved to metadata)",
    )


class TimeseriesPredictRequest(BaseModel):
    """Request to generate forecasts.

    For classical backends, the model must be fitted first.
    For zero-shot backends (chronos, chronos-bolt), provide historical data directly.
    """

    model: str = Field(
        ...,
        description="Model name (supports '-latest' suffix to use most recent version)",
    )
    horizon: int = Field(
        ...,
        ge=1,
        le=365,
        description="Number of periods to forecast",
    )
    confidence_level: float = Field(
        0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for prediction intervals",
    )
    data: list[TimeseriesDataPoint] | None = Field(
        None,
        description="Historical data (required for zero-shot backends)",
    )


class TimeseriesLoadRequest(BaseModel):
    """Request to load a saved model.

    Supports the '-latest' suffix to load the most recent version
    of a model (e.g., 'my-model-latest').
    """

    model: str = Field(
        ...,
        description="Model name (supports '-latest' suffix)",
    )
    backend: str | None = Field(
        None,
        description="Backend hint for file matching",
    )


# =============================================================================
# Response Types
# =============================================================================


class TimeseriesFitResponse(BaseModel):
    """Response from training a forecaster."""

    model: str = Field(..., description="Model name (generated if not provided)")
    backend: str = Field(..., description="Backend used for training")
    saved_path: str = Field(..., description="Path where model was saved")
    training_time_ms: float = Field(..., description="Training time in milliseconds")
    samples_fitted: int = Field(..., description="Number of data points used")
    description: str | None = Field(None, description="Model description if provided")


class TimeseriesPredictResponse(BaseModel):
    """Response from generating forecasts."""

    model_id: str = Field(..., description="Model identifier")
    backend: str = Field(..., description="Backend used for prediction")
    predictions: list[TimeseriesPrediction] = Field(
        ..., description="List of forecast predictions"
    )
    fit_time_ms: float | None = Field(
        None, description="Training time if model was just fitted"
    )
    predict_time_ms: float = Field(..., description="Prediction time in milliseconds")


class TimeseriesBackendsResponse(BaseModel):
    """Response listing available backends."""

    backends: list[TimeseriesBackendInfo] = Field(
        ..., description="List of available backends"
    )


class TimeseriesModelInfo(BaseModel):
    """Information about a saved model."""

    name: str = Field(..., description="Model name")
    filename: str = Field(..., description="Filename on disk")
    backend: str = Field(..., description="Backend used")
    path: str = Field(..., description="Full path to model file")
    size_bytes: int = Field(..., description="File size in bytes")
    created: str = Field(..., description="Creation timestamp (ISO 8601)")
    description: str | None = Field(None, description="Model description")


class TimeseriesModelsResponse(BaseModel):
    """Response listing saved models."""

    models: list[TimeseriesModelInfo] = Field(..., description="List of saved models")
    total: int = Field(..., description="Total number of models")


class TimeseriesDeleteResponse(BaseModel):
    """Response from deleting a model."""

    deleted: bool = Field(..., description="Whether the model was deleted")
    model: str = Field(..., description="Model name that was requested")
    message: str = Field(..., description="Status message")
