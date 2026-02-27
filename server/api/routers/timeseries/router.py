"""
Timeseries Router - Endpoints for time-series forecasting.

Provides access to:
- Classical forecasting (ARIMA, ExponentialSmoothing, Theta)
- Zero-shot forecasting (Chronos, Chronos-Bolt)
- Model management (list, load, delete)

All endpoints proxy to the Universal Runtime for actual computation.
"""

import logging

from fastapi import APIRouter, HTTPException
from server.services.ml_model_service import MLModelService
from server.services.universal_runtime_service import UniversalRuntimeService

from .types import (
    TimeseriesBackendsResponse,
    TimeseriesDeleteResponse,
    TimeseriesFitRequest,
    TimeseriesFitResponse,
    TimeseriesLoadRequest,
    TimeseriesModelInfo,
    TimeseriesModelsResponse,
    TimeseriesPredictRequest,
    TimeseriesPredictResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/timeseries", tags=["timeseries"])


# =============================================================================
# Backend Information
# =============================================================================


@router.get("/backends")
async def list_backends() -> TimeseriesBackendsResponse:
    """List available time-series forecasting backends.

    Returns information about each backend including:
    - Whether it requires training (classical) or is zero-shot (Chronos)
    - Whether it supports confidence intervals
    - Relative speed

    Example response:
    ```json
    {
        "backends": [
            {
                "name": "arima",
                "description": "Auto-ARIMA for stationary time series",
                "requires_training": true,
                "supports_confidence_intervals": true,
                "speed": "medium"
            },
            {
                "name": "chronos",
                "description": "Amazon Chronos T5-based foundation model",
                "requires_training": false,
                "supports_confidence_intervals": true,
                "speed": "medium"
            }
        ]
    }
    ```
    """
    result = await UniversalRuntimeService.timeseries_list_backends()
    return TimeseriesBackendsResponse(**result)


# =============================================================================
# Model Training
# =============================================================================


@router.post("/fit")
async def fit_timeseries(request: TimeseriesFitRequest) -> TimeseriesFitResponse:
    """Train a time-series forecaster.

    The model is automatically saved after training. If a model name is not
    provided, one is auto-generated (e.g., "timeseries-a1b2c3d4").

    For zero-shot backends (chronos, chronos-bolt), this validates the data
    format but no actual training occurs.

    Args:
        model: Base name for the model (auto-generated if not provided)
        backend: Forecasting algorithm ("arima", "exponential_smoothing", etc.)
        data: Training data as list of {timestamp, value} objects
        frequency: Time frequency (D, H, M, etc.), auto-detected if not provided
        overwrite: If True (default), overwrite existing model
        description: Optional model description

    Example request:
    ```json
    {
        "model": "sales-forecast",
        "backend": "arima",
        "data": [
            {"timestamp": "2024-01-01", "value": 100},
            {"timestamp": "2024-01-02", "value": 120},
            {"timestamp": "2024-01-03", "value": 110}
        ],
        "frequency": "D",
        "description": "Daily sales forecaster"
    }
    ```

    Model is automatically saved after fitting - no separate save step needed.
    Use "{model}-latest" in predict/load to get the most recent version.
    """
    # Generate model name if not provided
    model_name = request.model
    if not model_name:
        model_name = MLModelService.generate_model_name("timeseries")

    # Get versioned model name
    versioned_name = MLModelService.get_versioned_name(model_name, request.overwrite)
    logger.info(f"Training timeseries model: {model_name} -> {versioned_name}")

    # Convert data to list of dicts if needed
    data = [
        {"timestamp": d.timestamp, "value": d.value}
        if hasattr(d, "timestamp")
        else d
        for d in request.data
    ]

    result = await UniversalRuntimeService.timeseries_fit(
        model=versioned_name,
        backend=request.backend,
        data=data,
        frequency=request.frequency,
        overwrite=request.overwrite,
        description=request.description,
    )

    return TimeseriesFitResponse(
        model=result.get("model", versioned_name),
        backend=result.get("backend", request.backend),
        saved_path=result.get("saved_path", ""),
        training_time_ms=result.get("training_time_ms", 0.0),
        samples_fitted=result.get("samples_fitted", len(data)),
        description=request.description,
    )


# =============================================================================
# Prediction
# =============================================================================


@router.post("/predict")
async def predict_timeseries(
    request: TimeseriesPredictRequest,
) -> TimeseriesPredictResponse:
    """Generate time-series forecasts.

    For classical backends, the model must be fitted first.
    For zero-shot backends (chronos, chronos-bolt), provide historical data directly.

    Args:
        model: Model name (supports '-latest' suffix)
        horizon: Number of periods to forecast
        confidence_level: Confidence level for prediction intervals (0.5-0.99)
        data: Historical data (required for zero-shot backends)

    Example request (fitted model):
    ```json
    {
        "model": "sales-forecast",
        "horizon": 30,
        "confidence_level": 0.95
    }
    ```

    Example request (zero-shot):
    ```json
    {
        "model": "temp-chronos",
        "horizon": 7,
        "data": [
            {"timestamp": "2024-01-01", "value": 100},
            {"timestamp": "2024-01-02", "value": 120}
        ]
    }
    ```
    """
    # Resolve -latest suffix
    model_name = MLModelService.resolve_model_name("timeseries", request.model)

    # Convert data if provided
    data = None
    if request.data:
        data = [
            {"timestamp": d.timestamp, "value": d.value}
            if hasattr(d, "timestamp")
            else d
            for d in request.data
        ]

    result = await UniversalRuntimeService.timeseries_predict(
        model=model_name,
        horizon=request.horizon,
        confidence_level=request.confidence_level,
        data=data,
    )

    return TimeseriesPredictResponse(**result)


# =============================================================================
# Model Loading
# =============================================================================


@router.post("/load")
async def load_timeseries(request: TimeseriesLoadRequest) -> TimeseriesFitResponse:
    """Load a previously saved timeseries model.

    Supports the '-latest' suffix to load the most recent version
    (e.g., 'my-model-latest').

    Example request:
    ```json
    {
        "model": "sales-forecast-latest"
    }
    ```
    """
    # Resolve -latest suffix
    model_name = MLModelService.resolve_model_name("timeseries", request.model)

    result = await UniversalRuntimeService.timeseries_load(
        model=model_name,
        backend=request.backend,
    )

    return TimeseriesFitResponse(
        model=result.get("model", model_name),
        backend=result.get("backend", "unknown"),
        saved_path=result.get("saved_path", ""),
        training_time_ms=0.0,
        samples_fitted=0,
        description=result.get("description"),
    )


# =============================================================================
# Model Management
# =============================================================================


@router.get("/models")
async def list_models() -> TimeseriesModelsResponse:
    """List all saved timeseries models.

    Returns all models in ~/.llamafarm/models/timeseries/ with metadata.

    Example response:
    ```json
    {
        "models": [
            {
                "name": "sales-forecast",
                "filename": "sales-forecast_arima.joblib",
                "backend": "arima",
                "path": "/home/user/.llamafarm/models/timeseries/sales-forecast_arima.joblib",
                "size_bytes": 12345,
                "created": "2024-01-15T10:30:00",
                "description": "Daily sales forecaster"
            }
        ],
        "total": 1
    }
    ```
    """
    models = MLModelService.list_all_models("timeseries")

    return TimeseriesModelsResponse(
        models=[
            TimeseriesModelInfo(
                name=m.get("name", ""),
                filename=m.get("filename", ""),
                backend=m.get("backend", "unknown"),
                path=m.get("path", ""),
                size_bytes=m.get("size_bytes", 0),
                created=m.get("created", ""),
                description=m.get("description"),
            )
            for m in models
        ],
        total=len(models),
    )


@router.get("/models/{model_name}")
async def get_model(model_name: str) -> TimeseriesModelInfo:
    """Get information about a specific model."""
    models = MLModelService.list_all_models("timeseries")

    for m in models:
        if m.get("name") == model_name or m.get("filename", "").startswith(model_name):
            return TimeseriesModelInfo(
                name=m.get("name", ""),
                filename=m.get("filename", ""),
                backend=m.get("backend", "unknown"),
                path=m.get("path", ""),
                size_bytes=m.get("size_bytes", 0),
                created=m.get("created", ""),
                description=m.get("description"),
            )

    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


@router.delete("/models/{model_name}")
async def delete_model(model_name: str) -> TimeseriesDeleteResponse:
    """Delete a saved timeseries model.

    Args:
        model_name: Name of the model to delete
    """
    deleted = MLModelService.delete_model("timeseries", model_name)

    return TimeseriesDeleteResponse(
        deleted=deleted,
        model=model_name,
        message=f"Model '{model_name}' {'deleted successfully' if deleted else 'not found'}",
    )
