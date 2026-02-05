"""Time-series forecasting router with auto-save functionality.

This router handles:
- /v1/timeseries/backends - List available forecasting backends
- /v1/timeseries/fit - Train forecaster (auto-saves after training)
- /v1/timeseries/predict - Generate forecasts
- /v1/timeseries/load - Load pre-trained model from disk
- /v1/timeseries/models - List saved models
- /v1/timeseries/models/{model} - Delete a saved model

Backends:
- Classical (Darts): arima, exponential_smoothing, theta
- Zero-shot (Chronos): chronos, chronos-bolt
"""

import uuid
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from api_types.timeseries import (
    TimeseriesBackendInfo,
    TimeseriesBackendsResponse,
    TimeseriesDeleteResponse,
    TimeseriesFitRequest,
    TimeseriesFitResponse,
    TimeseriesLoadRequest,
    TimeseriesModelInfo,
    TimeseriesModelsResponse,
    TimeseriesPrediction,
    TimeseriesPredictRequest,
    TimeseriesPredictResponse,
)
from core.logging import UniversalRuntimeLogger
from models.timeseries_model import (
    TimeseriesModel,
    get_backends_info,
    is_valid_backend,
    list_saved_models,
)
from models.timeseries_model import (
    delete_model as delete_timeseries_model,
)
from services.error_handler import handle_endpoint_errors
from services.path_validator import (
    TIMESERIES_MODELS_DIR,
    sanitize_model_name,
)

logger = UniversalRuntimeLogger("timeseries-router")

router = APIRouter(tags=["timeseries"])

# Dependency injection: model loader function
_load_timeseries_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None

# Dependency injection: state management
_models: dict | None = None
_model_load_lock = None

# Model storage directory
_TIMESERIES_MODELS_DIR = TIMESERIES_MODELS_DIR


def set_timeseries_loader(
    load_timeseries_fn: Callable[..., Coroutine[Any, Any, Any]] | None,
) -> None:
    """Set the timeseries model loader function.

    Args:
        load_timeseries_fn: Async function that loads a timeseries model
    """
    global _load_timeseries_fn
    _load_timeseries_fn = load_timeseries_fn


def get_timeseries_loader() -> Callable[..., Coroutine[Any, Any, Any]] | None:
    """Get the current timeseries loader function (for testing)."""
    return _load_timeseries_fn


def set_models_dir(models_dir: Path) -> None:
    """Set the models directory for saving/loading models."""
    global _TIMESERIES_MODELS_DIR
    _TIMESERIES_MODELS_DIR = models_dir


def set_state(models: dict, model_load_lock) -> None:
    """Set shared state from the main server.

    Args:
        models: Model cache dictionary
        model_load_lock: Async lock for model loading
    """
    global _models, _model_load_lock
    _models = models
    _model_load_lock = model_load_lock


async def _get_timeseries_model(
    model_id: str,
    backend: str = "arima",
    **kwargs,
) -> TimeseriesModel:
    """Get or load a timeseries model."""
    if _load_timeseries_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Timeseries model loader not initialized. Server configuration error.",
        )
    return await _load_timeseries_fn(model_id=model_id, backend=backend, **kwargs)


def _make_cache_key(model_id: str, backend: str) -> str:
    """Generate a cache key for a timeseries model."""
    return f"timeseries:{backend}:{model_id}"


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/v1/timeseries/backends")
@handle_endpoint_errors("timeseries-backends")
async def list_backends() -> TimeseriesBackendsResponse:
    """List available timeseries forecasting backends.

    Returns information about each backend including:
    - Whether it requires training
    - Whether it supports confidence intervals
    - Relative speed
    """
    backends_info = get_backends_info()
    return TimeseriesBackendsResponse(
        backends=[
            TimeseriesBackendInfo(
                name=b.name,
                description=b.description,
                requires_training=b.requires_training,
                supports_confidence_intervals=b.supports_confidence_intervals,
                speed=b.speed,
            )
            for b in backends_info
        ]
    )


@router.post("/v1/timeseries/fit")
@handle_endpoint_errors("timeseries-fit")
async def fit_timeseries(request: TimeseriesFitRequest) -> TimeseriesFitResponse:
    """Train a time-series forecaster.

    The model is automatically saved after training. If a model name is not
    provided, one is auto-generated (e.g., "timeseries-a1b2c3d4").

    For zero-shot backends (chronos, chronos-bolt), this validates the data
    format but no actual training occurs.
    """
    backend = request.backend
    if not is_valid_backend(backend):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: {backend}. Use GET /v1/timeseries/backends to see available options.",
        )

    # Convert data to list of dicts if needed
    data = [
        {"timestamp": d.timestamp, "value": d.value}
        if hasattr(d, "timestamp")
        else d
        for d in request.data
    ]

    # Create and load model - use unique ID to avoid cache collisions
    model_id = request.model or f"timeseries-{uuid.uuid4().hex[:8]}"
    model = await _get_timeseries_model(model_id=model_id, backend=backend)

    # Fit the model
    result = await model.fit(
        data=data,
        frequency=request.frequency,
        model_name=request.model,
        overwrite=request.overwrite,
        description=request.description,
    )

    return TimeseriesFitResponse(
        model=result.model,
        backend=result.backend,
        saved_path=result.saved_path,
        training_time_ms=result.training_time_ms,
        samples_fitted=result.samples_fitted,
        description=result.description,
    )


@router.post("/v1/timeseries/predict")
@handle_endpoint_errors("timeseries-predict")
async def predict_timeseries(
    request: TimeseriesPredictRequest,
) -> TimeseriesPredictResponse:
    """Generate time-series forecasts.

    For classical backends, the model must be fitted first.
    For zero-shot backends (chronos, chronos-bolt), provide historical data.

    Supports the '-latest' suffix in model name to use the most recent version.
    """
    backend = None

    # Try to infer backend from model files if not a zero-shot request
    if request.data is None:
        # Must be a fitted model - try to load it
        saved_models = list_saved_models()
        for m in saved_models:
            if m["name"] == request.model or m["filename"].startswith(request.model):
                backend = m["backend"]
                break

    if backend is None and request.data is not None:
        # Zero-shot with data - default to chronos
        backend = "chronos"

    if backend is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found. Use GET /v1/timeseries/models to list saved models.",
        )

    # Load the model
    model = await _get_timeseries_model(model_id=request.model, backend=backend)

    # Convert data if provided
    data = None
    if request.data:
        data = [
            {"timestamp": d.timestamp, "value": d.value}
            if hasattr(d, "timestamp")
            else d
            for d in request.data
        ]

    # Generate predictions
    result = await model.predict(
        horizon=request.horizon,
        data=data,
        confidence_level=request.confidence_level,
    )

    return TimeseriesPredictResponse(
        model_id=result.model_id,
        backend=result.backend,
        predictions=[
            TimeseriesPrediction(
                timestamp=p.timestamp,
                value=p.value,
                lower=p.lower,
                upper=p.upper,
            )
            for p in result.predictions
        ],
        fit_time_ms=result.fit_time_ms,
        predict_time_ms=result.predict_time_ms,
    )


@router.post("/v1/timeseries/load")
@handle_endpoint_errors("timeseries-load")
async def load_timeseries(request: TimeseriesLoadRequest) -> TimeseriesFitResponse:
    """Load a previously saved timeseries model.

    Supports the '-latest' suffix to load the most recent version
    (e.g., 'my-model-latest').
    """
    # Find the model file
    saved_models = list_saved_models()

    # Handle -latest suffix
    model_name = request.model
    if model_name.endswith("-latest"):
        base_name = model_name[:-7]
        # Find the latest version
        matching = [m for m in saved_models if m["name"].startswith(base_name)]
        if matching:
            # Already sorted by creation time (newest first)
            model_name = matching[0]["name"]
            backend = matching[0]["backend"]
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No models found matching '{base_name}'",
            )
    else:
        # Find exact match
        matching = [m for m in saved_models if m["name"] == model_name]
        if not matching:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found",
            )
        backend = matching[0]["backend"]

    # Load the model to verify it exists
    model_path = _TIMESERIES_MODELS_DIR / matching[0]["filename"]
    await _get_timeseries_model(
        model_id=str(model_path),
        backend=backend,
    )

    return TimeseriesFitResponse(
        model=model_name,
        backend=backend,
        saved_path=str(model_path),
        training_time_ms=0.0,
        samples_fitted=0,
        description=matching[0].get("description"),
    )


@router.get("/v1/timeseries/models")
@handle_endpoint_errors("timeseries-list-models")
async def list_models() -> TimeseriesModelsResponse:
    """List all saved timeseries models."""
    models = list_saved_models()

    return TimeseriesModelsResponse(
        models=[
            TimeseriesModelInfo(
                name=m["name"],
                filename=m["filename"],
                backend=m["backend"],
                path=m["path"],
                size_bytes=m["size_bytes"],
                created=m["created"],
                description=m.get("description"),
            )
            for m in models
        ],
        total=len(models),
    )


@router.get("/v1/timeseries/models/{model_name}")
@handle_endpoint_errors("timeseries-get-model")
async def get_model(model_name: str) -> TimeseriesModelInfo:
    """Get information about a specific model."""
    models = list_saved_models()

    for m in models:
        if m["name"] == model_name or m["filename"].startswith(model_name):
            return TimeseriesModelInfo(
                name=m["name"],
                filename=m["filename"],
                backend=m["backend"],
                path=m["path"],
                size_bytes=m["size_bytes"],
                created=m["created"],
                description=m.get("description"),
            )

    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_name}' not found",
    )


@router.delete("/v1/timeseries/models/{model_name}")
@handle_endpoint_errors("timeseries-delete-model")
async def delete_model(model_name: str) -> TimeseriesDeleteResponse:
    """Delete a saved timeseries model."""
    # Sanitize the model name
    safe_name = sanitize_model_name(model_name)

    if delete_timeseries_model(safe_name):
        return TimeseriesDeleteResponse(
            deleted=True,
            model=model_name,
            message=f"Model '{model_name}' deleted successfully",
        )
    else:
        return TimeseriesDeleteResponse(
            deleted=False,
            model=model_name,
            message=f"Model '{model_name}' not found",
        )
