"""Anomaly detection router with auto-save functionality.

This router handles:
- /v1/anomaly/score - Score data for anomalies
- /v1/anomaly/fit - Fit anomaly detector (auto-saves after training)
- /v1/anomaly/detect - Detect anomalies (returns only anomalous points)
- /v1/anomaly/load - Load pre-trained model from disk
- /v1/anomaly/models - List saved models
- /v1/anomaly/models/{filename} - Delete a saved model

Note: The /v1/anomaly/save endpoint has been removed. Models are automatically
saved after fit to ensure data persistence without requiring an explicit save call.
"""

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from api_types.anomaly import (
    AnomalyFitRequest,
    AnomalyLoadRequest,
    AnomalyScoreRequest,
)
from core.logging import UniversalRuntimeLogger
from services.error_handler import handle_endpoint_errors
from services.path_validator import (
    ANOMALY_MODELS_DIR,
    PathValidationError,
    sanitize_filename,
    sanitize_model_name,
    validate_path_within_directory,
)
from utils.feature_encoder import FeatureEncoder

logger = UniversalRuntimeLogger("anomaly-router")

router = APIRouter(tags=["anomaly"])

# Dependency injection: model loader function
# Set via set_anomaly_loader() from the main server
_load_anomaly_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None

# Dependency injection: state management
# These are injected from the main server to share state
_models: dict | None = None
_encoders: dict[str, FeatureEncoder] | None = None
_model_load_lock = None

# Model storage directory - uses shared path_validator config
_ANOMALY_MODELS_DIR = ANOMALY_MODELS_DIR


def set_anomaly_loader(
    load_anomaly_fn: Callable[..., Coroutine[Any, Any, Any]] | None,
) -> None:
    """Set the anomaly model loader function.

    This should be called during app initialization to inject the model loading
    dependency from the main server.

    Args:
        load_anomaly_fn: Async function that loads an anomaly model
    """
    global _load_anomaly_fn
    _load_anomaly_fn = load_anomaly_fn


def get_anomaly_loader() -> Callable[..., Coroutine[Any, Any, Any]] | None:
    """Get the current anomaly loader function (for testing purposes)."""
    return _load_anomaly_fn


def set_models_dir(models_dir: Path) -> None:
    """Set the models directory for saving/loading models.

    Args:
        models_dir: Path to the anomaly models directory
    """
    global _ANOMALY_MODELS_DIR
    _ANOMALY_MODELS_DIR = models_dir


def set_state(
    models: dict,
    encoders: dict[str, FeatureEncoder],
    model_load_lock,
) -> None:
    """Set shared state from the main server.

    Args:
        models: Model cache dictionary
        encoders: Feature encoder cache dictionary
        model_load_lock: Async lock for model loading
    """
    global _models, _encoders, _model_load_lock
    _models = models
    _encoders = encoders
    _model_load_lock = model_load_lock


async def _get_anomaly_model(
    model_id: str,
    backend: str = "isolation_forest",
    **kwargs,
) -> Any:
    """Get or load an anomaly model."""
    if _load_anomaly_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Anomaly model loader not initialized. Server configuration error.",
        )
    return await _load_anomaly_fn(model_id=model_id, backend=backend, **kwargs)


def _make_cache_key(
    model_id: str, backend: str, normalization: str | None = None
) -> str:
    """Generate a cache key for an anomaly model."""
    if normalization:
        return f"anomaly:{backend}:{normalization}:{model_id}"
    return f"anomaly:{backend}:{model_id}"


def _get_model_path(model_name: str, backend: str) -> Path:
    """Get the path for a model file based on name and backend.

    The path is always within _ANOMALY_MODELS_DIR - users cannot control it.
    """
    safe_name = sanitize_model_name(model_name)
    safe_backend = sanitize_model_name(backend)
    filename = f"{safe_name}_{safe_backend}"
    return _ANOMALY_MODELS_DIR / filename


def _prepare_data(
    data: list[list[float]] | list[dict],
    schema: dict[str, str] | None,
    cache_key: str,
    fit_mode: bool = False,
) -> list[list[float]]:
    """Prepare data for anomaly detection by encoding if needed.

    Args:
        data: Raw data (numeric arrays or dicts)
        schema: Feature encoding schema (required for dict data during fit)
        cache_key: Cache key for storing/retrieving encoder
        fit_mode: If True, fit the encoder on the data. If False, use existing encoder.

    Returns:
        Encoded numeric data as list of lists
    """
    if _encoders is None:
        raise HTTPException(
            status_code=500,
            detail="Encoder state not initialized. Server configuration error.",
        )

    # If data is already numeric, return as-is
    if not data:
        return []

    if isinstance(data[0], list):
        return data

    # Dict-based data - need to encode
    if fit_mode:
        if schema is None:
            raise HTTPException(
                status_code=400,
                detail="Schema is required when fitting with dict-based data. "
                "Example: schema = {'time_ms': 'numeric', 'user_agent': 'hash'}",
            )
        encoder = FeatureEncoder()
        encoder.fit(data, schema)
        _encoders[cache_key] = encoder
        logger.info(f"Fitted feature encoder for {cache_key} with schema: {schema}")
    else:
        if cache_key not in _encoders:
            raise HTTPException(
                status_code=400,
                detail=f"No encoder found for model '{cache_key}'. "
                "Train with /v1/anomaly/fit using dict data first, or pass schema.",
            )
        encoder = _encoders[cache_key]

    encoded = encoder.transform(data)
    return encoded.tolist()


async def _auto_save_model(
    model: Any,
    model_name: str,
    backend: str,
    cache_key: str,
) -> str | None:
    """Auto-save anomaly model after fit to prevent data loss.

    Models are saved immediately after training to ensure they persist
    across server restarts without requiring an explicit /save call.

    Returns:
        Path to the saved model file, or None if save failed
    """
    _ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    save_path = _get_model_path(model_name, backend)
    await model.save(str(save_path))

    # Determine actual saved file path
    if backend == "autoencoder":
        actual_path = save_path.with_suffix(".pt")
    else:
        actual_path = save_path.with_suffix(".joblib")
        if not actual_path.exists():
            actual_path = save_path.with_suffix(".pkl")

    logger.debug(f"Model auto-saved to {actual_path}")

    # Save encoder if one exists for this model
    if _encoders and cache_key in _encoders:
        encoder = _encoders[cache_key]
        encoder_save_path = save_path.parent / f"{save_path.name}_encoder.json"
        encoder.save(encoder_save_path)
        logger.debug(f"Feature encoder saved to {encoder_save_path}")

    return str(actual_path)


@router.post("/v1/anomaly/score")
@handle_endpoint_errors("score_anomalies")
async def score_anomalies(request: AnomalyScoreRequest):
    """Score data points for anomalies.

    Detects anomalies in data using various algorithms:
    - isolation_forest: Fast tree-based method, good general purpose
    - one_class_svm: Support vector machine for outlier detection
    - local_outlier_factor: Density-based, good for clustering anomalies
    - autoencoder: Neural network, best for complex patterns

    Note: Model must be fitted first via /v1/anomaly/fit or loaded from disk.
    """
    cache_key = _make_cache_key(
        request.model, request.backend, request.normalization
    )

    model = await _get_anomaly_model(
        model_id=request.model,
        backend=request.backend,
        normalization=request.normalization,
    )

    if not model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Model not fitted. Call /v1/anomaly/fit first or load a pre-trained model.",
        )

    prepared_data = _prepare_data(
        data=request.data,
        schema=request.schema_,
        cache_key=cache_key,
        fit_mode=False,
    )

    results = await model.score(
        data=prepared_data,
        threshold=request.threshold,
    )

    data = [
        {
            "index": r.index,
            "score": r.score,
            "is_anomaly": r.is_anomaly,
            "raw_score": r.raw_score,
        }
        for r in results
    ]

    anomaly_count = sum(1 for r in results if r.is_anomaly)

    return {
        "object": "list",
        "data": data,
        "total_count": len(data),
        "model": request.model,
        "backend": request.backend,
        "summary": {
            "total_points": len(data),
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_count / len(data) if data else 0,
            "threshold": request.threshold or model.threshold,
        },
    }


@router.post("/v1/anomaly/fit")
@handle_endpoint_errors("fit_anomaly_detector")
async def fit_anomaly_detector(request: AnomalyFitRequest):
    """Fit an anomaly detector on training data.

    Train an anomaly detection model on data assumed to be mostly normal.
    The model learns what "normal" looks like and can then detect deviations.

    **Auto-Save**: Models are automatically saved after successful training
    to ensure persistence across server restarts. No explicit save call needed.

    **Overwrite Default**: By default, existing models with the same name
    are overwritten. Set overwrite=False to create versioned models.

    Backends:
    - isolation_forest: Fast, works well out of the box (recommended)
    - one_class_svm: Good for small datasets
    - local_outlier_factor: Density-based, good for clustering anomalies
    - autoencoder: Best for complex patterns, requires more data
    """
    cache_key = _make_cache_key(
        request.model, request.backend, request.normalization
    )

    prepared_data = _prepare_data(
        data=request.data,
        schema=request.schema_,
        cache_key=cache_key,
        fit_mode=True,
    )

    model = await _get_anomaly_model(
        model_id=request.model,
        backend=request.backend,
        contamination=request.contamination,
        normalization=request.normalization,
    )

    result = await model.fit(
        data=prepared_data,
        epochs=request.epochs,
        batch_size=request.batch_size,
    )

    # Include encoder info in response if used
    encoder_info = None
    if _encoders and cache_key in _encoders:
        encoder = _encoders[cache_key]
        encoder_info = {
            "schema": encoder.schema.features if encoder.schema else {},
            "features": list(encoder.schema.features.keys())
            if encoder.schema
            else [],
        }

    # Auto-save model to prevent data loss on restart
    saved_path = await _auto_save_model(
        model=model,
        model_name=request.model,
        backend=request.backend,
        cache_key=cache_key,
    )

    return {
        "object": "fit_result",
        "model": request.model,
        "backend": request.backend,
        "samples_fitted": result.samples_fitted,
        "training_time_ms": result.training_time_ms,
        "model_params": result.model_params,
        "encoder": encoder_info,
        "saved_path": saved_path,
        "status": "fitted",
    }


@router.post("/v1/anomaly/detect")
@handle_endpoint_errors("detect_anomalies")
async def detect_anomalies(request: AnomalyScoreRequest):
    """Detect anomalies in data (returns only anomalous points).

    Same as /v1/anomaly/score but filters to return only points
    classified as anomalies.
    """
    cache_key = _make_cache_key(
        request.model, request.backend, request.normalization
    )

    model = await _get_anomaly_model(
        model_id=request.model,
        backend=request.backend,
        normalization=request.normalization,
    )

    if not model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Model not fitted. Call /v1/anomaly/fit first.",
        )

    prepared_data = _prepare_data(
        data=request.data,
        schema=request.schema_,
        cache_key=cache_key,
        fit_mode=False,
    )

    results = await model.detect(
        data=prepared_data,
        threshold=request.threshold,
    )

    data = [
        {
            "index": r.index,
            "score": r.score,
            "raw_score": r.raw_score,
        }
        for r in results
    ]

    return {
        "object": "list",
        "data": data,
        "total_count": len(data),
        "model": request.model,
        "backend": request.backend,
        "summary": {
            "anomalies_detected": len(data),
            "threshold": request.threshold or model.threshold,
        },
    }


# Note: /v1/anomaly/save endpoint removed - auto-save handles persistence


@router.post("/v1/anomaly/load")
@handle_endpoint_errors("load_anomaly_model")
async def load_anomaly_model(request: AnomalyLoadRequest):
    """Load a pre-trained anomaly model from disk.

    Load a previously saved model for production inference without
    re-training. The model path is automatically determined from the
    model name and backend - no user control over file paths.
    """
    if _models is None or _model_load_lock is None:
        raise HTTPException(
            status_code=500,
            detail="Model state not initialized. Server configuration error.",
        )

    base_path = _get_model_path(request.model, request.backend)

    # Find the model file
    model_path = None
    for ext in [".joblib", ".pkl", ".pt"]:
        candidate = base_path.with_suffix(ext)
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        available = (
            [f.name for f in _ANOMALY_MODELS_DIR.glob("*") if f.is_file()]
            if _ANOMALY_MODELS_DIR.exists()
            else []
        )
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' with backend '{request.backend}' not found. "
            f"Available models: {available}",
        )

    # Load via the injected loader
    async with _model_load_lock:
        logger.info(f"Loading pre-trained anomaly model: {model_path}")

        # Import here to avoid circular imports
        from models import AnomalyModel
        from utils.device import get_optimal_device

        device = get_optimal_device()

        model = AnomalyModel(
            model_id=str(model_path),
            device=device,
            backend=request.backend,
        )

        await model.load()

        cache_key = _make_cache_key(
            request.model, request.backend, model.normalization
        )

        if cache_key in _models:
            await _models[cache_key].unload()
            del _models[cache_key]

        _models[cache_key] = model

    # Try to load encoder if one exists
    encoder_loaded = False
    encoder_schema = None
    encoder_path = base_path.parent / f"{base_path.name}_encoder.json"
    if encoder_path.exists() and _encoders is not None:
        encoder = FeatureEncoder.load(encoder_path)
        _encoders[cache_key] = encoder
        encoder_loaded = True
        encoder_schema = encoder.schema
        logger.info(f"Loaded feature encoder from {encoder_path}")

    return {
        "object": "load_result",
        "model": request.model,
        "backend": request.backend,
        "normalization": model.normalization,
        "filename": model_path.name,
        "is_fitted": model.is_fitted,
        "threshold": model.threshold,
        "encoder_loaded": encoder_loaded,
        "encoder_schema": encoder_schema,
        "status": "loaded",
    }


@router.get("/v1/anomaly/models")
@handle_endpoint_errors("list_anomaly_models")
async def list_anomaly_models():
    """List all saved anomaly models available for loading.

    Returns models saved in the anomaly models directory.
    """
    _ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = []
    for path in _ANOMALY_MODELS_DIR.glob("*"):
        if path.is_file() and path.suffix in (".pt", ".pkl", ".joblib"):
            stat = path.stat()
            backend = "autoencoder" if path.suffix == ".pt" else "sklearn"

            models.append(
                {
                    "filename": path.name,
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                    "backend": backend,
                }
            )

    models.sort(key=lambda x: x["modified"], reverse=True)

    return {
        "object": "list",
        "data": models,
        "models_dir": str(_ANOMALY_MODELS_DIR),
        "total": len(models),
    }


@router.delete("/v1/anomaly/models/{filename}")
@handle_endpoint_errors("delete_anomaly_model")
async def delete_anomaly_model(filename: str):
    """Delete a saved anomaly model.

    Removes the model file from disk. Does not affect cached models.
    """
    safe_filename = sanitize_filename(filename)
    if not safe_filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename",
        )

    if (
        "/" in filename
        or "\\" in filename
        or ".." in filename
        or safe_filename == "."
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename: path separators not allowed",
        )

    model_path = _ANOMALY_MODELS_DIR / safe_filename

    try:
        resolved_path = validate_path_within_directory(
            model_path, _ANOMALY_MODELS_DIR
        )
    except PathValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not resolved_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {safe_filename}",
        )

    resolved_path.unlink()

    return {
        "object": "delete_result",
        "filename": safe_filename,
        "deleted": True,
    }
