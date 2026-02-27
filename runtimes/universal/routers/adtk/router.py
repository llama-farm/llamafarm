"""ADTK (Anomaly Detection Toolkit) router for time-series anomaly detection.

Provides endpoints for detecting temporal anomalies:
- Level shifts (sudden baseline changes)
- Seasonal anomalies (pattern violations)
- Spikes/dips (IQR-based outliers)
- Volatility shifts (variance changes)
- Persist anomalies (stuck values)
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter

from api_types.adtk import (
    ADTKAnomaly,
    ADTKDeleteResponse,
    ADTKDetectorInfo,
    ADTKDetectorsResponse,
    ADTKDetectRequest,
    ADTKDetectResponse,
    ADTKFitRequest,
    ADTKFitResponse,
    ADTKLoadRequest,
    ADTKLoadResponse,
    ADTKModelInfo,
    ADTKModelsResponse,
)
from models.adtk_model import (
    ADTKModel,
    delete_model,
    get_detectors_info,
    list_saved_models,
)
from services.error_handler import handle_endpoint_errors
from services.path_validator import ADTK_MODELS_DIR, generate_model_name
from utils.model_cache import ModelCache

logger = logging.getLogger(__name__)

router = APIRouter()

# Type for model loader function
ADTKLoaderFunc = Callable[[str, str, dict[str, Any]], Awaitable[ADTKModel]]

# Module-level state (set by server.py)
_adtk_cache: ModelCache["ADTKModel"] | None = None
_model_load_lock: asyncio.Lock | None = None
_loader_func: ADTKLoaderFunc | None = None


def set_adtk_loader(loader: ADTKLoaderFunc) -> None:
    """Set the function to load ADTK models."""
    global _loader_func
    _loader_func = loader


def set_adtk_state(
    cache: ModelCache["ADTKModel"],
    lock: asyncio.Lock,
) -> None:
    """Set shared state from server.py."""
    global _adtk_cache, _model_load_lock
    _adtk_cache = cache
    _model_load_lock = lock


async def get_or_load_model(
    model_id: str,
    detector: str,
    params: dict[str, Any] | None = None,
) -> ADTKModel:
    """Get model from cache or load it.

    Args:
        model_id: Model identifier
        detector: Detector type
        params: Optional detector parameters

    Returns:
        ADTKModel instance

    Note:
        The loader function (load_adtk in server.py) handles its own locking,
        so we don't need to acquire the lock here. Doing so would cause a deadlock
        since asyncio locks are not reentrant.
    """
    if _loader_func is None:
        raise RuntimeError("ADTK router not initialized. Call set_adtk_loader first.")

    # The loader function handles caching and locking internally
    return await _loader_func(model_id, detector, params or {})


def _find_model_file(model_name: str, default_detector: str = "level_shift") -> tuple[Any, str]:
    """Find model file and extract detector type.

    Handles -latest suffix to find most recent version.

    Args:
        model_name: Model name, optionally with -latest suffix
        default_detector: Default detector type if not parseable from filename

    Returns:
        Tuple of (model_path, detector_type)

    Raises:
        FileNotFoundError: If no matching model file found
    """
    if model_name.endswith("-latest"):
        base_name = model_name[:-7]
        model_files = list(ADTK_MODELS_DIR.glob(f"{base_name}_*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model found matching: {base_name}")
        model_file = max(model_files, key=lambda p: p.stat().st_mtime)
    else:
        model_files = list(ADTK_MODELS_DIR.glob(f"{model_name}_*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model found: {model_name}")
        model_file = model_files[0]

    # Parse detector from filename
    parts = model_file.stem.rsplit("_", 1)
    detector = parts[1] if len(parts) == 2 else default_detector

    return model_file, detector


@router.get("/v1/adtk/detectors")
@handle_endpoint_errors("adtk-detectors")
async def list_detectors() -> ADTKDetectorsResponse:
    """List available ADTK detector types."""
    detectors = get_detectors_info()
    return ADTKDetectorsResponse(
        detectors=[
            ADTKDetectorInfo(
                name=d["name"],
                description=d["description"],
                requires_training=d["requires_training"],
                default_params=d["default_params"],
            )
            for d in detectors
        ]
    )


@router.post("/v1/adtk/fit")
@handle_endpoint_errors("adtk-fit")
async def fit_adtk(request: ADTKFitRequest) -> ADTKFitResponse:
    """Fit an ADTK detector on time-series data.

    Auto-saves after fitting and returns the saved path.
    """
    # Generate model name if not provided
    model_name = request.model or generate_model_name("adtk")

    # Get or create model
    model = await get_or_load_model(
        model_id=model_name,
        detector=request.detector,
        params=request.params,
    )

    # Convert data to list of dicts
    data = [
        d if isinstance(d, dict) else d.model_dump()
        for d in request.data
    ]

    # Fit the model
    result = await model.fit(
        data=data,
        autosave=True,
        overwrite=request.overwrite,
        description=request.description,
    )

    return ADTKFitResponse(
        model=model_name,
        detector=request.detector,
        saved_path=result.get("saved_path", ""),
        training_time_ms=result["training_time_ms"],
        samples_fitted=result["samples_fitted"],
        requires_training=result["requires_training"],
    )


@router.post("/v1/adtk/detect")
@handle_endpoint_errors("adtk-detect")
async def detect_anomalies(request: ADTKDetectRequest) -> ADTKDetectResponse:
    """Detect anomalies in time-series data.

    Can use a saved model or default detector settings.
    """
    # Use existing model or create ad-hoc detector
    model_name = request.model or f"adhoc-{request.detector}"

    model = await get_or_load_model(
        model_id=model_name,
        detector=request.detector,
        params=request.params,
    )

    # Convert data to list of dicts
    data = [
        d if isinstance(d, dict) else d.model_dump()
        for d in request.data
    ]

    # Detect anomalies
    result = await model.detect(data)

    return ADTKDetectResponse(
        model=request.model,
        detector=result.detector,
        anomalies=[
            ADTKAnomaly(
                timestamp=a.timestamp,
                value=a.value,
                anomaly_type=a.anomaly_type,
                score=a.score,
            )
            for a in result.anomalies
        ],
        total_points=result.total_points,
        anomaly_count=result.anomaly_count,
        detection_time_ms=result.detection_time_ms,
    )


@router.post("/v1/adtk/load")
@handle_endpoint_errors("adtk-load")
async def load_model(request: ADTKLoadRequest) -> ADTKLoadResponse:
    """Load a saved ADTK model."""
    model_name = request.model

    # Find model file and detector type
    model_file, detector = _find_model_file(model_name)

    # Load the model
    model = ADTKModel(model_id=model_name, detector=detector)
    await model.load_from_path(model_file)

    # Cache it
    if _adtk_cache is not None:
        cache_key = f"{model_name}_{detector}"
        _adtk_cache[cache_key] = model

    return ADTKLoadResponse(
        model=model_name,
        detector=detector,
        is_fitted=model.is_fitted,
    )


@router.get("/v1/adtk/models")
@handle_endpoint_errors("adtk-models")
async def list_models() -> ADTKModelsResponse:
    """List all saved ADTK models."""
    models = list_saved_models()
    return ADTKModelsResponse(
        models=[
            ADTKModelInfo(
                name=m.name,
                detector=m.detector,
                created_at=m.created_at,
                description=m.description,
                is_fitted=m.is_fitted,
            )
            for m in models
        ]
    )


@router.delete("/v1/adtk/models/{model_name}")
@handle_endpoint_errors("adtk-delete")
async def delete_adtk_model(model_name: str) -> ADTKDeleteResponse:
    """Delete a saved ADTK model."""
    deleted = delete_model(model_name)
    return ADTKDeleteResponse(
        model=model_name,
        deleted=deleted,
    )
