"""Drift detection router using Alibi Detect.

Provides endpoints for data drift monitoring:
- KS test: Univariate numeric drift
- MMD: Multivariate drift
- Chi-squared: Categorical drift
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from api_types.drift import (
    DriftDeleteResponse,
    DriftDetectorInfo,
    DriftDetectorsResponse,
    DriftDetectRequest,
    DriftDetectResponse,
    DriftFitRequest,
    DriftFitResponse,
    DriftLoadRequest,
    DriftLoadResponse,
    DriftModelInfo,
    DriftModelsResponse,
    DriftResetResponse,
    DriftResult,
    DriftStatus,
)
from models.drift_model import (
    DriftModel,
    delete_model,
    get_detectors_info,
    list_saved_models,
)
from services.error_handler import handle_endpoint_errors
from services.path_validator import DRIFT_MODELS_DIR, generate_model_name
from utils.model_cache import ModelCache

logger = logging.getLogger(__name__)

router = APIRouter()

# Type for model loader function
DriftLoaderFunc = Callable[[str, str, dict[str, Any]], Awaitable[DriftModel]]

# Module-level state (set by server.py)
_drift_cache: ModelCache["DriftModel"] | None = None
_model_load_lock: asyncio.Lock | None = None
_loader_func: DriftLoaderFunc | None = None


def set_drift_loader(loader: DriftLoaderFunc) -> None:
    """Set the function to load drift models."""
    global _loader_func
    _loader_func = loader


def set_drift_state(
    cache: ModelCache["DriftModel"],
    lock: asyncio.Lock,
) -> None:
    """Set shared state from server.py."""
    global _drift_cache, _model_load_lock
    _drift_cache = cache
    _model_load_lock = lock


async def get_or_load_model(
    model_id: str,
    detector: str,
    params: dict[str, Any] | None = None,
) -> DriftModel:
    """Get model from cache or load it.

    Args:
        model_id: Model identifier
        detector: Detector type
        params: Optional detector parameters

    Returns:
        DriftModel instance

    Note:
        The loader function (load_drift in server.py) handles its own locking,
        so we don't need to acquire the lock here. Doing so would cause a deadlock
        since asyncio locks are not reentrant.
    """
    if _loader_func is None:
        raise RuntimeError("Drift router not initialized. Call set_drift_loader first.")

    # The loader function handles caching and locking internally
    return await _loader_func(model_id, detector, params or {})


def _find_model_file(model_name: str, default_detector: str = "ks") -> tuple[Path, str]:
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
    # Prevent path traversal / glob injection
    safe_name = Path(model_name).name
    if safe_name != model_name or ".." in model_name:
        raise ValueError(f"Invalid model_name: {model_name}")

    if model_name.endswith("-latest"):
        base_name = model_name[:-7]
        model_files = list(DRIFT_MODELS_DIR.glob(f"{base_name}_*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model found matching: {base_name}")
        model_file = max(model_files, key=lambda p: p.stat().st_mtime)
    else:
        model_files = list(DRIFT_MODELS_DIR.glob(f"{model_name}_*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model found: {model_name}")
        model_file = model_files[0]

    # Parse detector from filename
    parts = model_file.stem.rsplit("_", 1)
    detector = parts[1] if len(parts) == 2 else default_detector

    return model_file, detector


@router.get("/v1/drift/detectors")
@handle_endpoint_errors("drift-detectors")
async def list_detectors() -> DriftDetectorsResponse:
    """List available drift detector types."""
    detectors = get_detectors_info()
    return DriftDetectorsResponse(
        detectors=[
            DriftDetectorInfo(
                name=d["name"],
                description=d["description"],
                multivariate=d["multivariate"],
                default_params=d["default_params"],
            )
            for d in detectors
        ]
    )


@router.post("/v1/drift/fit")
@handle_endpoint_errors("drift-fit")
async def fit_drift(request: DriftFitRequest) -> DriftFitResponse:
    """Fit a drift detector on reference data.

    Auto-saves after fitting and returns the saved path.
    """
    # Generate model name if not provided
    model_name = request.model or generate_model_name("drift")

    # Get or create model
    model = await get_or_load_model(
        model_id=model_name,
        detector=request.detector,
        params=request.params,
    )

    # Fit the model
    result = await model.fit(
        reference_data=request.reference_data,
        feature_names=request.feature_names,
        autosave=True,
        overwrite=request.overwrite,
        description=request.description,
    )

    return DriftFitResponse(
        model=model_name,
        detector=request.detector,
        saved_path=result.get("saved_path", ""),
        training_time_ms=result["training_time_ms"],
        reference_size=result["reference_size"],
        n_features=result["n_features"],
    )


@router.post("/v1/drift/detect")
@handle_endpoint_errors("drift-detect")
async def detect_drift(request: DriftDetectRequest) -> DriftDetectResponse:
    """Check for drift in new data."""
    start_time = time.time()

    # Find model file and detector type
    model_name = request.model
    model_file, detector = _find_model_file(model_name)

    # Load model
    model = await get_or_load_model(model_name, detector)
    if not model.is_fitted:
        await model.load_from_path(model_file)

    # Detect drift
    result = await model.detect(request.data)
    detection_time_ms = (time.time() - start_time) * 1000

    return DriftDetectResponse(
        model=model_name,
        detector=detector,
        result=DriftResult(
            is_drift=result.is_drift,
            p_value=result.p_value,
            threshold=result.threshold,
            distance=result.distance,
            p_values=result.p_values,
        ),
        detection_time_ms=detection_time_ms,
    )


@router.post("/v1/drift/load")
@handle_endpoint_errors("drift-load")
async def load_model(request: DriftLoadRequest) -> DriftLoadResponse:
    """Load a saved drift model."""
    model_name = request.model

    # Find model file and detector type
    model_file, detector = _find_model_file(model_name)

    # Load the model
    model = DriftModel(model_id=model_name, detector=detector)
    await model.load_from_path(model_file)

    # Cache it
    if _drift_cache is not None:
        cache_key = f"{model_name}_{detector}"
        _drift_cache[cache_key] = model

    return DriftLoadResponse(
        model=model_name,
        detector=detector,
        is_fitted=model.is_fitted,
        reference_size=model.reference_size,
    )


@router.get("/v1/drift/status/{model_name}")
@handle_endpoint_errors("drift-status")
async def get_status(model_name: str) -> DriftStatus:
    """Get the status of a drift detector."""
    # Find model file and detector type
    model_file, detector = _find_model_file(model_name)

    # Try to get from cache first
    if _drift_cache is not None:
        cache_key = f"{model_name}_{detector}"
        cached = _drift_cache.get(cache_key)
        if cached:
            status = cached.get_status()
            return DriftStatus(
                model=status.detector_id,
                detector=status.detector_type,
                is_fitted=status.is_fitted,
                reference_size=status.reference_size,
                detection_count=status.detection_count,
                drift_count=status.drift_count,
                last_detection=DriftResult(
                    is_drift=status.last_detection.is_drift,
                    p_value=status.last_detection.p_value,
                    threshold=status.last_detection.threshold,
                    distance=status.last_detection.distance,
                ) if status.last_detection else None,
            )

    # Load from disk
    model = DriftModel(model_id=model_name, detector=detector)
    await model.load_from_path(model_file)
    status = model.get_status()

    return DriftStatus(
        model=status.detector_id,
        detector=status.detector_type,
        is_fitted=status.is_fitted,
        reference_size=status.reference_size,
        detection_count=status.detection_count,
        drift_count=status.drift_count,
        last_detection=DriftResult(
            is_drift=status.last_detection.is_drift,
            p_value=status.last_detection.p_value,
            threshold=status.last_detection.threshold,
            distance=status.last_detection.distance,
        ) if status.last_detection else None,
    )


@router.post("/v1/drift/reset/{model_name}")
@handle_endpoint_errors("drift-reset")
async def reset_detector(model_name: str) -> DriftResetResponse:
    """Reset a drift detector (clear reference and stats)."""
    if _drift_cache is None:
        raise RuntimeError("Drift router not initialized")

    # Find model in cache
    for cache_key in list(_drift_cache.keys()):
        if cache_key.startswith(f"{model_name}_"):
            model = _drift_cache.get(cache_key)
            if model:
                await model.reset()
                return DriftResetResponse(model=model_name, success=True)

    return DriftResetResponse(model=model_name, success=False)


@router.get("/v1/drift/models")
@handle_endpoint_errors("drift-models")
async def list_models() -> DriftModelsResponse:
    """List all saved drift models."""
    models = list_saved_models()
    return DriftModelsResponse(
        models=[
            DriftModelInfo(
                name=m.name,
                detector=m.detector,
                created_at=m.created_at,
                description=m.description,
                is_fitted=m.is_fitted,
                reference_size=m.reference_size,
            )
            for m in models
        ]
    )


@router.delete("/v1/drift/models/{model_name}")
@handle_endpoint_errors("drift-delete")
async def delete_drift_model(model_name: str) -> DriftDeleteResponse:
    """Delete a saved drift model."""
    deleted = delete_model(model_name)
    return DriftDeleteResponse(
        model=model_name,
        deleted=deleted,
    )
