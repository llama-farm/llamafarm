"""
Service layer for anomaly detection operations.

Handles model loading, caching, data preparation, and model persistence.
"""

from pathlib import Path

from fastapi import HTTPException

from core.logging import UniversalRuntimeLogger
from models import AnomalyModel, BaseModel
from state import (
    ANOMALY_MODELS_DIR,
    get_device,
    get_encoders_cache,
    get_model_load_lock,
    get_models_cache,
    sanitize_model_name,
)
from utils.feature_encoder import FeatureEncoder

logger = UniversalRuntimeLogger("universal-runtime.anomaly")


def make_anomaly_cache_key(
    model_id: str, backend: str, normalization: str | None = None
) -> str:
    """Generate a cache key for an anomaly model.

    Args:
        model_id: Model identifier or path
        backend: Anomaly detection backend
        normalization: Score normalization method. If provided, it becomes part of
            the cache key to ensure models with different normalization methods
            are cached separately.

    Returns:
        Cache key string
    """
    if normalization:
        return f"anomaly:{backend}:{normalization}:{model_id}"
    return f"anomaly:{backend}:{model_id}"


def get_model_path(model_name: str, backend: str) -> Path:
    """Get the path for a model file based on name and backend.

    The path is always within ANOMALY_MODELS_DIR - users cannot control it.
    """
    safe_name = sanitize_model_name(model_name)
    safe_backend = sanitize_model_name(backend)
    filename = f"{safe_name}_{safe_backend}"
    return ANOMALY_MODELS_DIR / filename


async def load_anomaly(
    model_id: str,
    backend: str = "isolation_forest",
    contamination: float = 0.1,
    threshold: float | None = None,
    normalization: str = "standardization",
) -> AnomalyModel:
    """Load an anomaly detection model.

    Args:
        model_id: Model identifier or path to pre-trained model
        backend: Anomaly detection backend
        contamination: Expected proportion of anomalies
        threshold: Custom anomaly threshold
        normalization: Score normalization method (standardization, zscore, raw)

    Returns:
        Loaded AnomalyModel instance
    """
    models_cache = get_models_cache()
    model_load_lock = get_model_load_lock()

    cache_key = make_anomaly_cache_key(model_id, backend, normalization)

    if cache_key not in models_cache:
        async with model_load_lock:
            if cache_key not in models_cache:
                logger.info(f"Loading anomaly model ({backend}): {model_id}")
                device = get_device()

                model = AnomalyModel(
                    model_id=model_id,
                    device=device,
                    backend=backend,
                    contamination=contamination,
                    threshold=threshold,
                    normalization=normalization,
                )

                await model.load()
                models_cache[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return models_cache.get(cache_key)


def prepare_anomaly_data(
    data: list[list[float]] | list[dict],
    schema: dict[str, str] | None,
    cache_key: str,
    fit_mode: bool = False,
) -> list[list[float]]:
    """
    Prepare data for anomaly detection by encoding if needed.

    Args:
        data: Raw data (numeric arrays or dicts)
        schema: Feature encoding schema (required for dict data during fit)
        cache_key: Cache key for storing/retrieving encoder
        fit_mode: If True, fit the encoder on the data. If False, use existing encoder.

    Returns:
        Encoded numeric data as list of lists
    """
    encoders_cache = get_encoders_cache()

    # If data is already numeric, return as-is
    if not data:
        return []

    if isinstance(data[0], list):
        # Already numeric arrays
        return data

    # Dict-based data - need to encode
    if fit_mode:
        # Require schema for training
        if schema is None:
            raise HTTPException(
                status_code=400,
                detail="Schema is required when fitting with dict-based data. "
                "Example: schema = {'time_ms': 'numeric', 'user_agent': 'hash'}",
            )
        # Fit encoder on training data
        encoder = FeatureEncoder()
        encoder.fit(data, schema)
        encoders_cache[cache_key] = encoder
        logger.info(f"Fitted feature encoder for {cache_key} with schema: {schema}")
    else:
        # Use existing encoder (schema already learned during fit)
        if cache_key not in encoders_cache:
            raise HTTPException(
                status_code=400,
                detail=f"No encoder found for model '{cache_key}'. "
                "Train with /v1/anomaly/fit using dict data first, or pass schema.",
            )
        encoder = encoders_cache[cache_key]

    # Transform data
    encoded = encoder.transform(data)
    return encoded.tolist()


async def auto_save_anomaly_model(
    model: BaseModel,
    model_name: str,
    backend: str,
    cache_key: str,
) -> None:
    """Auto-save anomaly model after fit to prevent data loss.

    Models are saved immediately after training to ensure they persist
    across server restarts without requiring an explicit /save call.

    Raises:
        Exception: If model save fails. This is intentionally not caught
            because models MUST be persisted - a failed save should fail
            the entire fit operation.
    """
    encoders_cache = get_encoders_cache()

    # Create models directory if needed
    ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate path from model name
    save_path = get_model_path(model_name, backend)
    await model.save(str(save_path))

    # Determine actual saved file path for logging.
    # The model.save() method appends the appropriate extension based on backend:
    # - autoencoder backend: saves as PyTorch .pt file
    # - sklearn backends (isolation_forest, etc.): save as .joblib (preferred)
    #   or .pkl (legacy fallback for older scikit-learn versions)
    if backend == "autoencoder":
        actual_path = save_path.with_suffix(".pt")
    else:
        # sklearn-based backends prefer joblib for efficient array serialization,
        # but fall back to pickle (.pkl) for compatibility with older models
        actual_path = save_path.with_suffix(".joblib")
        if not actual_path.exists():
            actual_path = save_path.with_suffix(".pkl")

    logger.debug(f"Model saved to {actual_path}")

    # Save encoder if one exists for this model
    if cache_key in encoders_cache:
        encoder = encoders_cache[cache_key]
        encoder_save_path = save_path.parent / f"{save_path.name}_encoder.json"
        encoder.save(encoder_save_path)
        logger.debug(f"Feature encoder saved to {encoder_save_path}")
