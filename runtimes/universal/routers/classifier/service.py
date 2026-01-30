"""
Service layer for SetFit classifier operations.

Handles model loading, caching, and persistence for few-shot text classifiers.
"""

from pathlib import Path

from core.logging import UniversalRuntimeLogger
from models import ClassifierModel
from state import (
    CLASSIFIER_MODELS_DIR,
    get_classifiers_cache,
    get_device,
    get_model_load_lock,
    sanitize_model_name,
)

logger = UniversalRuntimeLogger("universal-runtime.classifier")


def make_classifier_cache_key(model_name: str) -> str:
    """Create a cache key for classifier models."""
    return f"classifier:{model_name}"


def get_classifier_path(model_name: str) -> Path:
    """Get the path for a classifier model directory.

    The path is always within CLASSIFIER_MODELS_DIR - users cannot control it.
    """
    safe_name = sanitize_model_name(model_name)
    return CLASSIFIER_MODELS_DIR / safe_name


async def load_classifier(
    model_id: str,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> ClassifierModel:
    """Load or get cached classifier model."""
    classifiers_cache = get_classifiers_cache()
    model_load_lock = get_model_load_lock()

    cache_key = make_classifier_cache_key(model_id)

    # Evict cached model if base_model changed (prevents returning a model
    # initialized with a different base_model for the same model_id)
    cached = classifiers_cache.get(cache_key) if cache_key in classifiers_cache else None
    if cached is not None and getattr(cached, "base_model", None) != base_model:
        logger.info(
            f"Evicting classifier '{model_id}': base_model changed "
            f"({cached.base_model} -> {base_model})"
        )
        classifiers_cache.pop(cache_key, None)
        await cached.unload()

    if cache_key not in classifiers_cache:
        async with model_load_lock:
            # Double-check after acquiring lock
            if cache_key not in classifiers_cache:
                logger.info(f"Loading classifier model: {model_id}")
                device = get_device()

                model = ClassifierModel(
                    model_id=model_id,
                    device=device,
                    base_model=base_model,
                )

                await model.load()
                classifiers_cache[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return classifiers_cache.get(cache_key)


async def auto_save_classifier_model(
    model: ClassifierModel,
    model_name: str,
) -> dict[str, str | None]:
    """Auto-save classifier model after fit to prevent data loss.

    Models are saved immediately after training to ensure they persist
    across server restarts without requiring an explicit /save call.

    Returns:
        Dict with saved file path
    """
    try:
        # Create models directory if needed
        CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate path from model name
        save_path = get_classifier_path(model_name)
        await model.save(str(save_path))

        logger.info(f"Auto-saved classifier model to {save_path}")
        return {"model_path": str(save_path)}

    except Exception as e:
        logger.warning(f"Auto-save failed (model still in memory): {e}")
        return {"model_path": None}
