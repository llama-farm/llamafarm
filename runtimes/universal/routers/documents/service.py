"""
Service layer for document understanding operations.

Handles model loading and caching for document extraction, VQA, and classification.
"""

from core.logging import UniversalRuntimeLogger
from models import DocumentModel
from state import get_device, get_model_load_lock, get_models_cache

logger = UniversalRuntimeLogger("universal-runtime.documents")


def make_document_cache_key(model_id: str, task: str) -> str:
    """Generate a cache key for a document model."""
    return f"document:{task}:{model_id}"


async def load_document(
    model_id: str,
    task: str = "extraction",
) -> DocumentModel:
    """Load a document understanding model.

    Args:
        model_id: HuggingFace model identifier
        task: Model task - "extraction", "vqa", or "classification"

    Returns:
        Loaded DocumentModel instance
    """
    models_cache = get_models_cache()
    model_load_lock = get_model_load_lock()

    cache_key = make_document_cache_key(model_id, task)

    if cache_key not in models_cache:
        async with model_load_lock:
            if cache_key not in models_cache:
                logger.info(f"Loading document model ({task}): {model_id}")
                device = get_device()

                model = DocumentModel(
                    model_id=model_id,
                    device=device,
                    task=task,
                )

                await model.load()
                models_cache[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return models_cache.get(cache_key)
