"""
Service layer for OCR operations.

Handles model loading and caching for OCR backends.
"""

from core.logging import UniversalRuntimeLogger
from models import OCRModel
from state import get_device, get_model_load_lock, get_models_cache

logger = UniversalRuntimeLogger("universal-runtime.ocr")


def make_ocr_cache_key(backend: str, languages: list[str]) -> str:
    """Generate a cache key for an OCR model.

    Args:
        backend: OCR backend (surya, easyocr, paddleocr, tesseract)
        languages: List of language codes

    Returns:
        A unique cache key string
    """
    lang_key = "_".join(sorted(languages))
    return f"ocr:{backend}:{lang_key}"


async def load_ocr(
    backend: str = "surya",
    languages: list[str] | None = None,
) -> OCRModel:
    """Load an OCR model with the specified backend.

    Args:
        backend: OCR backend to use (surya, easyocr, paddleocr, tesseract)
        languages: List of language codes (e.g., ['en', 'fr'])

    Returns:
        Loaded OCRModel instance
    """
    models_cache = get_models_cache()
    model_load_lock = get_model_load_lock()

    langs = languages or ["en"]
    cache_key = make_ocr_cache_key(backend, langs)

    if cache_key not in models_cache:
        async with model_load_lock:
            if cache_key not in models_cache:
                logger.info(f"Loading OCR model: {backend} (languages: {langs})")
                device = get_device()

                model = OCRModel(
                    model_id=f"ocr-{backend}",
                    device=device,
                    backend=backend,
                    languages=langs,
                )

                await model.load()
                models_cache[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return models_cache.get(cache_key)
