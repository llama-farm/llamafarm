"""
Service layer for encoder-based operations.

Handles model loading and caching for embeddings, reranking, classification, and NER.
"""

from core.logging import UniversalRuntimeLogger
from models import BaseModel, EncoderModel, GGUFEncoderModel
from state import get_device, get_model_load_lock, get_models_cache
from utils.model_format import detect_model_format

logger = UniversalRuntimeLogger("universal-runtime.embeddings")


def make_encoder_cache_key(
    model_id: str,
    task: str,
    model_format: str,
    preferred_quantization: str | None = None,
    max_length: int | None = None,
) -> str:
    """Generate a cache key for an encoder model.

    Args:
        model_id: HuggingFace model identifier
        task: Model task - "embedding", "classification", "reranking", or "ner"
        model_format: Model format - "gguf" or "transformers"
        preferred_quantization: Optional quantization preference for GGUF models
        max_length: Optional max sequence length override

    Returns:
        A unique cache key string that identifies this specific model configuration
    """
    quant_key = (
        preferred_quantization if preferred_quantization is not None else "default"
    )
    len_key = max_length if max_length is not None else "auto"
    return f"encoder:{task}:{model_format}:{model_id}:quant{quant_key}:len{len_key}"


async def load_encoder(
    model_id: str,
    task: str = "embedding",
    preferred_quantization: str | None = None,
    max_length: int | None = None,
    use_flash_attention: bool = True,
) -> BaseModel:
    """Load an encoder model for embeddings, classification, reranking, or NER.

    Automatically detects whether the model is in GGUF or transformers format
    and loads it with the appropriate backend. GGUF models use llama-cpp
    for optimized inference, while transformers models use the standard HuggingFace
    transformers library.

    Supports modern encoder features:
    - Configurable max_length (up to 8,192 for ModernBERT)
    - Flash Attention 2 for faster inference on CUDA

    Args:
        model_id: HuggingFace model identifier
        task: Model task - "embedding", "classification", "reranking", or "ner"
        preferred_quantization: Optional quantization preference for GGUF models
                                (e.g., "Q4_K_M", "Q8_0"). If None, defaults to Q4_K_M.
        max_length: Optional max sequence length override (auto-detected if None)
        use_flash_attention: Whether to use Flash Attention 2 if available (default True)

    Returns:
        Loaded encoder model instance
    """
    models_cache = get_models_cache()
    model_load_lock = get_model_load_lock()

    # Detect model format for proper caching and loading
    model_format = detect_model_format(model_id)
    # Include quantization and max_length in cache key for proper caching
    cache_key = make_encoder_cache_key(
        model_id, task, model_format, preferred_quantization, max_length
    )

    if cache_key not in models_cache:
        async with model_load_lock:
            # Double-check if model was loaded while waiting for the lock
            if cache_key not in models_cache:
                logger.info(
                    f"Loading encoder ({task}): {model_id} (format: {model_format})"
                )
                device = get_device()

                # Instantiate appropriate model class based on format
                model: BaseModel
                if model_format == "gguf":
                    if task != "embedding":
                        raise ValueError(
                            f"GGUF models only support embedding task, not '{task}'"
                        )
                    model = GGUFEncoderModel(
                        model_id, device, preferred_quantization=preferred_quantization
                    )
                else:
                    model = EncoderModel(
                        model_id,
                        device,
                        task=task,
                        max_length=max_length,
                        use_flash_attention=use_flash_attention,
                    )

                await model.load()
                models_cache[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return models_cache.get(cache_key)
