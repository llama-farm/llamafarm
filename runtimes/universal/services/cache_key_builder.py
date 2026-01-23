"""Cache key builder service for Universal Runtime.

Consolidates cache key generation that was previously duplicated across:
- server.py (_make_language_cache_key, _make_encoder_cache_key, etc.)

This service provides consistent, deterministic cache key generation
for all model types.
"""

from typing import Any


class CacheKeyBuilder:
    """Builder for generating consistent cache keys for model caching.

    Cache keys must be:
    - Deterministic: Same inputs always produce same key
    - Unique: Different inputs produce different keys
    - Readable: Keys should be human-readable for debugging
    """

    @staticmethod
    def _normalize_value(value: Any) -> str:
        """Normalize a value for inclusion in cache key.

        Handles None, lists, bools, and other types consistently.
        """
        if value is None:
            return "none"
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, list):
            return "_".join(sorted(str(v) for v in value))
        return str(value)

    @classmethod
    def language_model(
        cls,
        model_id: str,
        context_length: int | None = None,
        max_batch_size: int | None = None,
        use_flash_attn: bool = False,
        gguf_quantization: str | None = None,
        n_gpu_layers: int | None = None,
        n_threads: int | None = None,
        use_mmap: bool | None = None,
        use_mlock: bool | None = None,
        cache_type_k: str | None = None,
        cache_type_v: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate cache key for a language model.

        Args:
            model_id: HuggingFace model identifier
            context_length: Optional context window size
            max_batch_size: Optional max batch size
            use_flash_attn: Whether using flash attention
            gguf_quantization: Optional GGUF quantization type
            n_gpu_layers: Number of layers to offload to GPU (GGUF)
            n_threads: Number of CPU threads (GGUF)
            use_mmap: Whether to use memory mapping (GGUF)
            use_mlock: Whether to lock memory (GGUF)
            cache_type_k: KV cache type for keys (GGUF)
            cache_type_v: KV cache type for values (GGUF)

        Returns:
            Cache key string
        """
        ctx_key = cls._normalize_value(context_length)
        batch_key = cls._normalize_value(max_batch_size)
        quant_key = cls._normalize_value(gguf_quantization)
        gpu_key = cls._normalize_value(n_gpu_layers)
        threads_key = cls._normalize_value(n_threads)
        mmap_key = cls._normalize_value(use_mmap)
        mlock_key = cls._normalize_value(use_mlock)
        cachek_key = cls._normalize_value(cache_type_k)
        cachev_key = cls._normalize_value(cache_type_v)

        return (
            f"language:{model_id}:"
            f"ctx{ctx_key}:batch{batch_key}:"
            f"gpu{gpu_key}:threads{threads_key}:"
            f"flash{use_flash_attn}:"
            f"mmap{mmap_key}:mlock{mlock_key}:"
            f"cachek{cachek_key}:cachev{cachev_key}:"
            f"quant{quant_key}"
        )

    @classmethod
    def encoder_model(
        cls,
        model_id: str,
        task: str = "embedding",
        context_length: int | None = None,
        max_length: int | None = None,
        gguf_quantization: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate cache key for an encoder model.

        Args:
            model_id: HuggingFace model identifier
            task: Model task (embedding, reranking, classification, ner)
            context_length: Optional context window size
            max_length: Optional max sequence length
            gguf_quantization: Optional GGUF quantization type

        Returns:
            Cache key string
        """
        ctx_key = cls._normalize_value(context_length)
        len_key = cls._normalize_value(max_length)
        quant_key = cls._normalize_value(gguf_quantization)

        return f"encoder:{task}:{model_id}:ctx{ctx_key}:len{len_key}:quant{quant_key}"

    @classmethod
    def document_model(cls, model_id: str, task: str = "extraction") -> str:
        """Generate cache key for a document model.

        Args:
            model_id: HuggingFace model identifier
            task: Model task (extraction, vqa, classification)

        Returns:
            Cache key string
        """
        return f"document:{task}:{model_id}"

    @classmethod
    def ocr_model(cls, backend: str, languages: list[str] | None = None) -> str:
        """Generate cache key for an OCR model.

        Args:
            backend: OCR backend (surya, easyocr, paddleocr, tesseract)
            languages: List of language codes

        Returns:
            Cache key string
        """
        langs = languages or ["en"]
        lang_key = "_".join(sorted(langs))
        return f"ocr:{backend}:{lang_key}"

    @classmethod
    def anomaly_model(
        cls,
        model_id: str,
        backend: str = "isolation_forest",
        normalization: str | None = None,
    ) -> str:
        """Generate cache key for an anomaly model.

        Args:
            model_id: Model identifier or path
            backend: Anomaly detection backend
            normalization: Score normalization method

        Returns:
            Cache key string
        """
        if normalization:
            return f"anomaly:{backend}:{normalization}:{model_id}"
        return f"anomaly:{backend}:{model_id}"

    @classmethod
    def classifier_model(cls, model_id: str) -> str:
        """Generate cache key for a classifier model.

        Args:
            model_id: Model identifier

        Returns:
            Cache key string
        """
        return f"classifier:{model_id}"

    @classmethod
    def speech_model(cls, model_id: str, task: str = "transcribe") -> str:
        """Generate cache key for a speech model.

        Args:
            model_id: Whisper model identifier
            task: Model task (transcribe, translate)

        Returns:
            Cache key string
        """
        return f"speech:{task}:{model_id}"


# Convenience functions for backward compatibility
def make_language_cache_key(
    model_id: str,
    context_length: int | None = None,
    max_batch_size: int | None = None,
    use_flash_attn: bool = False,
    gguf_quantization: str | None = None,
    n_gpu_layers: int | None = None,
    n_threads: int | None = None,
    use_mmap: bool | None = None,
    use_mlock: bool | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    **kwargs: Any,
) -> str:
    """Generate cache key for a language model (backward compatible function)."""
    return CacheKeyBuilder.language_model(
        model_id=model_id,
        context_length=context_length,
        max_batch_size=max_batch_size,
        use_flash_attn=use_flash_attn,
        gguf_quantization=gguf_quantization,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        use_mmap=use_mmap,
        use_mlock=use_mlock,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        **kwargs,
    )


def make_encoder_cache_key(
    model_id: str,
    task: str = "embedding",
    context_length: int | None = None,
    max_length: int | None = None,
    gguf_quantization: str | None = None,
    **kwargs: Any,
) -> str:
    """Generate cache key for an encoder model (backward compatible function)."""
    return CacheKeyBuilder.encoder_model(
        model_id=model_id,
        task=task,
        context_length=context_length,
        max_length=max_length,
        gguf_quantization=gguf_quantization,
        **kwargs,
    )


def make_document_cache_key(model_id: str, task: str = "extraction") -> str:
    """Generate cache key for a document model (backward compatible function)."""
    return CacheKeyBuilder.document_model(model_id=model_id, task=task)


def make_ocr_cache_key(backend: str, languages: list[str] | None = None) -> str:
    """Generate cache key for an OCR model (backward compatible function)."""
    return CacheKeyBuilder.ocr_model(backend=backend, languages=languages)


def make_anomaly_cache_key(
    model_id: str, backend: str = "isolation_forest", normalization: str | None = None
) -> str:
    """Generate cache key for an anomaly model (backward compatible function)."""
    return CacheKeyBuilder.anomaly_model(
        model_id=model_id, backend=backend, normalization=normalization
    )


def make_classifier_cache_key(model_id: str) -> str:
    """Generate cache key for a classifier model (backward compatible function)."""
    return CacheKeyBuilder.classifier_model(model_id=model_id)


def make_speech_cache_key(model_id: str, task: str = "transcribe") -> str:
    """Generate cache key for a speech model (backward compatible function)."""
    return CacheKeyBuilder.speech_model(model_id=model_id, task=task)
