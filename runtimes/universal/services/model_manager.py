"""Unified model manager for all model types.

Consolidates the 10+ separate model caches in server.py into a single
typed service with consistent cache key generation and lifecycle management.
"""

import asyncio
import logging
import os
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

from utils.model_cache import ModelCache

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModelType(str, Enum):
    """Model type categories for cache key prefixes."""

    LANGUAGE = "language"
    ENCODER = "encoder"
    VISION = "vision"
    FEW_SHOT = "few_shot"
    OPEN_VOCAB = "open_vocab"
    OBJECT_DETECTION = "object_detection"
    BACKGROUND_REMOVAL = "background_removal"
    ANOMALY = "anomaly"
    CLASSIFIER = "classifier"
    LANG_DETECTION = "lang_detection"
    PII = "pii"
    TIMESERIES = "timeseries"
    SPEECH = "speech"
    TABLE_QA = "table_qa"


class ModelManager:
    """Unified manager for all model caches.

    Replaces the 10+ separate ModelCache instances in server.py with a single
    service that provides:
    - Consistent cache key generation
    - Type-safe model access
    - Centralized TTL management
    - Single cleanup loop for all model types

    Usage:
        manager = ModelManager(ttl=300)

        # Load or get cached model
        model = await manager.get_or_load(
            model_type=ModelType.VISION,
            cache_key="vision:openai/clip-vit-base-patch32",
            loader=lambda: load_clip_model("openai/clip-vit-base-patch32"),
        )

        # Cleanup expired models
        expired = manager.pop_all_expired()
        for key, model in expired:
            await model.unload()
    """

    def __init__(self, ttl: int = 300, maxsize: int = 1000):
        """Initialize the model manager.

        Args:
            ttl: Time-to-live in seconds for idle models (default: 5 minutes)
            maxsize: Maximum models per cache (default: 1000)
        """
        self._ttl = ttl
        self._maxsize = maxsize
        self._load_lock = asyncio.Lock()

        # Separate caches by model type for better organization
        # All use the same TTL for consistent behavior
        self._caches: dict[ModelType, ModelCache] = {
            model_type: ModelCache(ttl=ttl, maxsize=maxsize) for model_type in ModelType
        }

        # Feature encoder cache (not a model, but related state)
        self._encoders: dict[str, Any] = {}

        # Drift detector cache (stateful streaming objects)
        self._drift_detectors: dict[str, Any] = {}

        # Table QA models (separate due to different lifecycle)
        self._table_qa_models: dict[str, Any] = {}

    @property
    def ttl(self) -> int:
        """Get the TTL in seconds."""
        return self._ttl

    @property
    def load_lock(self) -> asyncio.Lock:
        """Get the async lock for model loading."""
        return self._load_lock

    # =========================================================================
    # Cache Key Generation (DRY: centralized key generation)
    # =========================================================================

    @staticmethod
    def make_language_cache_key(
        model_id: str,
        n_ctx: int | None = None,
        n_batch: int | None = None,
        n_gpu_layers: int | None = None,
        n_threads: int | None = None,
        flash_attn: bool | None = None,
        use_mmap: bool | None = None,
        use_mlock: bool | None = None,
        cache_type_k: str | None = None,
        cache_type_v: str | None = None,
        preferred_quantization: str | None = None,
    ) -> str:
        """Generate cache key for language models (GGUF/transformers)."""

        def _val(v, default="auto"):
            return v if v is not None else default

        return (
            f"language:{model_id}:"
            f"ctx{_val(n_ctx)}:batch{_val(n_batch)}:gpu{_val(n_gpu_layers)}:"
            f"threads{_val(n_threads)}:flash{_val(flash_attn, 'default')}:"
            f"mmap{_val(use_mmap, 'default')}:mlock{_val(use_mlock, 'default')}:"
            f"cachek{_val(cache_type_k, 'default')}:cachev{_val(cache_type_v, 'default')}:"
            f"quant{_val(preferred_quantization, 'default')}"
        )

    @staticmethod
    def make_encoder_cache_key(
        model_id: str,
        task: str,
        model_format: str,
        preferred_quantization: str | None = None,
        max_length: int | None = None,
    ) -> str:
        """Generate cache key for encoder models."""
        quant = preferred_quantization or "default"
        length = max_length if max_length is not None else "auto"
        return f"encoder:{task}:{model_format}:{model_id}:quant{quant}:len{length}"

    @staticmethod
    def make_vision_cache_key(model_name: str) -> str:
        """Generate cache key for CLIP vision models."""
        return f"vision:{model_name}"

    @staticmethod
    def make_few_shot_cache_key(classifier_id: str, model_name: str) -> str:
        """Generate cache key for few-shot classifiers."""
        return f"few_shot:{classifier_id}:{model_name}"

    @staticmethod
    def make_open_vocab_cache_key(model_name: str) -> str:
        """Generate cache key for open-vocabulary detection models."""
        return f"open_vocab:{model_name}"

    @staticmethod
    def make_object_detection_cache_key(model_name: str) -> str:
        """Generate cache key for object detection models."""
        return f"object_detection:{model_name}"

    @staticmethod
    def make_background_removal_cache_key(model_name: str) -> str:
        """Generate cache key for background removal models."""
        return f"background_removal:{model_name}"

    @staticmethod
    def make_anomaly_cache_key(
        model_id: str,
        backend: str,
        normalization: str = "standardization",
        scaler_type: str = "robust",
    ) -> str:
        """Generate cache key for anomaly detection models."""
        return f"anomaly:{model_id}:{backend}:{normalization}:{scaler_type}"

    @staticmethod
    def make_classifier_cache_key(model_id: str, embedding_model: str) -> str:
        """Generate cache key for SetFit-style classifiers."""
        return f"classifier:{model_id}:{embedding_model}"

    @staticmethod
    def make_lang_detection_cache_key(model_name: str) -> str:
        """Generate cache key for language detection models."""
        return f"lang_detection:{model_name}"

    @staticmethod
    def make_pii_cache_key(model_name: str) -> str:
        """Generate cache key for PII detection models."""
        return f"pii:{model_name}"

    @staticmethod
    def make_timeseries_cache_key(model_name: str) -> str:
        """Generate cache key for time series models."""
        return f"timeseries:{model_name}"

    @staticmethod
    def make_speech_cache_key(model_name: str) -> str:
        """Generate cache key for speech models."""
        return f"speech:{model_name}"

    # =========================================================================
    # Generic Cache Operations
    # =========================================================================

    def get_cache(self, model_type: ModelType) -> ModelCache:
        """Get the cache for a specific model type."""
        return self._caches[model_type]

    def contains(self, model_type: ModelType, cache_key: str) -> bool:
        """Check if a model is in the cache."""
        return cache_key in self._caches[model_type]

    def get(self, model_type: ModelType, cache_key: str) -> Any | None:
        """Get a model from cache (refreshes TTL)."""
        return self._caches[model_type].get(cache_key)

    def set(self, model_type: ModelType, cache_key: str, model: Any) -> None:
        """Store a model in cache."""
        self._caches[model_type][cache_key] = model

    def remove(self, model_type: ModelType, cache_key: str) -> Any | None:
        """Remove a model from cache."""
        cache = self._caches[model_type]
        if cache_key in cache:
            return cache.pop(cache_key)
        return None

    async def get_or_load(
        self,
        model_type: ModelType,
        cache_key: str,
        loader: Any,  # Callable[[], Awaitable[T]]
    ) -> Any:
        """Get model from cache or load it using the provided loader.

        This is the main method for loading models. It handles:
        - Cache lookup with TTL refresh
        - Async lock to prevent duplicate loads
        - Double-check pattern for thread safety

        Args:
            model_type: Type of model (for cache selection)
            cache_key: Unique cache key for this model
            loader: Async callable that loads and returns the model

        Returns:
            The loaded model
        """
        cache = self._caches[model_type]

        if cache_key not in cache:
            async with self._load_lock:
                # Double-check after acquiring lock
                if cache_key not in cache:
                    model = await loader()
                    cache[cache_key] = model

        return cache.get(cache_key)

    # =========================================================================
    # Cleanup Operations
    # =========================================================================

    def pop_all_expired(self) -> list[tuple[str, Any]]:
        """Pop expired models from all caches.

        Returns:
            List of (cache_key, model) tuples for all expired models
        """
        all_expired = []
        for model_type, cache in self._caches.items():
            expired = cache.pop_expired()
            if expired:
                logger.debug(f"Found {len(expired)} expired {model_type.value} models")
                all_expired.extend(expired)
        return all_expired

    def get_all_models(self) -> list[tuple[str, Any]]:
        """Get all models from all caches (for shutdown cleanup)."""
        all_models = []
        for cache in self._caches.values():
            all_models.extend(list(cache.items()))
        return all_models

    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self._caches.values():
            cache.clear()
        self._encoders.clear()
        self._drift_detectors.clear()
        self._table_qa_models.clear()

    def get_loaded_model_ids(self) -> list[str]:
        """Get list of all loaded model cache keys."""
        all_keys = []
        for cache in self._caches.values():
            all_keys.extend(list(cache.keys()))
        return all_keys

    # =========================================================================
    # Feature Encoder Management (for anomaly detection)
    # =========================================================================

    def get_encoder(self, cache_key: str) -> Any | None:
        """Get a feature encoder by cache key."""
        return self._encoders.get(cache_key)

    def set_encoder(self, cache_key: str, encoder: Any) -> None:
        """Store a feature encoder."""
        self._encoders[cache_key] = encoder

    def remove_encoder(self, cache_key: str) -> Any | None:
        """Remove a feature encoder."""
        return self._encoders.pop(cache_key, None)

    # =========================================================================
    # Drift Detector Management (for streaming)
    # =========================================================================

    def get_drift_detector(self, detector_id: str) -> Any | None:
        """Get a drift detector by ID."""
        return self._drift_detectors.get(detector_id)

    def set_drift_detector(self, detector_id: str, detector: Any) -> None:
        """Store a drift detector."""
        self._drift_detectors[detector_id] = detector

    def remove_drift_detector(self, detector_id: str) -> Any | None:
        """Remove a drift detector."""
        return self._drift_detectors.pop(detector_id, None)

    def list_drift_detectors(self) -> list[str]:
        """List all drift detector IDs."""
        return list(self._drift_detectors.keys())

    # =========================================================================
    # Table QA Model Management
    # =========================================================================

    def get_table_qa_model(self, model_name: str) -> Any | None:
        """Get a table QA model."""
        return self._table_qa_models.get(model_name)

    def set_table_qa_model(self, model_name: str, model: Any) -> None:
        """Store a table QA model."""
        self._table_qa_models[model_name] = model

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "ttl_seconds": self._ttl,
            "total_models": sum(len(c) for c in self._caches.values()),
            "by_type": {
                model_type.value: len(cache)
                for model_type, cache in self._caches.items()
                if len(cache) > 0
            },
            "encoders": len(self._encoders),
            "drift_detectors": len(self._drift_detectors),
            "table_qa_models": len(self._table_qa_models),
        }
        return stats


# Global singleton instance
# Initialize with environment variable or default TTL
_default_ttl = int(os.getenv("MODEL_UNLOAD_TIMEOUT", "300"))
model_manager = ModelManager(ttl=_default_ttl)
