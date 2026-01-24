"""Model loading service with caching and per-type locking.

This service provides a unified pattern for loading ML models with:
- Cache lookup to return existing models
- Per-model-type locks to reduce contention
- Integration with the TTL-based ModelCache

Issue addressed: server.py has 7 different model loading patterns
(one for each model type) with duplicated cache/lock logic.

Solution: Unified ModelLoader class that handles the common pattern:
1. Check cache for existing model
2. Acquire per-type lock if not cached
3. Double-check cache (another request may have loaded it)
4. Load model
5. Store in cache
6. Release lock
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any, Generic, TypeVar

from utils.model_cache import ModelCache

logger = logging.getLogger(__name__)

# Type variable for model types
M = TypeVar("M")


class ModelLoader(Generic[M]):
    """Generic model loader with cache and locking.

    This class provides a unified pattern for loading ML models with
    proper caching and concurrency handling.

    Features:
    - Cache-first lookup to avoid redundant loads
    - Per-type locking to reduce contention
    - Double-checked locking pattern for thread safety
    - Integration with TTL-based ModelCache

    Example:
        # Create a loader for anomaly models
        anomaly_loader = ModelLoader[AnomalyModel](
            cache=model_cache,
            model_type="anomaly"
        )

        # Load or get cached model
        model = await anomaly_loader.get_or_load(
            cache_key="anomaly:my-model:isolation_forest:zscore",
            loader_fn=lambda: create_anomaly_model(...)
        )
    """

    def __init__(
        self,
        cache: ModelCache[M],
        model_type: str,
    ):
        """Initialize the model loader.

        Args:
            cache: The ModelCache to store loaded models
            model_type: Name for this model type (for logging)
        """
        self._cache = cache
        self._model_type = model_type
        self._lock = asyncio.Lock()
        logger.debug(f"Initialized ModelLoader for {model_type}")

    @property
    def cache(self) -> ModelCache[M]:
        """Get the underlying cache."""
        return self._cache

    def get_cached(self, cache_key: str) -> M | None:
        """Get a model from cache without loading.

        Args:
            cache_key: The cache key to look up

        Returns:
            The cached model, or None if not found
        """
        return self._cache.get(cache_key)

    def is_cached(self, cache_key: str) -> bool:
        """Check if a model is in the cache.

        Args:
            cache_key: The cache key to check

        Returns:
            True if the model is cached, False otherwise
        """
        return cache_key in self._cache

    async def get_or_load(
        self,
        cache_key: str,
        loader_fn: Callable[[], Coroutine[Any, Any, M]],
    ) -> M:
        """Get a model from cache or load it.

        This method implements the double-checked locking pattern:
        1. Check cache (fast path, no lock)
        2. If not cached, acquire lock
        3. Check cache again (another request may have loaded it)
        4. Load the model
        5. Store in cache
        6. Release lock

        Args:
            cache_key: Unique key for this model configuration
            loader_fn: Async function that creates and initializes the model

        Returns:
            The loaded model (either from cache or freshly loaded)
        """
        # Fast path: check cache without lock
        model = self._cache.get(cache_key)
        if model is not None:
            logger.debug(f"Cache hit for {self._model_type}: {cache_key}")
            return model

        # Slow path: acquire lock and load
        async with self._lock:
            # Double-check: another request may have loaded it while we waited
            model = self._cache.get(cache_key)
            if model is not None:
                logger.debug(
                    f"Cache hit (after lock) for {self._model_type}: {cache_key}"
                )
                return model

            # Load the model
            logger.info(f"Loading {self._model_type} model: {cache_key}")
            model = await loader_fn()

            # Store in cache
            self._cache[cache_key] = model
            logger.info(f"Cached {self._model_type} model: {cache_key}")

            return model

    def remove(self, cache_key: str) -> M | None:
        """Remove a model from the cache.

        Args:
            cache_key: The cache key to remove

        Returns:
            The removed model, or None if not found
        """
        if cache_key in self._cache:
            return self._cache.pop(cache_key)
        return None

    def clear(self) -> None:
        """Clear all models from the cache."""
        self._cache.clear()
        logger.info(f"Cleared all {self._model_type} models from cache")


class ModelLoaderRegistry:
    """Registry for multiple model loaders.

    Provides a centralized way to manage loaders for different model types,
    each with their own cache and lock.

    Example:
        registry = ModelLoaderRegistry(ttl=300)

        # Get or create loader for a model type
        anomaly_loader = registry.get_loader("anomaly")
        classifier_loader = registry.get_loader("classifier")

        # Use the loaders
        model = await anomaly_loader.get_or_load(cache_key, loader_fn)
    """

    def __init__(self, ttl: float = 300, maxsize: int = 100):
        """Initialize the registry.

        Args:
            ttl: Time-to-live in seconds for cached models (default: 5 minutes)
            maxsize: Maximum number of models per cache (default: 100)
        """
        self._ttl = ttl
        self._maxsize = maxsize
        self._loaders: dict[str, ModelLoader[Any]] = {}
        self._lock = asyncio.Lock()

    async def get_loader(self, model_type: str) -> ModelLoader[Any]:
        """Get or create a loader for a model type.

        Args:
            model_type: The type of model (e.g., "anomaly", "classifier")

        Returns:
            The ModelLoader for this model type
        """
        if model_type not in self._loaders:
            async with self._lock:
                # Double-check after acquiring lock
                if model_type not in self._loaders:
                    cache: ModelCache[Any] = ModelCache(
                        ttl=self._ttl, maxsize=self._maxsize
                    )
                    self._loaders[model_type] = ModelLoader(
                        cache=cache, model_type=model_type
                    )
                    logger.info(f"Created ModelLoader for {model_type}")

        return self._loaders[model_type]

    def get_loader_sync(self, model_type: str) -> ModelLoader[Any]:
        """Get a loader synchronously (assumes it exists).

        This is useful when you know the loader has already been created
        (e.g., during application startup).

        Args:
            model_type: The type of model

        Returns:
            The ModelLoader for this model type

        Raises:
            KeyError: If the loader doesn't exist
        """
        return self._loaders[model_type]

    def register_loader(
        self,
        model_type: str,
        cache: ModelCache[Any] | None = None,
    ) -> ModelLoader[Any]:
        """Register a loader for a model type.

        This allows using custom caches or pre-creating loaders.

        Args:
            model_type: The type of model
            cache: Optional custom cache (default: new ModelCache)

        Returns:
            The registered ModelLoader
        """
        if cache is None:
            cache = ModelCache(ttl=self._ttl, maxsize=self._maxsize)

        loader: ModelLoader[Any] = ModelLoader(cache=cache, model_type=model_type)
        self._loaders[model_type] = loader
        logger.info(f"Registered ModelLoader for {model_type}")
        return loader

    def get_all_cached_keys(self) -> dict[str, list[str]]:
        """Get all cached model keys by type.

        Returns:
            Dict mapping model type to list of cache keys
        """
        return {
            model_type: list(loader.cache.keys())
            for model_type, loader in self._loaders.items()
        }

    async def clear_all(self) -> None:
        """Clear all caches across all loaders."""
        for loader in self._loaders.values():
            loader.clear()
        logger.info("Cleared all model caches")

    async def get_expired_models(self) -> dict[str, list[tuple[str, Any]]]:
        """Get all expired models from all loaders.

        Returns:
            Dict mapping model type to list of (key, model) tuples
        """
        return {
            model_type: loader.cache.pop_expired()
            for model_type, loader in self._loaders.items()
        }


# Global registry instance (can be customized at startup)
_registry: ModelLoaderRegistry | None = None


def get_model_registry(ttl: float = 300, maxsize: int = 100) -> ModelLoaderRegistry:
    """Get or create the global model loader registry.

    Args:
        ttl: Time-to-live for cached models (only used on first call)
        maxsize: Max models per cache (only used on first call)

    Returns:
        The global ModelLoaderRegistry
    """
    global _registry
    if _registry is None:
        _registry = ModelLoaderRegistry(ttl=ttl, maxsize=maxsize)
    return _registry


def reset_model_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _registry
    _registry = None
