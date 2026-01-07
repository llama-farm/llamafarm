"""TTL-based model cache using cachetools.

Provides a cache that:
- Automatically tracks last access time
- Refreshes TTL on access (not just on write)
- Supports async cleanup callbacks before expiration
"""

import time
from collections.abc import Iterator
from typing import Generic, TypeVar

from cachetools import TTLCache

T = TypeVar("T")


class ModelCache(Generic[T]):
    """TTL-based cache for models with async cleanup support.

    Uses cachetools.TTLCache internally but refreshes TTL on read access
    (not just write), and provides methods for async cleanup before items
    expire.

    This is designed for ML model caching where:
    - Models should stay loaded while being actively used
    - Idle models should be unloaded after a timeout
    - Unloading requires calling an async cleanup method

    Example:
        cache = ModelCache[BaseModel](ttl=300)  # 5 minute TTL

        # Set a model
        cache["encoder:model-id"] = model

        # Get model (refreshes TTL)
        model = cache.get("encoder:model-id")

        # In cleanup task:
        for key, model in cache.pop_expired():
            await model.unload()
    """

    def __init__(self, ttl: float, maxsize: int = 1000):
        """Initialize the cache.

        Args:
            ttl: Time-to-live in seconds. Items are considered expired
                after this many seconds of inactivity (no read or write).
            maxsize: Maximum number of items to store.
        """
        self._ttl = ttl
        self._maxsize = maxsize
        # Internal TTLCache with very long TTL - we manage expiry ourselves
        # to support async callbacks before removal
        self._cache: TTLCache[str, T] = TTLCache(maxsize=maxsize, ttl=ttl * 10)
        # Track access times ourselves for TTL-on-read behavior
        self._timer = time.monotonic
        self._access: dict[str, float] = {}

    @property
    def ttl(self) -> float:
        """Get the TTL in seconds."""
        return self._ttl

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __iter__(self) -> Iterator[str]:
        return iter(self._cache)

    def get(self, key: str, default: T | None = None) -> T | None:
        """Get item and refresh its TTL.

        Args:
            key: Cache key
            default: Value to return if key not found

        Returns:
            The cached item, or default if not found
        """
        if key not in self._cache:
            return default
        self._access[key] = self._timer()
        return self._cache[key]

    def __getitem__(self, key: str) -> T:
        """Get item and refresh TTL. Raises KeyError if not found."""
        if key not in self._cache:
            raise KeyError(key)
        self._access[key] = self._timer()
        return self._cache[key]

    def __setitem__(self, key: str, value: T) -> None:
        """Set item with fresh TTL."""
        self._cache[key] = value
        self._access[key] = self._timer()

    def __delitem__(self, key: str) -> None:
        """Remove item from cache."""
        del self._cache[key]
        self._access.pop(key, None)

    def pop(self, key: str, *args) -> T:
        """Remove and return item.

        Args:
            key: Cache key
            *args: Optional default value

        Returns:
            The removed item, or default if provided and key not found
        """
        self._access.pop(key, None)
        return self._cache.pop(key, *args)

    def keys(self):
        """Return view of cache keys."""
        return self._cache.keys()

    def values(self):
        """Return view of cache values."""
        return self._cache.values()

    def items(self):
        """Return view of cache items."""
        return self._cache.items()

    def clear(self) -> None:
        """Clear all items from cache."""
        self._cache.clear()
        self._access.clear()

    def get_idle_time(self, key: str) -> float | None:
        """Get seconds since last access for a key.

        Args:
            key: Cache key

        Returns:
            Seconds since last access, or None if key not found
        """
        if key not in self._access:
            return None
        return self._timer() - self._access[key]

    def is_expired(self, key: str) -> bool:
        """Check if an item has exceeded its TTL.

        Args:
            key: Cache key

        Returns:
            True if item exists and is expired, False otherwise
        """
        idle_time = self.get_idle_time(key)
        return idle_time is not None and idle_time > self._ttl

    def get_expired_keys(self) -> list[str]:
        """Get list of keys that have exceeded their TTL.

        Returns:
            List of expired cache keys
        """
        now = self._timer()
        cutoff = now - self._ttl
        return [k for k, t in self._access.items() if t < cutoff]

    def pop_expired(self) -> list[tuple[str, T]]:
        """Remove and return all expired items.

        This is the main method for cleanup tasks. It returns all expired
        items so the caller can perform async cleanup (like calling unload()).

        Returns:
            List of (key, value) tuples for expired items
        """
        expired_keys = self.get_expired_keys()
        result = []
        for key in expired_keys:
            if key in self._cache:
                value = self._cache.pop(key)
                self._access.pop(key, None)
                result.append((key, value))
        return result
