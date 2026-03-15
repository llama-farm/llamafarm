"""TTL-based model cache using cachetools with pin support and LRU eviction.

Provides a cache that:
- Automatically tracks last access time
- Refreshes TTL on access (not just on write)
- Supports async cleanup callbacks before expiration
- Supports pinning models to prevent eviction
- Implements LRU eviction when cache is full (respects pinned models)
"""

import time
from collections.abc import Iterator
from typing import Generic, TypeVar

from cachetools import TTLCache

T = TypeVar("T")


class ModelCache(Generic[T]):
    """TTL-based cache for models with async cleanup support and pinning.

    Uses cachetools.TTLCache internally but refreshes TTL on read access
    (not just write), and provides methods for async cleanup before items
    expire. Additionally supports pinning models to prevent eviction.

    This is designed for ML model caching where:
    - Models should stay loaded while being actively used
    - Idle models should be unloaded after a timeout
    - Unloading requires calling an async cleanup method
    - Critical models can be pinned to never be evicted

    Example:
        cache = ModelCache[BaseModel](ttl=300, maxsize=10)  # 5 minute TTL, max 10 models

        # Set a model
        cache["encoder:model-id"] = model

        # Get model (refreshes TTL)
        model = cache.get("encoder:model-id")

        # Pin a model (prevents eviction)
        cache.pin("encoder:model-id")

        # Unpin a model
        cache.unpin("encoder:model-id")

        # In cleanup task:
        for key, model in cache.pop_expired():
            await model.unload()
    """

    def __init__(self, ttl: float, maxsize: int = 1000):
        """Initialize the cache.

        Args:
            ttl: Time-to-live in seconds. Items are considered expired
                after this many seconds of inactivity (no read or write).
                Pinned items ignore TTL and are never expired.
            maxsize: Maximum number of items to store. When full, least
                recently used non-pinned items are evicted.
        """
        self._ttl = ttl
        self._maxsize = maxsize
        # Internal TTLCache with very long TTL - we manage expiry ourselves
        # to support async callbacks before removal
        self._cache: TTLCache[str, T] = TTLCache(maxsize=maxsize, ttl=ttl * 10)
        # Track access times ourselves for TTL-on-read behavior
        self._timer = time.monotonic
        self._access: dict[str, float] = {}
        # Track pinned models (never evict, ignore TTL)
        self._pinned: set[str] = set()

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
        """Get item and refresh its TTL (unless pinned).

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

    def peek(self, key: str, default: T | None = None) -> T | None:
        """Get item WITHOUT refreshing its TTL or access timestamp.

        Use this for read-only inspection (e.g. status endpoints) so that
        polling does not keep models alive or skew idle_time_seconds.

        Args:
            key: Cache key
            default: Value to return if key not found

        Returns:
            The cached item, or default if not found
        """
        if key not in self._cache:
            return default
        return self._cache[key]

    def __getitem__(self, key: str) -> T:
        """Get item and refresh TTL. Raises KeyError if not found."""
        if key not in self._cache:
            raise KeyError(key)
        self._access[key] = self._timer()
        return self._cache[key]

    def __setitem__(self, key: str, value: T) -> None:
        """Set item with fresh TTL.

        If cache is at capacity, evicts least recently used non-pinned item.
        If all items are pinned, raises ValueError.
        """
        # Check if we need to evict before adding
        if key not in self._cache and len(self._cache) >= self._maxsize:
            evicted = self._evict_lru()
            if not evicted:
                raise ValueError(
                    f"Cache full ({self._maxsize} items) and all items are pinned. "
                    "Cannot add new item without evicting."
                )

        self._cache[key] = value
        self._access[key] = self._timer()

    def __delitem__(self, key: str) -> None:
        """Remove item from cache."""
        del self._cache[key]
        self._access.pop(key, None)
        self._pinned.discard(key)

    def pop(self, key: str, *args) -> T:
        """Remove and return item.

        Args:
            key: Cache key
            *args: Optional default value

        Returns:
            The removed item, or default if provided and key not found
        """
        self._access.pop(key, None)
        self._pinned.discard(key)
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
        self._pinned.clear()

    def pin(self, key: str) -> None:
        """Pin a model to prevent eviction.

        Pinned models:
        - Are never expired (ignore TTL)
        - Are never evicted when cache is full

        Args:
            key: Cache key to pin

        Raises:
            KeyError: If key not in cache
        """
        if key not in self._cache:
            raise KeyError(f"Cannot pin non-existent key: {key}")
        self._pinned.add(key)

    def unpin(self, key: str) -> None:
        """Unpin a model, allowing it to be evicted/expired normally.

        Args:
            key: Cache key to unpin
        """
        self._pinned.discard(key)

    def is_pinned(self, key: str) -> bool:
        """Check if a key is pinned.

        Args:
            key: Cache key

        Returns:
            True if key is pinned, False otherwise
        """
        return key in self._pinned

    def get_pinned_keys(self) -> set[str]:
        """Get set of all pinned keys.

        Returns:
            Set of pinned cache keys
        """
        return self._pinned.copy()

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

        Pinned items are never considered expired.

        Args:
            key: Cache key

        Returns:
            True if item exists is not pinned and is expired, False otherwise
        """
        if key in self._pinned:
            return False
        idle_time = self.get_idle_time(key)
        return idle_time is not None and idle_time > self._ttl

    def get_expired_keys(self) -> list[str]:
        """Get list of keys that have exceeded their TTL.

        Pinned items are never included in expired keys.

        Returns:
            List of expired cache keys (excludes pinned items)
        """
        now = self._timer()
        cutoff = now - self._ttl
        return [
            k for k, t in self._access.items() if t < cutoff and k not in self._pinned
        ]

    def pop_expired(self) -> list[tuple[str, T]]:
        """Remove and return all expired items.

        This is the main method for cleanup tasks. It returns all expired
        items so the caller can perform async cleanup (like calling unload()).

        Pinned items are never expired.

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

    def _evict_lru(self) -> bool:
        """Evict the least recently used non-pinned item.

        Called automatically when cache is full and a new item is added.

        Returns:
            True if an item was evicted, False if no evictable items (all pinned)
        """
        # Find least recently used non-pinned item
        lru_key = None
        lru_time = float("inf")

        for key, access_time in self._access.items():
            if key not in self._pinned and access_time < lru_time:
                lru_key = key
                lru_time = access_time

        if lru_key is None:
            # All items are pinned
            return False

        # Evict the LRU item
        del self[lru_key]
        return True

    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring.

        Returns:
            Dictionary with cache statistics including:
            - total_items: Total number of cached items
            - pinned_items: Number of pinned items
            - evictable_items: Number of items that can be evicted
            - expired_items: Number of items past TTL (excludes pinned)
            - cache_full: Whether cache is at capacity
        """
        total = len(self._cache)
        pinned = len(self._pinned)
        expired = len(self.get_expired_keys())

        return {
            "total_items": total,
            "pinned_items": pinned,
            "evictable_items": total - pinned,
            "expired_items": expired,
            "cache_full": total >= self._maxsize,
            "max_size": self._maxsize,
            "ttl_seconds": self._ttl,
        }
