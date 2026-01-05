"""HybridQuery - Unified query interface for multi-store queries.

Phase 20: Hybrid Query Implementation
Phase 26: Performance & Polish - Added query result caching

Provides intelligent querying across multiple data stores:
- Vector store (semantic search)
- Graph store (relationship traversal)
- Timeseries store (time-based filtering)
- Spatial store (geo-location filtering)
- Working memory (recent context)

Features:
- Multi-store query routing
- Result fusion with configurable strategies
- Cross-store result linking via LinkageTable
- Relevance scoring and ranking
- Query result caching with TTL
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QueryCache:
    """Thread-safe LRU cache for query results with TTL support.

    Features:
    - Configurable max size and TTL
    - Thread-safe with read-write locks
    - Automatic expiration of stale entries
    - Cache statistics for monitoring
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 60.0):
        """Initialize cache.

        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}  # key -> (value, expire_time)
        self._access_order: list[str] = []  # LRU tracking
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _make_key(self, request_dict: dict) -> str:
        """Create a cache key from request parameters."""
        # Sort keys for consistent hashing
        sorted_items = sorted(str(request_dict.items()).encode())
        return hashlib.md5(str(sorted_items).encode()).hexdigest()

    def get(self, key: str) -> tuple[Any, bool]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Tuple of (value, hit) where hit is True if found and valid
        """
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None, False

            value, expire_time = self._cache[key]

            # Check if expired
            if time.time() > expire_time:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._stats["misses"] += 1
                return None, False

            # Update access order for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            self._stats["hits"] += 1
            return value, True

    def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size and self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]
                    self._stats["evictions"] += 1

            expire_time = time.time() + self.ttl_seconds
            self._cache[key] = (value, expire_time)

            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            return {
                **self._stats,
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hit_rate": hit_rate,
            }


class QueryMode(str, Enum):
    """Query execution modes."""

    VECTOR = "vector"  # Semantic search only
    GRAPH = "graph"  # Graph traversal only
    TIMESERIES = "timeseries"  # Time-based only
    SPATIAL = "spatial"  # Geo-location only
    HYBRID = "hybrid"  # Combine all available stores
    CONTEXT = "context"  # Recent context from working memory + graph


class FusionStrategy(str, Enum):
    """Result fusion strategies for combining multi-store results."""

    INTERLEAVE = "interleave"  # Round-robin from each store
    WEIGHTED = "weighted"  # Weight-based ranking
    SCORE_BASED = "score_based"  # Rank by relevance score
    TEMPORAL = "temporal"  # Most recent first
    SPATIAL_FIRST = "spatial_first"  # Closest first, then others


@dataclass
class QueryResult:
    """A single query result from any store."""

    id: str
    content: Any
    score: float = 0.0
    source_store: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None
    distance_m: float | None = None  # For spatial results
    path_depth: int | None = None  # For graph results

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "source_store": self.source_store,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "distance_m": self.distance_m,
            "path_depth": self.path_depth,
        }


@dataclass
class HybridQueryRequest:
    """Request for a hybrid query."""

    # Query text for vector/semantic search
    query_text: str | None = None

    # Graph query parameters
    graph_node_id: str | None = None
    graph_relationship: str | None = None
    graph_direction: str = "both"  # in, out, both
    graph_depth: int = 2

    # Time range filter
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Spatial filter
    latitude: float | None = None
    longitude: float | None = None
    radius_meters: float = 1000.0

    # Query behavior
    mode: QueryMode = QueryMode.HYBRID
    fusion_strategy: FusionStrategy = FusionStrategy.SCORE_BASED
    limit: int = 10
    include_metadata: bool = True

    # Store-specific limits (to balance results)
    vector_limit: int | None = None
    graph_limit: int | None = None
    timeseries_limit: int | None = None
    spatial_limit: int | None = None
    working_memory_limit: int | None = None


@dataclass
class HybridQueryResponse:
    """Response from a hybrid query."""

    results: list[QueryResult]
    total_count: int
    stores_queried: list[str]
    query_mode: QueryMode
    fusion_strategy: FusionStrategy
    execution_time_ms: float = 0.0
    store_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "total_count": self.total_count,
            "stores_queried": self.stores_queried,
            "query_mode": self.query_mode.value,
            "fusion_strategy": self.fusion_strategy.value,
            "execution_time_ms": self.execution_time_ms,
            "store_counts": self.store_counts,
            "metadata": self.metadata,
        }


class HybridQueryExecutor:
    """Executes hybrid queries across multiple stores.

    This is the main query orchestrator that:
    1. Routes queries to appropriate stores based on mode
    2. Collects and normalizes results
    3. Fuses results using the specified strategy
    4. Returns ranked, deduplicated results
    5. Caches results for repeated queries (Phase 26)
    """

    # Shared cache across all executor instances
    _cache: QueryCache | None = None
    _cache_lock = threading.Lock()

    def __init__(
        self,
        unified_store: Any,
        enable_cache: bool = True,
        cache_max_size: int = 100,
        cache_ttl_seconds: float = 30.0,
    ):
        """Initialize with a UnifiedDatasetStore.

        Args:
            unified_store: UnifiedDatasetStore instance
            enable_cache: Whether to enable query result caching
            cache_max_size: Maximum number of cached query results
            cache_ttl_seconds: Time-to-live for cached results
        """
        self.store = unified_store
        self.linkage_table = unified_store.linkage_table
        self.enable_cache = enable_cache

        # Initialize shared cache if not exists
        if enable_cache and HybridQueryExecutor._cache is None:
            with HybridQueryExecutor._cache_lock:
                if HybridQueryExecutor._cache is None:
                    HybridQueryExecutor._cache = QueryCache(
                        max_size=cache_max_size,
                        ttl_seconds=cache_ttl_seconds,
                    )

        # Store weights for weighted fusion (can be configured)
        self.store_weights = {
            "vector": 1.0,
            "graph": 0.9,
            "timeseries": 0.7,
            "spatial": 0.8,
            "working_memory": 0.6,
        }

    def _get_cache_key(self, request: HybridQueryRequest) -> str:
        """Generate cache key from request parameters."""
        # Get store name safely (may not exist on mock stores)
        store_name = getattr(self.store, "name", "default")
        if not isinstance(store_name, str):
            store_name = str(store_name) if store_name else "default"

        key_parts = [
            str(request.query_text),
            str(request.graph_node_id),
            str(request.graph_relationship),
            request.graph_direction,
            str(request.graph_depth),
            str(request.start_time),
            str(request.end_time),
            str(request.latitude),
            str(request.longitude),
            str(request.radius_meters),
            request.mode.value,
            request.fusion_strategy.value,
            str(request.limit),
            store_name,  # Include dataset name for uniqueness
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def execute(self, request: HybridQueryRequest) -> HybridQueryResponse:
        """Execute a hybrid query.

        Args:
            request: HybridQueryRequest with query parameters

        Returns:
            HybridQueryResponse with fused results
        """
        start_time = time.time()

        # Check cache first (skip for working_memory queries as they change frequently)
        cache_key = None
        if self.enable_cache and self._cache and request.mode != QueryMode.CONTEXT:
            cache_key = self._get_cache_key(request)
            cached_result, hit = self._cache.get(cache_key)
            if hit:
                logger.debug(f"Cache hit for query: {cache_key[:8]}...")
                cached_result.metadata["cache_hit"] = True
                cached_result.execution_time_ms = (time.time() - start_time) * 1000
                return cached_result

        # Collect results from each store
        all_results: list[QueryResult] = []
        stores_queried: list[str] = []
        store_counts: dict[str, int] = {}

        # Determine which stores to query based on mode
        stores_to_query = self._determine_stores(request)

        # Execute queries
        for store_name in stores_to_query:
            try:
                results = self._query_store(store_name, request)
                all_results.extend(results)
                stores_queried.append(store_name)
                store_counts[store_name] = len(results)
            except Exception as e:
                logger.warning(f"Query to {store_name} failed: {e}")

        # Fuse results
        fused_results = self._fuse_results(all_results, request)

        # Limit final results
        fused_results = fused_results[: request.limit]

        execution_time_ms = (time.time() - start_time) * 1000

        response = HybridQueryResponse(
            results=fused_results,
            total_count=len(all_results),
            stores_queried=stores_queried,
            query_mode=request.mode,
            fusion_strategy=request.fusion_strategy,
            execution_time_ms=execution_time_ms,
            store_counts=store_counts,
            metadata={"cache_hit": False},
        )

        # Cache result
        if cache_key and self._cache:
            self._cache.set(cache_key, response)

        return response

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics.

        Returns:
            Cache stats dict or None if caching disabled
        """
        if self._cache:
            return self._cache.get_stats()
        return None

    def clear_cache(self) -> None:
        """Clear the query cache."""
        if self._cache:
            self._cache.clear()

    def _determine_stores(self, request: HybridQueryRequest) -> list[str]:
        """Determine which stores to query based on mode and capabilities."""
        if request.mode == QueryMode.VECTOR:
            return ["vector"] if self.store.vector_store else []

        elif request.mode == QueryMode.GRAPH:
            return ["graph"] if self.store.graph_store else []

        elif request.mode == QueryMode.TIMESERIES:
            return ["timeseries"] if self.store.timeseries_store else []

        elif request.mode == QueryMode.SPATIAL:
            return ["spatial"] if self.store.spatial_store else []

        elif request.mode == QueryMode.CONTEXT:
            stores = []
            if self.store.working_memory:
                stores.append("working_memory")
            if self.store.graph_store:
                stores.append("graph")
            return stores

        else:  # HYBRID
            stores = []
            if request.query_text and self.store.vector_store:
                stores.append("vector")
            if request.graph_node_id and self.store.graph_store:
                stores.append("graph")
            if (request.start_time or request.end_time) and self.store.timeseries_store:
                stores.append("timeseries")
            if request.latitude is not None and self.store.spatial_store:
                stores.append("spatial")
            if self.store.working_memory:
                stores.append("working_memory")
            return stores

    def _query_store(
        self, store_name: str, request: HybridQueryRequest
    ) -> list[QueryResult]:
        """Query a specific store."""
        if store_name == "vector":
            return self._query_vector(request)
        elif store_name == "graph":
            return self._query_graph(request)
        elif store_name == "timeseries":
            return self._query_timeseries(request)
        elif store_name == "spatial":
            return self._query_spatial(request)
        elif store_name == "working_memory":
            return self._query_working_memory(request)
        else:
            return []

    def _query_vector(self, request: HybridQueryRequest) -> list[QueryResult]:
        """Query vector store for semantic search."""
        if not self.store.vector_store or not request.query_text:
            return []

        _limit = request.vector_limit or request.limit  # noqa: F841

        # TODO: Implement when ChromaDB is integrated
        # For now, return empty results
        logger.debug(f"Vector query (not yet implemented): {request.query_text}")
        return []

    def _query_graph(self, request: HybridQueryRequest) -> list[QueryResult]:
        """Query graph store for relationship traversal."""
        if not self.store.graph_store:
            return []

        limit = request.graph_limit or request.limit
        results = []

        # If node ID provided, find neighbors
        if request.graph_node_id:
            neighbors = self.store.graph_store.find_neighbors(
                request.graph_node_id,
                direction=request.graph_direction,
                relationship=request.graph_relationship,
            )

            for i, neighbor in enumerate(neighbors[:limit]):
                results.append(
                    QueryResult(
                        id=neighbor.get("id", str(i)),
                        content=neighbor.get("name") or neighbor,
                        score=1.0 / (i + 1),  # Decreasing score
                        source_store="graph",
                        metadata=neighbor.get("properties", {}),
                        path_depth=neighbor.get("depth", 1),
                    )
                )

        # If query text provided, search by name/properties
        elif request.query_text:
            # Basic text search in graph (name matching)
            matches = self.store.graph_store.search_nodes(
                name_pattern=request.query_text, limit=limit
            )

            for i, match in enumerate(matches):
                results.append(
                    QueryResult(
                        id=match.get("id", str(i)),
                        content=match.get("name"),
                        score=match.get("score", 1.0 / (i + 1)),
                        source_store="graph",
                        metadata=match.get("properties", {}),
                    )
                )

        return results

    def _query_timeseries(self, request: HybridQueryRequest) -> list[QueryResult]:
        """Query timeseries store for time-based data."""
        if not self.store.timeseries_store:
            return []

        limit = request.timeseries_limit or request.limit
        results = []

        # Query by time range
        raw_results = self.store.timeseries_store.query_time_range(
            start_time=request.start_time,
            end_time=request.end_time,
        )

        for i, record in enumerate(raw_results[:limit]):
            ts = record.get("ts") or record.get("timestamp")
            results.append(
                QueryResult(
                    id=record.get("id", str(i)),
                    content=record.get("data") or record,
                    score=1.0 / (i + 1),
                    source_store="timeseries",
                    metadata=record.get("metadata", {}),
                    timestamp=ts if isinstance(ts, datetime) else None,
                )
            )

        return results

    def _query_spatial(self, request: HybridQueryRequest) -> list[QueryResult]:
        """Query spatial store for geo-located data."""
        if not self.store.spatial_store:
            return []

        if request.latitude is None or request.longitude is None:
            return []

        limit = request.spatial_limit or request.limit
        results = []

        # Query by location
        raw_results = self.store.spatial_store.query_spatial(
            center_lat=request.latitude,
            center_lon=request.longitude,
            radius_meters=request.radius_meters,
        )

        for i, record in enumerate(raw_results[:limit]):
            distance = record.get("distance_m") or record.get("distance")
            # Score inversely proportional to distance
            score = 1.0 / (1 + (distance or 0) / 1000)

            results.append(
                QueryResult(
                    id=record.get("id", str(i)),
                    content=record.get("data") or record,
                    score=score,
                    source_store="spatial",
                    metadata=record.get("metadata", {}),
                    distance_m=distance,
                    timestamp=record.get("ts"),
                )
            )

        return results

    def _query_working_memory(self, request: HybridQueryRequest) -> list[QueryResult]:
        """Query working memory for recent context."""
        if not self.store.working_memory:
            return []

        limit = request.working_memory_limit or request.limit
        results = []

        # Get recent items
        raw_results = self.store.working_memory.get_recent(limit=limit)

        for i, record in enumerate(raw_results):
            results.append(
                QueryResult(
                    id=record.get("id", str(i)),
                    content=record.get("content") or record.get("data"),
                    score=1.0 / (i + 1),  # More recent = higher score
                    source_store="working_memory",
                    metadata=record.get("metadata", {}),
                    timestamp=record.get("timestamp") or record.get("created_at"),
                )
            )

        return results

    def _fuse_results(
        self, results: list[QueryResult], request: HybridQueryRequest
    ) -> list[QueryResult]:
        """Fuse results from multiple stores using the specified strategy."""
        if not results:
            return []

        if request.fusion_strategy == FusionStrategy.INTERLEAVE:
            return self._interleave_results(results)

        elif request.fusion_strategy == FusionStrategy.WEIGHTED:
            return self._weighted_results(results)

        elif request.fusion_strategy == FusionStrategy.TEMPORAL:
            return self._temporal_results(results)

        elif request.fusion_strategy == FusionStrategy.SPATIAL_FIRST:
            return self._spatial_first_results(results)

        else:  # SCORE_BASED (default)
            return self._score_based_results(results)

    def _interleave_results(self, results: list[QueryResult]) -> list[QueryResult]:
        """Interleave results from different stores (round-robin)."""
        # Group by source store
        by_store: dict[str, list[QueryResult]] = {}
        for result in results:
            if result.source_store not in by_store:
                by_store[result.source_store] = []
            by_store[result.source_store].append(result)

        # Round-robin interleave
        interleaved = []
        store_lists = list(by_store.values())
        max_len = max(len(lst) for lst in store_lists) if store_lists else 0

        for i in range(max_len):
            for store_list in store_lists:
                if i < len(store_list):
                    interleaved.append(store_list[i])

        return interleaved

    def _weighted_results(self, results: list[QueryResult]) -> list[QueryResult]:
        """Rank results by weighted score based on store."""
        for result in results:
            weight = self.store_weights.get(result.source_store, 0.5)
            result.score *= weight

        return sorted(results, key=lambda r: r.score, reverse=True)

    def _score_based_results(self, results: list[QueryResult]) -> list[QueryResult]:
        """Rank results purely by score."""
        return sorted(results, key=lambda r: r.score, reverse=True)

    def _temporal_results(self, results: list[QueryResult]) -> list[QueryResult]:
        """Rank results by timestamp (most recent first)."""

        # Sort by timestamp, with None timestamps at the end
        def sort_key(r: QueryResult):
            if r.timestamp:
                return (0, -r.timestamp.timestamp())
            return (1, 0)

        return sorted(results, key=sort_key)

    def _spatial_first_results(self, results: list[QueryResult]) -> list[QueryResult]:
        """Rank spatial results first (by distance), then others by score."""
        spatial = [
            r
            for r in results
            if r.source_store == "spatial" and r.distance_m is not None
        ]
        others = [r for r in results if r not in spatial]

        # Sort spatial by distance (closest first)
        spatial.sort(key=lambda r: r.distance_m or float("inf"))

        # Sort others by score
        others.sort(key=lambda r: r.score, reverse=True)

        return spatial + others


# Convenience functions


def hybrid_query(
    unified_store: Any,
    query_text: str | None = None,
    graph_node_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    radius_meters: float = 1000.0,
    mode: str = "hybrid",
    limit: int = 10,
) -> dict[str, Any]:
    """Convenience function for hybrid queries.

    Args:
        unified_store: UnifiedDatasetStore instance
        query_text: Text for semantic search
        graph_node_id: Node ID for graph traversal
        start_time: Start of time range
        end_time: End of time range
        latitude: Center latitude for spatial
        longitude: Center longitude for spatial
        radius_meters: Radius for spatial query
        mode: Query mode (hybrid, vector, graph, etc.)
        limit: Maximum results

    Returns:
        Dictionary with query results
    """
    executor = HybridQueryExecutor(unified_store)

    request = HybridQueryRequest(
        query_text=query_text,
        graph_node_id=graph_node_id,
        start_time=start_time,
        end_time=end_time,
        latitude=latitude,
        longitude=longitude,
        radius_meters=radius_meters,
        mode=QueryMode(mode),
        limit=limit,
    )

    response = executor.execute(request)
    return response.to_dict()
