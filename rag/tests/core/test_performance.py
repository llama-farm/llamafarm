"""Tests for Performance & Polish features (Phase 26).

Tests for:
- QueryCache with TTL and LRU eviction
- HybridQueryExecutor caching
- ConnectionPool for DuckDB
- Batch insert optimizations
"""

import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add rag to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.hybrid_query import (
    HybridQueryExecutor,
    HybridQueryRequest,
    QueryCache,
    QueryMode,
)


class TestQueryCache:
    """Test QueryCache class."""

    def test_cache_basic_operations(self):
        """Test basic get/set operations."""
        cache = QueryCache(max_size=10, ttl_seconds=60)

        # Set value
        cache.set("key1", "value1")

        # Get value
        value, hit = cache.get("key1")
        assert hit is True
        assert value == "value1"

    def test_cache_miss(self):
        """Test cache miss on unknown key."""
        cache = QueryCache(max_size=10, ttl_seconds=60)

        value, hit = cache.get("unknown")
        assert hit is False
        assert value is None

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = QueryCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL

        cache.set("key1", "value1")

        # Should hit immediately
        value, hit = cache.get("key1")
        assert hit is True

        # Wait for TTL
        time.sleep(0.15)

        # Should miss after TTL
        value, hit = cache.get("key1")
        assert hit is False

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = QueryCache(max_size=3, ttl_seconds=60)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (oldest)
        cache.set("key4", "value4")

        # key1 should still be present
        _, hit = cache.get("key1")
        assert hit is True

        # key2 should be evicted
        _, hit = cache.get("key2")
        assert hit is False

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = QueryCache(max_size=10, ttl_seconds=60)

        # Generate some hits and misses
        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        cache.get("key1")  # hit

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 2 / 3

    def test_cache_clear(self):
        """Test cache clear."""
        cache = QueryCache(max_size=10, ttl_seconds=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        _, hit = cache.get("key1")
        assert hit is False

        stats = cache.get_stats()
        assert stats["size"] == 0

    def test_cache_thread_safety(self):
        """Test cache is thread-safe."""
        cache = QueryCache(max_size=100, ttl_seconds=60)
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    key = f"thread_{thread_id}_key_{i}"
                    cache.set(key, f"value_{i}")
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestHybridQueryExecutorCaching:
    """Test HybridQueryExecutor caching functionality."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock UnifiedDatasetStore."""
        store = MagicMock()
        store.name = "test_dataset"
        store.vector_store = None
        store.graph_store = MagicMock()
        store.graph_store.find_neighbors.return_value = [
            {"id": "node1", "name": "Test Node", "properties": {}},
        ]
        store.timeseries_store = None
        store.spatial_store = None
        store.working_memory = MagicMock()
        store.working_memory.get_recent.return_value = []
        store.linkage_table = MagicMock()
        return store

    def test_executor_with_cache_enabled(self, mock_store):
        """Test executor with caching enabled."""
        executor = HybridQueryExecutor(
            mock_store,
            enable_cache=True,
            cache_max_size=10,
            cache_ttl_seconds=60,
        )

        request = HybridQueryRequest(
            graph_node_id="test:node",
            mode=QueryMode.GRAPH,
            limit=5,
        )

        # First execution should not be cached
        response1 = executor.execute(request)
        assert response1.metadata.get("cache_hit") is False

        # Second execution should be cached
        response2 = executor.execute(request)
        assert response2.metadata.get("cache_hit") is True

    def test_executor_with_cache_disabled(self, mock_store):
        """Test executor with caching disabled."""
        executor = HybridQueryExecutor(
            mock_store,
            enable_cache=False,
        )

        request = HybridQueryRequest(
            graph_node_id="test:node",
            mode=QueryMode.GRAPH,
            limit=5,
        )

        response1 = executor.execute(request)
        response2 = executor.execute(request)

        # Neither should be cached
        assert response1.metadata.get("cache_hit") is False
        assert response2.metadata.get("cache_hit") is False

    def test_context_mode_bypasses_cache(self, mock_store):
        """Test that CONTEXT mode bypasses cache."""
        executor = HybridQueryExecutor(
            mock_store,
            enable_cache=True,
        )

        request = HybridQueryRequest(
            mode=QueryMode.CONTEXT,
            limit=5,
        )

        response1 = executor.execute(request)
        response2 = executor.execute(request)

        # Context mode should never use cache
        assert response1.metadata.get("cache_hit") is False
        assert response2.metadata.get("cache_hit") is False

    def test_cache_stats(self, mock_store):
        """Test getting cache statistics."""
        executor = HybridQueryExecutor(
            mock_store,
            enable_cache=True,
        )

        request = HybridQueryRequest(
            graph_node_id="test:node",
            mode=QueryMode.GRAPH,
        )

        executor.execute(request)  # miss
        executor.execute(request)  # hit

        stats = executor.get_cache_stats()
        assert stats is not None
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1

    def test_clear_cache(self, mock_store):
        """Test clearing the cache."""
        executor = HybridQueryExecutor(
            mock_store,
            enable_cache=True,
        )

        request = HybridQueryRequest(
            graph_node_id="test:node",
            mode=QueryMode.GRAPH,
        )

        executor.execute(request)
        executor.clear_cache()

        # Should not be cached after clear
        response = executor.execute(request)
        assert response.metadata.get("cache_hit") is False


class TestConnectionPool:
    """Test DuckDB ConnectionPool."""

    def test_pool_initialization(self):
        """Test connection pool initialization."""
        from components.stores.duckdb_store import ConnectionPool

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = f"{temp_dir}/test.duckdb"
            pool = ConnectionPool(db_path=db_path, pool_size=3)

            try:
                stats = pool.get_stats()
                assert stats["pool_size"] == 3
                assert stats["created_count"] >= 1  # Pre-created connections
                assert stats["closed"] is False
            finally:
                pool.close()

    def test_pool_get_connection(self):
        """Test getting connection from pool."""
        from components.stores.duckdb_store import ConnectionPool

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = f"{temp_dir}/test.duckdb"
            pool = ConnectionPool(db_path=db_path, pool_size=3)

            try:
                with pool.get_connection() as conn:
                    result = conn.execute("SELECT 1 as value").fetchone()
                    assert result[0] == 1
            finally:
                pool.close()

    def test_pool_connection_reuse(self):
        """Test that connections are reused."""
        from components.stores.duckdb_store import ConnectionPool

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = f"{temp_dir}/test.duckdb"
            pool = ConnectionPool(db_path=db_path, pool_size=3)

            try:
                # Get and return connection multiple times
                for _ in range(5):
                    with pool.get_connection() as conn:
                        conn.execute("SELECT 1")

                stats = pool.get_stats()
                # Should not have created more than pool_size
                assert stats["created_count"] <= stats["pool_size"]
            finally:
                pool.close()

    def test_pool_close(self):
        """Test closing pool."""
        from components.stores.duckdb_store import ConnectionPool

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = f"{temp_dir}/test.duckdb"
            pool = ConnectionPool(db_path=db_path, pool_size=3)

            pool.close()

            stats = pool.get_stats()
            assert stats["closed"] is True

            # Should raise on get after close
            with pytest.raises(RuntimeError, match="closed"), pool.get_connection():
                pass


class TestBatchInsert:
    """Test batch insert optimizations."""

    def test_batch_insert_performance(self):
        """Test that batch insert handles large record sets."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = DuckDBStore(
                config={
                    "path": f"{temp_dir}/test.duckdb",
                    "batch_size": 100,  # Small batch for testing
                }
            )

            try:
                # Create 250 records (should trigger batch insert)
                records = [
                    {
                        "source": f"sensor_{i % 10}",
                        "ts": datetime.now(),
                        "data": {"value": i},
                        "metadata": {"batch": True},
                    }
                    for i in range(250)
                ]

                inserted = store.add_records(records)
                assert inserted == 250

                stats = store.get_stats()
                assert stats["record_count"] == 250

            finally:
                store.close()

    def test_small_insert_uses_regular_method(self):
        """Test that small inserts don't use batch method."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = DuckDBStore(
                config={
                    "path": f"{temp_dir}/test.duckdb",
                    "batch_size": 1000,
                }
            )

            try:
                records = [{"source": "test", "data": {"value": i}} for i in range(10)]

                inserted = store.add_records(records)
                assert inserted == 10

            finally:
                store.close()


class TestDuckDBStoreWithPool:
    """Test DuckDB store with connection pooling."""

    def test_store_with_pool_enabled(self):
        """Test store operations with pooling enabled."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = DuckDBStore(
                config={
                    "path": f"{temp_dir}/test.duckdb",
                    "use_pool": True,
                    "pool_size": 3,
                }
            )

            try:
                # Basic operations should work
                records = [
                    {"source": "test", "data": {"value": 1}},
                    {"source": "test", "data": {"value": 2}},
                ]
                inserted = store.add_records(records)
                assert inserted == 2

                stats = store.get_stats()
                assert stats["record_count"] == 2

            finally:
                store.close()

    def test_store_concurrent_access_with_pool(self):
        """Test concurrent access with connection pooling."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = DuckDBStore(
                config={
                    "path": f"{temp_dir}/test.duckdb",
                    "use_pool": True,
                    "pool_size": 5,
                }
            )

            errors = []

            def worker(thread_id):
                try:
                    for i in range(10):
                        store.add_records(
                            [{"source": f"thread_{thread_id}", "data": {"i": i}}]
                        )
                except Exception as e:
                    errors.append(e)

            try:
                threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                # Should have no errors
                assert len(errors) == 0

                # Should have all records
                stats = store.get_stats()
                assert stats["record_count"] == 30  # 3 threads * 10 records

            finally:
                store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
