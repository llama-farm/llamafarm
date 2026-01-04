"""Tests for MemoryStore - Unified interface for the Embedded Trinity Memory System.

These tests are written FIRST following TDD methodology.
The MemoryStore implementation should make these tests pass.
"""

import tempfile
from datetime import datetime, timedelta


class TestMemoryStoreInitialization:
    """Test MemoryStore initialization."""

    def test_memory_store_initializes_all_stores(self):
        """Test MemoryStore initializes all component stores."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "base_path": temp_dir,
                "vector_store": {"collection_name": "test_collection"},
                "timeseries_store": {"path": f"{temp_dir}/timeseries.duckdb"},
                "graph_store": {"path": f"{temp_dir}/graph.duckdb"},
                "working_memory": {
                    "path": f"{temp_dir}/working.duckdb",
                    "ttl_seconds": 3600,
                },
                "linkage_table": {"path": f"{temp_dir}/linkage.duckdb"},
            }
            store = MemoryStore(config=config)

            # All stores should be initialized
            assert store.is_connected()
            assert store.timeseries_store is not None
            assert store.graph_store is not None
            assert store.working_memory is not None
            assert store.linkage_table is not None
            store.close()

    def test_memory_store_initializes_with_minimal_config(self):
        """Test MemoryStore works with minimal configuration."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            assert store.is_connected()
            store.close()


class TestMemoryStoreAddOperations:
    """Test adding data to MemoryStore."""

    def test_add_telemetry_routes_to_timeseries(self):
        """Test add() routes telemetry to DuckDBStore."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            result = store.add(
                data='{"heart_rate": 72, "blood_oxygen": 98}',
                data_type="telemetry",
                metadata={"device": "watch", "user_id": "test_user"},
                latitude=35.7796,
                longitude=-78.6382,
            )

            assert result is not None
            assert "uuid" in result
            assert result["store"] == "timeseries"
            store.close()

    def test_add_relation_routes_to_graph(self):
        """Test add() routes relations to GraphStore."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            # Add a node
            result = store.add(
                data={"id": "person:alice", "name": "Alice", "role": "engineer"},
                data_type="node",
                metadata={"node_type": "person"},
            )

            assert result is not None
            assert "uuid" in result
            assert result["store"] == "graph"

            # Add an edge
            result = store.add(
                data={
                    "source": "person:alice",
                    "edge_type": "works_with",
                    "target": "person:bob",
                },
                data_type="edge",
            )

            assert result is not None
            assert result["store"] == "graph"
            store.close()

    def test_add_stream_routes_to_working_memory(self):
        """Test add() routes streaming data to WorkingMemory."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            result = store.add(
                data="User asked about weather",
                data_type="chat",
                metadata={"user_id": "test_user"},
            )

            assert result is not None
            assert result["store"] == "working_memory"
            store.close()

    def test_add_creates_linkage_entry(self):
        """Test add() creates linkage entry for cross-store references."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            result = store.add(
                data='{"heart_rate": 72}',
                data_type="telemetry",
                metadata={"source": "biometric_sensor"},
            )

            # Check linkage table has entry
            links = store.linkage_table.get_links(result["uuid"])
            assert links is not None
            assert links["timeseries_row_id"] is not None
            store.close()


class TestMemoryStoreQueryOperations:
    """Test query operations."""

    def test_query_timeseries_by_time_range(self):
        """Test query() retrieves time-series data by time range."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            # Add some telemetry
            now = datetime.now()
            store.add(
                data='{"heart_rate": 72}',
                data_type="telemetry",
                timestamp=now - timedelta(minutes=5),
            )
            store.add(
                data='{"heart_rate": 75}',
                data_type="telemetry",
                timestamp=now - timedelta(minutes=2),
            )

            # Query last 3 minutes
            results = store.query(
                time_range={"start": now - timedelta(minutes=3), "end": now},
                data_types=["telemetry"],
            )

            assert len(results) >= 1
            store.close()

    def test_query_graph_neighbors(self):
        """Test query() retrieves graph relationships."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            # Build a small graph
            store.add(
                data={"id": "person:alice", "name": "Alice"},
                data_type="node",
                metadata={"node_type": "person"},
            )
            store.add(
                data={"id": "person:bob", "name": "Bob"},
                data_type="node",
                metadata={"node_type": "person"},
            )
            store.add(
                data={
                    "source": "person:alice",
                    "edge_type": "knows",
                    "target": "person:bob",
                },
                data_type="edge",
            )

            # Query neighbors
            results = store.query(
                graph_query={"node_id": "person:alice", "direction": "outgoing"},
            )

            assert len(results) >= 1
            store.close()

    def test_query_working_memory_recent(self):
        """Test query() retrieves recent working memory records."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            # Add chat messages
            store.add(data="Hello", data_type="chat")
            store.add(data="How are you?", data_type="chat")

            # Query recent
            results = store.query(
                recent={"limit": 10, "data_type": "chat"},
            )

            assert len(results) >= 2
            store.close()

    def test_query_spatial(self):
        """Test query() retrieves data within spatial radius (if enabled)."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "base_path": temp_dir,
                "timeseries_store": {
                    "path": f"{temp_dir}/timeseries.duckdb",
                    "enable_spatial": True,  # Request spatial extension
                },
            }
            store = MemoryStore(config=config)

            # Add telemetry with location
            store.add(
                data='{"sensor": "s1"}',
                data_type="telemetry",
                latitude=35.7796,
                longitude=-78.6382,
            )

            # Query spatial - may return empty if spatial extension not available
            results = store.query(
                spatial={"latitude": 35.78, "longitude": -78.64, "radius_meters": 5000},
            )

            # If spatial is enabled and extension loaded, should find records
            # Otherwise returns empty (graceful degradation)
            assert isinstance(results, list)
            store.close()


class TestMemoryStoreDeleteOperations:
    """Test delete operations."""

    def test_delete_cascades_to_all_stores(self):
        """Test delete() uses LinkageTable to cascade delete."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            # Add data
            result = store.add(
                data='{"heart_rate": 72}',
                data_type="telemetry",
            )
            concept_uuid = result["uuid"]

            # Verify exists
            links = store.linkage_table.get_links(concept_uuid)
            assert links is not None

            # Delete - now returns dict with deleted_from list or None
            result = store.delete(concept_uuid)
            assert result is not None
            assert "deleted_from" in result
            assert "linkage" in result["deleted_from"]

            # Verify removed from linkage
            links = store.linkage_table.get_links(concept_uuid)
            assert links is None
            store.close()

    def test_delete_returns_none_for_nonexistent(self):
        """Test delete() returns None for nonexistent UUID."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            result = store.delete("nonexistent_uuid")
            assert result is None
            store.close()


class TestMemoryStoreContextBuilding:
    """Test context building operations."""

    def test_get_context_aggregates_all_stores(self):
        """Test get_context() builds aggregated context from all stores."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            # Add data to different stores
            store.add(data="User: What's the status?", data_type="chat")
            store.add(data='{"heart_rate": 72}', data_type="telemetry")
            store.add(
                data={"id": "person:alice", "name": "Alice"},
                data_type="node",
                metadata={"node_type": "person"},
            )

            # Get unified context
            context = store.get_context(
                recent_minutes=10,
                include_graph=True,
                include_working_memory=True,
            )

            assert context is not None
            assert "working_memory" in context
            assert "graph" in context or "timeseries" in context
            store.close()

    def test_get_context_respects_limits(self):
        """Test get_context() respects limit parameters."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            # Add many chat messages
            for i in range(20):
                store.add(data=f"Message {i}", data_type="chat")

            # Get context with limit
            context = store.get_context(limit=5)

            assert context is not None
            assert len(context.get("working_memory", [])) <= 5
            store.close()


class TestMemoryStoreStatistics:
    """Test statistics operations."""

    def test_get_stats_returns_all_store_stats(self):
        """Test get_stats() returns statistics from all stores."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            # Add some data
            store.add(data='{"test": 1}', data_type="telemetry")
            store.add(data="chat message", data_type="chat")

            stats = store.get_stats()

            assert "timeseries" in stats
            assert "graph" in stats
            assert "working_memory" in stats
            assert "linkage" in stats
            store.close()


class TestMemoryStoreCleanup:
    """Test cleanup operations."""

    def test_close_closes_all_stores(self):
        """Test close() closes all component stores."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            store.close()

            assert not store.is_connected()

    def test_close_is_idempotent(self):
        """Test close() can be called multiple times."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            store.close()
            store.close()  # Should not raise
            store.close()  # Should not raise

    def test_context_manager_closes_stores(self):
        """Test context manager properly closes stores."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            with MemoryStore(config=config) as store:
                assert store.is_connected()

            # After context exit, should be closed
            assert not store.is_connected()


class TestMemoryStorePruning:
    """Test pruning and cleanup operations."""

    def test_prune_working_memory(self):
        """Test prune_working_memory() removes expired records."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "base_path": temp_dir,
                "working_memory": {
                    "path": f"{temp_dir}/working.duckdb",
                    "ttl_seconds": 1,  # Very short TTL for testing
                },
            }
            store = MemoryStore(config=config)

            # Add data
            store.add(data="test message", data_type="chat")

            # Wait for expiration
            import time

            time.sleep(1.5)

            # Prune
            pruned = store.prune_working_memory()
            assert pruned >= 1
            store.close()

    def test_clear_working_memory(self):
        """Test clear_working_memory() removes all records."""
        from core.memory import MemoryStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_path": temp_dir}
            store = MemoryStore(config=config)

            # Add data
            store.add(data="test message 1", data_type="chat")
            store.add(data="test message 2", data_type="chat")

            # Clear
            store.clear_working_memory()

            # Verify empty
            results = store.query(recent={"limit": 10, "data_type": "chat"})
            assert len(results) == 0
            store.close()
