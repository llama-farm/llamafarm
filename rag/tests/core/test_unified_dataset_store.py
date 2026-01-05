#!/usr/bin/env python3
"""
Tests for UnifiedDatasetStore - Unified storage backend for typed datasets.

Phase 17: Unified Dataset Store
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add rag to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.unified_store import DATASET_TYPE_CAPABILITIES, UnifiedDatasetStore


class TestDatasetTypeCapabilities:
    """Test dataset type capability matrix."""

    def test_knowledge_type_capabilities(self):
        """Test knowledge type enables vector + graph only."""
        caps = DATASET_TYPE_CAPABILITIES["knowledge"]
        assert caps["vector"] is True
        assert caps["graph"] is True
        assert caps["timeseries"] is False
        assert caps["spatial"] is False
        assert caps["working_memory"] is False

    def test_realtime_type_capabilities(self):
        """Test realtime type enables all stores."""
        caps = DATASET_TYPE_CAPABILITIES["realtime"]
        assert caps["vector"] is True
        assert caps["graph"] is True
        assert caps["timeseries"] is True
        assert caps["spatial"] is True
        assert caps["working_memory"] is True

    def test_graph_type_capabilities(self):
        """Test graph type enables only graph store."""
        caps = DATASET_TYPE_CAPABILITIES["graph"]
        assert caps["vector"] is False
        assert caps["graph"] is True
        assert caps["timeseries"] is False
        assert caps["spatial"] is False
        assert caps["working_memory"] is False

    def test_timeseries_type_capabilities(self):
        """Test timeseries type enables timeseries + working memory."""
        caps = DATASET_TYPE_CAPABILITIES["timeseries"]
        assert caps["vector"] is False
        assert caps["graph"] is False
        assert caps["timeseries"] is True
        assert caps["spatial"] is False
        assert caps["working_memory"] is True

    def test_spatial_type_capabilities(self):
        """Test spatial type enables spatial + working memory."""
        caps = DATASET_TYPE_CAPABILITIES["spatial"]
        assert caps["vector"] is False
        assert caps["graph"] is False
        assert caps["timeseries"] is False
        assert caps["spatial"] is True
        assert caps["working_memory"] is True

    def test_hybrid_type_capabilities(self):
        """Test hybrid type enables all stores."""
        caps = DATASET_TYPE_CAPABILITIES["hybrid"]
        assert caps["vector"] is True
        assert caps["graph"] is True
        assert caps["timeseries"] is True
        assert caps["spatial"] is True
        assert caps["working_memory"] is True


class TestUnifiedDatasetStoreInit:
    """Test UnifiedDatasetStore initialization."""

    def test_init_knowledge_type(self, temp_project_dir):
        """Test initialization with knowledge type."""
        config = {"name": "test_knowledge", "type": "knowledge"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        assert store.name == "test_knowledge"
        assert store.dataset_type == "knowledge"
        assert store.graph_store is not None
        # Vector store not yet implemented
        assert store.timeseries_store is None
        assert store.spatial_store is None
        assert store.working_memory is None
        assert store.linkage_table is not None

        store.close()

    def test_init_realtime_type(self, temp_project_dir):
        """Test initialization with realtime type."""
        config = {"name": "test_realtime", "type": "realtime"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        assert store.dataset_type == "realtime"
        assert store.graph_store is not None
        assert store.timeseries_store is not None
        assert store.spatial_store is not None
        assert store.working_memory is not None

        store.close()

    def test_init_graph_only_type(self, temp_project_dir):
        """Test initialization with graph-only type."""
        config = {"name": "test_graph", "type": "graph"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        assert store.dataset_type == "graph"
        assert store.graph_store is not None
        assert store.timeseries_store is None
        assert store.spatial_store is None
        assert store.working_memory is None

        store.close()

    def test_init_spatial_type(self, temp_project_dir):
        """Test initialization with spatial type (top-level geo-tracking)."""
        config = {"name": "test_spatial", "type": "spatial"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        assert store.dataset_type == "spatial"
        assert store.graph_store is None
        assert store.timeseries_store is None
        assert store.spatial_store is not None
        assert store.working_memory is not None

        store.close()

    def test_init_creates_base_path(self, temp_project_dir):
        """Test that initialization creates the data directory."""
        config = {"name": "path_test", "type": "knowledge"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        expected_path = temp_project_dir / "lf_data" / "datasets" / "path_test"
        assert expected_path.exists()
        assert store.base_path == str(expected_path)

        store.close()

    def test_init_with_config_overrides(self, temp_project_dir):
        """Test that explicit config overrides type defaults."""
        config = {
            "name": "override_test",
            "type": "knowledge",  # Default: vector + graph
            "graph": {"enabled": False},  # Disable graph
            "working_memory": {"enabled": True, "ttl_seconds": 1800},  # Enable WM
        }
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        # Graph should be disabled despite knowledge type
        assert store.graph_store is None
        # Working memory should be enabled despite knowledge type
        assert store.working_memory is not None

        store.close()


class TestUnifiedDatasetStoreGraphOperations:
    """Test graph operations."""

    def test_add_node(self, temp_project_dir):
        """Test adding a node to graph store."""
        config = {"name": "graph_ops", "type": "knowledge"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        node_id = store.add_node(
            name="John Doe",
            node_type="person",
            properties={"role": "soldier"},
        )

        assert node_id is not None

        store.close()

    def test_add_edge(self, temp_project_dir):
        """Test adding an edge between nodes."""
        config = {"name": "edge_ops", "type": "knowledge"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        # Add two nodes
        node1 = store.add_node(name="Unit Alpha", node_type="unit")
        node2 = store.add_node(name="John Doe", node_type="person")

        # Add edge
        edge_id = store.add_edge(node1, node2, relationship="member_of")

        assert edge_id is not None

        store.close()

    def test_add_node_disabled_graph(self, temp_project_dir):
        """Test that add_node returns None when graph is disabled."""
        config = {"name": "no_graph", "type": "timeseries"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        result = store.add_node(name="Test", node_type="entity")

        assert result is None

        store.close()


class TestUnifiedDatasetStoreStreamOperations:
    """Test stream record operations."""

    def test_add_stream_record_timeseries(self, temp_project_dir):
        """Test adding stream record to timeseries store."""
        config = {"name": "stream_ts", "type": "timeseries"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        result = store.add_stream_record(
            data={"heart_rate": 72, "temperature": 98.6},
            data_type="biometrics",
            metadata={"soldier_id": "S001"},
        )

        assert "record_id" in result
        assert "timeseries" in result["stores"]
        assert "working_memory" in result["stores"]

        store.close()

    def test_add_stream_record_spatial(self, temp_project_dir):
        """Test adding stream record with location to spatial store."""
        config = {"name": "stream_spatial", "type": "spatial"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        result = store.add_stream_record(
            data={"vehicle_id": "V001"},
            data_type="location",
            latitude=34.5,
            longitude=-118.2,
        )

        assert "record_id" in result
        assert "spatial" in result["stores"]
        assert "working_memory" in result["stores"]

        store.close()

    def test_add_stream_record_realtime_all_stores(self, temp_project_dir):
        """Test realtime dataset writes to all stores."""
        config = {"name": "stream_realtime", "type": "realtime"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        result = store.add_stream_record(
            data={"status": "active", "heart_rate": 72},
            data_type="telemetry",
            latitude=34.5,
            longitude=-118.2,
            metadata={"unit": "Alpha"},
        )

        assert "timeseries" in result["stores"]
        assert "spatial" in result["stores"]
        assert "working_memory" in result["stores"]

        store.close()


class TestUnifiedDatasetStoreQuery:
    """Test query operations."""

    def test_query_graph(self, temp_project_dir):
        """Test graph query for neighbors."""
        config = {"name": "query_graph", "type": "knowledge"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        # Add nodes and edge
        unit_id = store.add_node(name="Unit Alpha", node_type="unit")
        person_id = store.add_node(name="John Doe", node_type="person")
        store.add_edge(unit_id, person_id, relationship="member_of")

        # Query neighbors
        result = store.query(graph_query={"node_id": unit_id, "direction": "out"})

        assert "graph" in result
        assert "graph" in result["stores_queried"]

        store.close()

    def test_query_hybrid_recent(self, temp_project_dir):
        """Test hybrid query includes working memory."""
        config = {"name": "query_hybrid", "type": "realtime"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        # Add some stream data
        store.add_stream_record(
            data={"status": "active"},
            data_type="telemetry",
        )

        # Query hybrid (should include working memory)
        result = store.query(query_type="hybrid")

        assert "working_memory" in result["stores_queried"]

        store.close()


class TestUnifiedDatasetStoreStats:
    """Test statistics aggregation."""

    def test_get_stats_all_stores(self, temp_project_dir):
        """Test get_stats returns info for all enabled stores."""
        config = {"name": "stats_test", "type": "realtime"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        stats = store.get_stats()

        assert stats["dataset_name"] == "stats_test"
        assert stats["dataset_type"] == "realtime"
        assert "graph" in stats["stores"]
        assert "timeseries" in stats["stores"]
        assert "spatial" in stats["stores"]
        assert "working_memory" in stats["stores"]

        store.close()

    def test_get_enabled_stores(self, temp_project_dir):
        """Test get_enabled_stores returns correct list."""
        config = {"name": "enabled_test", "type": "spatial"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        enabled = store.get_enabled_stores()

        assert "spatial" in enabled
        assert "working_memory" in enabled
        assert "graph" not in enabled
        assert "timeseries" not in enabled

        store.close()


class TestUnifiedDatasetStoreCleanup:
    """Test cleanup operations."""

    def test_clear_all_stores(self, temp_project_dir):
        """Test clear removes data from all stores."""
        config = {"name": "clear_test", "type": "realtime"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        # Add some data
        store.add_node(name="Test Node", node_type="entity")
        store.add_stream_record(data={"test": True}, data_type="test")

        # Clear all
        result = store.clear()

        assert result["graph"] is True
        assert result["timeseries"] is True
        assert result["spatial"] is True
        assert result["working_memory"] is True

        store.close()

    def test_close_idempotent(self, temp_project_dir):
        """Test that close can be called multiple times."""
        config = {"name": "close_test", "type": "knowledge"}
        store = UnifiedDatasetStore(config, str(temp_project_dir))

        store.close()
        store.close()  # Should not raise

        assert store._closed is True

    def test_context_manager(self, temp_project_dir):
        """Test context manager properly closes store."""
        config = {"name": "context_test", "type": "knowledge"}

        with UnifiedDatasetStore(config, str(temp_project_dir)) as store:
            assert store.is_connected()

        assert store._closed is True


# Fixtures
@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="unified_store_test_"))
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
