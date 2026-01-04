"""
Tests for MemoryStoreService - Per-project memory store management.

Phase 10: Memory Store Service Layer
"""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add server to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.datamodel import (
    ConsolidationConfig,
    GraphConfig,
    MemoryConfig,
    MemoryStoreConfig,
    TimeSeriesConfig,
    WorkingMemoryConfig,
)

from api.errors import MemoryStoreNotFoundError


class TestMemoryStoreServiceListStores:
    """Test MemoryStoreService.list_stores()."""

    def test_list_stores_empty(self, mock_project_service):
        """Test listing stores when no memory config exists."""
        from services.memory_store_service import MemoryStoreService

        # Config without memory section
        mock_project_service.load_config.return_value = MagicMock(memory=None)

        stores = MemoryStoreService.list_stores("ns", "proj")
        assert stores == []

    def test_list_stores_with_config(self, mock_project_service):
        """Test listing stores when memory config exists."""
        from services.memory_store_service import MemoryStoreService

        memory_config = MemoryConfig(
            stores=[
                MemoryStoreConfig(name="brain_memory"),
                MemoryStoreConfig(name="scenario_memory"),
            ],
            default_store="brain_memory",
        )
        mock_project_service.load_config.return_value = MagicMock(memory=memory_config)

        stores = MemoryStoreService.list_stores("ns", "proj")
        assert len(stores) == 2
        assert stores[0].name == "brain_memory"
        assert stores[1].name == "scenario_memory"


class TestMemoryStoreServiceGetStore:
    """Test MemoryStoreService.get_store()."""

    def test_get_store_not_configured(self, mock_project_service):
        """Test getting store when not configured raises error."""
        from services.memory_store_service import MemoryStoreService

        mock_project_service.load_config.return_value = MagicMock(memory=None)

        with pytest.raises(MemoryStoreNotFoundError):
            MemoryStoreService.get_store("ns", "proj", "nonexistent")

    def test_get_store_creates_directory(self, mock_project_service, temp_project_dir):
        """Test that getting a store creates the data directory."""
        from services.memory_store_service import MemoryStoreService, _store_cache

        # Clear cache
        _store_cache.clear()

        memory_config = MemoryConfig(
            stores=[MemoryStoreConfig(name="test_store")],
        )
        mock_project_service.load_config.return_value = MagicMock(memory=memory_config)
        mock_project_service.get_project_dir.return_value = str(temp_project_dir)

        store = MemoryStoreService.get_store("ns", "proj", "test_store")

        # Check directory was created
        expected_path = temp_project_dir / "lf_data" / "memory" / "test_store"
        assert expected_path.exists()

        # Close store
        store.close()
        _store_cache.clear()

    def test_get_store_path_pattern(self, mock_project_service, temp_project_dir):
        """Test that store path follows {project_dir}/lf_data/memory/{store_name}/."""
        from services.memory_store_service import MemoryStoreService, _store_cache

        _store_cache.clear()

        memory_config = MemoryConfig(
            stores=[MemoryStoreConfig(name="brain_memory")],
        )
        mock_project_service.load_config.return_value = MagicMock(memory=memory_config)
        mock_project_service.get_project_dir.return_value = str(temp_project_dir)

        store = MemoryStoreService.get_store("ns", "proj", "brain_memory")

        # Verify path pattern
        assert store.base_path == str(
            temp_project_dir / "lf_data" / "memory" / "brain_memory"
        )

        store.close()
        _store_cache.clear()

    def test_get_store_with_config(self, mock_project_service, temp_project_dir):
        """Test getting store with configuration options."""
        from services.memory_store_service import MemoryStoreService, _store_cache

        _store_cache.clear()

        memory_config = MemoryConfig(
            stores=[
                MemoryStoreConfig(
                    name="configured_store",
                    working_memory=WorkingMemoryConfig(
                        ttl_seconds=1800, max_records=5000
                    ),
                    timeseries=TimeSeriesConfig(retention_days=60),
                    graph=GraphConfig(max_path_depth=15),
                )
            ],
        )
        mock_project_service.load_config.return_value = MagicMock(memory=memory_config)
        mock_project_service.get_project_dir.return_value = str(temp_project_dir)

        store = MemoryStoreService.get_store("ns", "proj", "configured_store")
        assert store is not None
        assert store.is_connected()

        store.close()
        _store_cache.clear()

    def test_get_store_caching(self, mock_project_service, temp_project_dir):
        """Test that stores are cached and reused."""
        from services.memory_store_service import MemoryStoreService, _store_cache

        _store_cache.clear()

        memory_config = MemoryConfig(
            stores=[MemoryStoreConfig(name="cached_store")],
        )
        mock_project_service.load_config.return_value = MagicMock(memory=memory_config)
        mock_project_service.get_project_dir.return_value = str(temp_project_dir)

        # Get store twice
        store1 = MemoryStoreService.get_store("ns", "proj", "cached_store")
        store2 = MemoryStoreService.get_store("ns", "proj", "cached_store")

        # Should be the same instance
        assert store1 is store2

        store1.close()
        _store_cache.clear()


class TestMemoryStoreServiceGetDefaultStore:
    """Test MemoryStoreService.get_default_store()."""

    def test_get_default_store_when_set(self, mock_project_service, temp_project_dir):
        """Test getting default store when explicitly set."""
        from services.memory_store_service import MemoryStoreService, _store_cache

        _store_cache.clear()

        memory_config = MemoryConfig(
            stores=[
                MemoryStoreConfig(name="first_store"),
                MemoryStoreConfig(name="second_store"),
            ],
            default_store="second_store",
        )
        mock_project_service.load_config.return_value = MagicMock(memory=memory_config)
        mock_project_service.get_project_dir.return_value = str(temp_project_dir)

        store = MemoryStoreService.get_default_store("ns", "proj")
        assert store.base_path.endswith("second_store")

        store.close()
        _store_cache.clear()

    def test_get_default_store_falls_back_to_first(
        self, mock_project_service, temp_project_dir
    ):
        """Test that default store falls back to first store if not set."""
        from services.memory_store_service import MemoryStoreService, _store_cache

        _store_cache.clear()

        memory_config = MemoryConfig(
            stores=[
                MemoryStoreConfig(name="first_store"),
                MemoryStoreConfig(name="second_store"),
            ],
        )
        mock_project_service.load_config.return_value = MagicMock(memory=memory_config)
        mock_project_service.get_project_dir.return_value = str(temp_project_dir)

        store = MemoryStoreService.get_default_store("ns", "proj")
        assert store.base_path.endswith("first_store")

        store.close()
        _store_cache.clear()

    def test_get_default_store_no_stores_configured(self, mock_project_service):
        """Test getting default store when no stores are configured."""
        from services.memory_store_service import MemoryStoreService

        mock_project_service.load_config.return_value = MagicMock(memory=None)

        with pytest.raises(MemoryStoreNotFoundError):
            MemoryStoreService.get_default_store("ns", "proj")


class TestMemoryStoreServiceStats:
    """Test MemoryStoreService.get_store_stats()."""

    def test_get_store_stats(self, mock_project_service, temp_project_dir):
        """Test getting statistics for a store."""
        from services.memory_store_service import MemoryStoreService, _store_cache

        _store_cache.clear()

        memory_config = MemoryConfig(
            stores=[MemoryStoreConfig(name="stats_store")],
        )
        mock_project_service.load_config.return_value = MagicMock(memory=memory_config)
        mock_project_service.get_project_dir.return_value = str(temp_project_dir)

        stats = MemoryStoreService.get_store_stats("ns", "proj", "stats_store")

        assert "store_path" in stats
        assert "total_size_bytes" in stats
        assert stats["store_path"].endswith("stats_store")

        # Close store
        MemoryStoreService.close_all_stores()
        _store_cache.clear()


class TestMemoryStoreServiceClearDelete:
    """Test MemoryStoreService.clear_store() and delete_store()."""

    def test_clear_store(self, mock_project_service, temp_project_dir):
        """Test clearing a store removes data but keeps store."""
        from services.memory_store_service import MemoryStoreService, _store_cache

        _store_cache.clear()

        memory_config = MemoryConfig(
            stores=[MemoryStoreConfig(name="clear_store")],
        )
        mock_project_service.load_config.return_value = MagicMock(memory=memory_config)
        mock_project_service.get_project_dir.return_value = str(temp_project_dir)

        # Create store and add data
        store = MemoryStoreService.get_store("ns", "proj", "clear_store")
        store.add("test data", "chat", {"key": "value"})

        # Clear store
        result = MemoryStoreService.clear_store("ns", "proj", "clear_store")
        assert result["success"] is True
        assert result["store_name"] == "clear_store"

        # Store directory should still exist
        store_path = temp_project_dir / "lf_data" / "memory" / "clear_store"
        assert store_path.exists()

        MemoryStoreService.close_all_stores()
        _store_cache.clear()

    def test_delete_store(self, mock_project_service, temp_project_dir):
        """Test deleting a store removes data directory."""
        from services.memory_store_service import MemoryStoreService, _store_cache

        _store_cache.clear()

        memory_config = MemoryConfig(
            stores=[MemoryStoreConfig(name="delete_store")],
        )
        mock_project_service.load_config.return_value = MagicMock(memory=memory_config)
        mock_project_service.get_project_dir.return_value = str(temp_project_dir)

        # Create store (side effect creates directory)
        MemoryStoreService.get_store("ns", "proj", "delete_store")
        store_path = temp_project_dir / "lf_data" / "memory" / "delete_store"
        assert store_path.exists()

        # Delete store
        result = MemoryStoreService.delete_store(
            "ns", "proj", "delete_store", delete_data=True
        )
        assert result["success"] is True
        assert result["data_deleted"] is True

        # Store directory should be gone
        assert not store_path.exists()

        _store_cache.clear()


class TestMemoryStoreServiceConsolidator:
    """Test MemoryStoreService.get_consolidator()."""

    def test_get_consolidator(self, mock_project_service, temp_project_dir):
        """Test getting a consolidator for a store."""
        from services.memory_store_service import MemoryStoreService, _store_cache

        _store_cache.clear()

        memory_config = MemoryConfig(
            stores=[
                MemoryStoreConfig(
                    name="consolidate_store",
                    consolidation=ConsolidationConfig(
                        min_records=20,
                        batch_size=50,
                        prune_after_consolidate=False,
                    ),
                )
            ],
        )
        mock_project_service.load_config.return_value = MagicMock(memory=memory_config)
        mock_project_service.get_project_dir.return_value = str(temp_project_dir)

        consolidator = MemoryStoreService.get_consolidator(
            "ns", "proj", "consolidate_store"
        )
        assert consolidator is not None

        MemoryStoreService.close_all_stores()
        _store_cache.clear()


# Fixtures
@pytest.fixture
def mock_project_service():
    """Mock the ProjectService for testing."""
    with patch("services.memory_store_service.ProjectService") as mock:
        yield mock


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="memory_test_"))
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
