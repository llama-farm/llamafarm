"""Tests for TypedDatasetService.

Phase 21: Dataset Service Layer Updates
"""

from unittest.mock import MagicMock, patch

import pytest


class TestTypedDatasetServiceImport:
    """Test TypedDatasetService imports correctly."""

    def test_import_typed_dataset_service(self):
        """Test that TypedDatasetService can be imported."""
        from services.typed_dataset_service import TypedDatasetService

        assert TypedDatasetService is not None

    def test_inherits_from_dataset_service(self):
        """Test that TypedDatasetService inherits from DatasetService."""
        from services.dataset_service import DatasetService
        from services.typed_dataset_service import TypedDatasetService

        assert issubclass(TypedDatasetService, DatasetService)


class TestTypedDatasetServiceMocked:
    """Test TypedDatasetService with mocked stores."""

    @pytest.fixture
    def mock_get_store(self):
        """Mock the _get_store function."""
        with patch("services.typed_dataset_service._get_store") as mock:
            mock_store = MagicMock()
            mock_store.is_connected.return_value = True
            mock_store.get_enabled_stores.return_value = [
                "graph",
                "timeseries",
                "working_memory",
            ]
            mock_store.get_stats.return_value = {
                "dataset_name": "test",
                "dataset_type": "realtime",
                "stores": {"graph": {}, "timeseries": {}},
            }
            mock.return_value = mock_store
            yield mock, mock_store

    @pytest.fixture
    def mock_list_datasets(self):
        """Mock the list_datasets function."""
        with patch(
            "services.typed_dataset_service.DatasetService.list_datasets"
        ) as mock:
            mock_dataset = MagicMock()
            mock_dataset.name = "test_dataset"
            mock_dataset.type = "realtime"
            mock.return_value = [mock_dataset]
            yield mock

    def test_get_enabled_stores(self, mock_get_store):
        """Test get_enabled_stores method."""
        from services.typed_dataset_service import TypedDatasetService

        mock, mock_store = mock_get_store

        result = TypedDatasetService.get_enabled_stores("ns", "proj", "dataset")

        assert "graph" in result
        assert "timeseries" in result
        mock_store.get_enabled_stores.assert_called_once()

    def test_add_stream_record(self, mock_get_store):
        """Test add_stream_record method."""
        from services.typed_dataset_service import TypedDatasetService

        mock, mock_store = mock_get_store
        mock_store.add_stream_record.return_value = {
            "record_id": "rec-123",
            "stores": ["timeseries", "working_memory"],
        }

        result = TypedDatasetService.add_stream_record(
            namespace="ns",
            project="proj",
            dataset="dataset",
            data={"temperature": 72.5},
            data_type="sensor",
        )

        assert result["success"] is True
        assert result["record_id"] == "rec-123"
        mock_store.add_stream_record.assert_called_once()

    def test_add_stream_record_with_location(self, mock_get_store):
        """Test add_stream_record with location data."""
        from services.typed_dataset_service import TypedDatasetService

        mock, mock_store = mock_get_store
        mock_store.add_stream_record.return_value = {
            "record_id": "rec-456",
            "stores": ["spatial", "working_memory"],
        }

        result = TypedDatasetService.add_stream_record(
            namespace="ns",
            project="proj",
            dataset="dataset",
            data={"vehicle_id": "V001"},
            latitude=35.78,
            longitude=-78.64,
        )

        assert result["success"] is True
        call_kwargs = mock_store.add_stream_record.call_args[1]
        assert call_kwargs["latitude"] == 35.78
        assert call_kwargs["longitude"] == -78.64

    def test_add_stream_batch_success(self, mock_get_store):
        """Test successful batch ingestion."""
        from services.typed_dataset_service import TypedDatasetService

        mock, mock_store = mock_get_store
        mock_store.add_stream_record.return_value = {
            "record_id": "rec-123",
            "stores": [],
        }

        records = [
            {"data": {"temp": 72}},
            {"data": {"temp": 68}},
        ]

        result = TypedDatasetService.add_stream_batch(
            namespace="ns",
            project="proj",
            dataset="dataset",
            records=records,
        )

        assert result["success"] is True
        assert result["successful"] == 2
        assert result["failed"] == 0
        assert mock_store.add_stream_record.call_count == 2

    def test_add_stream_batch_partial_failure(self, mock_get_store):
        """Test batch with partial failures."""
        from services.typed_dataset_service import TypedDatasetService

        mock, mock_store = mock_get_store
        mock_store.add_stream_record.side_effect = [
            {"record_id": "rec-1", "stores": []},
            Exception("Database error"),
            {"record_id": "rec-3", "stores": []},
        ]

        records = [
            {"data": {"temp": 72}},
            {"data": {"temp": 68}},
            {"data": {"temp": 70}},
        ]

        result = TypedDatasetService.add_stream_batch(
            namespace="ns",
            project="proj",
            dataset="dataset",
            records=records,
        )

        assert result["success"] is False
        assert result["successful"] == 2
        assert result["failed"] == 1

    def test_add_stream_batch_fail_fast(self, mock_get_store):
        """Test batch with fail_fast option."""
        from services.typed_dataset_service import TypedDatasetService

        mock, mock_store = mock_get_store
        mock_store.add_stream_record.side_effect = [
            {"record_id": "rec-1", "stores": []},
            Exception("Database error"),
            {"record_id": "rec-3", "stores": []},
        ]

        records = [
            {"data": {"temp": 72}},
            {"data": {"temp": 68}},
            {"data": {"temp": 70}},
        ]

        result = TypedDatasetService.add_stream_batch(
            namespace="ns",
            project="proj",
            dataset="dataset",
            records=records,
            fail_fast=True,
        )

        # Should stop after first failure
        assert result["failed"] == 1
        assert result["successful"] == 1
        assert mock_store.add_stream_record.call_count == 2

    def test_add_graph_node(self, mock_get_store):
        """Test add_graph_node method."""
        from services.typed_dataset_service import TypedDatasetService

        mock, mock_store = mock_get_store
        mock_store.add_node.return_value = "node-123"

        result = TypedDatasetService.add_graph_node(
            namespace="ns",
            project="proj",
            dataset="dataset",
            name="John Doe",
            node_type="person",
            properties={"role": "engineer"},
        )

        assert result["success"] is True
        assert result["node_id"] == "node-123"

    def test_add_graph_edge(self, mock_get_store):
        """Test add_graph_edge method."""
        from services.typed_dataset_service import TypedDatasetService

        mock, mock_store = mock_get_store
        mock_store.add_edge.return_value = "edge-456"

        result = TypedDatasetService.add_graph_edge(
            namespace="ns",
            project="proj",
            dataset="dataset",
            source_id="node-1",
            target_id="node-2",
            relationship="works_at",
        )

        assert result["success"] is True
        assert result["edge_id"] == "edge-456"

    def test_get_dataset_stats(self, mock_get_store):
        """Test get_dataset_stats method."""
        from services.typed_dataset_service import TypedDatasetService

        mock, mock_store = mock_get_store

        result = TypedDatasetService.get_dataset_stats(
            namespace="ns",
            project="proj",
            dataset="dataset",
        )

        assert result["success"] is True
        assert result["dataset_name"] == "test"
        assert result["dataset_type"] == "realtime"

    def test_clear_dataset_stores(self, mock_get_store):
        """Test clear_dataset_stores method."""
        from services.typed_dataset_service import TypedDatasetService

        mock, mock_store = mock_get_store
        mock_store.clear.return_value = {
            "graph": True,
            "timeseries": True,
            "working_memory": True,
        }

        result = TypedDatasetService.clear_dataset_stores(
            namespace="ns",
            project="proj",
            dataset="dataset",
        )

        assert result["success"] is True
        assert result["cleared_stores"]["graph"] is True


class TestTypedDatasetServiceQueryMocked:
    """Test TypedDatasetService query methods with mocked stores."""

    @pytest.fixture
    def mock_hybrid_query(self):
        """Mock hybrid_query function."""
        with patch("services.typed_dataset_service._import_from_rag") as mock_import:
            mock_query = MagicMock()
            mock_query.return_value = {
                "results": [{"id": "1", "content": "test"}],
                "total_count": 1,
                "stores_queried": ["graph"],
            }
            mock_import.return_value = mock_query
            yield mock_import, mock_query

    @pytest.fixture
    def mock_get_store_for_query(self):
        """Mock the _get_store function."""
        with patch("services.typed_dataset_service._get_store") as mock:
            mock_store = MagicMock()
            mock_store.is_connected.return_value = True
            mock.return_value = mock_store
            yield mock, mock_store

    def test_hybrid_query(self, mock_get_store_for_query, mock_hybrid_query):
        """Test hybrid_query method."""
        from services.typed_dataset_service import TypedDatasetService

        _, mock_store = mock_get_store_for_query
        _, mock_query = mock_hybrid_query

        result = TypedDatasetService.hybrid_query(
            namespace="ns",
            project="proj",
            dataset="dataset",
            query_text="test query",
            limit=5,
        )

        assert result["success"] is True
        assert len(result["results"]) == 1


class TestTypedDatasetServiceValidation:
    """Test input validation in TypedDatasetService."""

    def test_create_typed_dataset_invalid_type(self):
        """Test that invalid dataset type raises error."""
        from services.typed_dataset_service import TypedDatasetService

        with pytest.raises(ValueError) as exc_info:
            TypedDatasetService.create_typed_dataset(
                namespace="ns",
                project="proj",
                name="test",
                dataset_type="invalid_type",
            )

        assert "Invalid dataset type" in str(exc_info.value)


class TestStoreCache:
    """Test store caching behavior."""

    def test_close_store_removes_from_cache(self):
        """Test that close_store removes store from cache."""
        from services.typed_dataset_service import TypedDatasetService, _store_cache

        # Add mock to cache
        mock_store = MagicMock()
        _store_cache["ns/proj/dataset"] = mock_store

        TypedDatasetService.close_store("ns", "proj", "dataset")

        assert "ns/proj/dataset" not in _store_cache
        mock_store.close.assert_called_once()

    def test_close_all_stores(self):
        """Test close_all_stores clears cache."""
        from services.typed_dataset_service import TypedDatasetService, _store_cache

        # Add mocks to cache
        mock1 = MagicMock()
        mock2 = MagicMock()
        _store_cache["ns/proj/ds1"] = mock1
        _store_cache["ns/proj/ds2"] = mock2

        TypedDatasetService.close_all_stores()

        assert len(_store_cache) == 0
        mock1.close.assert_called_once()
        mock2.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
