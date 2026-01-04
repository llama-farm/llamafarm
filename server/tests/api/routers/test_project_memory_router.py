"""
Tests for Per-Project Memory Router - API endpoints for per-project memory stores.

Phase 12: Per-Project Memory API Router
"""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add server to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


from api.routers.memory.project_memory_router import router


@pytest.fixture
def app():
    """Create a FastAPI test app with the project memory router."""
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="project_memory_api_test_"))
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestProjectMemoryAddEndpoint:
    """Test POST /v1/projects/{namespace}/{project}/memory/add."""

    def test_add_data(self, client, temp_project_dir):
        """Test adding data to project memory."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.add.return_value = {
                "success": True,
                "uuid": "test-uuid-123",
                "store": "working_memory",
                "component_id": "comp-123",
                "message": "Data added",
            }

            response = client.post(
                "/v1/projects/ns/proj/memory/add",
                json={
                    "data": "test data",
                    "data_type": "chat",
                    "metadata": {"key": "value"},
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["uuid"] == "test-uuid-123"
            assert data["store"] == "working_memory"

    def test_add_data_with_store_name(self, client, temp_project_dir):
        """Test adding data to a specific store."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.add.return_value = {
                "success": True,
                "uuid": "test-uuid-456",
                "store": "timeseries",
            }

            response = client.post(
                "/v1/projects/ns/proj/memory/add?store_name=brain_memory",
                json={
                    "data": {"value": 42.5},
                    "data_type": "telemetry",
                },
            )

            assert response.status_code == 200
            # Verify store_name was passed
            mock_service.add.assert_called_once()
            call_kwargs = mock_service.add.call_args[1]
            assert call_kwargs["store_name"] == "brain_memory"


class TestProjectMemoryQueryEndpoint:
    """Test GET /v1/projects/{namespace}/{project}/memory/query."""

    def test_query_empty(self, client):
        """Test querying empty store."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.query.return_value = {
                "success": True,
                "results": [],
                "total_count": 0,
            }

            response = client.get("/v1/projects/ns/proj/memory/query")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["total_count"] == 0
            assert data["results"] == []

    def test_query_with_filters(self, client):
        """Test querying with time and data type filters."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.query.return_value = {
                "success": True,
                "results": [
                    {"uuid": "1", "data_type": "chat", "content": "test"},
                ],
                "total_count": 1,
            }

            response = client.get(
                "/v1/projects/ns/proj/memory/query",
                params={
                    "data_types": "chat,audio",
                    "limit": 50,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] == 1

            # Verify filters were passed
            mock_service.query.assert_called_once()
            call_kwargs = mock_service.query.call_args[1]
            assert call_kwargs["data_types"] == ["chat", "audio"]
            assert call_kwargs["recent_limit"] == 50


class TestProjectMemoryContextEndpoint:
    """Test GET /v1/projects/{namespace}/{project}/memory/context."""

    def test_get_context(self, client):
        """Test getting aggregated context."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.get_context.return_value = {
                "success": True,
                "working_memory": [{"uuid": "1", "content": "test"}],
                "graph": [],
                "timeseries": [],
            }

            response = client.get(
                "/v1/projects/ns/proj/memory/context",
                params={"recent_minutes": 10},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["working_memory"]) == 1
            assert data["graph"] == []


class TestProjectMemoryDeleteEndpoint:
    """Test DELETE /v1/projects/{namespace}/{project}/memory/{uuid}."""

    def test_delete_existing(self, client):
        """Test deleting an existing record."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.delete.return_value = {
                "success": True,
                "uuid": "test-uuid",
                "deleted_from": ["working_memory"],
                "message": "Record deleted",
            }

            response = client.delete("/v1/projects/ns/proj/memory/test-uuid")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["uuid"] == "test-uuid"

    def test_delete_not_found(self, client):
        """Test deleting a non-existent record."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.delete.return_value = None

            response = client.delete("/v1/projects/ns/proj/memory/nonexistent")

            assert response.status_code == 404

    def test_delete_invalid_uuid(self, client):
        """Test deleting with invalid UUID format containing special characters."""
        # Use a UUID with special characters (not path traversal since FastAPI blocks that)
        response = client.delete("/v1/projects/ns/proj/memory/test@invalid!uuid")
        assert response.status_code == 400
        assert "Invalid UUID format" in response.json()["detail"]


class TestProjectMemoryClearEndpoint:
    """Test POST /v1/projects/{namespace}/{project}/memory/clear/{table}."""

    def test_clear_working_memory(self, client):
        """Test clearing working memory table."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.clear_table.return_value = {
                "success": True,
                "table": "working_memory",
                "cleared": {"working_memory": True},
            }

            response = client.post("/v1/projects/ns/proj/memory/clear/working_memory")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["table"] == "working_memory"

    def test_clear_all_tables(self, client):
        """Test clearing all tables."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.clear_table.return_value = {
                "success": True,
                "table": "all",
                "cleared": {
                    "working_memory": True,
                    "timeseries": 0,
                    "graph": {"nodes_deleted": 0, "edges_deleted": 0},
                    "linkage": 0,
                },
            }

            response = client.post("/v1/projects/ns/proj/memory/clear/all")

            assert response.status_code == 200
            data = response.json()
            assert data["table"] == "all"
            assert "working_memory" in data["cleared"]


class TestProjectMemoryConsolidateEndpoint:
    """Test POST /v1/projects/{namespace}/{project}/memory/consolidate."""

    def test_consolidate(self, client):
        """Test triggering consolidation."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.consolidate.return_value = {
                "success": True,
                "records_processed": 10,
                "facts_extracted": 5,
                "nodes_created": 3,
                "pruned": 10,
                "synthesis_method": "rule_based",
            }

            response = client.post(
                "/v1/projects/ns/proj/memory/consolidate",
                json={"use_llm": False},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["synthesis_method"] == "rule_based"


class TestProjectMemoryPruneEndpoint:
    """Test POST /v1/projects/{namespace}/{project}/memory/prune."""

    def test_prune(self, client):
        """Test pruning expired records."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.prune.return_value = {
                "success": True,
                "pruned_count": 5,
                "remaining_count": 100,
            }

            response = client.post("/v1/projects/ns/proj/memory/prune")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["pruned_count"] == 5
            assert data["remaining_count"] == 100


class TestProjectMemoryStatsEndpoint:
    """Test GET /v1/projects/{namespace}/{project}/memory/stats."""

    def test_get_stats(self, client):
        """Test getting storage statistics."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.get_stats.return_value = {
                "success": True,
                "working_memory": {"total_records": 150},
                "graph": {"node_count": 45, "edge_count": 78},
                "timeseries": {"record_count": 10000},
                "linkage": {"total_links": 250},
                "store_path": "/path/to/store",
                "total_size_bytes": 1048576,
            }

            response = client.get("/v1/projects/ns/proj/memory/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["working_memory"]["total_records"] == 150
            assert data["graph"]["node_count"] == 45


class TestProjectMemoryErrorHandling:
    """Test error handling for project memory endpoints."""

    def test_store_not_found(self, client):
        """Test handling of store not found error."""
        from api.errors import MemoryStoreNotFoundError

        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.add.side_effect = MemoryStoreNotFoundError("test_store")

            response = client.post(
                "/v1/projects/ns/proj/memory/add",
                json={"data": "test", "data_type": "chat"},
            )

            assert response.status_code == 404
            assert "test_store" in response.json()["detail"]

    def test_internal_error(self, client):
        """Test handling of internal errors."""
        with patch(
            "api.routers.memory.project_memory_router.MemoryDataService"
        ) as mock_service:
            mock_service.add.side_effect = Exception("Internal error")

            response = client.post(
                "/v1/projects/ns/proj/memory/add",
                json={"data": "test", "data_type": "chat"},
            )

            assert response.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
