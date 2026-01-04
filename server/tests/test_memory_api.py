"""Tests for Memory API endpoints - Embedded Trinity Memory System.

These tests follow TDD methodology - written FIRST before implementation.
The Memory API provides endpoints for unified memory operations.
"""

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from api.main import llama_farm_api


def _client() -> TestClient:
    """Create test client."""
    app = llama_farm_api()
    return TestClient(app)


class TestMemoryAddEndpoint:
    """Test POST /v1/memory/add endpoint."""

    def test_add_text_data(self, mocker):
        """Test adding text data to vector memory."""
        # Mock MemoryService
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.add.return_value = {
            "success": True,
            "uuid": "uuid-123",
            "store": "vector",
            "message": "Text data added to vector store",
        }

        client = _client()
        resp = client.post(
            "/v1/memory/add",
            json={
                "data": "This is a text message to remember",
                "data_type": "text",
                "metadata": {"source": "test"},
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["uuid"] == "uuid-123"
        assert data["store"] == "vector"

    def test_add_telemetry_data(self, mocker):
        """Test adding telemetry data to time-series store."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.add.return_value = {
            "success": True,
            "uuid": "uuid-456",
            "store": "timeseries",
            "message": "Telemetry data added to time-series store",
        }

        client = _client()
        resp = client.post(
            "/v1/memory/add",
            json={
                "data": {"heart_rate": 72, "temperature": 98.6},
                "data_type": "telemetry",
                "latitude": 35.7800,
                "longitude": -78.6400,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["store"] == "timeseries"

    def test_add_relation_data(self, mocker):
        """Test adding relationship data to graph store."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.add.return_value = {
            "success": True,
            "uuid": "uuid-789",
            "store": "graph",
            "nodes_created": 2,
            "edges_created": 1,
        }

        client = _client()
        resp = client.post(
            "/v1/memory/add",
            json={
                "data": {"source": "person:john", "target": "location:boston"},
                "data_type": "edge",
                "metadata": {"relationship": "lives_in"},
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["store"] == "graph"

    def test_add_chat_to_working_memory(self, mocker):
        """Test adding chat data to working memory."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.add.return_value = {
            "success": True,
            "uuid": "uuid-chat-001",
            "store": "working_memory",
            "expires_at": "2024-01-15T12:00:00Z",
        }

        client = _client()
        resp = client.post(
            "/v1/memory/add",
            json={
                "data": "User: What's the status?",
                "data_type": "chat",
                "metadata": {"channel": "tactical"},
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["store"] == "working_memory"

    def test_add_missing_data_returns_422(self, mocker):
        """Test that missing data field returns validation error."""
        client = _client()
        resp = client.post(
            "/v1/memory/add",
            json={
                "data_type": "text",
            },
        )

        assert resp.status_code == 422


class TestMemoryQueryEndpoint:
    """Test GET /v1/memory/query endpoint."""

    def test_query_returns_unified_context(self, mocker):
        """Test query returns unified context from all stores."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.query.return_value = {
            "results": [
                {"content": "Message 1", "store": "working_memory"},
                {"content": "Related entity", "store": "graph"},
                {"content": "Historical data", "store": "timeseries"},
            ],
            "total_count": 3,
        }

        client = _client()
        resp = client.get("/v1/memory/query")

        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 3

    def test_query_with_time_range(self, mocker):
        """Test query with time range filter."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.query.return_value = {
            "results": [
                {"content": "Recent event", "store": "timeseries"},
            ],
            "total_count": 1,
        }

        client = _client()
        resp = client.get(
            "/v1/memory/query",
            params={
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-02T00:00:00Z",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1

    def test_query_with_spatial_filter(self, mocker):
        """Test query with spatial (geo) filter."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.query.return_value = {
            "results": [
                {"content": "Nearby event", "store": "timeseries", "distance_m": 500},
            ],
            "total_count": 1,
        }

        client = _client()
        resp = client.get(
            "/v1/memory/query",
            params={
                "latitude": 35.7800,
                "longitude": -78.6400,
                "radius_m": 1000,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1

    def test_query_with_data_type_filter(self, mocker):
        """Test query filtered by data type."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.query.return_value = {
            "results": [
                {"content": "Chat 1", "store": "working_memory"},
                {"content": "Chat 2", "store": "working_memory"},
            ],
            "total_count": 2,
        }

        client = _client()
        resp = client.get(
            "/v1/memory/query",
            params={"data_types": "chat,audio"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 2

    def test_query_with_limit(self, mocker):
        """Test query respects limit parameter."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.query.return_value = {
            "results": [{"content": "Result 1"}],
            "total_count": 1,
        }

        client = _client()
        resp = client.get(
            "/v1/memory/query",
            params={"limit": 1},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) <= 1


class TestMemoryDeleteEndpoint:
    """Test DELETE /v1/memory/{uuid} endpoint."""

    def test_delete_performs_cascade_delete(self, mocker):
        """Test delete cascades to all stores via LinkageTable."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.delete.return_value = {
            "success": True,
            "uuid": "uuid-to-delete",
            "deleted_from": ["vector", "graph", "timeseries"],
            "message": "Record deleted from all linked stores",
        }

        client = _client()
        resp = client.delete("/v1/memory/uuid-to-delete")

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "vector" in data["deleted_from"]

    def test_delete_nonexistent_uuid_returns_404(self, mocker):
        """Test deleting non-existent UUID returns 404."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.delete.return_value = None

        client = _client()
        resp = client.delete("/v1/memory/nonexistent-uuid")

        assert resp.status_code == 404

    def test_delete_validates_uuid_format(self, mocker):
        """Test delete validates UUID format."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.delete.return_value = None

        client = _client()
        # UUID with invalid characters
        resp = client.delete("/v1/memory/invalid..uuid")

        # Should be rejected (400 for invalid format, or 404 if not found)
        assert resp.status_code in [400, 404]


class TestMemoryConsolidateEndpoint:
    """Test POST /v1/memory/consolidate endpoint."""

    def test_consolidate_triggers_synthesis(self, mocker):
        """Test consolidate triggers memory synthesis cycle."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.consolidate.return_value = {
            "success": True,
            "records_processed": 50,
            "facts_extracted": 12,
            "nodes_created": 8,
            "pruned": 35,
        }

        client = _client()
        resp = client.post("/v1/memory/consolidate")

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["records_processed"] == 50
        assert data["facts_extracted"] == 12

    def test_consolidate_with_llm_option(self, mocker):
        """Test consolidate with LLM synthesis option."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.consolidate.return_value = {
            "success": True,
            "records_processed": 25,
            "facts_extracted": 8,
            "nodes_created": 6,
            "pruned": 20,
            "synthesis_method": "llm",
        }

        client = _client()
        resp = client.post(
            "/v1/memory/consolidate",
            json={"use_llm": True},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["synthesis_method"] == "llm"

    def test_consolidate_skips_below_threshold(self, mocker):
        """Test consolidate skips when below buffer threshold."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.consolidate.return_value = {
            "success": True,
            "skipped": True,
            "records_processed": 0,
            "message": "Below buffer threshold",
        }

        client = _client()
        resp = client.post("/v1/memory/consolidate")

        assert resp.status_code == 200
        data = resp.json()
        assert data["skipped"] is True


class TestMemoryStatsEndpoint:
    """Test GET /v1/memory/stats endpoint."""

    def test_stats_returns_storage_statistics(self, mocker):
        """Test stats returns statistics from all stores."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.get_stats.return_value = {
            "working_memory": {
                "total_records": 150,
                "by_type": {"chat": 100, "audio": 50},
            },
            "graph": {
                "total_nodes": 45,
                "total_edges": 78,
            },
            "timeseries": {
                "total_records": 10000,
                "oldest_record": "2024-01-01T00:00:00Z",
            },
            "linkage": {
                "total_links": 250,
            },
        }

        client = _client()
        resp = client.get("/v1/memory/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert "working_memory" in data
        assert "graph" in data
        assert "timeseries" in data
        assert "linkage" in data
        assert data["working_memory"]["total_records"] == 150

    def test_stats_handles_empty_stores(self, mocker):
        """Test stats handles empty stores gracefully."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.get_stats.return_value = {
            "working_memory": {"total_records": 0},
            "graph": {"total_nodes": 0, "total_edges": 0},
            "timeseries": {"total_records": 0},
            "linkage": {"total_links": 0},
        }

        client = _client()
        resp = client.get("/v1/memory/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert data["working_memory"]["total_records"] == 0


class TestMemoryContextEndpoint:
    """Test GET /v1/memory/context endpoint."""

    def test_context_returns_aggregated_context(self, mocker):
        """Test context returns aggregated data from all stores."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.get_context.return_value = {
            "working_memory": [
                {"content": "Recent chat", "data_type": "chat"},
            ],
            "graph": [
                {
                    "source": "person:john",
                    "relationship": "located_at",
                    "target": "zone:alpha",
                },
            ],
            "timeseries": [
                {"heart_rate": 72, "timestamp": "2024-01-15T10:00:00Z"},
            ],
            "summary": "Recent activity from 3 sources",
        }

        client = _client()
        resp = client.get(
            "/v1/memory/context",
            params={"recent_minutes": 10},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "working_memory" in data
        assert "graph" in data
        assert "timeseries" in data

    def test_context_with_options(self, mocker):
        """Test context with include/exclude options."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.get_context.return_value = {
            "working_memory": [{"content": "Chat"}],
            "summary": "Working memory only",
        }

        client = _client()
        resp = client.get(
            "/v1/memory/context",
            params={
                "include_graph": False,
                "include_working_memory": True,
                "limit": 50,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "working_memory" in data


class TestMemoryPruneEndpoint:
    """Test POST /v1/memory/prune endpoint."""

    def test_prune_removes_expired_records(self, mocker):
        """Test prune removes expired records from working memory."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.prune.return_value = {
            "success": True,
            "pruned_count": 100,
            "remaining_count": 50,
        }

        client = _client()
        resp = client.post("/v1/memory/prune")

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["pruned_count"] == 100


class TestMemoryRequestValidation:
    """Test request validation for memory endpoints."""

    def test_add_requires_data(self, mocker):
        """Test add endpoint requires data field."""
        client = _client()
        resp = client.post(
            "/v1/memory/add",
            json={"data_type": "text"},
        )

        assert resp.status_code == 422

    def test_add_validates_data_type(self, mocker):
        """Test add endpoint validates data_type enum."""
        mock_memory_service = mocker.patch("api.routers.memory.router.MemoryService")
        mock_memory_service.add.return_value = {"success": True}

        client = _client()
        resp = client.post(
            "/v1/memory/add",
            json={
                "data": "test",
                "data_type": "invalid_type",
            },
        )

        # Should either work (flexible) or return 422 (strict enum)
        assert resp.status_code in [200, 422]

    def test_query_validates_time_format(self, mocker):
        """Test query validates ISO time format."""
        client = _client()
        resp = client.get(
            "/v1/memory/query",
            params={"start_time": "not-a-date"},
        )

        assert resp.status_code == 422


class TestMemoryResponseModels:
    """Test Pydantic response model serialization."""

    def test_add_response_serialization(self):
        """Test MemoryAddResponse model serializes correctly."""
        from api.routers.memory.types import MemoryAddResponse

        response = MemoryAddResponse(
            success=True,
            uuid="test-uuid",
            store="working_memory",
            message="Added successfully",
        )

        data = response.model_dump()
        assert data["success"] is True
        assert data["uuid"] == "test-uuid"
        assert data["store"] == "working_memory"

    def test_query_response_serialization(self):
        """Test MemoryQueryResponse model serializes correctly."""
        from api.routers.memory.types import MemoryQueryResponse, MemoryRecord

        record = MemoryRecord(
            uuid="rec-1",
            content="Test content",
            data_type="chat",
            store="working_memory",
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        )

        response = MemoryQueryResponse(
            results=[record],
            total_count=1,
        )

        data = response.model_dump()
        assert len(data["results"]) == 1
        assert data["total_count"] == 1
        assert data["results"][0]["uuid"] == "rec-1"

    def test_stats_response_serialization(self):
        """Test MemoryStatsResponse model serializes correctly."""
        from api.routers.memory.types import MemoryStatsResponse

        response = MemoryStatsResponse(
            working_memory={"total_records": 100},
            graph={"total_nodes": 50, "total_edges": 75},
            timeseries={"total_records": 1000},
            linkage={"total_links": 200},
        )

        data = response.model_dump()
        assert data["working_memory"]["total_records"] == 100
        assert data["graph"]["total_nodes"] == 50
