"""Tests for RAG stats endpoint."""

from datetime import UTC, datetime
from unittest.mock import Mock

from fastapi.testclient import TestClient

from api.main import llama_farm_api


def _client() -> TestClient:
    """Create test client."""
    app = llama_farm_api()
    return TestClient(app)


def test_rag_stats_endpoint_requires_rag_config(mocker):
    """Test that stats endpoint returns 400 when RAG is not configured."""
    # Mock ProjectService to return a project without RAG config
    mock_project = Mock()
    mock_project.config.rag = None

    mock_project_service = mocker.patch("api.routers.rag.router.ProjectService")
    mock_project_service.get_project.return_value = mock_project
    mock_project_service.get_project_dir.return_value = "/fake/path"

    client = _client()
    resp = client.get("/v1/projects/test-ns/test-project/rag/stats")

    assert resp.status_code == 400
    assert "RAG not configured" in resp.json()["detail"]


def test_rag_stats_endpoint_success(mocker):
    """Test successful stats retrieval."""
    # Mock ProjectService to return a project with RAG config
    mock_database = Mock()
    mock_database.name = "test_db"

    mock_rag_config = Mock()
    mock_rag_config.databases = [mock_database]

    mock_project = Mock()
    mock_project.config.rag = mock_rag_config

    mock_project_service = mocker.patch("api.routers.rag.router.ProjectService")
    mock_project_service.get_project.return_value = mock_project
    mock_project_service.get_project_dir.return_value = "/fake/path"

    # Mock the stats handler to return test data
    mock_stats_response = Mock()
    mock_stats_response.database = "test_db"
    mock_stats_response.vector_count = 1000
    mock_stats_response.document_count = 50
    mock_stats_response.chunk_count = 1000
    mock_stats_response.collection_size_bytes = 5242880
    mock_stats_response.index_size_bytes = 1048576
    mock_stats_response.embedding_dimension = 768
    mock_stats_response.distance_metric = "cosine"
    mock_stats_response.last_updated = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    mock_stats_response.metadata = {"collection_name": "test_db"}

    mock_handle_rag_stats = mocker.patch("api.routers.rag.router.handle_rag_stats")
    mock_handle_rag_stats.return_value = mock_stats_response

    client = _client()
    resp = client.get("/v1/projects/test-ns/test-project/rag/stats")

    assert resp.status_code == 200
    data = resp.json()
    assert data["database"] == "test_db"
    assert data["vector_count"] == 1000
    assert data["document_count"] == 50
    assert data["chunk_count"] == 1000
    assert data["embedding_dimension"] == 768
    assert data["distance_metric"] == "cosine"


def test_rag_stats_endpoint_with_database_param(mocker):
    """Test stats endpoint with specific database parameter."""
    # Mock ProjectService
    mock_database = Mock()
    mock_database.name = "specific_db"

    mock_rag_config = Mock()
    mock_rag_config.databases = [mock_database]

    mock_project = Mock()
    mock_project.config.rag = mock_rag_config

    mock_project_service = mocker.patch("api.routers.rag.router.ProjectService")
    mock_project_service.get_project.return_value = mock_project
    mock_project_service.get_project_dir.return_value = "/fake/path"

    # Mock the stats handler
    mock_stats_response = Mock()
    mock_stats_response.database = "specific_db"
    mock_stats_response.vector_count = 500
    mock_stats_response.document_count = 25
    mock_stats_response.chunk_count = 500
    mock_stats_response.collection_size_bytes = 2621440
    mock_stats_response.index_size_bytes = 524288
    mock_stats_response.embedding_dimension = 1536
    mock_stats_response.distance_metric = "l2"
    mock_stats_response.last_updated = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    mock_stats_response.metadata = None

    mock_handle_rag_stats = mocker.patch("api.routers.rag.router.handle_rag_stats")
    mock_handle_rag_stats.return_value = mock_stats_response

    client = _client()
    resp = client.get(
        "/v1/projects/test-ns/test-project/rag/stats?database=specific_db"
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["database"] == "specific_db"
    assert data["vector_count"] == 500
    assert data["embedding_dimension"] == 1536
    assert data["distance_metric"] == "l2"

    # Verify handle_rag_stats was called with the specific database
    mock_handle_rag_stats.assert_called_once()
    call_args = mock_handle_rag_stats.call_args
    assert call_args[0][2] == "specific_db"  # database parameter


def test_rag_stats_endpoint_empty_database(mocker):
    """Test stats endpoint with empty database (should use first configured database)."""
    # Mock ProjectService
    mock_database1 = Mock()
    mock_database1.name = "first_db"
    mock_database2 = Mock()
    mock_database2.name = "second_db"

    mock_rag_config = Mock()
    mock_rag_config.databases = [mock_database1, mock_database2]

    mock_project = Mock()
    mock_project.config.rag = mock_rag_config

    mock_project_service = mocker.patch("api.routers.rag.router.ProjectService")
    mock_project_service.get_project.return_value = mock_project
    mock_project_service.get_project_dir.return_value = "/fake/path"

    # Mock the stats handler
    mock_stats_response = Mock()
    mock_stats_response.database = "first_db"
    mock_stats_response.vector_count = 100
    mock_stats_response.document_count = 10
    mock_stats_response.chunk_count = 100
    mock_stats_response.collection_size_bytes = 1000000
    mock_stats_response.index_size_bytes = 100000
    mock_stats_response.embedding_dimension = 768
    mock_stats_response.distance_metric = "cosine"
    mock_stats_response.last_updated = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    mock_stats_response.metadata = {}

    mock_handle_rag_stats = mocker.patch("api.routers.rag.router.handle_rag_stats")
    mock_handle_rag_stats.return_value = mock_stats_response

    client = _client()
    resp = client.get("/v1/projects/test-ns/test-project/rag/stats")

    assert resp.status_code == 200
    # Should default to first database
    mock_handle_rag_stats.assert_called_once()
    call_args = mock_handle_rag_stats.call_args
    assert call_args[0][2] is None  # No database parameter passed


def test_rag_stats_response_model_serialization():
    """Test that RAGStatsResponse model serializes correctly."""
    from api.routers.rag.rag_stats import RAGStatsResponse

    response = RAGStatsResponse(
        database="test_db",
        vector_count=1000,
        document_count=50,
        chunk_count=1000,
        collection_size_bytes=5242880,
        index_size_bytes=1048576,
        embedding_dimension=768,
        distance_metric="cosine",
        last_updated=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        metadata={"key": "value"},
    )

    # Verify serialization produces expected fields
    data = response.model_dump()
    assert data["database"] == "test_db"
    assert data["vector_count"] == 1000
    assert data["document_count"] == 50
    assert data["chunk_count"] == 1000
    assert data["collection_size_bytes"] == 5242880
    assert data["index_size_bytes"] == 1048576
    assert data["embedding_dimension"] == 768
    assert data["distance_metric"] == "cosine"
    assert data["metadata"] == {"key": "value"}
