"""Tests for RAG health endpoint."""

from unittest.mock import Mock
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from api.main import llama_farm_api


def _client() -> TestClient:
    """Create test client."""
    app = llama_farm_api()
    return TestClient(app)


def test_rag_health_endpoint_requires_rag_config(mocker):
    """Test that health endpoint returns 400 when RAG is not configured."""
    # Mock ProjectService to return a project without RAG config
    mock_project = Mock()
    mock_project.config.rag = None

    mock_project_service = mocker.patch("api.routers.rag.router.ProjectService")
    mock_project_service.get_project.return_value = mock_project
    mock_project_service.get_project_dir.return_value = "/fake/path"

    client = _client()
    resp = client.get("/v1/projects/test-ns/test-project/rag/health")

    assert resp.status_code == 400
    assert "RAG not configured" in resp.json()["detail"]


def test_rag_health_endpoint_success(mocker):
    """Test successful health check."""
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

    # Mock the health check function to return successful health data
    mock_health_response = {
        "status": "healthy",
        "database": "test_db",
        "components": {
            "task_system": {
                "name": "task_system",
                "status": "healthy",
                "latency": 10.5,
                "message": "RAG worker processing tasks",
            }
        },
        "last_check": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "issues": None,
    }

    mock_handle_rag_health = mocker.patch("api.routers.rag.router.handle_rag_health")
    mock_handle_rag_health.return_value = mock_health_response

    client = _client()
    resp = client.get("/v1/projects/test-ns/test-project/rag/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["database"] == "test_db"
    assert "components" in data
    assert data["components"]["task_system"]["status"] == "healthy"


def test_rag_health_endpoint_with_database_param(mocker):
    """Test health endpoint with specific database parameter."""
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

    # Mock the health check function
    mock_health_response = {
        "status": "degraded",
        "database": "specific_db",
        "components": {
            "database_connection": {
                "name": "database_connection",
                "status": "degraded",
                "latency": 150.0,
                "message": "Database connection issues",
            }
        },
        "last_check": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "issues": ["Database connection error: Connection timeout"],
    }

    mock_handle_rag_health = mocker.patch("api.routers.rag.router.handle_rag_health")
    mock_handle_rag_health.return_value = mock_health_response

    client = _client()
    resp = client.get(
        "/v1/projects/test-ns/test-project/rag/health?database=specific_db"
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["database"] == "specific_db"
    assert len(data["issues"]) == 1
    assert "Database connection error" in data["issues"][0]

    # Verify handle_rag_health was called with the specific database
    mock_handle_rag_health.assert_called_once()
    call_args = mock_handle_rag_health.call_args
    assert call_args[0][2] == "specific_db"  # database parameter
