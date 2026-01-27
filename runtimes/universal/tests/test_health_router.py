"""Tests for Health router endpoints (health check, models list)."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def mock_device_info():
    """Mock device info."""
    return {
        "device": "cpu",
        "device_name": "CPU",
        "platform": "darwin",
        "python_version": "3.12.0",
    }


@pytest.fixture
def mock_models_cache():
    """Mock models cache with some models loaded."""
    mock_encoder = MagicMock()
    mock_encoder.model_type = "encoder"

    mock_language = MagicMock()
    mock_language.model_type = "language"

    return {
        "encoder:all-MiniLM-L6-v2": mock_encoder,
        "language:gpt2": mock_language,
    }


@pytest.fixture
def test_app(mock_device_info, mock_models_cache):
    """Create a test FastAPI app with the health router."""
    from routers.health import router, set_device_info_getter, set_models_cache

    app = FastAPI()
    app.include_router(router)

    # Set up dependency injection
    set_models_cache(mock_models_cache)
    set_device_info_getter(lambda: mock_device_info)

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_check_returns_healthy(self, client, mock_device_info):
        """Test GET /health returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data
        assert "loaded_models" in data
        assert "timestamp" in data
        assert "pid" in data

    def test_health_check_includes_device_info(self, client, mock_device_info):
        """Test GET /health includes device information."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["device"] == mock_device_info

    def test_health_check_lists_loaded_models(self, client, mock_models_cache):
        """Test GET /health lists loaded model IDs."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert set(data["loaded_models"]) == set(mock_models_cache.keys())


class TestModelsListEndpoint:
    """Test GET /v1/models endpoint."""

    def test_list_models_returns_loaded_models(self, client, mock_models_cache):
        """Test GET /v1/models returns list of loaded models."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2

    def test_list_models_includes_model_metadata(self, client, mock_models_cache):
        """Test GET /v1/models includes model metadata."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        # Find the encoder model
        encoder_model = next((m for m in data["data"] if "all-MiniLM" in m["id"]), None)
        assert encoder_model is not None
        assert encoder_model["object"] == "model"
        assert encoder_model["type"] == "encoder"
        assert "created" in encoder_model
        assert encoder_model["owned_by"] == "transformers-runtime"

    def test_list_models_empty_when_no_models(self, test_app):
        """Test GET /v1/models returns empty list when no models loaded."""
        from routers.health import set_models_cache

        set_models_cache({})
        client = TestClient(test_app)

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 0


class TestRouterInitialization:
    """Test router initialization and dependency injection."""

    def test_health_without_models_cache_raises_error(self):
        """Test that /health without models cache raises error."""
        from routers.health import router, set_device_info_getter, set_models_cache

        # Reset state
        set_models_cache(None)
        set_device_info_getter(None)

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"].lower()

    def test_models_without_cache_raises_error(self):
        """Test that /v1/models without models cache raises error."""
        from routers.health import router, set_models_cache

        # Reset state
        set_models_cache(None)

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/v1/models")

        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"].lower()
