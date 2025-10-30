"""Tests for models endpoint."""

from fastapi.testclient import TestClient

from api.main import llama_farm_api
from services.runtime_service.providers.base import CachedModel


def _client() -> TestClient:
    """Create test client."""
    app = llama_farm_api()
    return TestClient(app)


def test_list_models_empty(mocker):
    """Test models endpoint when no models are cached."""
    # Mock list_cached_models to return empty list
    mock_list = mocker.patch(
        "server.services.model_service.ModelService.list_cached_models"
    )
    mock_list.return_value = []

    client = _client()
    resp = client.get("/v1/models")

    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert data["data"] == []


def test_list_models_with_cached_models(mocker):
    """Test models endpoint with cached models."""
    # Mock list_cached_models to return test models
    test_models = [
        CachedModel(
            id="meta-llama/Llama-2-7b-hf",
            name="meta-llama/Llama-2-7b-hf",
            size=13476520960,
            path="/path/to/models/models--meta-llama--Llama-2-7b-hf",
        ),
        CachedModel(
            id="sentence-transformers/all-MiniLM-L6-v2",
            name="sentence-transformers/all-MiniLM-L6-v2",
            size=91627520,
            path="/path/to/models/models--sentence-transformers--all-MiniLM-L6-v2",
        ),
    ]

    mock_list = mocker.patch(
        "server.services.model_service.ModelService.list_cached_models"
    )
    mock_list.return_value = test_models

    client = _client()
    resp = client.get("/v1/models")

    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert len(data["data"]) == 2

    # Verify first model
    model1 = data["data"][0]
    assert model1["id"] == "meta-llama/Llama-2-7b-hf"
    assert model1["name"] == "meta-llama/Llama-2-7b-hf"
    assert model1["size"] == 13476520960
    assert "Llama-2-7b-hf" in model1["path"]

    # Verify second model
    model2 = data["data"][1]
    assert model2["id"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert model2["name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert model2["size"] == 91627520
    assert "all-MiniLM-L6-v2" in model2["path"]


def test_list_models_handles_errors(mocker):
    """Test models endpoint handles errors gracefully."""
    # Mock list_cached_models to raise an exception
    mock_list = mocker.patch(
        "server.services.model_service.ModelService.list_cached_models"
    )
    mock_list.side_effect = Exception("Cache directory not found")

    client = _client()
    resp = client.get("/v1/models")

    # Should return a 500 or appropriate error status
    assert resp.status_code >= 400


def test_delete_model_success(mocker):
    """Test deleting a model successfully."""
    # Mock delete_model to return success info
    mock_delete = mocker.patch(
        "server.services.model_service.ModelService.delete_model"
    )
    mock_delete.return_value = {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "revisions_deleted": 1,
        "size_freed": 13476520960,
        "path": "/path/to/models/models--meta-llama--Llama-2-7b-hf",
    }

    client = _client()
    resp = client.delete("/v1/models/meta-llama/Llama-2-7b-hf")

    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "meta-llama/Llama-2-7b-hf"
    assert data["revisions_deleted"] == 1
    assert data["size_freed"] == 13476520960
    assert "Llama-2-7b-hf" in data["path"]


def test_delete_model_not_found(mocker):
    """Test deleting a model that doesn't exist."""
    # Mock delete_model to raise ValueError with "not found"
    mock_delete = mocker.patch(
        "server.services.model_service.ModelService.delete_model"
    )
    mock_delete.side_effect = ValueError(
        "Model 'nonexistent/model' not found in cache."
    )

    client = _client()
    resp = client.delete("/v1/models/nonexistent/model")

    assert resp.status_code == 404
    data = resp.json()
    assert "not found" in data["detail"].lower()


def test_delete_model_invalid_provider(mocker):
    """Test deleting a model with an unsupported provider."""
    # Mock delete_model to raise ValueError for unsupported provider
    # Note: Using 'openai' as a valid enum value that will be handled by our code
    mock_delete = mocker.patch(
        "server.services.model_service.ModelService.delete_model"
    )
    mock_delete.side_effect = ValueError("Unsupported provider: openai")

    client = _client()
    resp = client.delete("/v1/models/some-model?provider=openai")

    assert resp.status_code == 400
    data = resp.json()
    assert "Unsupported provider" in data["detail"]


def test_delete_model_handles_errors(mocker):
    """Test delete endpoint handles unexpected errors gracefully."""
    # Mock delete_model to raise an unexpected exception
    mock_delete = mocker.patch(
        "server.services.model_service.ModelService.delete_model"
    )
    mock_delete.side_effect = Exception("Unexpected error")

    client = _client()
    resp = client.delete("/v1/models/some-model")

    assert resp.status_code == 500
    data = resp.json()
    # Should return generic error message for security (no internal details exposed)
    assert (
        "An error occurred while deleting the model" in data["detail"]
        or "contact support" in data["detail"]
    )
