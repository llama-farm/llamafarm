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
    mock_list = mocker.patch("api.routers.models.UniversalProvider.list_cached_models")
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

    mock_list = mocker.patch("api.routers.models.UniversalProvider.list_cached_models")
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
    mock_list = mocker.patch("api.routers.models.UniversalProvider.list_cached_models")
    mock_list.side_effect = Exception("Cache directory not found")

    client = _client()
    resp = client.get("/v1/models")

    # Should return a 500 or appropriate error status
    assert resp.status_code >= 400
