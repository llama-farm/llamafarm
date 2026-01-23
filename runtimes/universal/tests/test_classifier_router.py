"""Tests for Classifier router endpoints."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def mock_classifier_model():
    """Create a mock classifier model."""
    model = MagicMock()
    model.is_fitted = False
    model.labels = ["positive", "negative"]
    model.base_model = "sentence-transformers/all-MiniLM-L6-v2"

    # Mock fit result
    fit_result = MagicMock()
    fit_result.samples_fitted = 10
    fit_result.num_classes = 2
    fit_result.labels = ["positive", "negative"]
    fit_result.training_time_ms = 100.0
    fit_result.base_model = "sentence-transformers/all-MiniLM-L6-v2"

    async def mock_fit(*args, **kwargs):
        model.is_fitted = True
        return fit_result

    model.fit = mock_fit

    # Mock classify results
    class MockClassifyResult:
        def __init__(self, text, label, score, all_scores):
            self.text = text
            self.label = label
            self.score = score
            self.all_scores = all_scores

    async def mock_classify(texts):
        return [
            MockClassifyResult(
                text, "positive", 0.85, {"positive": 0.85, "negative": 0.15}
            )
            for text in texts
        ]

    model.classify = mock_classify

    async def mock_save(*args, **kwargs):
        pass

    model.save = mock_save

    async def mock_load(*args, **kwargs):
        model.is_fitted = True

    model.load = mock_load

    async def mock_unload(*args, **kwargs):
        pass

    model.unload = mock_unload

    return model


@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_app(mock_classifier_model, temp_models_dir):
    """Create a test FastAPI app with the classifier router."""
    from routers.classifier import (
        router,
        set_classifier_loader,
        set_models_dir,
        set_state,
    )

    app = FastAPI()
    app.include_router(router)

    # Configure the models directory
    set_models_dir(temp_models_dir)

    # Set up shared state
    classifiers = {}
    model_load_lock = asyncio.Lock()
    set_state(classifiers, model_load_lock)

    # Set up mock model loader
    async def mock_load_classifier(
        model_id, base_model="sentence-transformers/all-MiniLM-L6-v2"
    ):
        return mock_classifier_model

    set_classifier_loader(mock_load_classifier)
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestClassifierFitEndpoint:
    """Test /v1/classifier/fit endpoint."""

    def test_fit_returns_success(self, client, mock_classifier_model):
        """Test POST /v1/classifier/fit returns success with auto-save."""
        response = client.post(
            "/v1/classifier/fit",
            json={
                "model": "test-classifier",
                "training_data": [
                    {"text": "I love this!", "label": "positive"},
                    {"text": "This is great!", "label": "positive"},
                    {"text": "I hate this", "label": "negative"},
                    {"text": "This is awful", "label": "negative"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "fit_result"
        assert data["model"] == "test-classifier"
        assert data["samples_fitted"] == 10
        assert data["status"] == "fitted"

    def test_fit_autosaves_model(self, client, mock_classifier_model, temp_models_dir):
        """Test that fit automatically saves model to disk."""
        response = client.post(
            "/v1/classifier/fit",
            json={
                "model": "autosave-test",
                "training_data": [
                    {"text": "good", "label": "positive"},
                    {"text": "bad", "label": "negative"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Check that auto_saved is True
        assert (
            data.get("auto_saved", False) is True or data.get("saved_path") is not None
        )

    def test_fit_overwrite_default_is_true(self, client):
        """Test that overwrite defaults to True in fit request."""
        from api_types.classifier import ClassifierFitRequest

        request = ClassifierFitRequest(
            model="test",
            training_data=[{"text": "test", "label": "label"}],
        )
        assert request.overwrite is True

    def test_fit_requires_minimum_examples(self, client, mock_classifier_model):
        """Test that fit requires at least 2 training examples."""
        response = client.post(
            "/v1/classifier/fit",
            json={
                "model": "test-classifier",
                "training_data": [{"text": "only one", "label": "single"}],
            },
        )

        assert response.status_code == 400
        assert "2" in response.json()["detail"]


class TestClassifierPredictEndpoint:
    """Test /v1/classifier/predict endpoint."""

    def test_predict_returns_results(self, client, mock_classifier_model):
        """Test POST /v1/classifier/predict returns predictions."""
        # First fit the model to add it to the cache
        fit_response = client.post(
            "/v1/classifier/fit",
            json={
                "model": "test-classifier",
                "training_data": [
                    {"text": "good", "label": "positive"},
                    {"text": "bad", "label": "negative"},
                ],
            },
        )
        assert fit_response.status_code == 200

        response = client.post(
            "/v1/classifier/predict",
            json={
                "model": "test-classifier",
                "texts": ["This is amazing!", "This is terrible!"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["label"] == "positive"
        assert "score" in data["data"][0]
        assert "all_scores" in data["data"][0]

    def test_predict_unfitted_model_returns_error(self, client, mock_classifier_model):
        """Test predicting with unfitted model returns error."""
        mock_classifier_model.is_fitted = False

        response = client.post(
            "/v1/classifier/predict",
            json={
                "model": "unfitted-model",
                "texts": ["test text"],
            },
        )

        # Should return 404 (not found) or 400 (not fitted)
        assert response.status_code in (400, 404)


class TestClassifierSaveEndpointRemoved:
    """Test that /v1/classifier/save endpoint is removed (auto-save handles this)."""

    def test_save_endpoint_returns_404(self, client):
        """Test POST /v1/classifier/save returns 404 (removed)."""
        response = client.post(
            "/v1/classifier/save",
            json={"model": "test-model"},
        )

        # Should be 404 or 405 (not found or method not allowed)
        assert response.status_code in (404, 405)


class TestClassifierLoadEndpoint:
    """Test /v1/classifier/load endpoint."""

    def test_load_nonexistent_model_returns_404(self, client, temp_models_dir):
        """Test loading a model that doesn't exist returns 404."""
        response = client.post(
            "/v1/classifier/load",
            json={"model": "nonexistent-model"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_load_request_validation(self, client):
        """Test that load request requires model."""
        from api_types.classifier import ClassifierLoadRequest

        request = ClassifierLoadRequest(model="test")
        assert request.model == "test"


class TestClassifierModelsListEndpoint:
    """Test /v1/classifier/models endpoint."""

    def test_list_models_empty(self, client, temp_models_dir):
        """Test listing models when directory is empty."""
        response = client.get("/v1/classifier/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["total"] == 0

    def test_list_models_with_directories(self, client, temp_models_dir):
        """Test listing models when models exist."""
        # Create some fake model directories
        model1 = temp_models_dir / "classifier1"
        model1.mkdir()
        (model1 / "labels.txt").write_text("positive\nnegative")

        model2 = temp_models_dir / "classifier2"
        model2.mkdir()
        (model2 / "labels.txt").write_text("spam\nham")

        response = client.get("/v1/classifier/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["total"] == 2


class TestClassifierDeleteEndpoint:
    """Test DELETE /v1/classifier/models/{model_name} endpoint."""

    def test_delete_existing_model(self, client, temp_models_dir):
        """Test deleting an existing model directory."""
        # Create a model directory
        model_dir = temp_models_dir / "delete-me"
        model_dir.mkdir()
        (model_dir / "labels.txt").write_text("a\nb")

        response = client.delete("/v1/classifier/models/delete-me")

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert not model_dir.exists()

    def test_delete_nonexistent_model(self, client, temp_models_dir):
        """Test deleting a model that doesn't exist."""
        response = client.delete("/v1/classifier/models/nonexistent")

        assert response.status_code == 404

    def test_delete_path_traversal_blocked(self, client, temp_models_dir):
        """Test that path traversal attempts are blocked."""
        response = client.delete("/v1/classifier/models/..%2F..%2Fetc")

        # Should be 400 (blocked) or 404 (not found after sanitization)
        assert response.status_code in (400, 404)


class TestRouterInitialization:
    """Test router initialization and dependency injection."""

    def test_classifier_loader_not_set_raises_error(self):
        """Test that calling endpoints without setting loader raises error."""
        from routers.classifier import router, set_classifier_loader

        # Reset the loader
        set_classifier_loader(None)

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            "/v1/classifier/fit",
            json={
                "model": "test",
                "training_data": [
                    {"text": "a", "label": "x"},
                    {"text": "b", "label": "y"},
                ],
            },
        )

        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"].lower()
