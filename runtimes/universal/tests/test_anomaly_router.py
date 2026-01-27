"""Tests for Anomaly router endpoints."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def mock_anomaly_model():
    """Create a mock anomaly model."""
    model = MagicMock()
    model.is_fitted = False
    model.threshold = 0.5
    model.normalization = "standardization"
    model.backend = "isolation_forest"

    # Mock fit result
    fit_result = MagicMock()
    fit_result.samples_fitted = 100
    fit_result.training_time_ms = 50.0
    fit_result.model_params = {"n_estimators": 100}

    # After fit, model is fitted
    async def mock_fit(*args, **kwargs):
        model.is_fitted = True
        return fit_result

    model.fit = mock_fit

    # Mock score results
    class MockScoreResult:
        def __init__(self, idx, score, is_anomaly, raw_score):
            self.index = idx
            self.score = score
            self.is_anomaly = is_anomaly
            self.raw_score = raw_score

    async def mock_score(*args, **kwargs):
        return [
            MockScoreResult(0, 0.3, False, -0.5),
            MockScoreResult(1, 0.7, True, 0.8),
        ]

    model.score = mock_score

    async def mock_detect(*args, **kwargs):
        return [
            MockScoreResult(1, 0.7, True, 0.8),
        ]

    model.detect = mock_detect

    # Use AsyncMock for save to enable call tracking
    model.save = AsyncMock()

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
def test_app(mock_anomaly_model, temp_models_dir):
    """Create a test FastAPI app with the anomaly router."""
    from routers.anomaly import router, set_anomaly_loader, set_models_dir, set_state

    app = FastAPI()
    app.include_router(router)

    # Configure the models directory
    set_models_dir(temp_models_dir)

    # Set up shared state (models cache, encoders, lock)
    models = {}
    encoders = {}
    model_load_lock = asyncio.Lock()
    set_state(models, encoders, model_load_lock)

    # Set up mock model loader
    async def mock_load_anomaly(model_id, backend="isolation_forest", **kwargs):
        return mock_anomaly_model

    set_anomaly_loader(mock_load_anomaly)
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestAnomalyFitEndpoint:
    """Test /v1/anomaly/fit endpoint."""

    def test_fit_returns_success(self, client, mock_anomaly_model, temp_models_dir):
        """Test POST /v1/anomaly/fit returns success with auto-save."""
        response = client.post(
            "/v1/anomaly/fit",
            json={
                "model": "test-detector",
                "backend": "isolation_forest",
                "data": [[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]],
                "contamination": 0.1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "fit_result"
        assert data["model"] == "test-detector"
        assert data["backend"] == "isolation_forest"
        assert data["samples_fitted"] == 100
        assert data["status"] == "fitted"

    def test_fit_autosaves_model(self, client, mock_anomaly_model, temp_models_dir):
        """Test that fit automatically saves model to disk."""
        # Reset the save mock to track this specific test
        mock_anomaly_model.save.reset_mock()

        response = client.post(
            "/v1/anomaly/fit",
            json={
                "model": "autosave-test",
                "backend": "isolation_forest",
                "data": [[1.0, 2.0], [1.1, 2.1]],
            },
        )

        assert response.status_code == 200
        # Verify save was called (auto-save)
        assert mock_anomaly_model.save.called, "Model.save() should be called after fit"

    def test_fit_overwrite_default_is_true(self, client):
        """Test that overwrite defaults to True in fit request."""
        # The default is set in api_types/anomaly.py AnomalyFitRequest
        from api_types.anomaly import AnomalyFitRequest

        request = AnomalyFitRequest(
            model="test",
            data=[[1.0, 2.0]],
        )
        assert request.overwrite is True


class TestAnomalyScoreEndpoint:
    """Test /v1/anomaly/score endpoint."""

    def test_score_returns_results(self, client, mock_anomaly_model):
        """Test POST /v1/anomaly/score returns anomaly scores."""
        # First fit the model
        mock_anomaly_model.is_fitted = True

        response = client.post(
            "/v1/anomaly/score",
            json={
                "model": "test-detector",
                "backend": "isolation_forest",
                "data": [[1.0, 2.0], [100.0, 200.0]],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) == 2
        assert data["data"][0]["index"] == 0
        assert "score" in data["data"][0]
        assert "is_anomaly" in data["data"][0]

    def test_score_unfitted_model_returns_error(self, client, mock_anomaly_model):
        """Test scoring with unfitted model returns 400 error."""
        mock_anomaly_model.is_fitted = False

        response = client.post(
            "/v1/anomaly/score",
            json={
                "model": "unfitted-model",
                "backend": "isolation_forest",
                "data": [[1.0, 2.0]],
            },
        )

        assert response.status_code == 400
        assert "not fitted" in response.json()["detail"].lower()


class TestAnomalyDetectEndpoint:
    """Test /v1/anomaly/detect endpoint."""

    def test_detect_returns_only_anomalies(self, client, mock_anomaly_model):
        """Test POST /v1/anomaly/detect returns only anomalous points."""
        mock_anomaly_model.is_fitted = True

        response = client.post(
            "/v1/anomaly/detect",
            json={
                "model": "test-detector",
                "backend": "isolation_forest",
                "data": [[1.0, 2.0], [100.0, 200.0]],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        # Only anomalies returned
        assert len(data["data"]) == 1
        assert data["data"][0]["index"] == 1  # Second item was anomaly


class TestAnomalySaveEndpointRemoved:
    """Test that /v1/anomaly/save endpoint is removed (auto-save handles this)."""

    def test_save_endpoint_returns_404(self, client):
        """Test POST /v1/anomaly/save returns 404 (removed)."""
        response = client.post(
            "/v1/anomaly/save",
            json={
                "model": "test-model",
                "backend": "isolation_forest",
            },
        )

        # Should be 404 or 405 (not found or method not allowed)
        assert response.status_code in (404, 405)


class TestAnomalyLoadEndpoint:
    """Test /v1/anomaly/load endpoint."""

    def test_load_nonexistent_model_returns_404(self, client, temp_models_dir):
        """Test loading a model that doesn't exist returns 404."""
        response = client.post(
            "/v1/anomaly/load",
            json={
                "model": "nonexistent-model",
                "backend": "isolation_forest",
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_load_request_validation(self, client):
        """Test that load request requires model and backend."""
        from api_types.anomaly import AnomalyLoadRequest

        # Test default backend
        request = AnomalyLoadRequest(model="test")
        assert request.backend == "isolation_forest"


class TestAnomalyModelsListEndpoint:
    """Test /v1/anomaly/models endpoint."""

    def test_list_models_empty(self, client, temp_models_dir):
        """Test listing models when directory is empty."""
        response = client.get("/v1/anomaly/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["total"] == 0

    def test_list_models_with_files(self, client, temp_models_dir):
        """Test listing models when models exist."""
        # Create some fake model files
        (temp_models_dir / "model1_isolation_forest.joblib").write_bytes(b"data")
        (temp_models_dir / "model2_autoencoder.pt").write_bytes(b"data")

        response = client.get("/v1/anomaly/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["total"] == 2


class TestAnomalyDeleteEndpoint:
    """Test DELETE /v1/anomaly/models/{filename} endpoint."""

    def test_delete_existing_model(self, client, temp_models_dir):
        """Test deleting an existing model file."""
        # Create a model file
        model_file = temp_models_dir / "delete-me_isolation_forest.joblib"
        model_file.write_bytes(b"data")

        response = client.delete("/v1/anomaly/models/delete-me_isolation_forest.joblib")

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert not model_file.exists()

    def test_delete_nonexistent_model(self, client, temp_models_dir):
        """Test deleting a model that doesn't exist."""
        response = client.delete("/v1/anomaly/models/nonexistent.joblib")

        assert response.status_code == 404

    def test_delete_path_traversal_blocked(self, client, temp_models_dir):
        """Test that path traversal attempts are blocked."""
        # Test with double dots directly (not URL encoded)
        response = client.delete("/v1/anomaly/models/..%2Fpasswd")

        # Should either be 400 (blocked) or 404 (not found after sanitization)
        # The important thing is it doesn't delete anything outside the dir
        assert response.status_code in (400, 404)

        # Test with explicit path separator check
        response2 = client.delete("/v1/anomaly/models/sub/../../../etc/passwd")
        assert response2.status_code in (400, 404)


class TestRouterInitialization:
    """Test router initialization and dependency injection."""

    def test_anomaly_loader_not_set_raises_error(self):
        """Test that calling endpoints without setting anomaly loader raises error."""
        from routers.anomaly import (
            get_anomaly_loader,
            router,
            set_anomaly_loader,
        )

        # Store original loader to restore later
        original_loader = get_anomaly_loader()

        try:
            # Reset the loader
            set_anomaly_loader(None)

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.post(
                "/v1/anomaly/score",
                json={"model": "test", "data": [[1.0, 2.0]]},
            )

            assert response.status_code == 500
            assert "not initialized" in response.json()["detail"].lower()
        finally:
            # Restore original loader to prevent test pollution
            set_anomaly_loader(original_loader)
