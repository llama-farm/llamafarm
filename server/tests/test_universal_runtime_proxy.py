"""Tests for Universal Runtime proxy endpoints (NLP, auto-save, port configuration)."""

import pytest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def mock_runtime_service():
    """Mock the UniversalRuntimeService methods."""
    with patch("api.routers.nlp.router.UniversalRuntimeService") as mock:
        # Mock embeddings
        mock.embeddings = AsyncMock(return_value={
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        })

        # Mock rerank
        mock.rerank = AsyncMock(return_value={
            "object": "list",
            "data": [
                {"index": 0, "relevance_score": 0.9, "document": "doc1"},
                {"index": 1, "relevance_score": 0.5, "document": "doc2"},
            ],
        })

        # Mock classify
        mock.classify = AsyncMock(return_value={
            "object": "list",
            "data": [{"label": "positive", "score": 0.95}],
        })

        # Mock ner
        mock.ner = AsyncMock(return_value={
            "object": "list",
            "data": [[{"word": "Apple", "entity": "ORG", "score": 0.99}]],
        })

        yield mock


@pytest.fixture
def test_app(mock_runtime_service):
    """Create a test FastAPI app with the NLP router."""
    from api.routers.nlp import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestEmbeddingsEndpoint:
    """Test POST /v1/nlp/embeddings endpoint."""

    def test_embeddings_proxies_correctly(self, client, mock_runtime_service):
        """Test that embeddings request is proxied to Universal Runtime."""
        response = client.post(
            "/v1/nlp/embeddings",
            json={
                "input": "Hello world",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        mock_runtime_service.embeddings.assert_called_once()

    def test_embeddings_with_multiple_inputs(self, client, mock_runtime_service):
        """Test embeddings with list of inputs."""
        response = client.post(
            "/v1/nlp/embeddings",
            json={
                "input": ["Hello", "World"],
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        )

        assert response.status_code == 200


class TestRerankEndpoint:
    """Test POST /v1/nlp/rerank endpoint."""

    def test_rerank_proxies_correctly(self, client, mock_runtime_service):
        """Test that rerank request is proxied to Universal Runtime."""
        response = client.post(
            "/v1/nlp/rerank",
            json={
                "query": "What is AI?",
                "documents": ["AI is artificial intelligence", "The weather is nice"],
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        mock_runtime_service.rerank.assert_called_once()

    def test_rerank_with_top_n(self, client, mock_runtime_service):
        """Test rerank with top_n parameter."""
        response = client.post(
            "/v1/nlp/rerank",
            json={
                "query": "What is AI?",
                "documents": ["doc1", "doc2", "doc3"],
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "top_n": 2,
            },
        )

        assert response.status_code == 200


class TestClassifyEndpoint:
    """Test POST /v1/nlp/classify endpoint."""

    def test_classify_proxies_correctly(self, client, mock_runtime_service):
        """Test that classify request is proxied to Universal Runtime."""
        response = client.post(
            "/v1/nlp/classify",
            json={
                "input": "I love this product!",
                "model": "facebook/bart-large-mnli",
                "labels": ["positive", "negative", "neutral"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        mock_runtime_service.classify.assert_called_once()


class TestNEREndpoint:
    """Test POST /v1/nlp/ner endpoint."""

    def test_ner_proxies_correctly(self, client, mock_runtime_service):
        """Test that NER request is proxied to Universal Runtime."""
        response = client.post(
            "/v1/nlp/ner",
            json={
                "input": "Apple Inc. is based in California.",
                "model": "dslim/bert-base-NER",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        mock_runtime_service.ner.assert_called_once()


class TestPortConfiguration:
    """Test that port configuration uses settings, not hardcoded values."""

    def test_port_comes_from_settings(self):
        """Test that UniversalRuntimeService uses settings for port."""
        from server.services.universal_runtime_service import UniversalRuntimeService
        from server.core.settings import settings

        base_url = UniversalRuntimeService.get_base_url()

        assert str(settings.universal_port) in base_url
        assert settings.universal_host in base_url
        assert "11540" not in base_url or settings.universal_port == 11540


class TestMLAutoSave:
    """Test that ML fit endpoints document auto-save behavior."""

    def test_classifier_fit_docstring_mentions_autosave(self):
        """Test that classifier fit endpoint documents auto-save."""
        from api.routers.ml.router import fit_classifier

        docstring = fit_classifier.__doc__
        assert "automatically saved" in docstring.lower()
        assert "no separate save step" in docstring.lower()

    def test_anomaly_fit_docstring_mentions_autosave(self):
        """Test that anomaly fit endpoint documents auto-save."""
        from api.routers.ml.router import fit_anomaly_detector

        docstring = fit_anomaly_detector.__doc__
        assert "automatically saved" in docstring.lower()
        assert "no separate save step" in docstring.lower()


class TestOverwriteDefault:
    """Test that overwrite default is True in ML types."""

    def test_classifier_fit_overwrite_default_is_true(self):
        """Test that ClassifierFitRequest.overwrite defaults to True."""
        from api.routers.ml.types import ClassifierFitRequest

        # Create request with minimal required fields
        request = ClassifierFitRequest(
            model="test",
            training_data=[{"text": "hello", "label": "greeting"}],
        )
        assert request.overwrite is True

    def test_anomaly_fit_overwrite_default_is_true(self):
        """Test that AnomalyFitRequest.overwrite defaults to True."""
        from api.routers.ml.types import AnomalyFitRequest

        # Create request with minimal required fields
        request = AnomalyFitRequest(
            model="test",
            data=[[1.0, 2.0]],
        )
        assert request.overwrite is True
