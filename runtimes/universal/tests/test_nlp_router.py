"""Tests for NLP router endpoints."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def mock_encoder():
    """Create a mock encoder model."""
    encoder = AsyncMock()
    encoder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    encoder.rerank = AsyncMock(
        return_value=[
            {"index": 1, "relevance_score": 0.95, "document": "doc2"},
            {"index": 0, "relevance_score": 0.75, "document": "doc1"},
        ]
    )
    encoder.classify = AsyncMock(
        return_value=[
            {
                "label": "POSITIVE",
                "score": 0.95,
                "all_scores": {"POSITIVE": 0.95, "NEGATIVE": 0.05},
            },
        ]
    )

    class MockEntity:
        def __init__(self, text, label, start, end, score):
            self.text = text
            self.label = label
            self.start = start
            self.end = end
            self.score = score

    encoder.extract_entities = AsyncMock(
        return_value=[[MockEntity("John", "PERSON", 0, 4, 0.99)]]
    )
    return encoder


@pytest.fixture
def test_app(mock_encoder):
    """Create a test FastAPI app with the NLP router."""
    from routers.nlp import router, set_encoder_loader

    app = FastAPI()
    app.include_router(router)

    async def mock_load_encoder(model_id, task="embedding", **kwargs):
        return mock_encoder

    set_encoder_loader(mock_load_encoder)
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestEmbeddingsEndpoint:
    """Test /v1/embeddings endpoint."""

    def test_embeddings_endpoint(self, client, mock_encoder):
        """Test POST /v1/embeddings returns vectors."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-model",
                "input": ["hello", "world"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["object"] == "embedding"
        assert data["data"][0]["index"] == 0
        assert isinstance(data["data"][0]["embedding"], list)
        mock_encoder.embed.assert_called_once()

    def test_embeddings_single_input(self, client, mock_encoder):
        """Test embeddings with single string input."""
        mock_encoder.embed.return_value = [[0.1, 0.2, 0.3]]

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-model",
                "input": "single text",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1


class TestRerankEndpoint:
    """Test /v1/rerank endpoint."""

    def test_rerank_endpoint(self, client, mock_encoder):
        """Test POST /v1/rerank returns reordered docs."""
        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-reranker",
                "query": "what is AI?",
                "documents": ["doc1", "doc2"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        # First result should have higher score
        assert data["data"][0]["relevance_score"] > data["data"][1]["relevance_score"]
        mock_encoder.rerank.assert_called_once()

    def test_rerank_without_documents(self, client, mock_encoder):
        """Test rerank with return_documents=False."""
        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-reranker",
                "query": "what is AI?",
                "documents": ["doc1", "doc2"],
                "return_documents": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Documents should be None when return_documents=False
        for item in data["data"]:
            assert item.get("document") is None


class TestClassifyEndpoint:
    """Test /v1/classify endpoint."""

    def test_classify_endpoint(self, client, mock_encoder):
        """Test POST /v1/classify returns labels."""
        response = client.post(
            "/v1/classify",
            json={
                "model": "test-classifier",
                "texts": ["I love this!"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["total_count"] == 1
        assert data["data"][0]["label"] == "POSITIVE"
        assert data["data"][0]["score"] > 0.9
        mock_encoder.classify.assert_called_once()


class TestNEREndpoint:
    """Test /v1/ner endpoint."""

    def test_ner_endpoint(self, client, mock_encoder):
        """Test POST /v1/ner returns entities."""
        response = client.post(
            "/v1/ner",
            json={
                "model": "test-ner",
                "texts": ["John works at Google."],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert len(data["data"][0]["entities"]) == 1
        assert data["data"][0]["entities"][0]["text"] == "John"
        assert data["data"][0]["entities"][0]["label"] == "PERSON"
        mock_encoder.extract_entities.assert_called_once()


class TestRouterInitialization:
    """Test router initialization and dependency injection."""

    def test_encoder_loader_not_set_raises_error(self):
        """Test that calling endpoints without setting encoder loader raises error."""
        from routers.nlp import router, set_encoder_loader

        # Reset the loader
        set_encoder_loader(None)

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            "/v1/embeddings",
            json={"model": "test", "input": "hello"},
        )

        # Should return 500 error (internal details hidden for security)
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()
