#!/usr/bin/env python3
"""
Test suite for Router API endpoints.
Tests Phase 4: Router API Endpoints in Universal Runtime
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRouterTrainEndpoint:
    """Test POST /v1/router/train endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked router."""
        from server import app

        return TestClient(app)

    @pytest.fixture
    def route_config(self):
        """Sample route configuration for training."""
        return {
            "model": "test_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "general_llm",
            "similarity_threshold": 0.7,
            "routes": [
                {
                    "name": "billing",
                    "target_model": "billing_model",
                    "utterances": [
                        "what is my bill",
                        "how much do I owe",
                        "payment options",
                    ],
                },
                {
                    "name": "support",
                    "target_model": "support_model",
                    "utterances": [
                        "help with login",
                        "password reset",
                        "app not working",
                    ],
                },
            ],
        }

    def test_train_router_creates_router(self, client, route_config):
        """Test POST /v1/router/train creates a router from config."""
        response = client.post("/v1/router/train", json=route_config)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "train_result"
        assert data["model"] == "test_router"
        assert data["status"] == "trained"
        assert data["num_routes"] == 2
        assert "billing" in data["routes"]
        assert "support" in data["routes"]

    def test_train_router_requires_routes(self, client):
        """Test training fails without routes."""
        config = {
            "model": "test_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "routes": [],
        }
        response = client.post("/v1/router/train", json=config)

        assert response.status_code == 400
        assert "At least 1 route required" in response.json()["detail"]


class TestRouterRouteEndpoint:
    """Test POST /v1/router/route endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from server import app

        return TestClient(app)

    @pytest.fixture
    def trained_router(self, client):
        """Train a router for testing."""
        config = {
            "model": "route_test_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "general_llm",
            "similarity_threshold": 0.5,
            "routes": [
                {
                    "name": "billing",
                    "target_model": "billing_model",
                    "utterances": ["what is my bill", "billing inquiry"],
                },
            ],
        }
        response = client.post("/v1/router/train", json=config)
        assert response.status_code == 200
        return response.json()

    def test_route_query_returns_decision(self, client, trained_router):
        """Test POST /v1/router/route returns correct routing decision."""
        request = {
            "model": "route_test_router",
            "query": "what is my bill",
        }
        response = client.post("/v1/router/route", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "route_decision"
        assert data["route_name"] == "billing"
        assert data["target_model"] == "billing_model"
        assert data["similarity_score"] > 0.5

    def test_route_unknown_query_returns_default(self, client, trained_router):
        """Test routing unknown query returns default model."""
        request = {
            "model": "route_test_router",
            "query": "tell me about quantum physics",
        }
        response = client.post("/v1/router/route", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["target_model"] == "general_llm"

    def test_route_nonexistent_router_fails(self, client):
        """Test routing with nonexistent router fails."""
        request = {
            "model": "nonexistent_router",
            "query": "test query",
        }
        response = client.post("/v1/router/route", json=request)

        assert response.status_code == 404


class TestRouterModelsEndpoint:
    """Test GET /v1/router/models endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from server import app

        return TestClient(app)

    def test_list_models_returns_list(self, client):
        """Test GET /v1/router/models returns list of saved routers."""
        response = client.get("/v1/router/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)
        assert "total" in data


class TestRouterLoadEndpoint:
    """Test POST /v1/router/load endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from server import app

        return TestClient(app)

    @pytest.fixture
    def saved_router(self, client):
        """Train and save a router for testing."""
        # First train a router
        config = {
            "model": "load_test_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "general_llm",
            "similarity_threshold": 0.5,
            "routes": [
                {
                    "name": "test",
                    "target_model": "test_model",
                    "utterances": ["test utterance"],
                },
            ],
        }
        response = client.post("/v1/router/train", json=config)
        assert response.status_code == 200
        return "load_test_router"

    def test_load_saved_router(self, client, saved_router):
        """Test POST /v1/router/load loads a saved router."""
        request = {"model": saved_router}
        response = client.post("/v1/router/load", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "load_result"
        assert data["model"] == saved_router
        assert data["status"] == "loaded"

    def test_load_nonexistent_router_fails(self, client):
        """Test loading nonexistent router fails."""
        request = {"model": "nonexistent_router"}
        response = client.post("/v1/router/load", json=request)

        assert response.status_code == 404


class TestRouterDeleteEndpoint:
    """Test DELETE /v1/router/models/{name} endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from server import app

        return TestClient(app)

    @pytest.fixture
    def router_to_delete(self, client):
        """Train a router to delete."""
        config = {
            "model": "delete_test_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "general_llm",
            "routes": [
                {
                    "name": "test",
                    "target_model": "test_model",
                    "utterances": ["test"],
                },
            ],
        }
        response = client.post("/v1/router/train", json=config)
        assert response.status_code == 200
        return "delete_test_router"

    def test_delete_router_removes_model(self, client, router_to_delete):
        """Test DELETE /v1/router/models/{name} removes a router."""
        response = client.delete(f"/v1/router/models/{router_to_delete}")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "delete_result"
        assert data["model"] == router_to_delete
        assert data["deleted"] is True

        # Verify it's gone
        list_response = client.get("/v1/router/models")
        models = [m["name"] for m in list_response.json()["data"]]
        assert router_to_delete not in models

    def test_delete_nonexistent_router_fails(self, client):
        """Test deleting nonexistent router fails."""
        response = client.delete("/v1/router/models/nonexistent_router")

        assert response.status_code == 404

    def test_delete_path_traversal_blocked(self, client):
        """Test path traversal via .. is blocked in delete."""
        # Test that ".." is blocked (path traversal attempt)
        response = client.delete("/v1/router/models/test..router")

        # Should be blocked by path separator check (..)
        assert response.status_code == 400
        assert "path separators not allowed" in response.json()["detail"]
