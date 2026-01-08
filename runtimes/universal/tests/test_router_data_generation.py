#!/usr/bin/env python3
"""
Test suite for Router Synthetic Data Generation.
Tests Phase 5: Synthetic Data Generation Endpoint
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRouterDataGeneration:
    """Test POST /v1/router/generate-data endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from server import app

        return TestClient(app)

    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI completion response."""
        return {
            "choices": [
                {
                    "message": {
                        "content": """1. what is my bill
2. how much do I owe
3. payment options available
4. can I see my invoice
5. billing inquiry
6. account balance check
7. when is payment due
8. total amount owed
9. monthly charges
10. billing statement request"""
                    }
                }
            ]
        }

    def test_generate_data_returns_utterances(self, client, mock_openai_response):
        """Test POST /v1/router/generate-data generates utterances from description."""
        with patch("server.httpx.AsyncClient") as mock_client_class:
            # Create mock async client
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            mock_client_class.return_value = mock_client

            response = client.post(
                "/v1/router/generate-data",
                json={
                    "route_description": "billing inquiries about account balance and payments",
                    "count": 10,
                    "api_key": "test-key",  # Provide test API key
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "utterance_list"
            assert "utterances" in data
            assert len(data["utterances"]) >= 1

    def test_generate_data_with_custom_model(self, client, mock_openai_response):
        """Test generation with custom model parameter."""
        with patch("server.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            mock_client_class.return_value = mock_client

            response = client.post(
                "/v1/router/generate-data",
                json={
                    "route_description": "technical support questions",
                    "count": 5,
                    "model": "gpt-5-mini",
                    "api_key": "test-key",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "utterances" in data

    def test_generate_data_deduplicates_results(self, client):
        """Test that duplicate utterances are removed."""
        duplicate_response = {
            "choices": [
                {
                    "message": {
                        "content": """1. what is my bill
2. what is my bill
3. what is my bill
4. how much do I owe
5. how much do I owe"""
                    }
                }
            ]
        }

        with patch("server.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_response = MagicMock()
            mock_response.json.return_value = duplicate_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            mock_client_class.return_value = mock_client

            response = client.post(
                "/v1/router/generate-data",
                json={
                    "route_description": "billing questions",
                    "count": 5,
                    "api_key": "test-key",
                },
            )

            assert response.status_code == 200
            data = response.json()
            # Should have deduplicated to only unique utterances
            unique_utterances = set(data["utterances"])
            assert len(unique_utterances) == len(data["utterances"])

    def test_generate_data_batch_routes(self, client, mock_openai_response):
        """Test batch generation for multiple routes."""
        with patch("server.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            mock_client_class.return_value = mock_client

            response = client.post(
                "/v1/router/generate-data",
                json={
                    "routes": [
                        {
                            "route_name": "billing",
                            "description": "billing and payment questions",
                            "count": 5,
                        },
                        {
                            "route_name": "support",
                            "description": "technical support requests",
                            "count": 5,
                        },
                    ],
                    "api_key": "test-key",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "batch_utterance_list"
            assert "routes" in data
            assert len(data["routes"]) == 2

    def test_generate_data_requires_description_or_routes(self, client):
        """Test that either route_description or routes must be provided."""
        response = client.post(
            "/v1/router/generate-data",
            json={
                "count": 10,
                "api_key": "test-key",
            },
        )

        assert response.status_code == 400
        assert "route_description or routes required" in response.json()["detail"]

    def test_generate_data_filters_low_quality(self, client):
        """Test that low-quality utterances are filtered."""
        low_quality_response = {
            "choices": [
                {
                    "message": {
                        "content": """1. a
2.
3. test
4. what is my current account balance and payment status
5. xyz"""
                    }
                }
            ]
        }

        with patch("server.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_response = MagicMock()
            mock_response.json.return_value = low_quality_response
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            mock_client_class.return_value = mock_client

            response = client.post(
                "/v1/router/generate-data",
                json={
                    "route_description": "billing questions",
                    "count": 5,
                    "api_key": "test-key",
                },
            )

            assert response.status_code == 200
            data = response.json()
            # Short/empty utterances should be filtered
            for utterance in data["utterances"]:
                assert len(utterance.strip()) >= 3


class TestRouterDataGenerationPrompt:
    """Test the prompt template for data generation."""

    def test_prompt_includes_route_description(self):
        """Test that prompt includes the route description."""
        from server import _build_generation_prompt

        prompt = _build_generation_prompt(
            route_description="billing and payment inquiries",
            count=10,
        )

        assert "billing and payment inquiries" in prompt
        assert "10" in prompt

    def test_prompt_requests_diversity(self):
        """Test that prompt requests diverse utterances."""
        from server import _build_generation_prompt

        prompt = _build_generation_prompt(
            route_description="technical support",
            count=20,
        )

        # Should include diversity instructions
        assert "diverse" in prompt.lower() or "different" in prompt.lower()
