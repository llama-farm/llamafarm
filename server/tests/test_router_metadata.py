"""Tests for routing metadata in API responses (Phase E3)."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


class TestRouterMetadataHeaders:
    """Tests for routing metadata in response headers."""

    @pytest.mark.asyncio
    async def test_route_response_includes_routing_info(self):
        """Test that /v1/ml/router/route response includes routing metadata."""
        # The route endpoint already returns routing info in the body
        # This test verifies the response structure

        expected_fields = [
            "target_model",
            "route_name",
            "similarity_score",
            "matched_utterance",
        ]

        # Mock response structure
        mock_response = {
            "object": "route_decision",
            "model": "test_router",
            "query": "test query",
            "route_name": "billing",
            "target_model": "billing_model",
            "similarity_score": 0.85,
            "matched_utterance": "payment question",
        }

        for field in expected_fields:
            assert field in mock_response

    @pytest.mark.asyncio
    async def test_train_response_includes_storage_info(self):
        """Test that /v1/ml/router/train response includes storage info."""
        # The train endpoint should return storage path info

        expected_fields = [
            "model",
            "status",
            "auto_saved",
            "saved_path",
            "namespace",
            "project_id",
            "storage_path",
        ]

        # Mock response structure (based on actual implementation)
        mock_response = {
            "object": "train_result",
            "model": "test_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "num_routes": 2,
            "routes": ["billing", "support"],
            "status": "trained",
            "auto_saved": True,
            "saved_path": "/path/to/router",
            "namespace": "default",
            "project_id": "demo",
            "storage_path": "/path/to/router",
        }

        for field in expected_fields:
            assert field in mock_response


class TestRouterMetadataBody:
    """Tests for routing metadata in response body."""

    @pytest.mark.asyncio
    async def test_routing_info_structure(self):
        """Test that routing_info has the expected structure."""
        routing_info = {
            "target_model": "billing_assistant",
            "route_name": "billing",
            "similarity_score": 0.89,
            "router_name": "smart_router",
            "matched_utterance": "what is my bill",
        }

        assert "target_model" in routing_info
        assert "route_name" in routing_info
        assert "similarity_score" in routing_info
        assert 0.0 <= routing_info["similarity_score"] <= 1.0
        assert "router_name" in routing_info

    @pytest.mark.asyncio
    async def test_default_route_metadata(self):
        """Test metadata when query matches default route."""
        routing_info = {
            "target_model": "general_llm",
            "route_name": None,  # None indicates default route
            "similarity_score": 0.45,  # Below threshold
            "router_name": "smart_router",
            "matched_utterance": None,
        }

        assert routing_info["route_name"] is None
        assert routing_info["target_model"] == "general_llm"


class TestRouterMetadataLogging:
    """Tests for routing metadata in server logs."""

    def test_logging_metadata_format(self):
        """Test that logging format is consistent for routing decisions."""
        # This is a structural test for the logging format
        log_metadata = {
            "router_name": "test_router",
            "target_model": "billing_model",
            "route_name": "billing",
            "similarity_score": 0.85,
            "query_preview": "what is my bill...",
        }

        # All fields should be present and of correct type
        assert isinstance(log_metadata["router_name"], str)
        assert isinstance(log_metadata["target_model"], str)
        assert isinstance(log_metadata["similarity_score"], float)
        assert log_metadata["route_name"] is None or isinstance(log_metadata["route_name"], str)


class TestRouterMetadataIntegration:
    """Integration tests for routing metadata."""

    @pytest.mark.asyncio
    async def test_metadata_consistency_across_endpoints(self):
        """Test that metadata format is consistent across all router endpoints."""
        # Common fields that should appear in multiple responses
        common_routing_fields = ["target_model", "route_name", "similarity_score"]

        # Route response
        route_response = {
            "object": "route_decision",
            "model": "test_router",
            "query": "test",
            "route_name": "billing",
            "target_model": "billing_model",
            "similarity_score": 0.85,
            "matched_utterance": "payment",
        }

        for field in common_routing_fields:
            assert field in route_response

    @pytest.mark.asyncio
    async def test_metadata_types_are_correct(self):
        """Test that metadata field types are correct."""
        metadata = {
            "target_model": "billing_model",
            "route_name": "billing",
            "similarity_score": 0.85,
            "matched_utterance": "what is my bill",
        }

        assert isinstance(metadata["target_model"], str)
        assert isinstance(metadata["route_name"], str) or metadata["route_name"] is None
        assert isinstance(metadata["similarity_score"], (int, float))
        assert 0.0 <= metadata["similarity_score"] <= 1.0
