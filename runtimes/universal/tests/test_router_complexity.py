#!/usr/bin/env python3
"""
Test suite for RouterModel complexity classifier integration.
Tests Phase 8: Complexity Classifier Integration
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.router_model import RouterModel, RouteDecision


class TestRouterComplexityIntegration:
    """Test class for router complexity classifier integration."""

    @pytest.fixture
    def route_config_with_complexity(self):
        """Router configuration with complexity classifier."""
        return {
            "name": "complexity_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "fast_model",
            "similarity_threshold": 0.7,
            "complexity_classifier": "query_complexity",  # SetFit classifier name
            "complexity_threshold": 0.5,  # Above this = complex
            "complex_model": "powerful_model",  # Target for complex queries
            "routes": [
                {
                    "name": "billing",
                    "target_model": "billing_model",
                    "utterances": [
                        "what is my bill",
                        "payment options",
                        "invoice question",
                    ],
                },
            ],
        }

    @pytest.fixture
    def complexity_examples(self):
        """Sample complexity training data."""
        return {
            "simple": [
                "What time is it?",
                "Hello",
                "What is your name?",
                "How are you?",
                "What color is the sky?",
                "Count to 10",
            ],
            "complex": [
                "Analyze the socioeconomic factors contributing to urban decay in post-industrial cities and propose a comprehensive revitalization strategy",
                "Write a Python implementation of a distributed consensus algorithm using Raft",
                "Compare and contrast the philosophical frameworks of Kant and Nietzsche regarding the nature of morality",
                "Design a microservices architecture for a high-availability e-commerce platform with CQRS and event sourcing",
                "Explain the mathematical foundations of quantum computing and implement a simple quantum circuit simulator",
            ],
        }

    @pytest.mark.asyncio
    async def test_router_with_complexity_classifier_routes_simple_to_fast_model(
        self, route_config_with_complexity
    ):
        """Test that simple queries are routed to the fast model."""
        router = RouterModel(
            model_id="test_complexity_router",
            device="cpu",
            config=route_config_with_complexity,
        )

        # Mock the complexity classifier
        mock_classifier = MagicMock()
        # Return "simple" prediction for simple queries
        mock_classifier.predict.return_value = ["simple"]
        router._complexity_classifier = mock_classifier

        # Also need to mock the embedder loading
        with patch.object(router, "_encode_text") as mock_encode:
            # Return a fake embedding
            mock_encode.return_value = np.zeros(384)

            # Load router (mocked embedder)
            router._is_loaded = True
            router._route_embeddings = {"billing": np.random.randn(3, 384).astype(np.float32)}

            # Query that doesn't match topic routes (low similarity due to zero embedding)
            decision = await router.route("What is 2 + 2?")

            # Should route to fast model (default) for simple queries
            assert decision.target_model == "fast_model"
            assert decision.complexity_label == "simple"

    @pytest.mark.asyncio
    async def test_router_with_complexity_classifier_routes_complex_to_powerful_model(
        self, route_config_with_complexity
    ):
        """Test that complex queries are routed to the powerful model."""
        router = RouterModel(
            model_id="test_complexity_router",
            device="cpu",
            config=route_config_with_complexity,
        )

        mock_classifier = MagicMock()
        # Return "complex" prediction with high confidence
        mock_classifier.predict.return_value = ["complex"]
        mock_classifier.predict_proba.return_value = [[0.15, 0.85]]  # 85% complex
        router._complexity_classifier = mock_classifier

        with patch.object(router, "_encode_text") as mock_encode:
            mock_encode.return_value = np.zeros(384)
            router._is_loaded = True
            router._route_embeddings = {"billing": np.random.randn(3, 384).astype(np.float32)}

            # Complex query that doesn't match topic routes
            decision = await router.route(
                "Design a distributed system architecture with eventual consistency guarantees"
            )

            # Should route to powerful model for complex queries
            assert decision.target_model == "powerful_model"
            assert decision.complexity_label == "complex"

    @pytest.mark.asyncio
    async def test_topic_routing_takes_precedence_over_complexity(
        self, route_config_with_complexity
    ):
        """Test that topic routing takes precedence over complexity routing."""
        router = RouterModel(
            model_id="test_complexity_router",
            device="cpu",
            config=route_config_with_complexity,
        )

        mock_classifier = MagicMock()
        # Even if marked complex, topic routing should win
        mock_classifier.predict.return_value = ["complex"]
        mock_classifier.predict_proba.return_value = [[0.1, 0.9]]
        router._complexity_classifier = mock_classifier

        # Create a fake billing embedding that will match
        billing_embedding = np.array([[1.0, 0.0, 0.0]] * 3, dtype=np.float32)  # 3 utterances, 3D embedding

        with patch.object(router, "_encode_text") as mock_encode:
            # Return embedding that matches billing route perfectly
            mock_encode.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            router._is_loaded = True
            router._route_embeddings = {"billing": billing_embedding}

            # Query matches billing route
            decision = await router.route("What is my bill?")

            # Should route to billing model despite complexity
            assert decision.target_model == "billing_model"
            assert decision.route_name == "billing"

    @pytest.mark.asyncio
    async def test_complexity_only_routing(self):
        """Test router configured only with complexity classifier (no topic routes)."""
        config = {
            "name": "complexity_only_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "fast_model",
            "complexity_classifier": "query_complexity",
            "complexity_threshold": 0.5,
            "complex_model": "powerful_model",
            "routes": [],  # No topic routes
        }

        router = RouterModel(model_id="complexity_only", device="cpu", config=config)

        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = ["complex"]
        mock_classifier.predict_proba.return_value = [[0.2, 0.8]]
        router._complexity_classifier = mock_classifier

        router._is_loaded = True
        router._route_embeddings = {}

        with patch.object(router, "_encode_text") as mock_encode:
            mock_encode.return_value = np.zeros(384)

            decision = await router.route("Analyze this complex problem...")

            assert decision.target_model == "powerful_model"
            assert decision.complexity_label == "complex"

    @pytest.mark.asyncio
    async def test_router_without_complexity_classifier(self):
        """Test router works normally without complexity classifier."""
        config = {
            "name": "no_complexity_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "general_model",
            "similarity_threshold": 0.7,
            # No complexity_classifier field
            "routes": [
                {
                    "name": "billing",
                    "target_model": "billing_model",
                    "utterances": ["what is my bill"],
                },
            ],
        }

        router = RouterModel(model_id="no_complexity", device="cpu", config=config)
        router._is_loaded = True
        router._route_embeddings = {"billing": np.random.randn(1, 384).astype(np.float32)}

        with patch.object(router, "_encode_text") as mock_encode:
            mock_encode.return_value = np.zeros(384)

            # Query that doesn't match routes should go to default
            decision = await router.route("Random unrelated question")

            assert decision.target_model == "general_model"
            assert decision.complexity_label is None  # No complexity classification

    @pytest.mark.asyncio
    async def test_complexity_classifier_loading(self, route_config_with_complexity):
        """Test that complexity classifier is loaded correctly."""
        # Create a mock classifier file path
        with tempfile.TemporaryDirectory() as tmpdir:
            classifier_path = Path(tmpdir) / "query_complexity"
            classifier_path.mkdir()

            router = RouterModel(
                model_id="test_router",
                device="cpu",
                config=route_config_with_complexity,
            )

            # Verify complexity config was parsed
            assert router.complexity_classifier_name == "query_complexity"
            assert router.complexity_threshold == 0.5
            assert router.complex_model == "powerful_model"


class TestComplexityClassifierTraining:
    """Test class for complexity classifier training."""

    @pytest.fixture
    def training_data(self):
        """Sample training data for complexity classifier."""
        return {
            "texts": [
                "What time is it?",
                "Hello world",
                "Design a distributed database with ACID guarantees",
                "Implement a neural network from scratch in Python",
            ],
            "labels": ["simple", "simple", "complex", "complex"],
        }

    def test_complexity_labels_in_response(self):
        """Test that routing decision includes complexity labels."""
        # Create decision with complexity info
        decision = RouteDecision(
            target_model="powerful_model",
            route_name=None,
            similarity_score=0.0,
            matched_utterance=None,
            complexity_label="complex",
            complexity_score=0.85,
        )

        assert decision.complexity_label == "complex"
        assert decision.complexity_score == 0.85

    def test_route_decision_without_complexity(self):
        """Test that RouteDecision works without complexity fields."""
        decision = RouteDecision(
            target_model="default",
            route_name=None,
            similarity_score=0.5,
        )

        assert decision.complexity_label is None
        assert decision.complexity_score is None


class TestComplexityEdgeCases:
    """Test edge cases for complexity integration."""

    @pytest.mark.asyncio
    async def test_complexity_fallback_on_classifier_error(self):
        """Test fallback when complexity classifier fails."""
        config = {
            "name": "test_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "default_model",
            "complexity_classifier": "nonexistent_classifier",
            "routes": [],
        }

        router = RouterModel(model_id="fallback_test", device="cpu", config=config)
        router._is_loaded = True
        router._route_embeddings = {}
        # Classifier is None (failed to load)
        router._complexity_classifier = None

        with patch.object(router, "_encode_text") as mock_encode:
            mock_encode.return_value = np.zeros(384)

            # Should still work, just route to default
            decision = await router.route("Some query")

            assert decision.target_model == "default_model"
            assert decision.complexity_label is None

    @pytest.mark.asyncio
    async def test_complexity_score_threshold(self):
        """Test that complexity threshold is respected."""
        config = {
            "name": "threshold_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "fast_model",
            "complexity_classifier": "query_complexity",
            "complexity_threshold": 0.7,  # High threshold
            "complex_model": "powerful_model",
            "routes": [],
        }

        router = RouterModel(model_id="threshold_test", device="cpu", config=config)

        mock_classifier = MagicMock()
        # Score below threshold (0.5 < 0.7)
        mock_classifier.predict.return_value = ["complex"]
        mock_classifier.predict_proba.return_value = [[0.5, 0.5]]  # Only 50% confident
        router._complexity_classifier = mock_classifier

        router._is_loaded = True
        router._route_embeddings = {}

        with patch.object(router, "_encode_text") as mock_encode:
            mock_encode.return_value = np.zeros(384)

            decision = await router.route("Moderate complexity query")

            # Below threshold, should route to fast model
            assert decision.target_model == "fast_model"
            # Complexity label should still be set
            assert decision.complexity_label == "complex"

    @pytest.mark.asyncio
    async def test_complexity_prediction_exception_handling(self):
        """Test that exceptions in complexity prediction are handled gracefully."""
        config = {
            "name": "exception_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "default_model",
            "complexity_classifier": "query_complexity",
            "complex_model": "powerful_model",
            "routes": [],
        }

        router = RouterModel(model_id="exception_test", device="cpu", config=config)

        mock_classifier = MagicMock()
        # Classifier raises exception
        mock_classifier.predict.side_effect = RuntimeError("Prediction failed")
        router._complexity_classifier = mock_classifier

        router._is_loaded = True
        router._route_embeddings = {}

        with patch.object(router, "_encode_text") as mock_encode:
            mock_encode.return_value = np.zeros(384)

            # Should not raise, should fall back to default
            decision = await router.route("Some query")

            assert decision.target_model == "default_model"
            assert decision.complexity_label is None
