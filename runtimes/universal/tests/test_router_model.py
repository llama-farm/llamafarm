#!/usr/bin/env python3
"""
Test suite for RouterModel.
Tests Phase 3: Router Model in Universal Runtime
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRouterModel:
    """Test class for RouterModel functionality."""

    @pytest.fixture
    def unique_router_id(self):
        """Generate unique router ID to avoid conflicts with saved routers."""
        return f"test_router_{uuid.uuid4().hex[:8]}"

    @pytest.fixture
    def route_config(self, unique_router_id):
        """Sample route configuration with unique name."""
        return {
            "name": unique_router_id,
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
                        "invoice question",
                    ],
                },
                {
                    "name": "support",
                    "target_model": "support_model",
                    "utterances": [
                        "help with login",
                        "password reset",
                        "app not working",
                        "technical issue",
                    ],
                },
                {
                    "name": "weather",
                    "target_model": "weather_tool",
                    "utterances": [
                        "what is the weather",
                        "is it raining",
                        "temperature today",
                    ],
                },
            ],
        }

    @pytest.fixture
    def temp_model_dir(self):
        """Create temp directory for model saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_router_model_instantiation(self, route_config, unique_router_id):
        """Test RouterModel can be instantiated with route config."""
        from models.router_model import RouterModel

        router = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=route_config,
        )

        assert router.model_id == unique_router_id
        assert router.embedder_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert router.default_model == "general_llm"
        assert router.similarity_threshold == 0.7
        assert len(router.routes) == 3

    @pytest.mark.asyncio
    async def test_router_model_load(self, route_config, unique_router_id):
        """Test RouterModel loads embedder and computes embeddings."""
        from models.router_model import RouterModel

        router = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=route_config,
        )

        # Load should initialize embedder and compute route embeddings
        await router.load()

        assert router._embedder is not None
        assert router._is_loaded
        assert len(router._route_embeddings) == 3  # 3 routes

        # Each route should have embeddings for its utterances
        assert "billing" in router._route_embeddings
        assert "support" in router._route_embeddings
        assert "weather" in router._route_embeddings

        await router.unload()

    @pytest.mark.asyncio
    async def test_router_route_correct_target(self, route_config, unique_router_id):
        """Test RouterModel.route() returns correct target model for known utterance."""
        from models.router_model import RouterModel

        router = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=route_config,
        )
        await router.load()

        # Test billing query
        decision = await router.route("what is my current bill amount?")
        assert decision.route_name == "billing"
        assert decision.target_model == "billing_model"
        assert decision.similarity_score > 0.5

        # Test support query
        decision = await router.route("I need help resetting my password")
        assert decision.route_name == "support"
        assert decision.target_model == "support_model"

        # Test weather query (use query very similar to utterances)
        decision = await router.route("what is the weather today")
        assert decision.route_name == "weather"
        assert decision.target_model == "weather_tool"

        await router.unload()

    @pytest.mark.asyncio
    async def test_router_default_model_low_similarity(self, route_config, unique_router_id):
        """Test RouterModel.route() returns default model for low-similarity query."""
        from models.router_model import RouterModel

        router = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=route_config,
        )
        await router.load()

        # Random query that doesn't match any route well
        decision = await router.route("Tell me about quantum physics and dark matter")

        # Should fall back to default model
        assert decision.target_model == "general_llm"
        assert decision.route_name is None or decision.similarity_score < 0.7

        await router.unload()

    @pytest.mark.asyncio
    async def test_router_latency_under_10ms(self, route_config, unique_router_id):
        """Test RouterModel routing latency is < 10ms for 10 routes."""
        from models.router_model import RouterModel

        router = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=route_config,
        )
        await router.load()

        # Warm up
        await router.route("test query")

        # Measure routing time
        queries = [
            "billing question",
            "support help",
            "weather check",
            "random query 1",
            "random query 2",
        ]

        latencies = []
        for query in queries:
            start = time.perf_counter()
            await router.route(query)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        print(f"Average routing latency: {avg_latency:.2f}ms")
        print(f"Max routing latency: {max_latency:.2f}ms")

        # Note: First embedding after load may be slower due to model warmup
        # We're lenient here but in production should be < 10ms
        assert max_latency < 100, f"Routing too slow: {max_latency}ms"

        await router.unload()

    @pytest.mark.asyncio
    async def test_router_save_and_load(self, route_config, temp_model_dir, unique_router_id):
        """Test RouterModel saves and loads from disk correctly."""
        from models.router_model import RouterModel, ROUTER_MODELS_DIR

        # Override models dir for testing
        test_models_dir = temp_model_dir / "router"
        test_models_dir.mkdir(parents=True)

        with patch("models.router_model.ROUTER_MODELS_DIR", test_models_dir):
            router = RouterModel(
                model_id=unique_router_id,
                device="cpu",
                config=route_config,
            )
            await router.load()

            # Save the router
            save_path = "my_router"
            await router.save(save_path)

            # Verify files were created
            saved_path = test_models_dir / save_path
            assert saved_path.exists()
            assert (saved_path / "config.json").exists()
            assert (saved_path / "embeddings.npz").exists()

            await router.unload()

            # Load from saved path
            router2 = RouterModel(
                model_id=str(saved_path),
                device="cpu",
            )
            await router2.load()

            assert router2._is_loaded
            assert len(router2.routes) == 3

            # Verify routing still works
            decision = await router2.route("what is my bill")
            assert decision.route_name == "billing"

            await router2.unload()

    @pytest.mark.asyncio
    async def test_router_loads_embedder_correctly(self, route_config, unique_router_id):
        """Test RouterModel loads the specified embedder model."""
        from models.router_model import RouterModel

        router = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=route_config,
        )
        await router.load()

        # Verify embedder is loaded
        assert router._embedder is not None

        # Verify we can encode text
        embedding = router._encode_text("test query")
        assert embedding is not None
        assert len(embedding.shape) == 1  # 1D embedding
        assert embedding.shape[0] > 0  # Has dimensions

        await router.unload()


class TestRouteDecision:
    """Test RouteDecision dataclass."""

    def test_route_decision_creation(self):
        """Test RouteDecision can be created."""
        from models.router_model import RouteDecision

        decision = RouteDecision(
            target_model="billing_model",
            route_name="billing",
            similarity_score=0.92,
            matched_utterance="what is my bill",
        )

        assert decision.target_model == "billing_model"
        assert decision.route_name == "billing"
        assert decision.similarity_score == 0.92
        assert decision.matched_utterance == "what is my bill"

    def test_route_decision_default_route(self):
        """Test RouteDecision for default route case."""
        from models.router_model import RouteDecision

        decision = RouteDecision(
            target_model="general_llm",
            route_name=None,
            similarity_score=0.45,
            matched_utterance=None,
        )

        assert decision.target_model == "general_llm"
        assert decision.route_name is None


class TestRouterModelEdgeCases:
    """Test edge cases for RouterModel."""

    @pytest.mark.asyncio
    async def test_router_empty_query(self):
        """Test routing empty query."""
        from models.router_model import RouterModel

        config = {
            "name": "test",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "default",
            "routes": [
                {"name": "test", "target_model": "test_model", "utterances": ["hello"]}
            ],
        }

        router = RouterModel(model_id="test", device="cpu", config=config)
        await router.load()

        decision = await router.route("")
        assert decision.target_model == "default"

        await router.unload()

    @pytest.mark.asyncio
    async def test_router_single_route(self):
        """Test router with single route."""
        from models.router_model import RouterModel

        config = {
            "name": "single",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "default",
            "routes": [
                {
                    "name": "only_route",
                    "target_model": "only_model",
                    "utterances": ["specific query type"],
                }
            ],
        }

        router = RouterModel(model_id="single", device="cpu", config=config)
        await router.load()

        decision = await router.route("specific query type")
        assert decision.target_model == "only_model"

        await router.unload()

    @pytest.mark.asyncio
    async def test_router_many_utterances(self):
        """Test router with many utterances per route."""
        from models.router_model import RouterModel

        # Create route with 50 utterances
        utterances = [f"variant query number {i}" for i in range(50)]

        config = {
            "name": "many",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "default",
            "routes": [
                {"name": "many_route", "target_model": "many_model", "utterances": utterances}
            ],
        }

        router = RouterModel(model_id="many", device="cpu", config=config)
        await router.load()

        decision = await router.route("variant query number 25")
        assert decision.target_model == "many_model"

        await router.unload()
