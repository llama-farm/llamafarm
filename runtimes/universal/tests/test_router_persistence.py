"""Tests for router persistence - auto-save and auto-load functionality."""

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from models.router_model import RouterModel


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def unique_router_id():
    """Generate a unique router ID to avoid conflicts with existing saved routers."""
    return f"test_router_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def router_config(unique_router_id):
    """Sample router configuration with unique name."""
    return {
        "name": unique_router_id,
        "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "default_model": "default_llm",
        "similarity_threshold": 0.7,
        "routes": [
            {
                "name": "billing",
                "target_model": "billing_model",
                "utterances": ["what is my bill", "payment question"],
            },
            {
                "name": "support",
                "target_model": "support_model",
                "utterances": ["help with login", "password reset"],
            },
        ],
    }


class TestRouterAutoSave:
    """Tests for router auto-save after training."""

    @pytest.mark.asyncio
    async def test_router_saves_config_json(
        self, temp_storage_dir, router_config, unique_router_id
    ):
        """Test that router saves config.json after training."""
        # Create and load router with unique ID to avoid loading from disk
        router = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=router_config,
        )
        await router.load()

        # Save to temp directory
        save_path = await router.save(temp_storage_dir)

        # Verify config.json exists
        config_path = Path(save_path) / "config.json"
        assert config_path.exists()

        # Verify content
        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config["embedder_model"] == router_config["embedder_model"]
        assert saved_config["default_model"] == router_config["default_model"]
        assert saved_config["similarity_threshold"] == router_config["similarity_threshold"]
        assert len(saved_config["routes"]) == 2

        await router.unload()

    @pytest.mark.asyncio
    async def test_router_saves_embeddings(
        self, temp_storage_dir, router_config, unique_router_id
    ):
        """Test that router saves pre-computed embeddings."""
        # Create and load router with unique ID
        router = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=router_config,
        )
        await router.load()

        # Save to temp directory
        save_path = await router.save(temp_storage_dir)

        # Verify embeddings.npz exists
        embeddings_path = Path(save_path) / "embeddings.npz"
        assert embeddings_path.exists()

        # Verify content
        data = np.load(embeddings_path)
        assert "billing" in data.files
        assert "support" in data.files

        await router.unload()


class TestRouterAutoLoad:
    """Tests for router auto-load from disk."""

    @pytest.mark.asyncio
    async def test_router_loads_from_disk(
        self, temp_storage_dir, router_config, unique_router_id
    ):
        """Test that router can be loaded from saved files."""
        # Create and save router with unique ID
        router1 = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=router_config,
        )
        await router1.load()
        save_path = await router1.save(temp_storage_dir)
        await router1.unload()

        # Load new router from disk
        router2 = RouterModel(
            model_id=str(save_path),
            device="cpu",
        )
        await router2.load()

        # Verify configuration loaded correctly
        assert router2.embedder_model == router_config["embedder_model"]
        assert router2.default_model == router_config["default_model"]
        assert router2.similarity_threshold == router_config["similarity_threshold"]
        assert len(router2.routes) == 2

        await router2.unload()

    @pytest.mark.asyncio
    async def test_loaded_router_produces_same_results(
        self, temp_storage_dir, router_config, unique_router_id
    ):
        """Test that loaded router produces same routing decisions as trained router."""
        # Create and save router with unique ID
        router1 = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=router_config,
        )
        await router1.load()

        # Route a query with original router
        decision1 = await router1.route("what is my current bill balance")

        save_path = await router1.save(temp_storage_dir)
        await router1.unload()

        # Load new router from disk
        router2 = RouterModel(
            model_id=str(save_path),
            device="cpu",
        )
        await router2.load()

        # Route same query with loaded router
        decision2 = await router2.route("what is my current bill balance")

        # Results should match
        assert decision1.target_model == decision2.target_model
        assert decision1.route_name == decision2.route_name
        # Similarity scores should be identical (using same embeddings)
        assert abs(decision1.similarity_score - decision2.similarity_score) < 0.001

        await router2.unload()

    @pytest.mark.asyncio
    async def test_loaded_router_routes_correctly(
        self, temp_storage_dir, router_config, unique_router_id
    ):
        """Test that loaded router routes queries to correct targets."""
        # Create and save router with unique ID
        router = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=router_config,
        )
        await router.load()
        save_path = await router.save(temp_storage_dir)
        await router.unload()

        # Load from disk
        loaded_router = RouterModel(
            model_id=str(save_path),
            device="cpu",
        )
        await loaded_router.load()

        # Test billing query
        billing_decision = await loaded_router.route("I have a question about my payment")
        assert billing_decision.route_name == "billing"
        assert billing_decision.target_model == "billing_model"

        # Test support query
        support_decision = await loaded_router.route("I need help resetting my password")
        assert support_decision.route_name == "support"
        assert support_decision.target_model == "support_model"

        await loaded_router.unload()


class TestRouterPersistenceIntegration:
    """Integration tests for router persistence."""

    @pytest.mark.asyncio
    async def test_router_survives_unload_reload_cycle(
        self, temp_storage_dir, router_config, unique_router_id
    ):
        """Test that router state survives unload/reload cycle."""
        # Train router with unique ID
        router = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=router_config,
        )
        await router.load()

        # Get initial routing decision
        initial_decision = await router.route("billing question")
        initial_target = initial_decision.target_model

        # Save and unload
        save_path = await router.save(temp_storage_dir)
        await router.unload()

        # Verify router is unloaded
        assert not router._is_loaded

        # Reload from disk
        router2 = RouterModel(
            model_id=str(save_path),
            device="cpu",
        )
        await router2.load()

        # Verify routing still works
        reloaded_decision = await router2.route("billing question")
        assert reloaded_decision.target_model == initial_target

        await router2.unload()

    @pytest.mark.asyncio
    async def test_router_with_no_embeddings_recomputes(
        self, temp_storage_dir, router_config, unique_router_id
    ):
        """Test that router recomputes embeddings if file is missing."""
        # Create and save router with unique ID
        router = RouterModel(
            model_id=unique_router_id,
            device="cpu",
            config=router_config,
        )
        await router.load()
        save_path = await router.save(temp_storage_dir)
        await router.unload()

        # Delete embeddings file
        embeddings_path = Path(save_path) / "embeddings.npz"
        if embeddings_path.exists():
            embeddings_path.unlink()

        # Load router - should recompute embeddings
        router2 = RouterModel(
            model_id=str(save_path),
            device="cpu",
        )
        await router2.load()

        # Should still work
        decision = await router2.route("payment help")
        assert decision.target_model in ["billing_model", "support_model", "default_llm"]

        await router2.unload()
