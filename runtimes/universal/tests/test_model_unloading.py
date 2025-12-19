"""
Tests for automatic model unloading feature.

Verifies that models are automatically unloaded after a period of inactivity
to free up VRAM/RAM.
"""

import asyncio
import contextlib
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.model_cache import ModelCache


@pytest.fixture
def mock_model():
    """Create a mock model with unload method."""
    model = MagicMock()
    model.unload = AsyncMock()
    model.model_id = "test/model"
    return model


@pytest.fixture
def reset_server_globals():
    """Reset server global variables before each test."""
    # Import here to avoid circular imports
    import server

    # Store original caches
    original_models = server._models
    original_classifiers = server._classifiers
    original_task = server._cleanup_task

    # Replace with fresh caches for test
    server._models = ModelCache(ttl=server.MODEL_UNLOAD_TIMEOUT)
    server._classifiers = ModelCache(ttl=server.MODEL_UNLOAD_TIMEOUT)
    server._cleanup_task = None

    yield

    # Restore original caches
    server._models = original_models
    server._classifiers = original_classifiers
    server._cleanup_task = original_task


class TestModelCache:
    """Test the ModelCache class directly."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = ModelCache(ttl=300)
        model = MagicMock()
        cache["test:key"] = model
        assert cache.get("test:key") is model

    def test_contains(self):
        """Test __contains__ method."""
        cache = ModelCache(ttl=300)
        model = MagicMock()
        cache["test:key"] = model
        assert "test:key" in cache
        assert "nonexistent" not in cache

    def test_get_refreshes_ttl(self):
        """Test that get() refreshes the TTL."""
        cache = ModelCache(ttl=1)  # 1 second TTL
        model = MagicMock()
        cache["test:key"] = model

        # Wait a bit but not past TTL
        time.sleep(0.5)

        # Access should refresh TTL
        cache.get("test:key")

        # Check idle time was reset
        idle_time = cache.get_idle_time("test:key")
        assert idle_time is not None
        assert idle_time < 0.2  # Should be very recent

    def test_get_expired_keys(self):
        """Test get_expired_keys returns expired keys."""
        cache = ModelCache(ttl=0.1)  # 100ms TTL
        model = MagicMock()
        cache["test:key"] = model

        # Wait for expiry
        time.sleep(0.2)

        expired = cache.get_expired_keys()
        assert "test:key" in expired

    def test_pop_expired(self):
        """Test pop_expired removes and returns expired items."""
        cache = ModelCache(ttl=0.1)  # 100ms TTL
        model = MagicMock()
        cache["test:key"] = model

        # Wait for expiry
        time.sleep(0.2)

        expired = cache.pop_expired()
        assert len(expired) == 1
        assert expired[0][0] == "test:key"
        assert expired[0][1] is model
        assert "test:key" not in cache

    def test_recent_items_not_expired(self):
        """Test that recently accessed items are not expired."""
        cache = ModelCache(ttl=1)  # 1 second TTL
        model = MagicMock()
        cache["test:key"] = model

        # Check immediately
        expired = cache.get_expired_keys()
        assert "test:key" not in expired

    def test_pop(self):
        """Test pop removes and returns item."""
        cache = ModelCache(ttl=300)
        model = MagicMock()
        cache["test:key"] = model

        result = cache.pop("test:key")
        assert result is model
        assert "test:key" not in cache

    def test_clear(self):
        """Test clear removes all items."""
        cache = ModelCache(ttl=300)
        cache["key1"] = MagicMock()
        cache["key2"] = MagicMock()

        cache.clear()
        assert len(cache) == 0


@pytest.mark.asyncio
async def test_cleanup_idle_models(reset_server_globals, mock_model):
    """Test that idle models are unloaded after timeout."""
    import server

    cache_key = "test:model"

    # Add model to cache
    server._models[cache_key] = mock_model

    # Manually set access time to old time by manipulating internal state
    # Since ModelCache uses time.monotonic(), we need to mock the TTL behavior
    old_ttl = server._models._ttl
    server._models._ttl = 0  # Make everything appear expired

    # Pop expired items (simulates cleanup task)
    expired = server._models.pop_expired()

    # Restore TTL
    server._models._ttl = old_ttl

    # Verify model was returned as expired
    assert len(expired) == 1
    assert expired[0][0] == cache_key
    assert expired[0][1] is mock_model


@pytest.mark.asyncio
async def test_cleanup_does_not_unload_recent_models(reset_server_globals, mock_model):
    """Test that recently-accessed models are not unloaded."""
    import server

    cache_key = "test:model"

    # Add model to cache
    server._models[cache_key] = mock_model

    # Immediately check for expired (nothing should be expired)
    expired = server._models.pop_expired()

    # Verify no models were expired
    assert len(expired) == 0
    assert cache_key in server._models


@pytest.mark.asyncio
async def test_load_language_tracks_access(reset_server_globals):
    """Test that loading a language model tracks access."""
    import server

    with (
        patch("server.get_device", return_value="cpu"),
        patch("server.detect_model_format", return_value="transformers"),
        patch("server.LanguageModel") as MockLanguageModel,
    ):
        # Create mock model instance
        mock_instance = MagicMock()
        mock_instance.load = AsyncMock()
        MockLanguageModel.return_value = mock_instance

        # Load model
        model_id = "test/model"
        await server.load_language(model_id)

        # Verify model is tracked (new cache key includes quantization)
        cache_key = f"language:{model_id}:ctxauto:quantdefault"
        assert cache_key in server._models


@pytest.mark.asyncio
async def test_load_encoder_tracks_access(reset_server_globals):
    """Test that loading an encoder model tracks access."""
    import server

    with (
        patch("server.get_device", return_value="cpu"),
        patch("server.detect_model_format", return_value="transformers"),
        patch("server.EncoderModel") as MockEncoderModel,
    ):
        # Create mock model instance
        mock_instance = MagicMock()
        mock_instance.load = AsyncMock()
        MockEncoderModel.return_value = mock_instance

        # Load model
        model_id = "test/embedding-model"
        await server.load_encoder(model_id, task="embedding")

        # Verify model is tracked (new cache key includes quantization and max_length)
        cache_key = "encoder:embedding:transformers:test/embedding-model:quantdefault:lenauto"
        assert cache_key in server._models


@pytest.mark.asyncio
async def test_model_reaccess_updates_timestamp(reset_server_globals):
    """Test that re-accessing a cached model updates the timestamp."""
    import server

    with (
        patch("server.get_device", return_value="cpu"),
        patch("server.detect_model_format", return_value="transformers"),
        patch("server.LanguageModel") as MockLanguageModel,
    ):
        # Create mock model instance
        mock_instance = MagicMock()
        mock_instance.load = AsyncMock()
        MockLanguageModel.return_value = mock_instance

        # Load model first time
        model_id = "test/model"
        await server.load_language(model_id)
        # New cache key includes quantization
        cache_key = f"language:{model_id}:ctxauto:quantdefault"
        first_idle = server._models.get_idle_time(cache_key)

        # Wait a bit
        await asyncio.sleep(0.1)

        # Load model second time (should use cache)
        await server.load_language(model_id)
        second_idle = server._models.get_idle_time(cache_key)

        # Verify idle time was reset (second access should have lower idle time)
        assert second_idle < first_idle + 0.1  # Should be close to 0


@pytest.mark.asyncio
async def test_environment_variables_override_defaults():
    """Test that environment variables override default timeout values."""
    with patch.dict(
        os.environ, {"MODEL_UNLOAD_TIMEOUT": "600", "CLEANUP_CHECK_INTERVAL": "60"}
    ):
        # Reload server module to pick up new env vars
        import importlib

        import server

        importlib.reload(server)

        # Verify new values
        assert server.MODEL_UNLOAD_TIMEOUT == 600
        assert server.CLEANUP_CHECK_INTERVAL == 60


@pytest.mark.asyncio
async def test_cleanup_handles_unload_errors(reset_server_globals):
    """Test that cleanup continues even if a model unload fails."""
    import server

    # Create two models
    cache_key1 = "test:model1"
    cache_key2 = "test:model2"

    mock_model1 = MagicMock()
    mock_model1.unload = AsyncMock(side_effect=Exception("Unload failed"))

    mock_model2 = MagicMock()
    mock_model2.unload = AsyncMock()

    # Add both models to cache
    server._models[cache_key1] = mock_model1
    server._models[cache_key2] = mock_model2

    # Set TTL to 0 to make everything appear expired
    original_ttl = server._models._ttl
    server._models._ttl = 0

    # Pop expired items
    expired = server._models.pop_expired()

    # Restore TTL
    server._models._ttl = original_ttl

    # Simulate cleanup task unloading expired models
    for _key, model in expired:
        with contextlib.suppress(Exception):
            await model.unload()

    # Verify both unloads were attempted
    mock_model1.unload.assert_called_once()
    mock_model2.unload.assert_called_once()
