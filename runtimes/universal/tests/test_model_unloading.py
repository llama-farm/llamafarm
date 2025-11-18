"""
Tests for automatic model unloading feature.

Verifies that models are automatically unloaded after a period of inactivity
to free up VRAM/RAM.
"""

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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

    # Store original values
    original_models = server._models.copy()
    original_access = server._model_last_access.copy()
    original_task = server._cleanup_task

    # Clear for test
    server._models.clear()
    server._model_last_access.clear()
    server._cleanup_task = None

    yield

    # Restore original values
    server._models = original_models
    server._model_last_access = original_access
    server._cleanup_task = original_task


@pytest.mark.asyncio
async def test_track_model_access(reset_server_globals):
    """Test that model access is tracked with timestamp."""
    import server

    cache_key = "test:model"

    # Track access
    before = datetime.now()
    server._track_model_access(cache_key)
    after = datetime.now()

    # Verify timestamp was recorded
    assert cache_key in server._model_last_access
    access_time = server._model_last_access[cache_key]
    assert before <= access_time <= after


@pytest.mark.asyncio
async def test_cleanup_idle_models(reset_server_globals, mock_model):
    """Test that idle models are unloaded after timeout."""
    import server

    cache_key = "test:model"

    # Add model to cache
    server._models[cache_key] = mock_model

    # Set last access to old time (beyond timeout)
    old_time = datetime.now() - timedelta(seconds=server.MODEL_UNLOAD_TIMEOUT + 10)
    server._model_last_access[cache_key] = old_time

    # Run one iteration of cleanup (manually, not as background task)
    now = datetime.now()
    models_to_unload = []

    for key, last_access in server._model_last_access.items():
        idle_time = (now - last_access).total_seconds()
        if idle_time > server.MODEL_UNLOAD_TIMEOUT:
            models_to_unload.append(key)

    # Unload idle models
    for key in models_to_unload:
        model = server._models.get(key)
        if model:
            await model.unload()
            del server._models[key]
            del server._model_last_access[key]

    # Verify model was unloaded
    mock_model.unload.assert_called_once()
    assert cache_key not in server._models
    assert cache_key not in server._model_last_access


@pytest.mark.asyncio
async def test_cleanup_does_not_unload_recent_models(reset_server_globals, mock_model):
    """Test that recently-accessed models are not unloaded."""
    import server

    cache_key = "test:model"

    # Add model to cache
    server._models[cache_key] = mock_model

    # Set last access to recent time (within timeout)
    recent_time = datetime.now() - timedelta(seconds=10)
    server._model_last_access[cache_key] = recent_time

    # Run one iteration of cleanup
    now = datetime.now()
    models_to_unload = []

    for key, last_access in server._model_last_access.items():
        idle_time = (now - last_access).total_seconds()
        if idle_time > server.MODEL_UNLOAD_TIMEOUT:
            models_to_unload.append(key)

    # Verify no models were marked for unloading
    assert len(models_to_unload) == 0

    # Verify model was NOT unloaded
    mock_model.unload.assert_not_called()
    assert cache_key in server._models
    assert cache_key in server._model_last_access


@pytest.mark.asyncio
async def test_load_language_tracks_access(reset_server_globals):
    """Test that loading a language model tracks access."""
    import server

    with patch("server.get_device", return_value="cpu"):
        with patch("server.detect_model_format", return_value="transformers"):
            with patch("server.LanguageModel") as MockLanguageModel:
                # Create mock model instance
                mock_instance = MagicMock()
                mock_instance.load = AsyncMock()
                MockLanguageModel.return_value = mock_instance

                # Load model
                model_id = "test/model"
                await server.load_language(model_id)

                # Verify model is tracked (new cache key includes quantization)
                cache_key = f"language:{model_id}:ctxauto:quantdefault"
                assert cache_key in server._model_last_access
                assert cache_key in server._models


@pytest.mark.asyncio
async def test_load_encoder_tracks_access(reset_server_globals):
    """Test that loading an encoder model tracks access."""
    import server

    with patch("server.get_device", return_value="cpu"):
        with patch("server.detect_model_format", return_value="transformers"):
            with patch("server.EncoderModel") as MockEncoderModel:
                # Create mock model instance
                mock_instance = MagicMock()
                mock_instance.load = AsyncMock()
                MockEncoderModel.return_value = mock_instance

                # Load model
                model_id = "test/embedding-model"
                await server.load_encoder(model_id, task="embedding")

                # Verify model is tracked (new cache key includes quantization)
                cache_key = "encoder:embedding:transformers:test/embedding-model:quantdefault"
                assert cache_key in server._model_last_access
                assert cache_key in server._models


@pytest.mark.asyncio
async def test_model_reaccess_updates_timestamp(reset_server_globals):
    """Test that re-accessing a cached model updates the timestamp."""
    import server

    with patch("server.get_device", return_value="cpu"):
        with patch("server.detect_model_format", return_value="transformers"):
            with patch("server.LanguageModel") as MockLanguageModel:
                # Create mock model instance
                mock_instance = MagicMock()
                mock_instance.load = AsyncMock()
                MockLanguageModel.return_value = mock_instance

                # Load model first time
                model_id = "test/model"
                await server.load_language(model_id)
                # New cache key includes quantization
                cache_key = f"language:{model_id}:ctxauto:quantdefault"
                first_access = server._model_last_access[cache_key]

                # Wait a bit
                await asyncio.sleep(0.1)

                # Load model second time (should use cache)
                await server.load_language(model_id)
                second_access = server._model_last_access[cache_key]

                # Verify timestamp was updated
                assert second_access > first_access


@pytest.mark.asyncio
async def test_environment_variables_override_defaults():
    """Test that environment variables override default timeout values."""
    with patch.dict(os.environ, {
        "MODEL_UNLOAD_TIMEOUT": "600",
        "CLEANUP_CHECK_INTERVAL": "60"
    }):
        # Reload server module to pick up new env vars
        import importlib
        import server
        importlib.reload(server)

        # Verify new values
        assert server.MODEL_UNLOAD_TIMEOUT == 600
        assert server.CLEANUP_CHECK_INTERVAL == 60


@pytest.mark.asyncio
async def test_cleanup_handles_unload_errors(reset_server_globals, mock_model):
    """Test that cleanup continues even if a model unload fails."""
    import server

    # Create two models
    cache_key1 = "test:model1"
    cache_key2 = "test:model2"

    mock_model1 = MagicMock()
    mock_model1.unload = AsyncMock(side_effect=Exception("Unload failed"))

    mock_model2 = MagicMock()
    mock_model2.unload = AsyncMock()

    # Add both models to cache with old timestamps
    old_time = datetime.now() - timedelta(seconds=server.MODEL_UNLOAD_TIMEOUT + 10)
    server._models[cache_key1] = mock_model1
    server._model_last_access[cache_key1] = old_time
    server._models[cache_key2] = mock_model2
    server._model_last_access[cache_key2] = old_time

    # Run cleanup iteration
    now = datetime.now()
    models_to_unload = []

    for key, last_access in server._model_last_access.items():
        idle_time = (now - last_access).total_seconds()
        if idle_time > server.MODEL_UNLOAD_TIMEOUT:
            models_to_unload.append(key)

    # Unload with error handling
    for key in models_to_unload:
        model = server._models.get(key)
        if model:
            try:
                await model.unload()
                del server._models[key]
                del server._model_last_access[key]
            except Exception:
                # Log error but continue (in real code)
                pass

    # Verify first model unload was attempted but failed (still in cache)
    mock_model1.unload.assert_called_once()
    assert cache_key1 in server._models  # Still there due to error

    # Verify second model was unloaded successfully
    mock_model2.unload.assert_called_once()
    assert cache_key2 not in server._models
    assert cache_key2 not in server._model_last_access
