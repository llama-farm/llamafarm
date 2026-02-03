"""
Shared state management for the Universal Runtime.

This module centralizes all global state including:
- Model caches (with TTL-based expiration)
- Device management
- Locks for thread-safe model loading
- Configuration constants

All routers should import shared state from this module.
"""

import asyncio
import os
from contextlib import suppress
from pathlib import Path

from core.logging import UniversalRuntimeLogger
from models import BaseModel, ClassifierModel
from utils.device import get_optimal_device
from utils.feature_encoder import FeatureEncoder
from utils.model_cache import ModelCache
from utils.safe_home import get_data_dir

logger = UniversalRuntimeLogger("universal-runtime")

# ============================================================================
# Configuration
# ============================================================================

# Model unload timeout configuration (in seconds)
# Default: 5 minutes (300 seconds)
MODEL_UNLOAD_TIMEOUT = int(os.getenv("MODEL_UNLOAD_TIMEOUT", "300"))

# Cleanup check interval (in seconds) - how often to check for idle models
# Default: 30 seconds
CLEANUP_CHECK_INTERVAL = int(os.getenv("CLEANUP_CHECK_INTERVAL", "30"))

# Maximum file upload size (100 MB by default, configurable via env var)
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 100 * 1024 * 1024))

# Model storage directories - uses standard LlamaFarm data directory structure
_LF_DATA_DIR = get_data_dir()
ANOMALY_MODELS_DIR = _LF_DATA_DIR / "models" / "anomaly"
CLASSIFIER_MODELS_DIR = _LF_DATA_DIR / "models" / "classifier"


# ============================================================================
# Global State
# ============================================================================

# Model caches using TTL-based caching (via cachetools)
# Models are automatically tracked for idle time and cleaned up by background task
_models: ModelCache[BaseModel] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
_classifiers: ModelCache[ClassifierModel] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)

# Lock for thread-safe model loading
_model_load_lock = asyncio.Lock()

# Current device (lazily initialized)
_current_device = None

# Feature encoder cache for anomaly detection with mixed data types
_encoders: dict[str, FeatureEncoder] = {}

# Background cleanup task reference
_cleanup_task: asyncio.Task | None = None


# ============================================================================
# Device Management
# ============================================================================


def get_device() -> str:
    """Get the optimal device for the current platform.

    Returns the cached device if already determined, otherwise detects
    the optimal device (MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise).
    """
    global _current_device
    if _current_device is None:
        _current_device = get_optimal_device()
        logger.info(f"Using device: {_current_device}")
    return _current_device


# ============================================================================
# Model Cache Access
# ============================================================================


def get_models_cache() -> ModelCache[BaseModel]:
    """Get the models cache."""
    return _models


def get_classifiers_cache() -> ModelCache[ClassifierModel]:
    """Get the classifiers cache."""
    return _classifiers


def get_encoders_cache() -> dict[str, FeatureEncoder]:
    """Get the feature encoders cache."""
    return _encoders


def get_model_load_lock() -> asyncio.Lock:
    """Get the model loading lock."""
    return _model_load_lock


# ============================================================================
# Cleanup Task Management
# ============================================================================


async def cleanup_idle_models() -> None:
    """Background task that periodically unloads idle models.

    Uses ModelCache's TTL-based expiration to find and unload models that
    haven't been accessed in MODEL_UNLOAD_TIMEOUT seconds.
    """
    logger.info(
        f"Model cleanup task started (timeout={MODEL_UNLOAD_TIMEOUT}s, "
        f"check_interval={CLEANUP_CHECK_INTERVAL}s)"
    )

    while True:
        try:
            await asyncio.sleep(CLEANUP_CHECK_INTERVAL)

            # Cleanup expired models from both caches
            for cache, cache_name in [
                (_models, "models"),
                (_classifiers, "classifiers"),
            ]:
                expired_items = cache.pop_expired()
                if expired_items:
                    logger.info(f"Unloading {len(expired_items)} idle {cache_name}")
                    for cache_key, model in expired_items:
                        try:
                            await model.unload()
                            logger.info(f"Successfully unloaded: {cache_key}")
                        except Exception as e:
                            logger.error(
                                f"Error unloading model {cache_key}: {e}", exc_info=True
                            )

        except asyncio.CancelledError:
            logger.info("Model cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}", exc_info=True)
            # Continue running despite errors


def set_cleanup_task(task: asyncio.Task | None) -> None:
    """Set the cleanup task reference."""
    global _cleanup_task
    _cleanup_task = task


def get_cleanup_task() -> asyncio.Task | None:
    """Get the cleanup task reference."""
    return _cleanup_task


async def shutdown_models() -> None:
    """Unload all models during shutdown."""
    # Stop cleanup task
    task = get_cleanup_task()
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        logger.info("Model cleanup task stopped")

    # Unload all remaining models
    if _models:
        logger.info(f"Unloading {len(_models)} remaining model(s)")
        for cache_key, model in list(_models.items()):
            try:
                await model.unload()
                logger.info(f"Unloaded model: {cache_key}")
            except Exception as e:
                logger.error(f"Error unloading model {cache_key}: {e}")
        _models.clear()

    if _classifiers:
        logger.info(f"Unloading {len(_classifiers)} remaining classifier(s)")
        for cache_key, model in list(_classifiers.items()):
            try:
                await model.unload()
                logger.info(f"Unloaded classifier: {cache_key}")
            except Exception as e:
                logger.error(f"Error unloading classifier {cache_key}: {e}")
        _classifiers.clear()


# ============================================================================
# Path Security Utilities
# ============================================================================


def sanitize_model_name(name: str) -> str:
    """Sanitize model name to create a safe filename.

    Only allows alphanumeric characters, hyphens, and underscores.
    This prevents path traversal and ensures consistent naming.
    """
    return "".join(c for c in name if c.isalnum() or c in "-_")


def sanitize_filename(name: str) -> str:
    """Sanitize a filename, preserving extension dots.

    Only allows alphanumeric characters, hyphens, underscores, and dots.
    This prevents path traversal while allowing file extensions like .joblib
    """
    return "".join(c for c in name if c.isalnum() or c in "-_.")


def validate_path_within_directory(path: Path, safe_dir: Path) -> Path:
    """Validate that a path is within the allowed directory.

    This is a security function to prevent path traversal attacks.
    Returns the resolved (absolute) path if valid.

    Raises:
        ValueError: If path is outside the allowed directory
    """
    resolved = path.resolve()
    safe_resolved = safe_dir.resolve()

    # Use Path.is_relative_to for Python 3.9+ compatibility
    try:
        resolved.relative_to(safe_resolved)
    except ValueError:
        raise ValueError(
            f"Security error: Path '{path}' resolves outside allowed directory"
        ) from None

    return resolved
