"""Health router for health check and models list endpoints."""

import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException

from core.logging import UniversalRuntimeLogger

logger = UniversalRuntimeLogger("universal-runtime.health")

router = APIRouter(tags=["health"])

# Dependency injection for models cache and device info
_models: dict | None = None
_get_device_info_fn: Callable[[], dict[str, Any]] | None = None


def set_models_cache(models: dict | None) -> None:
    """Set the models cache for health check."""
    global _models
    _models = models


def set_device_info_getter(
    get_device_info_fn: Callable[[], dict[str, Any]] | None,
) -> None:
    """Set the device info getter function."""
    global _get_device_info_fn
    _get_device_info_fn = get_device_info_fn


@router.get("/health")
async def health_check():
    """Health check endpoint with device information."""
    if _models is None or _get_device_info_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Health router not initialized. Call set_models_cache() and set_device_info_getter() first.",
        )

    device_info = _get_device_info_fn()
    return {
        "status": "healthy",
        "device": device_info,
        "loaded_models": list(_models.keys()),
        "timestamp": datetime.utcnow().isoformat(),
        "pid": os.getpid(),
    }


@router.get("/v1/models")
async def list_models():
    """List currently loaded models."""
    if _models is None:
        raise HTTPException(
            status_code=500,
            detail="Health router not initialized. Call set_models_cache() first.",
        )

    models_list = []
    for model_id, model in _models.items():
        models_list.append(
            {
                "id": model_id,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "transformers-runtime",
                "type": model.model_type,
            }
        )

    return {"object": "list", "data": models_list}
