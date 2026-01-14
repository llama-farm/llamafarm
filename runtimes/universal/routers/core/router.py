"""
Core endpoints for health checks and model listing.
"""

import os
from datetime import datetime

from fastapi import APIRouter

from state import get_models_cache
from utils.device import get_device_info

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint with device information."""
    device_info = get_device_info()
    models_cache = get_models_cache()
    return {
        "status": "healthy",
        "device": device_info,
        "loaded_models": list(models_cache.keys()),
        "timestamp": datetime.utcnow().isoformat(),
        "pid": os.getpid(),
    }


@router.get("/v1/models")
async def list_models():
    """List currently loaded models."""
    models_cache = get_models_cache()
    models_list = []
    for model_id, model in models_cache.items():
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
