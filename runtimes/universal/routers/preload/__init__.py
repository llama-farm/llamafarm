"""
Preload router for Universal Runtime.

Provides endpoints to manually trigger model preloading and check status.
These endpoints run directly on the Universal Runtime.

The /v1/preload endpoint does NOT accept user-supplied paths.
It only reads from the runtime's working directory to prevent path traversal attacks.
The main LlamaFarm server handles path validation and sends preload requests.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/preload", tags=["preload"])


class PreloadRequest(BaseModel):
    """Request to manually trigger model preload.

    No config_path parameter - reads from working directory only.
    This prevents path traversal attacks entirely.
    """

    pass  # No parameters - always uses default config search


class PreloadResponse(BaseModel):
    """Response from preload operation."""

    status: str
    """Overall status: success, partial, or failed"""

    results: dict[str, dict]
    """Per-model results with status, pinned, load_time_seconds, error_message"""

    summary: dict
    """Aggregate statistics: loaded, failed, already_loaded, skipped, total_time_seconds, concurrency_used"""

    resources: dict | None = None
    """Resource information if available: device, cpu_count, available_ram_gb, available_vram_gb"""


# Dependency injection - set by server.py during startup
_preload_fn = None
_preload_lock = asyncio.Lock()


def set_preload_function(fn):
    """Set the preload function to call.

    This should be called from server.py with preload_models_from_config.
    """
    global _preload_fn
    _preload_fn = fn


@router.post("", response_model=PreloadResponse)
async def trigger_preload(request: PreloadRequest | None = None):
    """Manually trigger model preloading.

    This reads llamafarm.yaml from the runtime's working directory,
    finds models with preload: true, and loads them with optimal
    concurrency based on system resources.

    This endpoint does NOT accept a config_path parameter.
    It always reads from the working directory to prevent path traversal attacks.
    The main LlamaFarm server is responsible for validating project paths
    and triggering preloads for specific projects.

    **Note**: This is idempotent - models already in cache are marked as
    "already_loaded" and not reloaded.

    **Example Request**:
    ```json
    {}
    ```

    **Example Response**:
    ```json
    {
        "status": "success",
        "results": {
            "fast": {
                "status": "loaded",
                "pinned": true,
                "load_time_seconds": 8.45
            },
            "embedder": {
                "status": "already_loaded",
                "pinned": false
            }
        },
        "summary": {
            "loaded": 1,
            "failed": 0,
            "already_loaded": 1,
            "skipped": 0,
            "total_time_seconds": 8.52,
            "concurrency_used": 3
        },
        "resources": {
            "device": "cuda",
            "cpu_count": 16,
            "available_ram_gb": 28.4,
            "available_vram_gb": 22.1,
            "gpu_name": "NVIDIA GeForce RTX 4090"
        }
    }
    ```
    """
    if _preload_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Preload function not initialized. This is a server configuration error.",
        )

    # Initialize request if not provided
    if request is None:
        request = PreloadRequest()

    if _preload_lock.locked():
        raise HTTPException(
            status_code=429,
            detail="A model preload operation is already in progress. Please wait for it to complete.",
        )

    logger.info("Manual preload triggered (using working directory config)")

    try:
        async with _preload_lock:
            # ALWAYS use None (default search in working directory)
            result = await _preload_fn(config_path=None)

            summary = result.get("summary", {})
            loaded = summary.get("loaded", 0)
            failed = summary.get("failed", 0)

            if failed > 0 and loaded == 0:
                status = "failed"
            elif failed > 0:
                status = "partial"
            else:
                status = "success"

            return PreloadResponse(
                status=status,
                results=result.get("results", {}),
                summary=summary,
                resources=result.get("resources"),
            )

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status codes
        raise
    except Exception as e:
        logger.error(f"Preload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Model preload failed. Check server logs for details.",
        ) from e


@router.get("/status")
async def get_preload_status():
    """Get current preload status and loaded models.

    Returns information about which models are currently loaded in cache,
    which are pinned, and cache statistics.

    **Example Response**:
    ```json
    {
        "loaded_models": [
            {
                "cache_key": "language:microsoft/phi-2:...",
                "model_id": "microsoft/phi-2",
                "pinned": true,
                "idle_time_seconds": 120.5
            },
            {
                "cache_key": "encoder:embedding:sentence-transformers/all-MiniLM-L6-v2:...",
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "pinned": false,
                "idle_time_seconds": 45.2
            }
        ],
        "cache_stats": {
            "total_items": 2,
            "pinned_items": 1,
            "evictable_items": 1,
            "expired_items": 0,
            "cache_full": false,
            "max_size": 1000,
            "ttl_seconds": 300
        }
    }
    ```
    """
    # Use state.get_models_cache() to access the correct cache instance
    from state import get_models_cache

    _models = get_models_cache()

    # Get loaded models info
    loaded_models = []
    for cache_key in _models:
        model = _models.get(cache_key)
        if model:
            idle_time = _models.get_idle_time(cache_key)
            loaded_models.append(
                {
                    "cache_key": cache_key,
                    "model_id": model.model_id,
                    "pinned": _models.is_pinned(cache_key),
                    "idle_time_seconds": round(idle_time, 1) if idle_time else 0.0,
                }
            )

    # Get cache statistics
    cache_stats = _models.get_cache_stats()

    return {
        "loaded_models": loaded_models,
        "cache_stats": cache_stats,
    }


@router.get("/resources")
async def get_resources():
    """Get system resource information for preloading.

    Returns detected CPU cores, RAM, VRAM, and recommended concurrency levels.
    Useful for understanding preload performance and tuning.

    **Example Response**:
    ```json
    {
        "device": "cuda",
        "cpu_count": 16,
        "available_ram_gb": 28.4,
        "total_ram_gb": 32.0,
        "available_vram_gb": 22.1,
        "total_vram_gb": 24.0,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_count": 1,
        "optimal_concurrency": 4,
        "max_concurrency": 8
    }
    ```
    """
    from utils.device import get_optimal_device
    from utils.resource_detect import get_resource_info

    device = get_optimal_device()
    resource_info = get_resource_info(device)

    return {
        "device": resource_info.device,
        "cpu_count": resource_info.cpu_count,
        "available_ram_gb": round(resource_info.available_ram_gb, 1),
        "total_ram_gb": round(resource_info.total_ram_gb, 1),
        "available_vram_gb": round(resource_info.available_vram_gb, 1),
        "total_vram_gb": round(resource_info.total_vram_gb, 1),
        "gpu_name": resource_info.gpu_name,
        "gpu_count": resource_info.gpu_count,
        "optimal_concurrency": resource_info.optimal_concurrency,
        "max_concurrency": resource_info.max_concurrency,
    }
