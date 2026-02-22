"""
Preload router for LlamaFarm server.

Provides project-scoped endpoints to trigger model preloading via the Universal Runtime.
These endpoints run on the main LlamaFarm server and proxy to the
Universal Runtime.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from services.universal_runtime_service import UniversalRuntimeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models/preload", tags=["models"])


class ProjectPreloadRequest(BaseModel):
    """Request to trigger model preload for a project."""

    pass  # No parameters needed - reads from project's llamafarm.yaml


class ProjectPreloadResponse(BaseModel):
    """Response from project preload operation."""

    status: str
    """Overall status: success, partial, or failed"""

    results: dict[str, dict]
    """Per-model results"""

    summary: dict
    """Aggregate statistics"""

    resources: dict | None = None
    """Resource information"""


@router.post("/{namespace}/{project}")
async def trigger_project_preload(
    namespace: str,
    project: str,
) -> ProjectPreloadResponse:
    """Trigger model preloading for a specific project.

    This endpoint:
    1. Reads the project's llamafarm.yaml configuration
    2. Finds models with `preload: true`
    3. Loads them concurrently using optimal resource detection
    4. Returns detailed per-model status

    **Path Parameters**:
    - `namespace`: Project namespace (e.g., "default")
    - `project`: Project name (e.g., "my-chatbot")

    **Example**: `POST /v1/models/preload/default/my-chatbot`

    **Response**:
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
            "available_vram_gb": 22.1
        }
    }
    ```
    """
    from pathlib import Path

    from core.settings import settings

    logger.info(f"Triggering preload for project {namespace}/{project}")

    project_dir = Path(settings.lf_data_dir) / "projects" / namespace / project
    config_path = project_dir / "llamafarm.yaml"

    if not config_path.exists():
        logger.warning(f"Config not found at {config_path}")

        return ProjectPreloadResponse(
            status="success",
            results={},
            summary={
                "loaded": 0,
                "failed": 0,
                "already_loaded": 0,
                "skipped": 0,
                "total_time_seconds": 0.0,
                "concurrency_used": 0,
                "message": f"No config found at {config_path}",
            },
        )

    # Proxy to Universal Runtime's preload endpoint
    payload = {
        "config_path": str(config_path),
    }

    result = await UniversalRuntimeService._make_request(
        "POST",
        "/v1/preload",
        json=payload,
        timeout=600.0,  # 10 minutes for preload (can be slow on first run)
    )

    return ProjectPreloadResponse(
        status=result.get("status", "unknown"),
        results=result.get("results", {}),
        summary=result.get("summary", {}),
        resources=result.get("resources"),
    )


@router.get("/{namespace}/{project}/status")
async def get_project_preload_status(namespace: str, project: str):
    """Get preload status for a specific project.

    Returns information about which models are loaded for this project.

    **Path Parameters**:
    - `namespace`: Project namespace
    - `project`: Project name

    **Example**: `GET /v1/models/preload/default/my-chatbot/status`

    **Response**:
    ```json
    {
        "loaded_models": [
            {
                "cache_key": "language:microsoft/phi-2:...",
                "model_id": "microsoft/phi-2",
                "pinned": true,
                "idle_time_seconds": 120.5
            }
        ],
        "cache_stats": {
            "total_items": 2,
            "pinned_items": 1,
            "evictable_items": 1,
            "cache_full": false
        }
    }
    ```
    """
    logger.info(f"Getting preload status for project {namespace}/{project}")

    result = await UniversalRuntimeService._make_request(
        "GET",
        "/v1/preload/status",
        timeout=10.0,
    )

    return result


@router.get("/resources")
async def get_preload_resources():
    """Get system resource information for model preloading.

    Returns detected hardware capabilities and recommended concurrency.
    This is **not** project-scoped (hardware is shared across all projects).

    **Example**: `GET /v1/models/preload/resources`

    **Response**:
    ```json
    {
        "device": "cuda",
        "cpu_count": 16,
        "available_ram_gb": 28.4,
        "total_ram_gb": 32.0,
        "available_vram_gb": 22.1,
        "total_vram_gb": 24.0,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "optimal_concurrency": 4,
        "max_concurrency": 8
    }
    ```
    """

    result = await UniversalRuntimeService._make_request(
        "GET",
        "/v1/preload/resources",
        timeout=10.0,
    )

    return result
