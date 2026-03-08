"""
Preload router for LlamaFarm server.

Provides project-scoped endpoints to trigger model preloading via the Universal Runtime.
These endpoints run on the main LlamaFarm server and proxy to the
Universal Runtime.

Since the runtime doesn't accept config_path for security,
this server handles preload differently:
- Option A: Copy project's llamafarm.yaml to runtime's working directory, then trigger preload
- Option B: Trigger separate runtime instance per project
- Option C: Have runtime support project-based preloading via different mechanism

For now, this implements a basic approach where we document the limitation.
"""

import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
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


def _validate_path_component(component: str, name: str) -> None:
    """Validate that a path component doesn't contain traversal sequences.

    This prevents path traversal attacks by rejecting any component that:
    - Contains ".." (parent directory reference)
    - Starts with "/" or "\\" (absolute path)
    - Contains special characters that could be used for traversal

    Args:
        component: The path component to validate (e.g., namespace or project name)
        name: Human-readable name for error messages (e.g., "namespace", "project")

    Raises:
        HTTPException: If the component contains invalid characters or patterns
    """
    if not component:
        raise HTTPException(status_code=400, detail=f"Invalid {name}: cannot be empty")

    # reject any path traversal attempts
    if ".." in component:
        raise HTTPException(
            status_code=400, detail=f"Invalid {name}: path traversal not allowed (..)"
        )

    # reject absolute paths
    if component.startswith("/") or component.startswith("\\"):
        raise HTTPException(
            status_code=400, detail=f"Invalid {name}: absolute paths not allowed"
        )

    # reject null bytes (used in some path traversal attacks)
    if "\0" in component:
        raise HTTPException(
            status_code=400, detail=f"Invalid {name}: null bytes not allowed"
        )

    # only allow safe characters: alphanumeric, dash, underscore, dot
    if not re.match(r"^[a-zA-Z0-9._-]+$", component):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {name}: only alphanumeric, dash, underscore, and dot allowed",
        )

    # reject components that are only dots
    # (e.g., ".", "..", "...", etc.)
    if component.replace(".", "") == "":
        raise HTTPException(
            status_code=400, detail=f"Invalid {name}: cannot consist only of dots"
        )


def _validate_project_path(
    namespace: str, project: str, base_dir: Path
) -> tuple[Path, Path]:
    """Validate and construct project paths with traversal protection.

    Args:
        namespace: Project namespace (validated)
        project: Project name (validated)
        base_dir: Base directory for projects (e.g., ~/.llamafarm/projects)

    Returns:
        Tuple of (project_dir, config_path) where both are validated Path objects

    Raises:
        HTTPException: If validation fails or path traversal is detected
    """

    _validate_path_component(namespace, "namespace")
    _validate_path_component(project, "project")

    project_dir = base_dir / "projects" / namespace / project
    config_path = project_dir / "llamafarm.yaml"

    # resolve to canonical paths to detect traversal
    try:
        resolved_project_dir = project_dir.resolve(strict=False)
        allowed_base = (base_dir / "projects").resolve(strict=False)
    except (ValueError, OSError) as e:
        logger.warning(f"Path resolution failed for {namespace}/{project}: {e}")
        raise HTTPException(
            status_code=400, detail="Invalid project path: cannot resolve"
        ) from e

    # ensure the resolved path is still within the allowed directory
    try:
        # Use relative_to() to ensure resolved path is child of allowed_base
        resolved_project_dir.relative_to(allowed_base)
    except ValueError as e:
        # relative_to() raises ValueError if path is not a child
        logger.warning(
            f"Path traversal attempt detected: {namespace}/{project} "
            f"resolves to {resolved_project_dir} which is outside {allowed_base}"
        )
        raise HTTPException(
            status_code=400, detail="Invalid project path: traversal detected"
        ) from e

    # ensure we're still in the expected namespace/project structure
    expected_parts = ["projects", namespace, project]
    actual_parts = resolved_project_dir.parts[-len(expected_parts) :]

    if list(actual_parts) != expected_parts:
        logger.warning(
            f"Path structure mismatch for {namespace}/{project}: "
            f"expected {expected_parts}, got {actual_parts}"
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid project path: unexpected structure (possible symlink attack)",
        )

    return resolved_project_dir, config_path


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
    try:
        result = await UniversalRuntimeService._make_request(
            "GET",
            "/v1/preload/resources",
            timeout=10.0,
        )
    except Exception as e:
        logger.error(f"Failed to get resources from runtime: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to communicate with Universal Runtime: {str(e)}",
        ) from e

    return result


@router.post("/projects/{namespace}/{project}")
async def trigger_project_preload(
    namespace: str,
    project: str,
) -> ProjectPreloadResponse:
    """Trigger model preloading for a specific project.

    **CURRENT LIMITATION**: The Universal Runtime doesn't accept config_path
    for security reasons (prevents path traversal). This endpoint currently
    returns a placeholder response.

    **Future Implementation Options**:
    1. Copy project's llamafarm.yaml to runtime's working directory temporarily
    2. Add internal-only API with authentication for config_path
    3. Implement project-aware preloading in runtime using project IDs

    **Security**: Path parameters are validated to prevent directory traversal attacks.

    **Path Parameters**:
    - `namespace`: Project namespace (e.g., "default")
        - Must match pattern: `[a-zA-Z0-9._-]+`
        - Cannot contain ".." or start with "/"
    - `project`: Project name (e.g., "my-chatbot")
        - Must match pattern: `[a-zA-Z0-9._-]+`
        - Cannot contain ".." or start with "/"

    **Example**: `POST /v1/models/preload/projects/default/my-chatbot`
    """
    from core.settings import settings

    logger.info(f"Triggering preload for project {namespace}/{project}")

    # validate and construct paths with traversal protection
    try:
        project_dir, config_path = _validate_project_path(
            namespace, project, Path(settings.lf_data_dir)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error validating project path: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal error validating project path"
        ) from e

    # check if config exists
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
                "message": f"No config found for project {namespace}/{project}",
            },
        )

    if not config_path.is_file():
        logger.warning(f"Config path is not a regular file: {config_path}")
        raise HTTPException(status_code=400, detail="Config path is not a regular file")

    # TODO: Implement project-specific preloading
    # Options:
    # 1. Temporarily copy config to runtime's working directory
    # 2. Add authenticated internal API that accepts config_path
    # 3. Add project-based preloading mechanism to runtime

    logger.warning(
        f"Project-specific preload not yet implemented for {namespace}/{project}. "
        f"Runtime endpoint doesn't accept config_path for security. "
        f"Calling default preload instead."
    )

    # For now, trigger default preload (uses runtime's working directory)
    try:
        result = await UniversalRuntimeService._make_request(
            "POST",
            "/v1/preload",
            json={},  # No config_path - uses runtime's working directory
            timeout=600.0,
        )
    except Exception as e:
        logger.error(f"Failed to call runtime preload endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to communicate with Universal Runtime: {str(e)}",
        ) from e

    # Add note about limitation
    if "summary" in result:
        result["summary"]["message"] = (
            "Loaded from runtime's working directory (project-specific preload not yet implemented)"
        )

    return ProjectPreloadResponse(
        status=result.get("status", "unknown"),
        results=result.get("results", {}),
        summary=result.get("summary", {}),
        resources=result.get("resources"),
    )


@router.get("/projects/{namespace}/{project}/status")
async def get_project_preload_status(namespace: str, project: str):
    """Get preload status for a specific project.

    **Note**: Returns global status (all loaded models) since project-specific
    tracking is not yet implemented.

    **Security**: Path parameters are validated to prevent directory traversal attacks.

    **Example**: `GET /v1/models/preload/projects/default/my-chatbot/status`
    """

    _validate_path_component(namespace, "namespace")
    _validate_path_component(project, "project")

    logger.info(f"Getting preload status for project {namespace}/{project}")

    try:
        result = await UniversalRuntimeService._make_request(
            "GET",
            "/v1/preload/status",
            timeout=10.0,
        )
    except Exception as e:
        logger.error(f"Failed to get preload status from runtime: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to communicate with Universal Runtime: {str(e)}",
        ) from e

    return result
