"""Addon management API endpoints."""

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from .service import AddonService
from .types import (
    AddonInfo,
    AddonInstallRequest,
    AddonInstallResponse,
    AddonTaskStatus,
)

router = APIRouter(prefix="/v1/addons", tags=["addons"])

# Global service instance
addon_service = AddonService()


@router.get("", response_model=list[AddonInfo])
async def list_addons():
    """List all available addons and their installation status."""
    return addon_service.list_addons()


@router.post("/install", response_model=AddonInstallResponse)
async def install_addon(request: AddonInstallRequest, background_tasks: BackgroundTasks):
    """Install an addon (async operation)."""

    # Validate addon exists
    if not addon_service.addon_exists(request.name):
        raise HTTPException(404, f"Addon '{request.name}' not found")

    # Create background task
    task_id = f"addon-install-{uuid.uuid4()}"
    background_tasks.add_task(
        addon_service.install_addon_task, task_id, request.name, request.restart_service
    )

    return AddonInstallResponse(task_id=task_id, status="in_progress", addon=request.name)


@router.get("/tasks/{task_id}", response_model=AddonTaskStatus)
async def get_task_status(task_id: str):
    """Get the status of an addon installation task."""
    status = await addon_service.get_task_status(task_id)
    if not status:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return status


@router.post("/uninstall")
async def uninstall_addon(request: AddonInstallRequest):
    """Uninstall an addon."""
    await addon_service.uninstall_addon(request.name)
    return {"status": "success"}
