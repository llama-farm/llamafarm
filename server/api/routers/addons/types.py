"""Addon API types."""

from datetime import datetime

from pydantic import BaseModel


class AddonInfo(BaseModel):
    """Addon metadata."""

    name: str
    display_name: str
    description: str
    component: str
    version: str
    dependencies: list[str] = []
    installed: bool = False
    installed_at: datetime | None = None


class AddonInstallRequest(BaseModel):
    """Request to install an addon."""

    name: str
    restart_service: bool = True


class AddonInstallResponse(BaseModel):
    """Response after initiating addon installation."""

    task_id: str
    status: str
    addon: str


class AddonTaskStatus(BaseModel):
    """Status of an addon installation task."""

    status: str  # "in_progress", "completed", "failed"
    progress: int
    message: str
    error: str | None = None
