"""Model versioning and deployment tracking for edge/drone fleets.

Endpoints:
- GET    /v1/vision/models/{model_id}/deployments          — List deployments
- POST   /v1/vision/models/{model_id}/deployments          — Register deployment
- PATCH  /v1/vision/models/{model_id}/deployments/{device}  — Update deployment status
- DELETE /v1/vision/models/{model_id}/deployments/{device}  — Remove deployment
- GET    /v1/vision/models/{model_id}/versions              — List model versions
- POST   /v1/vision/models/{model_id}/versions/promote      — Promote version to current

Storage: deployment.json per model in ~/.llamafarm/models/vision/{model_id}/
"""

import contextlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.error_handler import handle_endpoint_errors

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-deployments"])

_VISION_MODELS_DIR: Path | None = None


def set_deployments_models_dir(d: Path) -> None:
    global _VISION_MODELS_DIR
    _VISION_MODELS_DIR = d


def _models_dir() -> Path:
    if _VISION_MODELS_DIR is None:
        raise HTTPException(status_code=500, detail="Vision models dir not configured")
    return _VISION_MODELS_DIR


def _model_dir(model_id: str) -> Path:
    safe_id = Path(model_id).name
    if safe_id != model_id or ".." in model_id or ":" in model_id or "\\" in model_id:
        raise HTTPException(status_code=400, detail="Invalid model_id")
    d = _models_dir() / safe_id
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return d


# =============================================================================
# Deployment Types
# =============================================================================

DeploymentStatus = Literal["pending", "deployed", "failed", "outdated", "offline"]


class DeploymentRecord(BaseModel):
    """A single device deployment record."""
    device_id: str = Field(..., description="Unique device identifier (e.g., drone serial)")
    device_name: str | None = Field(None, description="Human-friendly device name")
    model_version: int = Field(..., description="Deployed model version number")
    export_format: str = Field("pt", description="Deployed format (pt, onnx, engine, tflite)")
    status: DeploymentStatus = Field("pending")
    deployed_at: str | None = None
    last_seen: str | None = None
    last_inference_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeploymentFile(BaseModel):
    """The deployment.json file contents."""
    model_id: str
    deployments: dict[str, DeploymentRecord] = Field(default_factory=dict)
    updated_at: str = ""


class RegisterDeploymentRequest(BaseModel):
    device_id: str
    device_name: str | None = None
    model_version: int | None = None  # None = latest
    export_format: str = "pt"
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateDeploymentRequest(BaseModel):
    status: DeploymentStatus | None = None
    last_inference_count: int | None = None
    metadata: dict[str, Any] | None = None


class PromoteVersionRequest(BaseModel):
    version: int = Field(..., description="Version number to promote to current")


# =============================================================================
# Deployment File I/O
# =============================================================================

def _load_deployments(model_dir: Path, model_id: str) -> DeploymentFile:
    dep_file = model_dir / "deployment.json"
    if dep_file.exists():
        with contextlib.suppress(Exception):
            data = json.loads(dep_file.read_text())
            recs = {}
            for k, v in data.get("deployments", {}).items():
                recs[k] = DeploymentRecord(**v)
            return DeploymentFile(
                model_id=data.get("model_id", model_id),
                deployments=recs,
                updated_at=data.get("updated_at", ""),
            )
    return DeploymentFile(model_id=model_id)


def _save_deployments(model_dir: Path, dep: DeploymentFile) -> None:
    dep.updated_at = datetime.now(timezone.utc).isoformat()
    data = {
        "model_id": dep.model_id,
        "deployments": {k: v.model_dump() for k, v in dep.deployments.items()},
        "updated_at": dep.updated_at,
    }
    (model_dir / "deployment.json").write_text(json.dumps(data, indent=2))


# =============================================================================
# Version Helpers
# =============================================================================

def _list_versions(model_dir: Path) -> list[dict[str, Any]]:
    """List all versioned model files in a model directory."""
    versions: list[dict[str, Any]] = []
    for f in sorted(model_dir.iterdir()):
        if not f.is_file():
            continue
        name = f.stem
        # Match v{N}.{ext} pattern
        if name.startswith("v") and name[1:].isdigit():
            ver_num = int(name[1:])
            versions.append({
                "version": ver_num,
                "filename": f.name,
                "format": f.suffix.lstrip("."),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(
                    f.stat().st_mtime, tz=timezone.utc,
                ).isoformat(),
            })
    # Sort by version
    versions.sort(key=lambda v: v["version"])

    # Check if current.pt exists
    has_current = any(
        (model_dir / f"current{ext}").exists()
        for ext in (".pt", ".onnx", ".engine", ".tflite")
    )

    return versions


# =============================================================================
# Endpoints — Versions
# =============================================================================

@router.get("/v1/vision/models/{model_id}/versions")
@handle_endpoint_errors("vision_list_versions")
async def list_versions(model_id: str) -> dict[str, Any]:
    """List all versions of a vision model."""
    model_dir = _model_dir(model_id)
    versions = _list_versions(model_dir)

    current_version = None
    for ext in (".pt", ".onnx"):
        current = model_dir / f"current{ext}"
        if current.is_symlink() or current.exists():
            target = current.resolve().stem  # e.g., "v3"
            if target.startswith("v") and target[1:].isdigit():
                current_version = int(target[1:])
            break

    return {
        "model_id": model_id,
        "versions": versions,
        "current_version": current_version,
        "total": len(versions),
    }


@router.post("/v1/vision/models/{model_id}/versions/promote")
@handle_endpoint_errors("vision_promote_version")
async def promote_version(model_id: str, request: PromoteVersionRequest) -> dict[str, Any]:
    """Promote a specific version to current (active).

    Creates/updates the current.{ext} symlink to point to v{N}.{ext}.
    Also marks all deployments with older versions as 'outdated'.
    """
    model_dir = _model_dir(model_id)
    ver = request.version

    # Find the version file
    source = None
    for ext in (".pt", ".onnx", ".engine", ".tflite"):
        candidate = model_dir / f"v{ver}{ext}"
        if candidate.exists():
            source = candidate
            break

    if source is None:
        raise HTTPException(status_code=404, detail=f"Version {ver} not found for model '{model_id}'")

    # Create symlink
    current = model_dir / f"current{source.suffix}"
    if current.exists() or current.is_symlink():
        current.unlink()
    current.symlink_to(source.name)

    # Mark outdated deployments
    dep = _load_deployments(model_dir, model_id)
    outdated_count = 0
    for rec in dep.deployments.values():
        if rec.model_version < ver and rec.status == "deployed":
            rec.status = "outdated"
            outdated_count += 1
    if outdated_count > 0:
        _save_deployments(model_dir, dep)

    return {
        "model_id": model_id,
        "promoted_version": ver,
        "format": source.suffix.lstrip("."),
        "current_file": f"current{source.suffix}",
        "outdated_deployments": outdated_count,
    }


# =============================================================================
# Endpoints — Deployments
# =============================================================================

@router.get("/v1/vision/models/{model_id}/deployments")
@handle_endpoint_errors("vision_list_deployments")
async def list_deployments(model_id: str) -> dict[str, Any]:
    """List all device deployments for a model."""
    model_dir = _model_dir(model_id)
    dep = _load_deployments(model_dir, model_id)

    return {
        "model_id": model_id,
        "deployments": [rec.model_dump() for rec in dep.deployments.values()],
        "total": len(dep.deployments),
        "status_summary": _status_summary(dep),
    }


@router.post("/v1/vision/models/{model_id}/deployments")
@handle_endpoint_errors("vision_register_deployment")
async def register_deployment(model_id: str, request: RegisterDeploymentRequest) -> dict[str, Any]:
    """Register a device deployment for a model.

    If model_version is not specified, uses the latest available version.
    """
    model_dir = _model_dir(model_id)

    # Resolve version
    version = request.model_version
    if version is None:
        versions = _list_versions(model_dir)
        if not versions:
            raise HTTPException(status_code=422, detail="No versions available for this model")
        version = versions[-1]["version"]

    dep = _load_deployments(model_dir, model_id)

    safe_device = Path(request.device_id).name
    if safe_device != request.device_id or ".." in request.device_id:
        raise HTTPException(status_code=400, detail="Invalid device_id")

    now = datetime.now(timezone.utc).isoformat()
    dep.deployments[request.device_id] = DeploymentRecord(
        device_id=request.device_id,
        device_name=request.device_name,
        model_version=version,
        export_format=request.export_format,
        status="pending",
        deployed_at=now,
        last_seen=now,
        metadata=request.metadata,
    )
    _save_deployments(model_dir, dep)

    return {
        "model_id": model_id,
        "device_id": request.device_id,
        "model_version": version,
        "status": "pending",
    }


@router.patch("/v1/vision/models/{model_id}/deployments/{device_id}")
@handle_endpoint_errors("vision_update_deployment")
async def update_deployment(
    model_id: str, device_id: str, request: UpdateDeploymentRequest,
) -> dict[str, Any]:
    """Update deployment status (e.g., device reports back after OTA update)."""
    model_dir = _model_dir(model_id)
    dep = _load_deployments(model_dir, model_id)

    rec = dep.deployments.get(device_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"No deployment for device '{device_id}'")

    if request.status is not None:
        rec.status = request.status
    if request.last_inference_count is not None:
        rec.last_inference_count = request.last_inference_count
    if request.metadata is not None:
        rec.metadata.update(request.metadata)
    rec.last_seen = datetime.now(timezone.utc).isoformat()

    _save_deployments(model_dir, dep)

    return {"model_id": model_id, "device_id": device_id, "updated": rec.model_dump()}


@router.delete("/v1/vision/models/{model_id}/deployments/{device_id}")
@handle_endpoint_errors("vision_delete_deployment")
async def delete_deployment(model_id: str, device_id: str) -> dict[str, Any]:
    """Remove a device deployment record."""
    model_dir = _model_dir(model_id)
    dep = _load_deployments(model_dir, model_id)

    if device_id not in dep.deployments:
        raise HTTPException(status_code=404, detail=f"No deployment for device '{device_id}'")

    del dep.deployments[device_id]
    _save_deployments(model_dir, dep)

    return {"model_id": model_id, "device_id": device_id, "deleted": True}


def _status_summary(dep: DeploymentFile) -> dict[str, int]:
    """Count deployments by status."""
    summary: dict[str, int] = {}
    for rec in dep.deployments.values():
        summary[rec.status] = summary.get(rec.status, 0) + 1
    return summary
