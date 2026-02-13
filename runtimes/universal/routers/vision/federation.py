"""Federation router for multi-node vision cascade.

Endpoints for:
- Peer management (add/remove/list)
- Inbound escalation (another node sends us an image to analyze)
- Model package management (create/list/import/push)
- Federation health status
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/vision/federation", tags=["vision-federation"])


# ============================================================
# Request/Response models
# ============================================================


class PeerInfo(BaseModel):
    name: str
    url: str
    models: list[str] = []
    gpu_vram_gb: float = 0
    priority: int = 0
    timeout: float = 30.0


class PeerStatus(BaseModel):
    name: str
    url: str
    healthy: bool = False
    latency_ms: float = 0.0
    models: list[str] = []


class EscalateRequest(BaseModel):
    """Inbound escalation: another node wants our opinion."""

    image: str = Field(..., description="Base64-encoded image")
    model: str = Field(default="", description="Preferred model to use")
    confidence_threshold: float = 0.5
    opinions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Prior model opinions in the cascade",
    )


class EscalateResponse(BaseModel):
    model: str
    detections: list[dict[str, Any]]
    inference_time_ms: float
    node_id: str = "local"


class PackageInfo(BaseModel):
    model_id: str
    version: int
    task: str = "detection"
    class_names: list[str] = []
    accuracy: float = 0.0
    created_at: str = ""
    package_path: str = ""
    size_kb: float = 0.0


class CreatePackageRequest(BaseModel):
    model_id: str
    model_path: str = ""
    description: str = ""


class ImportPackageRequest(BaseModel):
    source: str = Field(..., description="Local file path to a model package archive")
    model_id: str = ""


# ============================================================
# Peer management
# ============================================================


@router.get("/peers", response_model=list[PeerStatus])
async def list_peers() -> list[dict[str, Any]]:
    """List all configured federation peers."""
    try:
        from config.federation import FederationConfig
    except ImportError:
        return []

    # For now return empty -- peers are registered dynamically
    return []


@router.post("/peers")
async def register_peer(peer: PeerInfo) -> dict[str, Any]:
    """Register a federation peer."""
    logger.info(f"Registered peer: {peer.name} at {peer.url}")
    return {
        "name": peer.name,
        "url": peer.url,
        "registered": True,
    }


@router.delete("/peers/{name}")
async def remove_peer(name: str) -> dict[str, Any]:
    """Remove a federation peer."""
    logger.info(f"Removed peer: {name}")
    return {"name": name, "removed": True}


@router.get("/status")
async def federation_status() -> dict[str, Any]:
    """Get federation health status."""
    return {
        "enabled": False,
        "peers": 0,
        "healthy_peers": 0,
    }


# ============================================================
# Inbound escalation
# ============================================================


@router.post("/escalate", response_model=EscalateResponse)
async def handle_escalation(request: EscalateRequest) -> dict[str, Any]:
    """Handle an inbound escalation from another node.

    Another LlamaFarm instance sends us an image because its local
    models were uncertain. We run our model and return our opinion.
    """
    import base64
    import time

    start = time.time()

    # Decode image
    try:
        image_bytes = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # TODO: Load requested model and run detection
    # For now, return empty result (this will be wired to model registry)
    elapsed = (time.time() - start) * 1000

    return {
        "model": request.model or "none",
        "detections": [],
        "inference_time_ms": elapsed,
        "node_id": "local",
    }


# ============================================================
# Model packages
# ============================================================


@router.get("/packages", response_model=list[PackageInfo])
async def list_packages() -> list[dict[str, Any]]:
    """List available model packages."""
    try:
        from pathlib import Path

        from vision_training.model_package import ModelPackager

        packager = ModelPackager()
        packages = packager.list_packages()

        return [
            {
                "model_id": pkg.metadata.model_id,
                "version": pkg.metadata.version,
                "task": pkg.metadata.task,
                "class_names": pkg.metadata.class_names,
                "accuracy": pkg.metadata.accuracy,
                "created_at": pkg.metadata.created_at,
                "package_path": pkg.package_path,
                "size_kb": Path(pkg.package_path).stat().st_size / 1024,
            }
            for pkg in packages
        ]
    except Exception as e:
        logger.error(f"Failed to list packages: {e}")
        return []


@router.post("/packages")
def create_package(request: CreatePackageRequest) -> dict[str, Any]:
    """Create a model package from a trained model."""
    from pathlib import Path

    from .models import _get_models_dir

    # Validate model_path is within the models directory
    models_dir = _get_models_dir()
    resolved = Path(request.model_path).resolve()
    if not str(resolved).startswith(str(models_dir.resolve())):
        raise HTTPException(
            status_code=400, detail="Model path must be within the models directory"
        )

    try:
        from vision_training.model_package import ModelPackager, PackageMetadata

        packager = ModelPackager()
        metadata = PackageMetadata(
            model_id=request.model_id,
            description=request.description,
        )

        package = packager.create_package(
            model_path=request.model_path,
            metadata=metadata,
        )

        return {
            "model_id": package.metadata.model_id,
            "package_path": package.package_path,
            "created": True,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/packages/import")
def import_package(request: ImportPackageRequest) -> dict[str, Any]:
    """Import a model package from a local file path."""
    try:
        from vision_training.model_package import ModelPackager

        packager = ModelPackager()
        package = packager.extract_package(request.source)

        return {
            "model_id": package.metadata.model_id,
            "version": package.metadata.version,
            "model_path": package.model_path,
            "imported": True,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Package not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
