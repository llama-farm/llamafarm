"""Bundle management API endpoints."""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from core.logging import FastAPIStructLogger

from . import service
from .types import BundleEstimate, BundleRequest, BundleSummary

logger = FastAPIStructLogger()

router = APIRouter(tags=["bundle"])


@router.get("/bundle/version")
async def get_bundle_version():
    """Get the version that will be used for bundling."""
    ver = await service.get_latest_version()
    return {"version": ver}


@router.post("/bundle", response_class=StreamingResponse)
async def create_bundle(request: BundleRequest):
    """Create a new bundle. Returns SSE stream of progress events."""
    error = service.validate_request(request)
    if error:
        raise HTTPException(400, error)

    return StreamingResponse(
        service.create_bundle(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/bundle/estimate", response_model=BundleEstimate)
def estimate_bundle_size(request: BundleRequest):
    """Estimate bundle size based on configuration."""
    error = service.validate_request(request)
    if error:
        raise HTTPException(400, error)

    components = service.estimate_size(request)
    return BundleEstimate(
        estimated_bytes=sum(components.values()),
        components=components,
    )


@router.get("/bundles", response_model=list[BundleSummary])
def list_bundles():
    """List all completed bundles."""
    return service.list_bundles()


@router.get("/bundles/{bundle_id}/download")
def download_bundle(bundle_id: str):
    """Download a bundle archive."""
    try:
        path = service.get_bundle_path(bundle_id)
    except Exception as err:
        logger.exception("Error retrieving bundle path")
        raise HTTPException(500, "Internal server error") from err
    if not path:
        raise HTTPException(404, "Bundle not found")

    # Inline containment check so CodeQL can verify the path is safe.
    safe_root = os.path.realpath(service._bundles_dir())
    real_path = os.path.realpath(path)
    if not real_path.startswith(safe_root + os.sep):
        raise HTTPException(404, "Bundle not found")

    return FileResponse(
        path=real_path,
        media_type="application/gzip",
        filename=os.path.basename(real_path),
    )


@router.delete("/bundles/{bundle_id}")
def delete_bundle(bundle_id: str):
    """Delete a bundle."""
    try:
        deleted = service.delete_bundle(bundle_id)
    except Exception as err:
        logger.exception("Error deleting bundle")
        raise HTTPException(500, "Internal server error") from err
    if not deleted:
        raise HTTPException(404, "Bundle not found")
    return {"status": "deleted"}
