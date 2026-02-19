"""Bundle management API endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from core.logging import FastAPIStructLogger

from . import service
from .types import BundleEstimate, BundleRequest, BundleSummary

logger = FastAPIStructLogger()

router = APIRouter(prefix="/v1/bundle", tags=["bundle"])


@router.post("", response_class=StreamingResponse)
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


@router.post("/estimate", response_model=BundleEstimate)
async def estimate_bundle_size(request: BundleRequest):
    """Estimate bundle size based on configuration."""
    error = service.validate_request(request)
    if error:
        raise HTTPException(400, error)

    components = service.estimate_size(request)
    return BundleEstimate(
        estimated_bytes=sum(components.values()),
        components=components,
    )


@router.get("s", response_model=list[BundleSummary])
async def list_bundles():
    """List all completed bundles."""
    return service.list_bundles()


@router.get("s/{bundle_id}/download")
async def download_bundle(bundle_id: str):
    """Download a bundle archive."""
    path = service.get_bundle_path(bundle_id)
    if not path:
        raise HTTPException(404, f"Bundle '{bundle_id}' not found")

    return FileResponse(
        path=str(path),
        media_type="application/gzip",
        filename=path.name,
    )


@router.delete("s/{bundle_id}")
async def delete_bundle(bundle_id: str):
    """Delete a bundle."""
    if not service.delete_bundle(bundle_id):
        raise HTTPException(404, f"Bundle '{bundle_id}' not found")
    return {"status": "deleted"}
