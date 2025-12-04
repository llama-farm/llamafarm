"""Disk space API endpoints."""

import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from server.services.disk_space_service import DiskSpaceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/disk")
def get_disk_space():
    """Get disk space information for cache and system disk.

    Returns:
        Dict with disk space information for both cache and system disk
    """
    try:
        cache_info, system_info = DiskSpaceService.check_both_disks()

        return {
            "cache": asdict(cache_info),
            "system": asdict(system_info),
        }
    except OSError as e:
        logger.error(f"Failed to check disk space: {e}", exc_info=True)
        # Don't expose filesystem paths or error details to clients
        raise HTTPException(
            status_code=500,
            detail="Failed to check disk space. Please try again later.",
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error checking disk space: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while checking disk space.",
        ) from e

