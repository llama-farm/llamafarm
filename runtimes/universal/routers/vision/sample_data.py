"""Sample data management — clone/check vision-sample-data repo."""

import asyncio
import logging
import os
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-sample-data"])

SAMPLE_REPO = "https://github.com/llama-farm/vision-sample-data.git"
_data_dir: Path = Path.home()  # will be overridden by set_data_dir


def set_data_dir(path: Path) -> None:
    global _data_dir
    _data_dir = path


def _sample_dir() -> Path:
    return _data_dir / "vision-sample-data"


class SampleDataStatus(BaseModel):
    installed: bool
    path: str
    categories: list[str] = Field(default_factory=list)


class CloneResponse(BaseModel):
    success: bool
    path: str
    message: str


@router.get("/v1/vision/sample-data/status", response_model=SampleDataStatus)
async def sample_data_status() -> SampleDataStatus:
    """Check if sample data repo is cloned."""
    sd = _sample_dir()
    if sd.is_dir():
        cats = sorted(
            d.name for d in sd.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        return SampleDataStatus(installed=True, path=str(sd), categories=cats)
    return SampleDataStatus(installed=False, path=str(sd))


@router.post("/v1/vision/sample-data/clone", response_model=CloneResponse)
async def clone_sample_data() -> CloneResponse:
    """Clone the vision-sample-data repo if not already present."""
    sd = _sample_dir()
    if sd.is_dir():
        return CloneResponse(success=True, path=str(sd), message="Already installed")

    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", SAMPLE_REPO, str(sd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            err = stderr.decode().strip() if stderr else "Unknown error"
            return CloneResponse(success=False, path=str(sd), message=f"Clone failed: {err}")
        return CloneResponse(success=True, path=str(sd), message="Cloned successfully")
    except asyncio.TimeoutError:
        return CloneResponse(success=False, path=str(sd), message="Clone timed out (120s)")
    except Exception as e:
        return CloneResponse(success=False, path=str(sd), message=str(e))
