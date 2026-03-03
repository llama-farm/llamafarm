"""Sample data management — clone/check vision-sample-data repo."""

import asyncio
import logging
import os
import random
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-sample-data"])

SAMPLE_REPO = "https://github.com/llama-farm/vision-sample-data.git"
_data_dir: Path = Path.home()  # will be overridden by set_data_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def set_data_dir(path: Path) -> None:
    global _data_dir
    _data_dir = path


def _sample_dir() -> Path:
    return _data_dir / "vision-sample-data"


def _create_train_val_split(sd: Path, train_ratio: float = 0.8) -> None:
    """Create train/val symlink split for YOLO classification."""
    train_dir = sd / "train"
    val_dir = sd / "val"
    if train_dir.exists():
        return  # already created

    categories = [d for d in sd.iterdir() if d.is_dir() and not d.name.startswith(".") and not d.is_symlink()]
    rng = random.Random(42)

    for cat in categories:
        images = sorted(f for f in cat.iterdir() if f.suffix.lower() in IMAGE_EXTS and not f.is_symlink())
        if not images:
            continue
        rng.shuffle(images)
        split = max(1, int(len(images) * train_ratio))

        (train_dir / cat.name).mkdir(parents=True, exist_ok=True)
        (val_dir / cat.name).mkdir(parents=True, exist_ok=True)

        for img in images[:split]:
            dest = train_dir / cat.name / img.name
            if not dest.exists():
                os.symlink(img.resolve(), dest)
        for img in images[split:]:
            dest = val_dir / cat.name / img.name
            if not dest.exists():
                os.symlink(img.resolve(), dest)

    logger.info(f"Created train/val split in {sd}")


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
            logger.error("git clone failed: %s", err)
            return CloneResponse(success=False, path=str(sd), message="Failed to clone sample data repository")
        # Create train/val split for YOLO classification compatibility
        _create_train_val_split(sd)
        return CloneResponse(success=True, path=str(sd), message="Cloned successfully")
    except asyncio.TimeoutError:
        return CloneResponse(success=False, path=str(sd), message="Clone timed out (120s)")
    except Exception:
        logger.exception("Unexpected error cloning sample data")
        return CloneResponse(success=False, path=str(sd), message="Failed to clone sample data repository")
