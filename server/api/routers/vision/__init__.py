"""Vision router module."""

from fastapi import APIRouter

from .pipeline import router as pipeline_router
from .review import router as review_router
from .router import router as main_vision_router

# Main vision router includes all sub-routers
# Note: main_vision_router already has prefix="/vision"
vision_router = main_vision_router
vision_router.include_router(review_router)
vision_router.include_router(pipeline_router)

__all__ = ["vision_router"]
