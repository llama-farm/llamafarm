"""Vision router module."""

from fastapi import APIRouter

from .router import router as main_vision_router
from .review import router as review_router

# Main vision router includes both existing routes and review
# Note: main_vision_router already has prefix="/vision"
vision_router = main_vision_router
vision_router.include_router(review_router)

__all__ = ["vision_router"]
