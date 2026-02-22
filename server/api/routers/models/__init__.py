from fastapi import APIRouter

from .preload import router as preload_router
from .services import router as services_router

router = APIRouter()
router.include_router(services_router)
router.include_router(preload_router)

__all__ = ["router"]
