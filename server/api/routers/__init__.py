from .projects import router as projects_router
from .datasets import router as datasets_router
from .inference import router as inference_router
from .health import router as health_router
from .rag import router as rag_router
from .system import upgrades_router

__all__ = [
    "projects_router",
    "datasets_router",
    "inference_router",
    "health_router",
    "rag_router",
    "upgrades_router",
]
