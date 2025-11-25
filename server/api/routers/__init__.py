from .projects import router as projects_router
from .datasets import router as datasets_router
from .health import router as health_router
from .rag import router as rag_router
from .system import upgrades_router
from .examples import router as examples_router
from .event_logs import router as event_logs_router
from .models import router as models_router

__all__ = [
    "projects_router",
    "datasets_router",
    "health_router",
    "rag_router",
    "upgrades_router",
    "examples_router",
    "event_logs_router",
    "models_router",
]
