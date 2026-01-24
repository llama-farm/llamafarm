from .adtk import router as adtk_router
from .catboost import router as catboost_router
from .datasets import router as datasets_router
from .drift import router as drift_router
from .event_logs import router as event_logs_router
from .examples import router as examples_router
from .explain import router as explain_router
from .health import router as health_router
from .ml import router as ml_router
from .models import router as models_router
from .nlp import router as nlp_router
from .projects import router as projects_router
from .rag import router as rag_router
from .system import disk_router, upgrades_router
from .timeseries import router as timeseries_router
from .vision import vision_router

__all__ = [
    "adtk_router",
    "catboost_router",
    "datasets_router",
    "drift_router",
    "event_logs_router",
    "examples_router",
    "explain_router",
    "health_router",
    "ml_router",
    "models_router",
    "nlp_router",
    "projects_router",
    "rag_router",
    "disk_router",
    "upgrades_router",
    "timeseries_router",
    "vision_router",
]
