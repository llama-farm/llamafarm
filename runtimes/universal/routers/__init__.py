"""Universal Runtime routers."""

from .anomaly import router as anomaly_router
from .audio import router as audio_router
from .classifier import router as classifier_router
from .files import router as files_router
from .health import router as health_router
from .nlp import router as nlp_router
from .vision import router as vision_router

__all__ = [
    "anomaly_router",
    "audio_router",
    "classifier_router",
    "files_router",
    "health_router",
    "nlp_router",
    "vision_router",
]
