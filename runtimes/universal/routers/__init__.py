"""Universal Runtime routers."""

from .adtk import router as adtk_router
from .anomaly import router as anomaly_router
from .audio import router as audio_router
from .catboost import router as catboost_router
from .classifier import router as classifier_router
from .drift import router as drift_router
from .explain import router as explain_router
from .files import router as files_router
from .health import router as health_router
from .nlp import router as nlp_router
from .timeseries import router as timeseries_router
from .vision import router as vision_router

__all__ = [
    "adtk_router",
    "anomaly_router",
    "audio_router",
    "catboost_router",
    "classifier_router",
    "drift_router",
    "explain_router",
    "files_router",
    "health_router",
    "nlp_router",
    "timeseries_router",
    "vision_router",
]
