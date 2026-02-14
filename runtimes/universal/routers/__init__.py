"""Universal Runtime routers."""

import importlib.util

from .anomaly import router as anomaly_router
from .audio import router as audio_router
from .classifier import router as classifier_router
from .explain import router as explain_router
from .files import router as files_router
from .health import router as health_router
from .nlp import router as nlp_router
from .vision import router as vision_router

__all__ = [
    "anomaly_router",
    "audio_router",
    "classifier_router",
    "explain_router",
    "files_router",
    "health_router",
    "nlp_router",
    "vision_router",
]

# Addon routers (conditionally available when addon packages are installed)
if importlib.util.find_spec("darts") is not None:
    from .timeseries import router as timeseries_router

    __all__.append("timeseries_router")

if importlib.util.find_spec("adtk") is not None:
    from .adtk import router as adtk_router

    __all__.append("adtk_router")

if importlib.util.find_spec("alibi_detect") is not None:
    from .drift import router as drift_router

    __all__.append("drift_router")

if importlib.util.find_spec("catboost") is not None:
    from .catboost import router as catboost_router

    __all__.append("catboost_router")
