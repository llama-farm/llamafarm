"""Vision routers for edge runtime — detection, classification, and streaming only.

Excludes: OCR, document extraction, training, evaluation, tracking, sample data, models.
"""

from fastapi import APIRouter

from .classification import router as classification_router
from .classification import set_classification_loader
from .detect_classify import router as detect_classify_router
from .detect_classify import set_detect_classify_loaders
from .detection import router as detection_router
from .detection import set_detection_loader
from .streaming import router as streaming_router
from .streaming import (
    set_streaming_detection_loader,
    start_session_cleanup,
    stop_session_cleanup,
)

# Combined router — edge subset only
router = APIRouter(tags=["vision"])
router.include_router(detection_router)
router.include_router(classification_router)
router.include_router(detect_classify_router)
router.include_router(streaming_router)

__all__ = [
    "router",
    "set_detection_loader",
    "set_classification_loader",
    "set_detect_classify_loaders",
    "set_streaming_detection_loader",
    "start_session_cleanup",
    "stop_session_cleanup",
]
