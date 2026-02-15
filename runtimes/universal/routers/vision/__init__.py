"""Vision routers for OCR, detection, classification, streaming, training, models."""

from fastapi import APIRouter

from .classification import router as classification_router
from .classification import set_classification_loader
from .detection import router as detection_router
from .detection import set_detection_loader
from .models import router as models_router
from .models import set_model_export_loader, set_vision_models_dir
from .router import router as legacy_ocr_router
from .router import set_document_loader, set_file_image_getter, set_ocr_loader
from .streaming import router as streaming_router
from .streaming import set_streaming_detection_loader
from .training import router as training_router

# Combined router
router = APIRouter(tags=["vision"])
router.include_router(legacy_ocr_router)
router.include_router(detection_router)
router.include_router(classification_router)
router.include_router(streaming_router)
router.include_router(training_router)
router.include_router(models_router)

__all__ = [
    "router",
    "set_ocr_loader", "set_document_loader", "set_file_image_getter",
    "set_detection_loader", "set_classification_loader",
    "set_streaming_detection_loader",
    "set_vision_models_dir", "set_model_export_loader",
]
