"""Vision routers for OCR, document extraction, detection, classification, and embedding.

This module provides endpoints for:
- OCR text extraction (/v1/ocr - legacy, /v1/vision/ocr - new)
- Document understanding (/v1/documents/extract)
- Object detection via YOLO (/v1/vision/detect)
- Image classification via CLIP (/v1/vision/classify)
- Image/text embeddings via CLIP (/v1/vision/embed)
- Streaming cascade pipeline (/v1/vision/stream/*)
- Training pipeline (/v1/vision/train/*)
- Model management (/v1/vision/models/*)
- Segmentation (/v1/vision/segment)
- Federation (/v1/vision/federation/*)
"""

from fastapi import APIRouter

from .classification import (
    router as classification_router,
)
from .classification import (
    set_classification_loader,
)

# Import other routers
from .detection import (
    router as detection_router,
)
from .detection import (
    set_detection_loader,
)
from .embedding import (
    router as embedding_router,
)
from .embedding import (
    set_embedding_loader,
)
from .federation import router as federation_router
from .models import router as models_router
from .models import set_model_export_loader, set_vision_models_dir

# Import new vision OCR router (/v1/vision/ocr)
from .ocr import (
    router as vision_ocr_router,
)
from .ocr import (
    set_ocr_loader as set_vision_ocr_loader,
)

# Import legacy OCR & document router (/v1/ocr, /v1/documents/extract)
from .router import (
    router as legacy_ocr_router,
)
from .router import (
    set_document_loader,
    set_file_image_getter,
)
from .router import (
    set_ocr_loader as set_legacy_ocr_loader,
)
from .segmentation import router as segmentation_router
from .streaming import router as streaming_router
from .training import router as training_router

# Create combined vision router
vision_router = APIRouter(tags=["vision"])

# Include all sub-routers
vision_router.include_router(legacy_ocr_router)    # /v1/ocr, /v1/documents/extract
vision_router.include_router(vision_ocr_router)     # /v1/vision/ocr
vision_router.include_router(detection_router)       # /v1/vision/detect
vision_router.include_router(classification_router)  # /v1/vision/classify
vision_router.include_router(embedding_router)       # /v1/vision/embed
vision_router.include_router(streaming_router)       # /v1/vision/stream/*
vision_router.include_router(training_router)        # /v1/vision/train/*
vision_router.include_router(models_router)          # /v1/vision/models/*
vision_router.include_router(segmentation_router)    # /v1/vision/segment
vision_router.include_router(federation_router)      # /v1/vision/federation/*

# For backward compatibility, also export 'router' as alias
router = vision_router


def set_ocr_loader(load_fn):
    """Set OCR loader on both legacy and new OCR routers."""
    set_legacy_ocr_loader(load_fn)
    set_vision_ocr_loader(load_fn)


__all__ = [
    # Combined router
    "vision_router",
    "router",
    # Loader setters (OCR/documents)
    "set_ocr_loader",
    "set_document_loader",
    "set_file_image_getter",
    # Loader setters (vision)
    "set_detection_loader",
    "set_classification_loader",
    "set_embedding_loader",
    # Vision models
    "set_vision_models_dir",
    "set_model_export_loader",
]
