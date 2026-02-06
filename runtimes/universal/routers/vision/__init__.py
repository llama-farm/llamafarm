"""Vision routers for OCR, document extraction, detection, classification, and embedding.

This module provides endpoints for:
- OCR text extraction (existing)
- Document understanding (existing)
- Object detection via YOLO (new)
- Image classification via CLIP (new)
- Image/text embeddings via CLIP (new)
"""

from fastapi import APIRouter

# Import existing router (OCR & documents)
from .router import (
    router as ocr_router,
    set_document_loader,
    set_file_image_getter,
    set_ocr_loader,
)

# Import new routers
from .detection import (
    router as detection_router,
    set_detection_loader,
)
from .classification import (
    router as classification_router,
    set_classification_loader,
)
from .embedding import (
    router as embedding_router,
    set_embedding_loader,
)
from .streaming import router as streaming_router
from .training import router as training_router

# Create combined vision router
vision_router = APIRouter(tags=["vision"])

# Include all sub-routers
# Note: OCR router has /v1/ocr and /v1/documents/extract prefixes in its routes
vision_router.include_router(ocr_router)
# Detection, classification, embedding, streaming have /v1/vision/* prefixes
vision_router.include_router(detection_router)
vision_router.include_router(classification_router)
vision_router.include_router(embedding_router)
vision_router.include_router(streaming_router)
vision_router.include_router(training_router)

# For backward compatibility, also export 'router' as alias
router = vision_router

__all__ = [
    # Combined router
    "vision_router",
    "router",
    # Loader setters (OCR/documents)
    "set_ocr_loader",
    "set_document_loader",
    "set_file_image_getter",
    # Loader setters (new)
    "set_detection_loader",
    "set_classification_loader",
    "set_embedding_loader",
]
