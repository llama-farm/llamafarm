"""Vision router for OCR and document extraction endpoints."""

from .router import (
    router,
    set_document_loader,
    set_file_image_getter,
    set_ocr_loader,
)

__all__ = [
    "router",
    "set_ocr_loader",
    "set_document_loader",
    "set_file_image_getter",
]
