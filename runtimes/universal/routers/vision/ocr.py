"""OCR router for text extraction endpoints."""

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, HTTPException

from api_types.vision import (
    OCRBox,
    OCRRequest,
    OCRResponse,
    OCRResult,
)
from services.error_handler import handle_endpoint_errors

from .utils import decode_base64_image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision-ocr"])

# Dependency injection for model loader
_load_ocr_model_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None


def set_ocr_loader(
    load_fn: Callable[..., Coroutine[Any, Any, Any]] | None,
) -> None:
    """Set the OCR model loader function."""
    global _load_ocr_model_fn
    _load_ocr_model_fn = load_fn


def _get_loader():
    """Get OCR loader or raise if not initialized."""
    if _load_ocr_model_fn is None:
        raise HTTPException(
            status_code=500,
            detail="OCR model loader not initialized.",
        )
    return _load_ocr_model_fn


@router.post("/v1/vision/ocr", response_model=OCRResponse)
@handle_endpoint_errors("vision_ocr")
async def ocr_image(request: OCRRequest) -> OCRResponse:
    """Extract text from images using OCR.

    Supports multiple backends (surya, easyocr, paddleocr, tesseract).
    """
    loader = _get_loader()
    model = await loader(request.model)

    results = []

    if request.images:
        for i, img_str in enumerate(request.images):
            img_bytes = decode_base64_image(img_str)
            ocr_result = await model.predict(img_bytes)

            api_boxes = []
            if request.return_boxes and ocr_result.boxes:
                for box in ocr_result.boxes:
                    api_boxes.append(OCRBox(
                        x1=box.x1,
                        y1=box.y1,
                        x2=box.x2,
                        y2=box.y2,
                        text=box.text,
                        confidence=box.confidence or 0.0,
                    ))

            results.append(OCRResult(
                index=i,
                text=ocr_result.text,
                confidence=ocr_result.confidence or 0.0,
                boxes=api_boxes if request.return_boxes else [],
            ))

    return OCRResponse(
        data=results,
        model=request.model,
        usage={"total_images": len(results)},
    )
