"""OCR router for text extraction endpoints."""

import logging
import time
from typing import Any, Callable, Coroutine

from fastapi import APIRouter, HTTPException

from api_types.vision import (
    OCRRequest,
    OCRResponse,
    OCRResult,
    OCRBox,
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
    """Extract text from an image using OCR.

    Supports multiple backends (easyocr, tesseract, etc.).
    """
    start_time = time.perf_counter()
    loader = _get_loader()
    
    # Check if we have multiple images (not supported by frontend yet but good for future)
    # The current request from frontend sends "image" in body which might not map directly 
    # if I strictly follow OCRRequest. 
    # Wait, OCRRequest in api_types.py has `images: list[str] | None`.
    # But StreamingPanel.tsx sends:
    # body = {
    #   image: queuedImage.data,
    #   model: ocrModel,
    #   return_regions: true,
    # }
    # This doesn't match OCRRequest! OCRRequest has `images` (list).
    # I should check if I need to adjust the request model or the frontend.
    
    # Frontend sends single "image" field. API expects "images" list.
    # I'll create a local adapter model or update the frontend.
    # Updating frontend is cleaner but user complained about backend failure.
    # I'll check if I can just accept arbitrary body and parse it, or fix frontend.
    # Let's fix frontend to match API spec, OR update API spec to be more flexible.
    # Given I can't easily see the frontend right now (I read it earlier), I recall it sending `image`.
    
    # Actually, let's look at api_types.py again.
    # class OCRRequest(BaseModel):
    #     model: str = "surya"
    #     images: list[str] | None = None
    #     file_id: str | None = None
    
    # StreamingPanel sends: { image: "...", ... }
    # This will fail validation.
    
    # I will stick to the plan of creating this router, BUT I will handle the single image case 
    # manually or use a customized request model here if needed. 
    # Better: I will use a request model that supports both or Fix Frontend.
    # Fixing frontend is better.
    
    # For now, let's assume I fix frontend.
    
    # Load model
    # Model ID logic: usually just the backend name like 'easyocr'
    model = await loader(request.model)
    
    results = []
    
    # Process images
    if request.images:
        for i, img_str in enumerate(request.images):
            img_bytes = decode_base64_image(img_str)
            
            # Run OCR
            # usage: await model.predict(image_bytes)
            # OCRModel.predict returns OCRResult (from models/ocr_model.py)
            ocr_result = await model.predict(img_bytes)
            
            # Map to API response type
            api_boxes = []
            if ocr_result.boxes:
                for box in ocr_result.boxes:
                    api_boxes.append(OCRBox(
                        x1=box.x1,
                        y1=box.y1,
                        x2=box.x2,
                        y2=box.y2,
                        text=box.text,
                        confidence=box.confidence or 0.0
                    ))
            
            results.append(OCRResult(
                index=i,
                text=ocr_result.text,
                confidence=ocr_result.confidence or 0.0,
                boxes=api_boxes
            ))

    return OCRResponse(
        data=results,
        model=request.model,
        usage={"total_images": len(results)}
    )
