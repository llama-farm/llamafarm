"""
OCR endpoints for text extraction from images.
"""

from fastapi import APIRouter, HTTPException

from core.logging import UniversalRuntimeLogger
from utils.file_handler import get_file_images

from .service import load_ocr
from .types import OCRRequest

router = APIRouter()
logger = UniversalRuntimeLogger("universal-runtime.ocr")


@router.post("/v1/ocr")
async def extract_text_from_images(request: OCRRequest):
    """
    OCR endpoint for text extraction from images.

    Supports multiple OCR backends:
    - surya: Best accuracy, transformer-based, layout-aware (recommended)
    - easyocr: Good multilingual support (80+ languages), widely used
    - paddleocr: Fast, optimized for production, excellent for Asian languages
    - tesseract: Classic OCR engine, CPU-only, widely deployed

    You can provide images either as:
    1. Base64-encoded strings in the `images` field
    2. A file ID from a previous upload via `file_id` field

    Example with base64:
    ```json
    {
        "model": "surya",
        "images": ["base64_encoded_image..."],
        "languages": ["en"],
        "return_boxes": false
    }
    ```

    Example with file_id (from /v1/files upload):
    ```json
    {
        "model": "surya",
        "file_id": "file_abc123_def456",
        "languages": ["en"]
    }
    ```
    """
    try:
        # Resolve images from file_id or direct base64
        images = request.images
        if request.file_id:
            images = get_file_images(request.file_id)
            if not images:
                raise HTTPException(
                    status_code=400,
                    detail=f"No images found for file_id: {request.file_id}",
                )
        elif not images:
            raise HTTPException(
                status_code=400,
                detail="Either 'images' or 'file_id' must be provided",
            )

        # Load OCR model
        model = await load_ocr(
            backend=request.model,
            languages=request.languages,
        )

        # Run OCR
        results = await model.recognize(
            images=images,
            languages=request.languages,
            return_boxes=request.return_boxes,
        )

        # Format response
        data = []
        for idx, result in enumerate(results):
            item = {
                "index": idx,
                "text": result.text,
                "confidence": result.confidence,
            }
            if request.return_boxes and result.boxes:
                item["boxes"] = [
                    {
                        "x1": box.x1,
                        "y1": box.y1,
                        "x2": box.x2,
                        "y2": box.y2,
                        "text": box.text,
                        "confidence": box.confidence,
                    }
                    for box in result.boxes
                ]
            data.append(item)

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "images_processed": len(images),
            },
        }

    except ImportError as e:
        logger.error(f"OCR backend not installed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"OCR backend '{request.model}' not installed. {str(e)}",
        ) from e
    except Exception as e:
        logger.error(f"Error in extract_text_from_images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
