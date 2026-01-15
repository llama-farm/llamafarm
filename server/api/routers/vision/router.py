"""
Vision Router - Endpoints for OCR and Document extraction.

Provides access to:
- OCR (text extraction from images/PDFs)
- Document Extraction (structured data from forms/invoices)

All endpoints accept multipart form data with either:
- file: Upload a PDF or image file directly
- images: Base64-encoded image data URIs (comma-separated or JSON array)
"""

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from server.services.universal_runtime_service import UniversalRuntimeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["vision"])


# Supported file extensions for upload
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}

# Type alias for optional file upload (avoids B008 linting error)
OptionalFileUpload = Annotated[
    UploadFile | None, File(description="PDF or image file to process")
]


async def _get_images_from_input(
    file: UploadFile | None,
    images: str | None,
) -> list[str]:
    """Extract base64 images from either file upload or images parameter.

    Args:
        file: Optional uploaded file (PDF or image)
        images: Optional base64 images (JSON array or comma-separated)

    Returns:
        List of base64 data URIs

    Raises:
        HTTPException: If neither file nor images provided, or file type unsupported
    """
    from pathlib import Path

    from core.image_utils import (
        IMAGE_MIME_TYPES,
        image_bytes_to_base64,
        pdf_bytes_to_base64_images,
    )

    # Case 1: File upload provided
    if file is not None and file.filename:
        filename = file.filename
        ext = Path(filename).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

        content = await file.read()

        if ext == ".pdf":
            return pdf_bytes_to_base64_images(content)
        else:
            mime_type = IMAGE_MIME_TYPES.get(ext, "image/png")
            return [image_bytes_to_base64(content, mime_type)]

    # Case 2: Base64 images provided
    if images:
        # Try to parse as JSON array first
        try:
            parsed = json.loads(images)
            if isinstance(parsed, list):
                # Validate all elements are strings (base64 data URIs)
                if not all(isinstance(item, str) for item in parsed):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid 'images' format: all array elements must be base64 data URI strings",
                    )
                if not parsed:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid 'images' format: array cannot be empty",
                    )
                return parsed
        except json.JSONDecodeError:
            pass

        # Fall back to comma-separated (but be careful with base64 which contains commas in data)
        # If it starts with "data:", treat as single image or split on "],["
        if images.startswith("data:"):
            # Single base64 image
            return [images]
        else:
            # Try splitting - but this is tricky with base64
            # Best to require JSON array format for multiple images
            return [images]

    # Neither provided
    raise HTTPException(
        status_code=400,
        detail="Either 'file' (upload) or 'images' (base64) must be provided",
    )


# =============================================================================
# OCR Endpoint
# =============================================================================


@router.post("/ocr")
async def extract_text(
    file: OptionalFileUpload = None,
    images: str | None = Form(
        default=None,
        description='Base64-encoded images as JSON array (e.g., ["data:image/png;base64,..."])',
    ),
    model: str = Form(
        default="surya",
        description="OCR backend: surya, easyocr, paddleocr, tesseract",
    ),
    languages: str = Form(
        default="en",
        description="Comma-separated language codes (e.g., 'en,fr')",
    ),
    return_boxes: bool = Form(
        default=False,
        description="Return bounding boxes for detected text",
    ),
    parse_by_page: bool = Form(
        default=False,
        description="If true, return separate results per page. If false (default), combine all text into single result.",
    ),
) -> dict[str, Any]:
    """OCR endpoint for text extraction from images.

    Accepts either a file upload OR base64-encoded images.

    Supports multiple OCR backends:
    - surya: Best accuracy, transformer-based, layout-aware (recommended)
    - easyocr: Good multilingual support (80+ languages), widely used
    - paddleocr: Fast, optimized for production, excellent for Asian languages
    - tesseract: Classic OCR engine, CPU-only, widely deployed

    Example with file upload (curl):
    ```bash
    curl -X POST http://localhost:8000/v1/vision/ocr \\
      -F "file=@document.pdf" \\
      -F "model=easyocr" \\
      -F "languages=en"
    ```

    Example with base64 images (curl):
    ```bash
    curl -X POST http://localhost:8000/v1/vision/ocr \\
      -F 'images=["data:image/png;base64,iVBORw0KGgo..."]' \\
      -F "model=surya" \\
      -F "languages=en"
    ```

    Response:
    ```json
    {
        "object": "list",
        "data": [
            {
                "index": 0,
                "text": "Extracted text from image...",
                "confidence": 0.95
            }
        ],
        "model": "surya",
        "usage": {"images_processed": 1}
    }
    ```
    """
    # Get images from file or base64 input
    image_list = await _get_images_from_input(file, images)

    # Parse languages
    lang_list = [lang.strip() for lang in languages.split(",") if lang.strip()]

    result = await UniversalRuntimeService.ocr(
        model=model,
        images=image_list,
        languages=lang_list if lang_list else None,
        return_boxes=return_boxes,
    )

    # If not parsing by page and we have multiple results, combine them
    if not parse_by_page and len(result.get("data", [])) > 1:
        data = result["data"]
        combined_text = "\n\n".join(item["text"] for item in data)
        avg_confidence = sum(item["confidence"] for item in data) / len(data)
        original_page_count = len(data)
        result["data"] = [
            {
                "index": 0,
                "text": combined_text,
                "confidence": avg_confidence,
            }
        ]
        result["usage"]["pages_combined"] = original_page_count

    return result


# =============================================================================
# Document Extraction Endpoint
# =============================================================================


@router.post("/documents/extract")
async def extract_from_documents(
    file: OptionalFileUpload = None,
    images: str | None = Form(
        default=None,
        description='Base64-encoded images as JSON array (e.g., ["data:image/png;base64,..."])',
    ),
    model: str = Form(
        ...,
        description="HuggingFace model ID (e.g., naver-clova-ix/donut-base-finetuned-docvqa)",
    ),
    prompts: str = Form(
        default="",
        description="Comma-separated prompts for VQA task",
    ),
    task: str = Form(
        default="extraction",
        description="Task: extraction, vqa, or classification",
    ),
) -> dict[str, Any]:
    """Document understanding endpoint.

    Accepts either a file upload OR base64-encoded images.

    Extract structured information from documents using vision-language models.
    Supports forms, invoices, receipts, and other document types.

    Model types:
    - Donut models: End-to-end, no OCR needed (naver-clova-ix/donut-*)
    - LayoutLM models: Uses OCR + layout features (microsoft/layoutlmv3-*)

    Tasks:
    - extraction: Extract key-value pairs from documents
    - vqa: Answer questions about document content
    - classification: Classify document types

    Example with file upload (curl):
    ```bash
    curl -X POST http://localhost:8000/v1/vision/documents/extract \\
      -F "file=@receipt.pdf" \\
      -F "model=naver-clova-ix/donut-base-finetuned-docvqa" \\
      -F "prompts=What is the total amount?" \\
      -F "task=vqa"
    ```

    Example with base64 images (curl):
    ```bash
    curl -X POST http://localhost:8000/v1/vision/documents/extract \\
      -F 'images=["data:image/png;base64,iVBORw0KGgo..."]' \\
      -F "model=naver-clova-ix/donut-base-finetuned-cord-v2" \\
      -F "task=extraction"
    ```

    Response:
    ```json
    {
        "object": "list",
        "data": [
            {
                "index": 0,
                "text": "...",
                "structured": {...}
            }
        ],
        "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "task": "extraction"
    }
    ```
    """
    # Validate task
    valid_tasks = {"extraction", "vqa", "classification"}
    if task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task: {task}. Must be one of: {', '.join(valid_tasks)}",
        )

    # Get images from file or base64 input
    image_list = await _get_images_from_input(file, images)

    # Parse prompts
    prompt_list = (
        [p.strip() for p in prompts.split(",") if p.strip()] if prompts else None
    )

    return await UniversalRuntimeService.extract_documents(
        model=model,
        images=image_list,
        prompts=prompt_list,
        task=task,  # type: ignore
    )
