"""
Vision Router - Endpoints for OCR and Document extraction.

Provides access to:
- OCR (text extraction from images/PDFs)
- Document Extraction (structured data from forms/invoices)

Supports two input methods:
1. Base64-encoded images in JSON body (for /ocr and /documents/extract)
2. File upload via multipart form (for /ocr/upload and /documents/extract/upload)
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from server.services.universal_runtime_service import UniversalRuntimeService

from .types import DocumentExtractRequest, OCRRequest

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

# Type aliases for file upload parameters (avoids B008 linting error)
FileUpload = Annotated[UploadFile, File(description="PDF or image file to process")]


# =============================================================================
# OCR Endpoints
# =============================================================================


@router.post("/ocr")
async def extract_text(request: OCRRequest) -> dict[str, Any]:
    """OCR endpoint for text extraction from images.

    Supports multiple OCR backends:
    - surya: Best accuracy, transformer-based, layout-aware (recommended)
    - easyocr: Good multilingual support (80+ languages), widely used
    - paddleocr: Fast, optimized for production, excellent for Asian languages
    - tesseract: Classic OCR engine, CPU-only, widely deployed

    Provide images as base64-encoded strings in the `images` field.

    Example request:
    ```json
    {
        "model": "surya",
        "images": ["data:image/png;base64,iVBORw0KGgo..."],
        "languages": ["en"]
    }
    ```

    Response:
    ```json
    {
        "object": "list",
        "data": [
            {
                "index": 0,
                "text": "Extracted text from image...",
                "boxes": []
            }
        ],
        "model": "surya",
        "total_count": 1
    }
    ```
    """
    return await UniversalRuntimeService.ocr(
        model=request.model,
        images=request.images,
        languages=request.languages,
        return_boxes=request.return_boxes,
    )


# =============================================================================
# Document Extraction Endpoints
# =============================================================================


@router.post("/documents/extract")
async def extract_from_documents(request: DocumentExtractRequest) -> dict[str, Any]:
    """Document understanding endpoint.

    Extract structured information from documents using vision-language models.
    Supports forms, invoices, receipts, and other document types.

    Model types:
    - Donut models: End-to-end, no OCR needed (naver-clova-ix/donut-*)
    - LayoutLM models: Uses OCR + layout features (microsoft/layoutlmv3-*)

    Tasks:
    - extraction: Extract key-value pairs from documents
    - vqa: Answer questions about document content
    - classification: Classify document types

    Example for extraction:
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "images": ["data:image/png;base64,iVBORw0KGgo..."],
        "task": "extraction"
    }
    ```

    Example for VQA (Visual Question Answering):
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-docvqa",
        "images": ["data:image/png;base64,iVBORw0KGgo..."],
        "prompts": ["What is the total amount?", "What is the date?"],
        "task": "vqa"
    }
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
    return await UniversalRuntimeService.extract_documents(
        model=request.model,
        images=request.images,
        prompts=request.prompts,
        task=request.task,
    )


# =============================================================================
# File Upload Endpoints
# =============================================================================


async def _validate_and_convert_file(file: UploadFile) -> list[str]:
    """Validate uploaded file and convert to base64 images.

    Args:
        file: Uploaded file (PDF or image)

    Returns:
        List of base64 data URIs

    Raises:
        HTTPException: If file type is not supported
    """
    from pathlib import Path

    from core.image_utils import (
        IMAGE_MIME_TYPES,
        image_bytes_to_base64,
        pdf_bytes_to_base64_images,
    )

    # Get file extension
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Read file content
    content = await file.read()

    # Convert based on file type
    if ext == ".pdf":
        return pdf_bytes_to_base64_images(content)
    else:
        # Image file - get MIME type and convert
        mime_type = IMAGE_MIME_TYPES.get(ext, "image/png")
        return [image_bytes_to_base64(content, mime_type)]


@router.post("/ocr/upload")
async def extract_text_upload(
    file: FileUpload,
    model: str = Form(
        default="surya", description="OCR backend: surya, easyocr, paddleocr, tesseract"
    ),
    languages: str = Form(
        default="en", description="Comma-separated language codes (e.g., 'en,fr')"
    ),
    return_boxes: bool = Form(
        default=False, description="Return bounding boxes for detected text"
    ),
) -> dict[str, Any]:
    """OCR endpoint with file upload.

    Upload a PDF or image file directly. PDFs are converted to images automatically.

    Supported formats: PDF, PNG, JPEG, GIF, WebP, BMP, TIFF

    Example using curl:
    ```bash
    curl -X POST http://localhost:8000/v1/vision/ocr/upload \\
      -F "file=@document.pdf" \\
      -F "model=surya" \\
      -F "languages=en"
    ```
    """
    # Convert file to base64 images
    images = await _validate_and_convert_file(file)

    # Parse languages
    lang_list = [lang.strip() for lang in languages.split(",") if lang.strip()]

    return await UniversalRuntimeService.ocr(
        model=model,
        images=images,
        languages=lang_list if lang_list else None,
        return_boxes=return_boxes,
    )


@router.post("/documents/extract/upload")
async def extract_from_documents_upload(
    file: FileUpload,
    model: str = Form(
        ...,
        description="HuggingFace model ID (e.g., naver-clova-ix/donut-base-finetuned-docvqa)",
    ),
    prompts: str = Form(default="", description="Comma-separated prompts for VQA task"),
    task: str = Form(
        default="extraction", description="Task: extraction, vqa, or classification"
    ),
) -> dict[str, Any]:
    """Document extraction endpoint with file upload.

    Upload a PDF or image file directly. PDFs are converted to images automatically.

    Supported formats: PDF, PNG, JPEG, GIF, WebP, BMP, TIFF

    Example using curl:
    ```bash
    curl -X POST http://localhost:8000/v1/vision/documents/extract/upload \\
      -F "file=@receipt.pdf" \\
      -F "model=naver-clova-ix/donut-base-finetuned-docvqa" \\
      -F "prompts=What is the total amount?,What is the date?" \\
      -F "task=vqa"
    ```
    """
    # Validate task
    valid_tasks = {"extraction", "vqa", "classification"}
    if task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task: {task}. Must be one of: {', '.join(valid_tasks)}",
        )

    # Convert file to base64 images
    images = await _validate_and_convert_file(file)

    # Parse prompts
    prompt_list = (
        [p.strip() for p in prompts.split(",") if p.strip()] if prompts else None
    )

    return await UniversalRuntimeService.extract_documents(
        model=model,
        images=images,
        prompts=prompt_list,
        task=task,  # type: ignore
    )
