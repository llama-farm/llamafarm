"""
Vision Router - Endpoints for OCR and Document extraction.

Provides access to:
- OCR (text extraction from images/PDFs)
- Document Extraction (structured data from forms/invoices)
- File management for vision tasks
"""

import logging
from typing import Any

from fastapi import APIRouter, Form, UploadFile
from server.services.universal_runtime_service import UniversalRuntimeService

from .types import DocumentExtractRequest, OCRRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["vision"])


# =============================================================================
# File Management
# =============================================================================


@router.post("/files")
async def upload_file(
    file: UploadFile,
    convert_pdf: bool = Form(default=True),
    pdf_dpi: int = Form(default=150),
) -> dict[str, Any]:
    """Upload a file for use with OCR or document extraction.

    Uploaded files are stored temporarily (5 minutes TTL) and can be referenced
    by their file ID in subsequent API calls.

    For PDFs, pages are automatically converted to images for OCR/document processing.

    Args:
        file: The file to upload (images, PDFs supported, max 100MB)
        convert_pdf: If True, convert PDF pages to images (default: True)
        pdf_dpi: DPI for PDF to image conversion (default: 150)

    Returns:
        File metadata including ID for referencing in other endpoints
    """
    return await UniversalRuntimeService.upload_file(
        file=file,
        convert_pdf=convert_pdf,
        pdf_dpi=pdf_dpi,
    )


@router.get("/files")
async def list_files() -> dict[str, Any]:
    """List all uploaded files with their metadata."""
    return await UniversalRuntimeService.list_files()


@router.get("/files/{file_id}")
async def get_file(file_id: str) -> dict[str, Any]:
    """Get metadata for a specific uploaded file."""
    return await UniversalRuntimeService.get_file(file_id)


@router.get("/files/{file_id}/images")
async def get_file_images(file_id: str) -> dict[str, Any]:
    """Get base64-encoded images for a file.

    For PDFs, returns one image per page.
    For image files, returns the image itself.
    """
    return await UniversalRuntimeService.get_file_images(file_id)


@router.delete("/files/{file_id}")
async def delete_file(file_id: str) -> dict[str, Any]:
    """Delete an uploaded file."""
    return await UniversalRuntimeService.delete_file(file_id)


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

    You can provide images either as:
    1. Base64-encoded strings in the `images` field
    2. A file ID from a previous upload via `file_id` field

    Example with base64 images:
    ```json
    {
        "model": "surya",
        "images": ["data:image/png;base64,..."],
        "languages": ["en"]
    }
    ```

    Example with file_id (from /v1/vision/files upload):
    ```json
    {
        "model": "surya",
        "file_id": "file_abc123_def456",
        "languages": ["en"]
    }
    ```
    """
    return await UniversalRuntimeService.ocr(
        model=request.model,
        images=request.images,
        file_id=request.file_id,
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

    Example with base64 images:
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "images": ["data:image/png;base64,..."],
        "task": "extraction"
    }
    ```

    Example with file_id:
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "file_id": "file_abc123_def456",
        "task": "extraction"
    }
    ```

    For VQA, include prompts:
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-docvqa",
        "file_id": "file_abc123_def456",
        "prompts": ["What is the total amount?"],
        "task": "vqa"
    }
    ```
    """
    return await UniversalRuntimeService.extract_documents(
        model=request.model,
        images=request.images,
        file_id=request.file_id,
        prompts=request.prompts,
        task=request.task,
    )
