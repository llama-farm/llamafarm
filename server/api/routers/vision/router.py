"""
Vision Router - Endpoints for OCR and Document extraction.

Provides access to:
- OCR (text extraction from images/PDFs)
- Document Extraction (structured data from forms/invoices)

Images are passed directly as base64-encoded strings.
"""

import logging
from typing import Any

from fastapi import APIRouter
from server.services.universal_runtime_service import UniversalRuntimeService

from .types import DocumentExtractRequest, OCRRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["vision"])


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
        file_id=request.file_id,
        prompts=request.prompts,
        task=request.task,
    )
