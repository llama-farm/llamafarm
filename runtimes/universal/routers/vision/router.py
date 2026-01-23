"""Vision router for OCR and document extraction endpoints.

This router provides endpoints for:
- OCR text extraction from images
- Document understanding and extraction (forms, invoices, etc.)
"""

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, HTTPException

from api_types.vision import (
    DocumentExtractRequest,
    OCRRequest,
)
from services.error_handler import handle_endpoint_errors

logger = logging.getLogger(__name__)

# Router with vision prefix (some endpoints don't have a common prefix)
router = APIRouter(tags=["vision"])

# =============================================================================
# Dependency Injection
# =============================================================================

# Injected loader functions
_load_ocr_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None
_load_document_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None

# Injected file image getter (for resolving file_id to images)
_get_file_images_fn: Callable[[str], list[str] | None] | None = None


def set_ocr_loader(load_ocr_fn: Callable[..., Coroutine[Any, Any, Any]] | None):
    """Set the OCR model loading function.

    The function should have signature: async def load_ocr(backend, languages) -> OCRModel
    """
    global _load_ocr_fn
    _load_ocr_fn = load_ocr_fn


def set_document_loader(
    load_document_fn: Callable[..., Coroutine[Any, Any, Any]] | None,
):
    """Set the document model loading function.

    The function should have signature: async def load_document(model_id, task) -> DocumentModel
    """
    global _load_document_fn
    _load_document_fn = load_document_fn


def set_file_image_getter(get_file_images_fn: Callable[[str], list[str] | None] | None):
    """Set the file image getter function.

    The function should have signature: def get_file_images(file_id) -> list[str] | None
    """
    global _get_file_images_fn
    _get_file_images_fn = get_file_images_fn


def _get_ocr_loader():
    """Get the OCR loader, raising error if not initialized."""
    if _load_ocr_fn is None:
        raise HTTPException(
            status_code=500,
            detail="OCR loader not initialized. Server configuration error.",
        )
    return _load_ocr_fn


def _get_document_loader():
    """Get the document loader, raising error if not initialized."""
    if _load_document_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Document loader not initialized. Server configuration error.",
        )
    return _load_document_fn


def _get_file_images(file_id: str) -> list[str]:
    """Get images for a file ID, raising error if not found."""
    if _get_file_images_fn is None:
        raise HTTPException(
            status_code=500,
            detail="File image getter not initialized. Server configuration error.",
        )
    images = _get_file_images_fn(file_id)
    if not images:
        raise HTTPException(
            status_code=400,
            detail=f"No images found for file_id: {file_id}",
        )
    return images


# =============================================================================
# OCR Endpoint
# =============================================================================


@router.post("/v1/ocr")
@handle_endpoint_errors("extract_text_from_images")
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
    # Resolve images from file_id or direct base64
    images = request.images
    if request.file_id:
        images = _get_file_images(request.file_id)
    elif not images:
        raise HTTPException(
            status_code=400,
            detail="Either 'images' or 'file_id' must be provided",
        )

    # Load OCR model
    load_ocr = _get_ocr_loader()
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


# =============================================================================
# Document Extraction Endpoint
# =============================================================================


@router.post("/v1/documents/extract")
@handle_endpoint_errors("extract_from_documents")
async def extract_from_documents(request: DocumentExtractRequest):
    """
    Document understanding endpoint.

    Extract structured information from documents using vision-language models.
    Supports forms, invoices, receipts, and other document types.

    Model types:
    - Donut models: End-to-end, no OCR needed (naver-clova-ix/donut-*)
    - LayoutLM models: Uses OCR + layout features (microsoft/layoutlmv3-*)

    Tasks:
    - extraction: Extract key-value pairs from documents
    - vqa: Answer questions about document content
    - classification: Classify document types

    You can provide images either as:
    1. Base64-encoded strings in the `images` field
    2. A file ID from a previous upload via `file_id` field

    Example with base64:
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "images": ["base64_encoded_image..."],
        "task": "extraction"
    }
    ```

    Example with file_id (from /v1/files upload):
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
        "model": "microsoft/layoutlmv3-base-finetuned-docvqa",
        "file_id": "file_abc123_def456",
        "prompts": ["What is the total amount?"],
        "task": "vqa"
    }
    ```
    """
    # Resolve images from file_id or direct base64
    images = request.images
    if request.file_id:
        images = _get_file_images(request.file_id)
    elif not images:
        raise HTTPException(
            status_code=400,
            detail="Either 'images' or 'file_id' must be provided",
        )

    # Load document model
    load_document = _get_document_loader()
    model = await load_document(
        model_id=request.model,
        task=request.task,
    )

    # Extract from documents
    results = await model.extract(
        images=images,
        prompts=request.prompts,
    )

    # Format response
    data = []
    for idx, result in enumerate(results):
        item = {
            "index": idx,
            "confidence": result.confidence,
        }

        if result.text:
            item["text"] = result.text

        if result.fields:
            item["fields"] = [
                {
                    "key": f.key,
                    "value": f.value,
                    "confidence": f.confidence,
                    "bbox": f.bbox,
                }
                for f in result.fields
            ]

        if result.answer:
            item["answer"] = result.answer

        if result.classification:
            item["classification"] = result.classification
            item["classification_scores"] = result.classification_scores

        data.append(item)

    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "task": request.task,
        "usage": {
            "documents_processed": len(images),
        },
    }
