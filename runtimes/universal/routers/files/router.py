"""Files router for file upload, list, get, and delete endpoints."""

import os

from fastapi import APIRouter, Form, HTTPException, UploadFile

from core.logging import UniversalRuntimeLogger
from services.error_handler import handle_endpoint_errors
from utils.file_handler import (
    delete_file,
    get_file,
    get_file_images,
    list_files,
    store_file,
)

logger = UniversalRuntimeLogger("universal-runtime.files")

router = APIRouter(tags=["files"])

# Maximum file upload size (100 MB by default, configurable via env var)
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 100 * 1024 * 1024))


# Chunk size for streaming uploads (64KB)
UPLOAD_CHUNK_SIZE = 64 * 1024


@router.post("/v1/files")
@handle_endpoint_errors("upload_file")
async def upload_file(
    file: UploadFile,
    convert_pdf: bool = Form(default=True),
    pdf_dpi: int = Form(default=150),
):
    """
    Upload a file for use with OCR, document extraction, or image generation.

    Uploaded files are stored temporarily (5 minutes TTL) and can be referenced
    by their file ID in subsequent API calls.

    For PDFs, pages are automatically converted to images for OCR/document processing.

    Args:
        file: The file to upload (images, PDFs supported, max 100MB)
        convert_pdf: If True, convert PDF pages to images (default: True)
        pdf_dpi: DPI for PDF to image conversion (default: 150)

    Returns:
        File metadata including ID for referencing in other endpoints

    Example:
        ```bash
        curl -X POST http://localhost:8000/v1/files \\
            -F "file=@document.pdf" \\
            -F "convert_pdf=true" \\
            -F "pdf_dpi=150"
        ```
    """
    # Stream file in chunks to enforce size limit without loading entire file
    chunks = []
    total_size = 0
    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
            )
        chunks.append(chunk)
    content = b"".join(chunks)
    stored = await store_file(
        content=content,
        filename=file.filename or "unknown",
        content_type=file.content_type,
        convert_pdf_to_images=convert_pdf,
        pdf_dpi=pdf_dpi,
    )

    return {
        "id": stored.id,
        "object": "file",
        "filename": stored.filename,
        "content_type": stored.content_type,
        "size": stored.size,
        "created_at": stored.created_at,
        "has_images": stored.page_images is not None
        or stored.filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif")
        ),
        "page_count": len(stored.page_images) if stored.page_images else None,
    }


@router.get("/v1/files")
async def get_uploaded_files():
    """
    List all uploaded files with their metadata.

    Returns:
        List of file metadata
    """
    return {"object": "list", "data": list_files()}


@router.get("/v1/files/{file_id}")
async def get_uploaded_file(file_id: str):
    """
    Get metadata for a specific uploaded file.

    Args:
        file_id: The file ID returned from upload

    Returns:
        File metadata
    """
    stored = get_file(file_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    return {
        "id": stored.id,
        "object": "file",
        "filename": stored.filename,
        "content_type": stored.content_type,
        "size": stored.size,
        "created_at": stored.created_at,
        "has_images": stored.page_images is not None,
        "page_count": len(stored.page_images) if stored.page_images else None,
    }


@router.get("/v1/files/{file_id}/images")
async def get_file_as_images(file_id: str):
    """
    Get base64-encoded images for a file.

    For PDFs, returns one image per page.
    For image files, returns the image itself.

    Args:
        file_id: The file ID returned from upload

    Returns:
        List of base64-encoded images
    """
    stored = get_file(file_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    images = get_file_images(file_id)
    if not images:
        raise HTTPException(
            status_code=400,
            detail=f"File {file_id} cannot be converted to images",
        )

    return {
        "object": "list",
        "file_id": file_id,
        "data": [{"index": i, "base64": img} for i, img in enumerate(images)],
    }


@router.delete("/v1/files/{file_id}")
async def delete_uploaded_file(file_id: str):
    """
    Delete an uploaded file.

    Args:
        file_id: The file ID to delete

    Returns:
        Deletion confirmation
    """
    if delete_file(file_id):
        return {"deleted": True, "id": file_id}
    raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
