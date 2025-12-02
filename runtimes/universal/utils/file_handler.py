"""
Shared file handling utilities for Universal Runtime.

Provides:
- File upload with automatic base64 encoding
- Temporary file storage with TTL
- Support for images, PDFs, and other file types
- PDF to image conversion for OCR/document processing
"""

import asyncio
import base64
import hashlib
import logging
import mimetypes
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# File storage TTL (seconds) - files are cleaned up after this time
FILE_TTL = 300  # 5 minutes

# Supported file types
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
DOCUMENT_TYPES = {".pdf"}
ALL_SUPPORTED = IMAGE_TYPES | DOCUMENT_TYPES


@dataclass
class StoredFile:
    """A file stored in the temporary cache."""

    id: str
    filename: str
    content_type: str
    size: int
    base64_data: str  # Base64-encoded content
    created_at: float = field(default_factory=time.time)
    # For PDFs converted to images, store page images
    page_images: list[str] | None = None  # List of base64-encoded page images


# In-memory file storage
_file_cache: dict[str, StoredFile] = {}
_cleanup_task: asyncio.Task | None = None


def _generate_file_id(content: bytes, filename: str) -> str:
    """Generate a unique file ID based on content hash and UUID."""
    content_hash = hashlib.sha256(content[:1024]).hexdigest()[:8]
    unique_id = uuid.uuid4().hex[:8]
    return f"file_{content_hash}_{unique_id}"


async def store_file(
    content: bytes,
    filename: str,
    content_type: str | None = None,
    convert_pdf_to_images: bool = True,
    pdf_dpi: int = 150,
) -> StoredFile:
    """
    Store a file and return its metadata.

    Args:
        content: Raw file bytes
        filename: Original filename
        content_type: MIME type (auto-detected if not provided)
        convert_pdf_to_images: If True, convert PDF pages to images
        pdf_dpi: DPI for PDF to image conversion

    Returns:
        StoredFile with ID and base64 data
    """
    # Auto-detect content type
    if content_type is None:
        content_type, _ = mimetypes.guess_type(filename)
        content_type = content_type or "application/octet-stream"

    # Generate file ID
    file_id = _generate_file_id(content, filename)

    # Base64 encode
    base64_data = base64.b64encode(content).decode("utf-8")

    # Check if PDF and needs conversion
    page_images = None
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf" and convert_pdf_to_images:
        page_images = await _convert_pdf_to_images(content, dpi=pdf_dpi)
        logger.info(f"Converted PDF to {len(page_images)} page images")

    # Create stored file
    stored = StoredFile(
        id=file_id,
        filename=filename,
        content_type=content_type,
        size=len(content),
        base64_data=base64_data,
        page_images=page_images,
    )

    # Store in cache
    _file_cache[file_id] = stored

    # Ensure cleanup task is running
    _ensure_cleanup_task()

    logger.info(f"Stored file: {file_id} ({filename}, {len(content)} bytes)")
    return stored


async def _convert_pdf_to_images(pdf_content: bytes, dpi: int = 150) -> list[str]:
    """
    Convert PDF pages to base64-encoded PNG images.

    Args:
        pdf_content: Raw PDF bytes
        dpi: Resolution for rendering

    Returns:
        List of base64-encoded PNG images (one per page)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed, cannot convert PDF to images")
        return []

    images = []

    try:
        # Open PDF from bytes
        pdf = fitz.open(stream=pdf_content, filetype="pdf")

        for page_num in range(len(pdf)):
            page = pdf[page_num]

            # Render page to image
            # zoom = dpi / 72 (PDF default is 72 DPI)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PNG bytes
            png_bytes = pix.tobytes("png")

            # Base64 encode
            base64_image = base64.b64encode(png_bytes).decode("utf-8")
            images.append(base64_image)

        pdf.close()

    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")

    return images


def get_file(file_id: str) -> StoredFile | None:
    """
    Retrieve a stored file by ID.

    Args:
        file_id: The file ID returned from store_file

    Returns:
        StoredFile or None if not found/expired
    """
    stored = _file_cache.get(file_id)

    if stored is None:
        return None

    # Check if expired
    if time.time() - stored.created_at > FILE_TTL:
        # Use pop for atomic, race-condition-safe deletion
        _file_cache.pop(file_id, None)
        return None

    return stored


def get_file_images(file_id: str) -> list[str]:
    """
    Get images for a file (page images for PDFs, or the file itself for images).

    Args:
        file_id: The file ID

    Returns:
        List of base64-encoded images
    """
    stored = get_file(file_id)

    if stored is None:
        return []

    # If it's a PDF with converted page images, return those
    if stored.page_images:
        return stored.page_images

    # If it's an image file, return the base64 data
    suffix = Path(stored.filename).suffix.lower()
    if suffix in IMAGE_TYPES:
        return [stored.base64_data]

    return []


def delete_file(file_id: str) -> bool:
    """
    Delete a stored file.

    Args:
        file_id: The file ID

    Returns:
        True if deleted, False if not found
    """
    # Use pop for atomic, race-condition-safe deletion
    return _file_cache.pop(file_id, None) is not None


def list_files() -> list[dict]:
    """
    List all stored files with their metadata.

    Returns:
        List of file metadata dicts
    """
    now = time.time()
    result = []

    for file_id, stored in list(_file_cache.items()):
        # Check if expired
        if now - stored.created_at > FILE_TTL:
            # Use pop for atomic, race-condition-safe deletion
            _file_cache.pop(file_id, None)
            continue

        result.append(
            {
                "id": stored.id,
                "filename": stored.filename,
                "content_type": stored.content_type,
                "size": stored.size,
                "created_at": stored.created_at,
                "ttl_remaining": FILE_TTL - (now - stored.created_at),
                "has_page_images": stored.page_images is not None,
                "page_count": len(stored.page_images) if stored.page_images else None,
            }
        )

    return result


async def _cleanup_expired_files():
    """Background task to clean up expired files."""
    while True:
        await asyncio.sleep(60)  # Check every minute

        now = time.time()
        expired = [
            file_id
            for file_id, stored in _file_cache.items()
            if now - stored.created_at > FILE_TTL
        ]

        for file_id in expired:
            # Use pop for atomic, race-condition-safe deletion
            _file_cache.pop(file_id, None)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired files")


def _ensure_cleanup_task():
    """Ensure the cleanup background task is running."""
    global _cleanup_task

    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_expired_files())
