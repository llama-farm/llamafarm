"""
Image utilities for file-to-base64 conversion.

Provides helper functions for:
- Converting PDFs to base64-encoded images
- Encoding images (PNG, JPEG, etc.) to base64 data URIs
- Detecting file types and applying appropriate conversions

Usage:
    from core.image_utils import file_to_base64_images, image_to_base64

    # Convert a PDF or image file to base64 data URIs
    images = file_to_base64_images("/path/to/document.pdf", dpi=150)

    # Encode a single image file
    data_uri = image_to_base64("/path/to/image.png")
"""

import base64
import logging
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)

# Supported image MIME types
IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

# File extensions that are PDFs
PDF_EXTENSIONS = {".pdf"}


def get_mime_type(file_path: str | Path) -> str | None:
    """Get MIME type for a file based on extension.

    Args:
        file_path: Path to the file

    Returns:
        MIME type string or None if not a supported image type
    """
    ext = Path(file_path).suffix.lower()
    return IMAGE_MIME_TYPES.get(ext)


def is_pdf(file_path: str | Path) -> bool:
    """Check if a file is a PDF based on extension.

    Args:
        file_path: Path to the file

    Returns:
        True if the file is a PDF
    """
    ext = Path(file_path).suffix.lower()
    return ext in PDF_EXTENSIONS


def is_supported_image(file_path: str | Path) -> bool:
    """Check if a file is a supported image type.

    Args:
        file_path: Path to the file

    Returns:
        True if the file is a supported image type
    """
    return get_mime_type(file_path) is not None


def bytes_to_base64_data_uri(data: bytes, mime_type: str) -> str:
    """Convert raw bytes to a base64 data URI.

    Args:
        data: Raw image bytes
        mime_type: MIME type (e.g., "image/png")

    Returns:
        Data URI string: "data:{mime_type};base64,{encoded_data}"
    """
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def image_to_base64(
    file_path: str | Path,
    mime_type: str | None = None,
) -> str:
    """Convert an image file to a base64 data URI.

    Args:
        file_path: Path to the image file
        mime_type: Optional MIME type override. If not provided, detected from extension.

    Returns:
        Data URI string: "data:{mime_type};base64,{encoded_data}"

    Raises:
        ValueError: If the file type is not supported
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if mime_type is None:
        mime_type = get_mime_type(path)
        if mime_type is None:
            raise ValueError(
                f"Unsupported image type: {path.suffix}. "
                f"Supported types: {', '.join(IMAGE_MIME_TYPES.keys())}"
            )

    with open(path, "rb") as f:
        data = f.read()

    return bytes_to_base64_data_uri(data, mime_type)


def image_bytes_to_base64(
    data: bytes | BinaryIO,
    mime_type: str = "image/png",
) -> str:
    """Convert image bytes or file-like object to a base64 data URI.

    Args:
        data: Raw image bytes or file-like object
        mime_type: MIME type of the image (default: image/png)

    Returns:
        Data URI string: "data:{mime_type};base64,{encoded_data}"
    """
    if hasattr(data, "read"):
        data = data.read()
    return bytes_to_base64_data_uri(data, mime_type)


def pdf_to_base64_images(
    file_path: str | Path,
    dpi: int = 150,
    output_format: str = "png",
) -> list[str]:
    """Convert a PDF file to a list of base64-encoded images (one per page).

    Uses PyMuPDF (fitz) for PDF rendering.

    Args:
        file_path: Path to the PDF file
        dpi: Resolution for rendering (default: 150 DPI)
        output_format: Output image format, "png" or "jpeg" (default: png)

    Returns:
        List of data URI strings, one per page

    Raises:
        FileNotFoundError: If the file doesn't exist
        ImportError: If PyMuPDF is not installed
        ValueError: If the file is not a valid PDF
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required for PDF conversion. "
            "Install it with: pip install pymupdf"
        ) from e

    # Determine output format and MIME type
    output_format = output_format.lower()
    if output_format == "jpeg" or output_format == "jpg":
        output_format = "jpeg"
        mime_type = "image/jpeg"
    else:
        output_format = "png"
        mime_type = "image/png"

    # Calculate zoom factor from DPI (72 is PDF's base DPI)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    images: list[str] = []

    try:
        doc = fitz.open(path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}") from e

    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=matrix)

            # Convert to bytes
            if output_format == "jpeg":
                img_bytes = pix.tobytes("jpeg")
            else:
                img_bytes = pix.tobytes("png")

            # Convert to base64 data URI
            data_uri = bytes_to_base64_data_uri(img_bytes, mime_type)
            images.append(data_uri)

            logger.debug(
                f"Converted PDF page {page_num + 1}/{len(doc)} to {output_format}"
            )
    finally:
        doc.close()

    logger.info(f"Converted PDF with {len(images)} pages to base64 images")
    return images


def pdf_bytes_to_base64_images(
    data: bytes | BinaryIO,
    dpi: int = 150,
    output_format: str = "png",
) -> list[str]:
    """Convert PDF bytes to a list of base64-encoded images.

    Args:
        data: PDF file bytes or file-like object
        dpi: Resolution for rendering (default: 150 DPI)
        output_format: Output image format, "png" or "jpeg" (default: png)

    Returns:
        List of data URI strings, one per page

    Raises:
        ImportError: If PyMuPDF is not installed
        ValueError: If the data is not a valid PDF
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required for PDF conversion. "
            "Install it with: pip install pymupdf"
        ) from e

    if hasattr(data, "read"):
        data = data.read()

    # Determine output format and MIME type
    output_format = output_format.lower()
    if output_format == "jpeg" or output_format == "jpg":
        output_format = "jpeg"
        mime_type = "image/jpeg"
    else:
        output_format = "png"
        mime_type = "image/png"

    # Calculate zoom factor from DPI
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    images: list[str] = []

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Failed to open PDF from bytes: {e}") from e

    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=matrix)

            if output_format == "jpeg":
                img_bytes = pix.tobytes("jpeg")
            else:
                img_bytes = pix.tobytes("png")

            data_uri = bytes_to_base64_data_uri(img_bytes, mime_type)
            images.append(data_uri)
    finally:
        doc.close()

    return images


def file_to_base64_images(
    file_path: str | Path,
    dpi: int = 150,
    output_format: str = "png",
) -> list[str]:
    """Convert any supported file (PDF or image) to base64-encoded images.

    This is the main entry point for file conversion. It automatically
    detects the file type and applies the appropriate conversion.

    Args:
        file_path: Path to the file (PDF or image)
        dpi: Resolution for PDF rendering (default: 150 DPI, ignored for images)
        output_format: Output format for PDFs, "png" or "jpeg" (default: png)

    Returns:
        List of data URI strings. For images, returns a single-element list.
        For PDFs, returns one element per page.

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file type is not supported

    Example:
        >>> images = file_to_base64_images("document.pdf", dpi=150)
        >>> print(f"Converted {len(images)} pages")

        >>> images = file_to_base64_images("photo.jpg")
        >>> print(images[0][:50])  # "data:image/jpeg;base64,..."
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if is_pdf(path):
        return pdf_to_base64_images(path, dpi=dpi, output_format=output_format)
    elif is_supported_image(path):
        return [image_to_base64(path)]
    else:
        supported = list(IMAGE_MIME_TYPES.keys()) + list(PDF_EXTENSIONS)
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported types: {', '.join(supported)}"
        )
