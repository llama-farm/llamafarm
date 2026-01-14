"""
Request/response types for OCR endpoints.
"""

from pydantic import BaseModel


class OCRRequest(BaseModel):
    """OCR request for text extraction from images."""

    model: str = "surya"  # Backend: surya, easyocr, paddleocr, tesseract
    images: list[str] | None = None  # Base64-encoded images
    file_id: str | None = None  # File ID from /v1/files upload
    languages: list[str] | None = None  # Language codes (e.g., ['en', 'fr'])
    return_boxes: bool = False  # Return bounding boxes for detected text
