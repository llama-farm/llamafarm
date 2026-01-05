"""
Pydantic models for Vision endpoints (OCR, Document extraction).
"""

from typing import Literal

from pydantic import BaseModel


class OCRRequest(BaseModel):
    """OCR request for text extraction from images."""

    model: str = "surya"  # Backend: surya, easyocr, paddleocr, tesseract
    images: list[str]  # Base64-encoded images (required)
    languages: list[str] | None = None  # Language codes (e.g., ['en', 'fr'])
    return_boxes: bool = False  # Return bounding boxes for detected text


class DocumentExtractRequest(BaseModel):
    """Document extraction request."""

    model: str  # HuggingFace model ID
    images: list[str]  # Base64-encoded document images (required)
    prompts: list[str] | None = None  # Optional prompts for each image
    task: Literal["extraction", "vqa", "classification"] = "extraction"
