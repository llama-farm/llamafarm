"""
Pydantic models for Vision endpoints (OCR, Document extraction).
"""

from typing import Literal

from pydantic import BaseModel, model_validator


class OCRRequest(BaseModel):
    """OCR request for text extraction from images."""

    model: str = "surya"  # Backend: surya, easyocr, paddleocr, tesseract
    images: list[str] | None = None  # Base64-encoded images
    file_id: str | None = None  # File ID (deprecated, use images instead)
    languages: list[str] | None = None  # Language codes (e.g., ['en', 'fr'])
    return_boxes: bool = False  # Return bounding boxes for detected text

    @model_validator(mode="after")
    def check_input_provided(self) -> "OCRRequest":
        """Ensure at least one input source is provided."""
        if not self.images and not self.file_id:
            raise ValueError("Either 'images' or 'file_id' must be provided")
        return self


class DocumentExtractRequest(BaseModel):
    """Document extraction request."""

    model: str  # HuggingFace model ID
    images: list[str] | None = None  # Base64-encoded document images
    file_id: str | None = None  # File ID (deprecated, use images instead)
    prompts: list[str] | None = None  # Optional prompts for each image
    task: Literal["extraction", "vqa", "classification"] = "extraction"

    @model_validator(mode="after")
    def check_input_provided(self) -> "DocumentExtractRequest":
        """Ensure at least one input source is provided."""
        if not self.images and not self.file_id:
            raise ValueError("Either 'images' or 'file_id' must be provided")
        return self
