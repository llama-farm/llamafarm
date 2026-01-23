"""Vision types for OCR and document extraction endpoints."""

from typing import Literal

from pydantic import BaseModel

# =============================================================================
# OCR Types
# =============================================================================


class OCRRequest(BaseModel):
    """OCR request for text extraction from images.

    Supports multiple OCR backends:
    - surya: Best accuracy, transformer-based, layout-aware (recommended)
    - easyocr: Good multilingual support (80+ languages), widely used
    - paddleocr: Fast, optimized for production, excellent for Asian languages
    - tesseract: Classic OCR engine, CPU-only, widely deployed
    """

    model: str = "surya"  # Backend: surya, easyocr, paddleocr, tesseract
    images: list[str] | None = None  # Base64-encoded images
    file_id: str | None = None  # File ID from /v1/files upload
    languages: list[str] | None = None  # Language codes (e.g., ['en', 'fr'])
    return_boxes: bool = False  # Return bounding boxes for detected text


class OCRBox(BaseModel):
    """Bounding box for detected text."""

    x1: float
    y1: float
    x2: float
    y2: float
    text: str
    confidence: float


class OCRResult(BaseModel):
    """OCR result for a single image."""

    index: int
    text: str
    confidence: float
    boxes: list[OCRBox] | None = None


class OCRResponse(BaseModel):
    """OCR response."""

    object: Literal["list"] = "list"
    data: list[OCRResult]
    model: str
    usage: dict[str, int]


# =============================================================================
# Document Extraction Types
# =============================================================================


class DocumentExtractRequest(BaseModel):
    """Document extraction request.

    Model types:
    - Donut models: End-to-end, no OCR needed (naver-clova-ix/donut-*)
    - LayoutLM models: Uses OCR + layout features (microsoft/layoutlmv3-*)

    Tasks:
    - extraction: Extract key-value pairs from documents
    - vqa: Answer questions about document content
    - classification: Classify document types
    """

    model: str  # HuggingFace model ID
    images: list[str] | None = None  # Base64-encoded document images
    file_id: str | None = None  # File ID from /v1/files upload
    prompts: list[str] | None = None  # Optional prompts for each image (VQA)
    task: str = "extraction"  # extraction, vqa, classification


class DocumentField(BaseModel):
    """Extracted field from a document."""

    key: str
    value: str
    confidence: float
    bbox: list[float] | None = None


class DocumentResult(BaseModel):
    """Document extraction result for a single image."""

    index: int
    confidence: float
    text: str | None = None
    fields: list[DocumentField] | None = None
    answer: str | None = None  # For VQA task
    classification: str | None = None  # For classification task
    classification_scores: dict[str, float] | None = None


class DocumentResponse(BaseModel):
    """Document extraction response."""

    object: Literal["list"] = "list"
    data: list[DocumentResult]
    model: str
    task: str
    usage: dict[str, int]
