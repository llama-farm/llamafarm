"""
Pydantic models for Vision endpoints.

Includes:
- OCR and Document extraction
- Object detection (standard and open-vocabulary)
- Image classification (zero-shot and few-shot)
- Background removal and segmentation
"""

from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# OCR & Document Extraction Types
# =============================================================================


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


# =============================================================================
# Object Detection Types
# =============================================================================


class ObjectDetectRequest(BaseModel):
    """Request to detect objects in an image using standard object detection."""

    image: str  # Base64-encoded image
    threshold: float = Field(default=0.5, ge=0, le=1)
    labels: list[str] | None = None  # Optional label filter
    model: str = "hustvl/yolos-tiny"


class ObjectDetectBatchRequest(BaseModel):
    """Request to detect objects in multiple images."""

    images: list[str]  # Base64-encoded images
    threshold: float = Field(default=0.5, ge=0, le=1)
    labels: list[str] | None = None
    model: str = "hustvl/yolos-tiny"


class OpenVocabDetectRequest(BaseModel):
    """Request for open-vocabulary object detection (detect arbitrary objects by text description)."""

    image: str  # Base64-encoded image
    labels: list[str]  # Text descriptions of objects to detect (e.g., ["a red car", "a person wearing a hat"])
    threshold: float = Field(default=0.1, ge=0, le=1)
    model: str = "google/owlvit-base-patch32"


class OpenVocabDetectBatchRequest(BaseModel):
    """Request for open-vocabulary detection on multiple images."""

    images: list[str]  # Base64-encoded images
    labels: list[str]  # Text descriptions to detect across all images
    threshold: float = Field(default=0.1, ge=0, le=1)
    model: str = "google/owlvit-base-patch32"


class OpenVocabDetectByImageRequest(BaseModel):
    """Request for open-vocabulary detection using reference images (one-shot detection)."""

    query_image: str  # Base64-encoded query image to search in
    reference_images: list[str]  # Base64-encoded reference images showing objects to find
    threshold: float = Field(default=0.1, ge=0, le=1)
    model: str = "google/owlvit-base-patch32"


# =============================================================================
# Image Classification Types
# =============================================================================


class ZeroShotClassifyRequest(BaseModel):
    """Request for zero-shot image classification (classify without training)."""

    image: str  # Base64-encoded image
    labels: list[str]  # Candidate class labels
    model: str = "openai/clip-vit-base-patch32"


class ZeroShotClassifyBatchRequest(BaseModel):
    """Request for zero-shot classification on multiple images."""

    images: list[str]  # Base64-encoded images
    labels: list[str]  # Candidate class labels (same for all images)
    model: str = "openai/clip-vit-base-patch32"


class FewShotClassifyFitRequest(BaseModel):
    """Request to train a few-shot image classifier."""

    classifier_id: str  # Unique identifier for this classifier
    training_data: list[dict[str, str]]  # [{"image": "<base64>", "label": "cat"}, ...]
    model: str = "openai/clip-vit-base-patch32"
    num_iterations: int = Field(default=20, ge=1, le=100)


class FewShotClassifyPredictRequest(BaseModel):
    """Request to classify images using a fitted few-shot classifier."""

    classifier_id: str  # Classifier ID from fit request
    image: str  # Base64-encoded image


class FewShotClassifyPredictBatchRequest(BaseModel):
    """Request to classify multiple images using a fitted few-shot classifier."""

    classifier_id: str
    images: list[str]  # Base64-encoded images


class FewShotClassifyRefineRequest(BaseModel):
    """Request to refine a few-shot classifier with additional training data."""

    classifier_id: str
    training_data: list[dict[str, str]]  # Additional labeled examples
    num_iterations: int = Field(default=10, ge=1, le=100)


# =============================================================================
# Segmentation & Background Removal Types
# =============================================================================


class BackgroundRemoveRequest(BaseModel):
    """Request to remove background from an image."""

    image: str  # Base64-encoded image
    model: str = "briaai/RMBG-1.4"


class SegmentRequest(BaseModel):
    """Request for image segmentation (SAM - Segment Anything Model)."""

    image: str  # Base64-encoded image
    model: str = "facebook/sam-vit-base"
    points: list[list[float]] | None = None  # Optional point prompts [[x, y], ...]
    boxes: list[list[float]] | None = None  # Optional box prompts [[x1, y1, x2, y2], ...]


class SegmentBatchRequest(BaseModel):
    """Request for segmentation on multiple images."""

    images: list[str]  # Base64-encoded images
    model: str = "facebook/sam-vit-base"
    points: list[list[list[float]]] | None = None  # Points per image
    boxes: list[list[list[float]]] | None = None  # Boxes per image
