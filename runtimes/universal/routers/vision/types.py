"""Pydantic models for vision endpoints."""

from pydantic import BaseModel, Field

# =============================================================================
# Zero-Shot Classification (CLIP)
# =============================================================================


class ZeroShotClassifyRequest(BaseModel):
    """Zero-shot image classification request."""

    image: str  # Base64-encoded image or file path
    labels: list[str]  # Labels to classify against
    model: str = "openai/clip-vit-base-patch32"  # CLIP model to use


class ZeroShotClassifyBatchRequest(BaseModel):
    """Zero-shot image classification batch request."""

    images: list[str]  # List of base64-encoded images or file paths
    labels: list[str]  # Labels to classify against
    model: str = "openai/clip-vit-base-patch32"  # CLIP model to use


# =============================================================================
# Few-Shot Classification
# =============================================================================


class FewShotTrainRequest(BaseModel):
    """Request to train a few-shot classifier."""

    classifier_id: str  # Unique ID for this classifier
    images: list[str]  # Base64-encoded images or file paths
    labels: list[str]  # Labels for each image (same length as images)
    model: str = "openai/clip-vit-base-patch32"  # CLIP model for embeddings
    epochs: int = 100  # Training epochs
    learning_rate: float = 0.001  # Learning rate


class FewShotRefineRequest(BaseModel):
    """Request to refine an existing few-shot classifier with more data."""

    classifier_id: str  # Classifier ID to refine
    images: list[str]  # Additional images
    labels: list[str]  # Labels for additional images
    model: str = "openai/clip-vit-base-patch32"  # CLIP model (must match original)
    epochs: int = 50  # Refinement epochs
    learning_rate: float = 0.0005  # Lower learning rate for refinement


class FewShotPredictRequest(BaseModel):
    """Request to classify an image with a trained few-shot classifier."""

    classifier_id: str  # Classifier ID to use
    image: str  # Base64-encoded image or file path
    model: str = "openai/clip-vit-base-patch32"  # CLIP model (must match training)


class FewShotPredictBatchRequest(BaseModel):
    """Request to classify multiple images with a trained few-shot classifier."""

    classifier_id: str  # Classifier ID to use
    images: list[str]  # Base64-encoded images or file paths
    model: str = "openai/clip-vit-base-patch32"  # CLIP model (must match training)


class FewShotLoadRequest(BaseModel):
    """Request to load a saved few-shot classifier."""

    classifier_id: str  # Classifier ID to load
    model: str = "openai/clip-vit-base-patch32"  # CLIP model (must match saved)


# =============================================================================
# Object Detection (YOLOS)
# =============================================================================


class ObjectDetectionRequest(BaseModel):
    """Object detection request."""

    image: str  # Base64-encoded image or file path
    threshold: float = Field(
        default=0.5, ge=0, le=1, description="Confidence threshold (0-1)"
    )
    labels: list[str] | None = None  # Optional filter to specific labels
    model: str = "hustvl/yolos-tiny"  # HuggingFace model name


class ObjectDetectionBatchRequest(BaseModel):
    """Batch object detection request."""

    images: list[str]  # List of base64-encoded images or file paths
    threshold: float = Field(
        default=0.5, ge=0, le=1, description="Confidence threshold (0-1)"
    )
    labels: list[str] | None = None
    model: str = "hustvl/yolos-tiny"


# =============================================================================
# Open-Vocabulary Detection (OWL-ViT)
# =============================================================================


class OpenVocabDetectTextRequest(BaseModel):
    """Open-vocabulary detection using text queries."""

    image: str  # Base64-encoded image or file path
    queries: list[str]  # Text queries describing what to find
    threshold: float = Field(default=0.1, ge=0.0, le=1.0)  # Confidence threshold [0,1]
    top_k: int | None = None  # Limit number of detections
    model: str = "google/owlvit-base-patch32"


class OpenVocabDetectTextBatchRequest(BaseModel):
    """Batch open-vocabulary detection using text queries."""

    images: list[str]  # List of base64-encoded images or file paths
    queries: list[str]  # Text queries (applied to all images)
    threshold: float = Field(default=0.1, ge=0.0, le=1.0)  # Confidence threshold [0,1]
    top_k: int | None = None
    model: str = "google/owlvit-base-patch32"


class OpenVocabDetectImageRequest(BaseModel):
    """Open-vocabulary detection using reference images."""

    image: str  # Target image to search in
    query_images: list[str]  # Reference images showing what to find
    threshold: float = Field(default=0.9, ge=0.0, le=1.0)  # Similarity threshold [0,1]
    top_k: int | None = None
    model: str = "google/owlvit-base-patch32"


# =============================================================================
# Background Removal (RMBG)
# =============================================================================


class BackgroundRemovalRequest(BaseModel):
    """Background removal request."""

    image: str  # Base64-encoded image or file path
    return_mask: bool = False  # Whether to also return the alpha mask
    model: str = "briaai/RMBG-1.4"  # HuggingFace model name


class BackgroundRemovalBatchRequest(BaseModel):
    """Batch background removal request."""

    images: list[str]  # List of base64-encoded images or file paths
    return_mask: bool = False
    model: str = "briaai/RMBG-1.4"
