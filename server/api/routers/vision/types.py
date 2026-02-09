"""
Pydantic models for Vision endpoints.

Includes:
- OCR (existing)
- Document extraction (existing)
- Object detection (new)
- Image classification (new)
- Image segmentation (new)
- Streaming vision (new)
- Training (new)
- Model management (new)
- Review queue (new)

These types mirror the Universal Runtime types for consistency.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# OCR Types (existing)
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
# Common Vision Types (new)
# =============================================================================


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


class Point(BaseModel):
    """Point prompt for segmentation."""

    x: float
    y: float
    label: Literal[0, 1] = 1  # 0=background, 1=foreground


class Mask(BaseModel):
    """Segmentation mask result."""

    mask_base64: str
    box: BoundingBox
    confidence: float
    area: int


# =============================================================================
# Detection Types (new)
# =============================================================================


class Detection(BaseModel):
    """Single object detection result."""

    box: BoundingBox
    class_name: str
    class_id: int
    confidence: float


class DetectRequest(BaseModel):
    """Object detection request."""

    image: str = Field(..., description="Base64-encoded image")
    model: str = Field(default="yolov8n", description="Model ID")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    classes: list[str] | None = Field(default=None, description="Filter to specific classes")


class DetectResponse(BaseModel):
    """Object detection response."""

    detections: list[Detection]
    model: str
    inference_time_ms: float


# =============================================================================
# Classification Types (new)
# =============================================================================


class ImageClassifyRequest(BaseModel):
    """Image classification request (CLIP-based)."""

    image: str = Field(..., description="Base64-encoded image")
    model: str = Field(default="clip-vit-base", description="Model ID")
    classes: list[str] | None = Field(default=None, description="Classes for zero-shot")
    top_k: int = Field(default=5, ge=1, le=100)


class ImageClassifyResponse(BaseModel):
    """Image classification response."""

    class_name: str
    class_id: int
    confidence: float
    all_scores: dict[str, float]
    model: str
    inference_time_ms: float


# =============================================================================
# Segmentation Types (new)
# =============================================================================


class SegmentRequest(BaseModel):
    """Image segmentation request (SAM-based)."""

    image: str = Field(..., description="Base64-encoded image")
    model: str = Field(default="mobilesam", description="Model ID")
    points: list[Point] | None = None
    boxes: list[BoundingBox] | None = None
    multimask_output: bool = False


class SegmentResponse(BaseModel):
    """Image segmentation response."""

    masks: list[Mask]
    model: str
    inference_time_ms: float


# =============================================================================
# Embedding Types (new)
# =============================================================================


class ImageEmbedRequest(BaseModel):
    """Image/text embedding request."""

    model: str = Field(default="clip-vit-base", description="CLIP model ID")
    images: list[str] | None = Field(default=None, description="Base64-encoded images")
    texts: list[str] | None = Field(default=None, description="Text strings to embed")


class ImageEmbedResponse(BaseModel):
    """Image/text embedding response."""

    embeddings: list[list[float]]
    model: str
    dimensions: int
    inference_time_ms: float


# =============================================================================
# Streaming Types (new)
# =============================================================================


class StreamingConfig(BaseModel):
    """Configuration for streaming vision detection."""

    target_fps: float = Field(default=1.0, ge=0.1, le=30.0)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    escalation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    action_classes: list[str] | None = None
    cooldown_seconds: float = Field(default=5.0, ge=0.0)


class StreamStartRequest(BaseModel):
    """Start streaming session request."""

    model: str = Field(default="yolov8n")
    config: StreamingConfig = Field(default_factory=StreamingConfig)


class StreamStartResponse(BaseModel):
    """Start streaming session response."""

    session_id: str
    config: StreamingConfig


class StreamFrameRequest(BaseModel):
    """Process single frame request."""

    session_id: str
    image: str  # Base64-encoded


class StreamFrameResponse(BaseModel):
    """Process single frame response."""

    status: Literal["ok", "action", "review"]
    detections: list[Detection] | None = None
    confidence: float | None = None
    image_id: str | None = None


# =============================================================================
# Training Types (new)
# =============================================================================


class TrainingConfig(BaseModel):
    """Training configuration."""

    epochs: int = Field(default=10, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=0.001, ge=0.0)
    use_ewc: bool = True
    ewc_lambda: float = Field(default=0.4, ge=0.0)
    use_replay: bool = True
    replay_ratio: float = Field(default=0.3, ge=0.0, le=1.0)


class TrainRequest(BaseModel):
    """Training request."""

    model: str
    dataset: str
    task: Literal["detection", "classification", "segmentation"]
    config: TrainingConfig = Field(default_factory=TrainingConfig)
    base_model: str | None = None


class TrainResponse(BaseModel):
    """Training job response."""

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: float = 0.0
    metrics: dict | None = None


# =============================================================================
# Model Management Types (new)
# =============================================================================


class VisionModelInfo(BaseModel):
    """Information about a vision model."""

    model_id: str
    name: str
    task: Literal["detection", "classification", "segmentation", "embedding"]
    version: str
    size_mb: float
    loaded: bool
    device: str | None


class ModelExportRequest(BaseModel):
    """Model export request."""

    model_id: str
    format: Literal["onnx", "coreml", "tensorrt", "tflite", "openvino"]
    quantization: Literal["fp32", "fp16", "int8"] = "fp16"


class ModelExportResponse(BaseModel):
    """Model export response."""

    export_path: str
    format: str
    size_mb: float


# =============================================================================
# Review Queue Types (new)
# =============================================================================


class ReviewItem(BaseModel):
    """Item in the review queue."""

    image_id: str
    image_url: str
    thumbnail_url: str
    timestamp: datetime
    prediction: Detection | None
    confidence: float
    model: str
    source: str
    status: Literal["pending", "approved", "rejected", "corrected"]


class ReviewDecision(BaseModel):
    """Human review decision."""

    image_id: str
    decision: Literal["correct", "wrong", "adjusted"]
    corrections: list[Detection] | None = None


class ReviewListResponse(BaseModel):
    """List of review items response."""

    items: list[ReviewItem]
    total: int
    pending: int
