"""Vision types for OCR, document extraction, detection, classification, and segmentation.

This module contains types for:
- OCR text extraction (existing)
- Document understanding (existing)
- Object detection (NEW)
- Image classification (NEW)
- Image segmentation (NEW)
- Streaming vision (NEW)
- Training pipeline (NEW)
- Model management (NEW)
- Review queue (NEW)
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

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


# =============================================================================
# Common Vision Types (shared across detection, classification, segmentation)
# =============================================================================


class BoundingBox(BaseModel):
    """Bounding box coordinates (normalized or pixel)."""

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

    mask_base64: str  # Base64-encoded binary mask (PNG)
    box: BoundingBox
    confidence: float
    area: int


# =============================================================================
# Object Detection Types
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
    classes: list[str] | None = Field(
        default=None, description="Filter to specific classes"
    )


class DetectResponse(BaseModel):
    """Object detection response."""

    detections: list[Detection]
    model: str
    inference_time_ms: float


# =============================================================================
# Image Classification Types (distinct from document classification)
# =============================================================================


class ImageClassifyRequest(BaseModel):
    """Image classification request (CLIP-based)."""

    image: str = Field(..., description="Base64-encoded image")
    model: str = Field(default="clip-vit-base", description="Model ID")
    classes: list[str] | None = Field(
        default=None, description="Classes for zero-shot classification"
    )
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
# Image Segmentation Types
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
# Image Embedding Types (for RAG and similarity search)
# =============================================================================


class ImageEmbedRequest(BaseModel):
    """Image/text embedding request (CLIP-based)."""

    model: str = Field(default="clip-vit-base", description="CLIP model ID")
    images: list[str] | None = Field(
        default=None, description="Base64-encoded images"
    )
    texts: list[str] | None = Field(
        default=None, description="Text strings to embed"
    )


class ImageEmbedResponse(BaseModel):
    """Image/text embedding response."""

    embeddings: list[list[float]]
    model: str
    dimensions: int
    inference_time_ms: float


# =============================================================================
# Streaming Vision Types
# =============================================================================


class ModelOpinionResponse(BaseModel):
    """What one model thought about a detection (API response)."""

    model_id: str
    node_id: str = "local"
    class_name: str = ""
    confidence: float = 0.0
    bbox: BoundingBox | None = None
    mask_polygon: list[list[float]] | None = None
    inference_time_ms: float = 0.0


class CascadeConfig(BaseModel):
    """Configuration for model cascade behavior.

    Supports both the simple secondary_model_id (backward compatible)
    and the full cascade_chain for multi-hop escalation.
    """

    secondary_model_id: str | None = Field(
        default=None,
        description="Fallback model for uncertain detections (e.g., yolov8m, yolov8l)",
    )
    cascade_chain: list[str] | None = Field(
        default=None,
        description="Ordered list of model IDs for multi-hop cascade. Overrides secondary_model_id if set.",
    )
    feedback_to_primary: bool = Field(
        default=True,
        description="Auto-add successful secondary results to replay buffer for training",
    )
    save_uncertain_images: bool = Field(
        default=True,
        description="Save images that fail both models to review queue",
    )
    segmentation_model_id: str | None = Field(
        default=None,
        description="Run segmentation on uncertain bboxes before escalating",
    )
    classification_model_id: str | None = Field(
        default=None,
        description="Run CLIP classification on uncertain crops before escalating",
    )
    enrich_on_escalation: bool = Field(
        default=True,
        description="Attach seg masks and classification scores when escalating",
    )
    max_hops: int = Field(
        default=3,
        description="Circuit breaker: max cascade hops before sending to review",
    )


class StreamingConfig(BaseModel):
    """Configuration for streaming vision detection with cascade."""

    target_fps: float = Field(default=1.0, ge=0.1, le=30.0)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    escalation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    action_classes: list[str] | None = None
    cooldown_seconds: float = Field(default=5.0, ge=0.0)
    cascade: CascadeConfig | None = Field(
        default=None,
        description="Cascade configuration for automatic escalation to fallback model",
    )


class StreamStartRequest(BaseModel):
    """Start streaming session request."""

    model: str = Field(default="yolov8n", description="Primary (fast) model")
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

    status: Literal["ok", "action", "review", "escalated"]
    detections: list[Detection] | None = None
    confidence: float | None = None
    image_id: str | None = None  # For review queue
    suppressed: bool = False  # True if action was suppressed due to cooldown
    escalated_to: str | None = None  # Model that handled escalation (if any)
    added_to_replay: bool = False  # True if result was added to replay buffer
    hop_count: int = 0  # How many models saw this frame
    cascade_resolved_by: str | None = None  # Which model resolved it (if cascaded)
    opinions: list[ModelOpinionResponse] | None = None  # All model opinions


class StreamStopRequest(BaseModel):
    """Stop streaming session request."""

    session_id: str


class StreamStopResponse(BaseModel):
    """Stop streaming session response."""

    session_id: str
    frames_processed: int
    actions_triggered: int
    duration_seconds: float


# =============================================================================
# Training Types
# =============================================================================


class TrainingConfig(BaseModel):
    """Training configuration."""

    epochs: int = Field(default=10, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=0.001, ge=0.0)
    # Continual learning
    use_ewc: bool = True
    ewc_lambda: float = Field(default=0.4, ge=0.0)
    use_replay: bool = True
    replay_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    # Validation
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5)


class TrainRequest(BaseModel):
    """Training request."""

    model: str
    dataset: str  # Dataset path or ID
    task: Literal["detection", "classification", "segmentation"]
    config: TrainingConfig = Field(default_factory=TrainingConfig)
    base_model: str | None = None  # For fine-tuning


class TrainResponse(BaseModel):
    """Training job response."""

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: float = 0.0
    metrics: dict | None = None
    error: str | None = None


class TrainStatusRequest(BaseModel):
    """Get training job status request."""

    job_id: str


class TrainStatusResponse(BaseModel):
    """Training job status response."""

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: float
    current_epoch: int | None = None
    total_epochs: int | None = None
    metrics: dict | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


# =============================================================================
# Model Management Types
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


class ModelLoadRequest(BaseModel):
    """Load model into memory request."""

    model_id: str
    device: str = "auto"  # auto, cpu, cuda, mps


class ModelLoadResponse(BaseModel):
    """Load model response."""

    model_id: str
    loaded: bool
    device: str
    load_time_ms: float


class ModelUnloadRequest(BaseModel):
    """Unload model from memory request."""

    model_id: str


class ModelUnloadResponse(BaseModel):
    """Unload model response."""

    model_id: str
    unloaded: bool


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
    export_time_seconds: float


class ModelImportRequest(BaseModel):
    """Model import request."""

    path: str
    name: str
    task: Literal["detection", "classification", "segmentation"]
    metadata: dict | None = None


class ModelImportResponse(BaseModel):
    """Model import response."""

    model_id: str
    name: str
    task: str
    imported: bool


class VisionModelsListResponse(BaseModel):
    """List of vision models response."""

    models: list[VisionModelInfo]
    total: int


class VisionBackendInfo(BaseModel):
    """Information about a vision backend."""

    name: str
    version: str
    supported_tasks: list[str]
    device: str
    available: bool


class VisionBackendsResponse(BaseModel):
    """List of vision backends response."""

    backends: list[VisionBackendInfo]


# =============================================================================
# Review Queue Types (for human-in-the-loop)
# =============================================================================


class ReviewItem(BaseModel):
    """Item in the review queue."""

    image_id: str
    image_url: str  # URL to fetch image
    thumbnail_url: str
    timestamp: datetime
    prediction: Detection | None
    confidence: float
    model: str
    source: str  # e.g., "stream:camera1", "upload:batch123"
    status: Literal["pending", "approved", "rejected", "corrected"]
    all_opinions: list[ModelOpinionResponse] | None = None  # Every model's take


class ReviewDecision(BaseModel):
    """Human or model review decision."""

    image_id: str
    decision: Literal["correct", "wrong", "adjusted"]
    corrections: list[Detection] | None = None  # If adjusted
    reviewer_type: Literal["human", "model"] = "human"
    reviewer_model_id: str | None = None  # If reviewed by a model (audit)
    reviewer_confidence: float | None = None  # Model's confidence in the review


class ReviewDecisionResponse(BaseModel):
    """Review decision response."""

    image_id: str
    decision: str
    processed: bool
    next_image_id: str | None = None


class ReviewListRequest(BaseModel):
    """List review queue request."""

    status: Literal["pending", "approved", "rejected", "corrected", "all"] = "pending"
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    source: str | None = None  # Filter by source


class ReviewListResponse(BaseModel):
    """List of review items response."""

    items: list[ReviewItem]
    total: int
    pending: int


class ReviewBatchRequest(BaseModel):
    """Batch review request."""

    image_ids: list[str]
    decision: Literal["correct", "wrong"]


class ReviewBatchResponse(BaseModel):
    """Batch review response."""

    processed: int
    failed: int
    errors: list[str] | None = None


# =============================================================================
# Correction Feedback Types (for auto-learning)
# =============================================================================


class CorrectionRequest(BaseModel):
    """Submit a correction for a detection."""

    image_id: str = Field(..., description="Image ID from review queue or detection")
    corrected_class: str = Field(..., description="Correct class name")
    box: BoundingBox | None = Field(default=None, description="Corrected bounding box")
    original_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    session_id: str | None = Field(default=None, description="Session that produced the detection")


class CorrectionResponse(BaseModel):
    """Correction submission response."""

    image_id: str
    added_to_replay: bool
    replay_buffer_size: int
    training_triggered: bool = False


# =============================================================================
# Replay Buffer Types
# =============================================================================


class ReplayBufferStats(BaseModel):
    """Replay buffer statistics."""

    size: int
    max_size: int
    by_source: dict[str, int]
    avg_priority: float


class ReplayBufferResponse(BaseModel):
    """Replay buffer status response."""

    stats: ReplayBufferStats
    auto_train_threshold: int
    training_eligible: bool


# =============================================================================
# Auto-Training Types
# =============================================================================


class AutoTrainConfig(BaseModel):
    """Auto-training configuration."""

    enabled: bool = True
    threshold: int = Field(default=50, description="Samples before auto-training")
    min_interval_hours: float = Field(default=6.0, description="Min hours between training")
    epochs: int = Field(default=5, ge=1, le=100)
    use_ewc: bool = True
    use_replay: bool = True


class AutoTrainStatus(BaseModel):
    """Auto-training status."""

    enabled: bool
    last_training_at: datetime | None
    next_eligible_at: datetime | None
    buffer_size: int
    threshold: int
    training_in_progress: bool
    current_job_id: str | None = None
