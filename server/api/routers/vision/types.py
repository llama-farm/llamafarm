"""
Pydantic models for Vision endpoints.
"""

from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# OCR Types (existing)
# =============================================================================

class OCRRequest(BaseModel):
    model: str = "surya"
    images: list[str]
    languages: list[str] | None = None
    return_boxes: bool = False


class DocumentExtractRequest(BaseModel):
    model: str
    images: list[str]
    prompts: list[str] | None = None
    task: Literal["extraction", "vqa", "classification"] = "extraction"


# =============================================================================
# Detection Types
# =============================================================================

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionItem(BaseModel):
    box: BoundingBox
    class_name: str
    class_id: int
    confidence: float

class DetectRequest(BaseModel):
    model: str = "yolov8n"
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    classes: list[str] | None = None

class DetectResponse(BaseModel):
    detections: list[DetectionItem]
    model: str
    inference_time_ms: float


# =============================================================================
# Classification Types
# =============================================================================

class ClassifyRequest(BaseModel):
    model: str = "clip-vit-base"
    classes: list[str]
    top_k: int = Field(default=5, ge=1)

class ClassifyResponse(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    all_scores: dict[str, float]
    model: str
    inference_time_ms: float


# =============================================================================
# Training Types
# =============================================================================

class TrainConfigRequest(BaseModel):
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = 0.001

class TrainRequest(BaseModel):
    model: str
    dataset: str
    task: Literal["detection", "classification"] = "detection"
    config: TrainConfigRequest = Field(default_factory=TrainConfigRequest)

class TrainResponse(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0


# =============================================================================
# Streaming Types
# =============================================================================

class CascadeConfigRequest(BaseModel):
    chain: list[str] = Field(default=["yolov8n"])
    confidence_threshold: float = 0.7

class StreamStartRequest(BaseModel):
    config: CascadeConfigRequest = Field(default_factory=CascadeConfigRequest)
    target_fps: float = 1.0
    action_classes: list[str] | None = None

class StreamFrameResponse(BaseModel):
    status: str
    detections: list[DetectionItem] | None = None
    confidence: float | None = None
    resolved_by: str | None = None


# =============================================================================
# Model Management Types
# =============================================================================

class ModelExportRequest(BaseModel):
    model_id: str
    format: Literal["onnx", "coreml", "tensorrt", "tflite", "openvino"]
    quantization: Literal["fp32", "fp16", "int8"] = "fp16"


# =============================================================================
# Review Types
# =============================================================================

class ReviewItem(BaseModel):
    image_id: str
    confidence: float
    class_name: str
    source: str
    status: str = "pending"

class ReviewDecision(BaseModel):
    image_id: str
    decision: Literal["correct", "wrong", "adjusted"]
    corrected_class: str | None = None
