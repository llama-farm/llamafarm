"""Detect+Classify combo endpoint — YOLO detect → crop → CLIP classify per crop."""

import io
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from services.error_handler import handle_endpoint_errors

from .utils import decode_base64_image

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-detect-classify"])

_load_detection_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None
_load_classification_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None


def set_detect_classify_loaders(
    detection_fn: Callable[..., Coroutine[Any, Any, Any]] | None,
    classification_fn: Callable[..., Coroutine[Any, Any, Any]] | None,
) -> None:
    global _load_detection_fn, _load_classification_fn
    _load_detection_fn = detection_fn
    _load_classification_fn = classification_fn


# =============================================================================
# Request/Response models
# =============================================================================

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ClassifiedDetection(BaseModel):
    """A detection with classification results."""
    box: BoundingBox
    detection_class: str
    detection_confidence: float
    classification: str
    classification_confidence: float
    all_scores: dict[str, float]


class DetectClassifyRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    detection_model: str = Field(default="yolov8n", description="YOLO model for detection")
    classification_model: str = Field(default="clip-vit-base", description="CLIP model for classification")
    classes: list[str] = Field(..., description="Classes for zero-shot classification of each crop")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
    detection_classes: list[str] | None = Field(default=None, description="Filter detections to these YOLO classes")
    top_k: int = Field(default=3, ge=1, le=100, description="Top-K classification results per crop")
    min_crop_px: int = Field(default=16, ge=1, description="Minimum crop dimension in pixels (skip tiny detections)")


class DetectClassifyResponse(BaseModel):
    results: list[ClassifiedDetection]
    total_detections: int
    classified_count: int
    detection_model: str
    classification_model: str
    detection_time_ms: float
    classification_time_ms: float
    total_time_ms: float


# =============================================================================
# Endpoint
# =============================================================================

@router.post("/v1/vision/detect_classify", response_model=DetectClassifyResponse)
@handle_endpoint_errors("vision_detect_classify")
async def detect_and_classify(request: DetectClassifyRequest) -> DetectClassifyResponse:
    """Detect objects then classify each crop — single round-trip.

    Runs YOLO detection → crops each bounding box → CLIP classifies each crop.
    Returns unified results with both detection and classification info.
    """
    if _load_detection_fn is None or _load_classification_fn is None:
        raise HTTPException(status_code=500, detail="Model loaders not initialized")
    if not request.classes:
        raise HTTPException(status_code=400, detail="Classes required for classification")

    total_start = time.perf_counter()
    image_bytes = decode_base64_image(request.image)

    # Step 1: Detect
    det_start = time.perf_counter()
    det_model = await _load_detection_fn(request.detection_model)
    det_result = await det_model.detect(
        image=image_bytes,
        confidence_threshold=request.confidence_threshold,
        classes=request.detection_classes,
    )
    det_time = (time.perf_counter() - det_start) * 1000

    total_detections = len(det_result.boxes)
    if total_detections == 0:
        return DetectClassifyResponse(
            results=[], total_detections=0, classified_count=0,
            detection_model=request.detection_model,
            classification_model=request.classification_model,
            detection_time_ms=det_time, classification_time_ms=0.0,
            total_time_ms=(time.perf_counter() - total_start) * 1000,
        )

    # Step 2: Crop each detection and classify
    cls_start = time.perf_counter()
    cls_model = await _load_classification_fn(request.classification_model)

    # Convert image once for cropping
    pil_image = Image.open(io.BytesIO(image_bytes))
    results: list[ClassifiedDetection] = []

    for box in det_result.boxes:
        # Crop the detection region
        x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
        x2, y2 = min(pil_image.width, int(box.x2)), min(pil_image.height, int(box.y2))

        # Skip tiny crops
        if (x2 - x1) < request.min_crop_px or (y2 - y1) < request.min_crop_px:
            continue

        crop = pil_image.crop((x1, y1, x2, y2))

        # Ensure RGB mode for JPEG encoding (handles RGBA, P, L, etc.)
        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        # Convert crop to bytes for the classifier
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=90)
        crop_bytes = buf.getvalue()

        # Classify the crop
        cls_result = await cls_model.classify(
            image=crop_bytes,
            classes=request.classes,
            top_k=request.top_k,
        )

        results.append(ClassifiedDetection(
            box=BoundingBox(x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2),
            detection_class=box.class_name,
            detection_confidence=box.confidence,
            classification=cls_result.class_name,
            classification_confidence=cls_result.confidence,
            all_scores=cls_result.all_scores,
        ))

    cls_time = (time.perf_counter() - cls_start) * 1000

    return DetectClassifyResponse(
        results=results,
        total_detections=total_detections,
        classified_count=len(results),
        detection_model=request.detection_model,
        classification_model=request.classification_model,
        detection_time_ms=det_time,
        classification_time_ms=cls_time,
        total_time_ms=(time.perf_counter() - total_start) * 1000,
    )
