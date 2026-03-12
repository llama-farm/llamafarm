"""Detection router — POST /v1/vision/detect"""

import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.error_handler import handle_endpoint_errors

from .utils import decode_base64_image

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-detection"])

_load_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None


def set_detection_loader(load_fn: Callable[..., Coroutine[Any, Any, Any]] | None) -> None:
    global _load_fn
    _load_fn = load_fn


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    box: BoundingBox
    class_name: str
    class_id: int
    confidence: float

class DetectRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    model: str = "yolov8n"
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    classes: list[str] | None = None

class DetectResponse(BaseModel):
    detections: list[Detection]
    model: str
    inference_time_ms: float


@router.post("/v1/vision/detect", response_model=DetectResponse)
@handle_endpoint_errors("vision_detect")
async def detect_objects(request: DetectRequest) -> DetectResponse:
    """Detect objects in an image using YOLO."""
    if _load_fn is None:
        raise HTTPException(status_code=500, detail="Detection loader not initialized")

    start = time.perf_counter()
    model = await _load_fn(request.model)
    image_bytes = decode_base64_image(request.image)

    result = await model.detect(
        image=image_bytes,
        confidence_threshold=request.confidence_threshold,
        classes=request.classes,
    )

    return DetectResponse(
        detections=[
            Detection(
                box=BoundingBox(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2),
                class_name=b.class_name, class_id=b.class_id, confidence=b.confidence,
            ) for b in result.boxes
        ],
        model=request.model,
        inference_time_ms=(time.perf_counter() - start) * 1000,
    )
