"""Segmentation router for instance/semantic segmentation endpoints.

Provides YOLO-based segmentation with pixel-level masks.
"""

import logging
import time
from typing import Any, Callable, Coroutine

from fastapi import APIRouter, HTTPException

from api_types.vision import (
    BoundingBox,
    Detection,
)
from services.error_handler import handle_endpoint_errors
from .utils import decode_base64_image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision-segmentation"])

# Dependency injection for model loader
_load_segmentation_model_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None


def set_segmentation_loader(
    load_fn: Callable[..., Coroutine[Any, Any, Any]] | None,
) -> None:
    """Set the segmentation model loader function."""
    global _load_segmentation_model_fn
    _load_segmentation_model_fn = load_fn


def _get_loader():
    """Get segmentation loader or raise if not initialized."""
    if _load_segmentation_model_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Segmentation model loader not initialized.",
        )
    return _load_segmentation_model_fn


from pydantic import BaseModel, Field

class SegmentRequest(BaseModel):
    """Request for image segmentation."""
    image: str = Field(..., description="Base64 encoded image")
    model: str = Field(default="yolov8n-seg", description="Segmentation model")
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    return_masks: bool = Field(default=True, description="Return pixel masks")


class Segment(BaseModel):
    """A detected segment with mask."""
    box: BoundingBox
    class_name: str
    class_id: int
    confidence: float
    mask: list[list[float]] | None = None  # Simplified mask representation


class SegmentResponse(BaseModel):
    """Response from segmentation."""
    segments: list[Segment]
    model: str
    inference_time_ms: float


@router.post("/v1/vision/segment", response_model=SegmentResponse)
@handle_endpoint_errors("vision_segment")
async def segment_image(request: SegmentRequest) -> SegmentResponse:
    """Segment objects in an image with pixel-level masks.

    Uses YOLO segmentation models for instance segmentation.
    Returns bounding boxes with class labels, confidence, and masks.

    Supported models:
    - yolov8n-seg, yolov8s-seg, yolov8m-seg (YOLOv8 segmentation)
    
    Example request:
    ```json
    {
        "image": "data:image/jpeg;base64,...",
        "model": "yolov8n-seg",
        "confidence_threshold": 0.5,
        "return_masks": true
    }
    ```
    """
    start_time = time.perf_counter()

    # Load model - use detection loader for now since seg models are YOLO
    # In production, you'd have a dedicated segmentation loader
    from .detection import _load_detection_model_fn
    
    if _load_detection_model_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Detection model loader not initialized (used for segmentation).",
        )
    
    # Ensure model name ends with -seg
    model_name = request.model
    if not model_name.endswith('-seg'):
        model_name = model_name + '-seg'
    
    try:
        model = await _load_detection_model_fn(model_name)
    except Exception as e:
        # Fallback to regular detection if seg model not found
        logger.warning(f"Segmentation model {model_name} not found, falling back to detection: {e}")
        model_name = request.model.replace('-seg', '')
        model = await _load_detection_model_fn(model_name)

    # Decode image
    image_bytes = decode_base64_image(request.image)

    # Run detection (YOLO seg models return masks via detect)
    result = await model.detect(
        image=image_bytes,
        confidence_threshold=request.confidence_threshold,
    )

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    # Convert to segment format
    segments = [
        Segment(
            box=BoundingBox(
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
            ),
            class_name=box.class_name,
            class_id=box.class_id,
            confidence=box.confidence,
            mask=getattr(box, 'mask', None) if request.return_masks else None,
        )
        for box in result.boxes
    ]

    return SegmentResponse(
        segments=segments,
        model=model_name,
        inference_time_ms=inference_time_ms,
    )


