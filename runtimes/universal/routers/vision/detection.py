"""Detection router for object detection endpoints.

Provides YOLO-based object detection with support for:
- Multiple YOLO variants (v8, v11)
- Confidence threshold filtering
- Class filtering
- Custom fine-tuned models
"""

import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, HTTPException

from api_types.vision import (
    BoundingBox,
    Detection,
    DetectRequest,
    DetectResponse,
)
from services.error_handler import handle_endpoint_errors

from .utils import decode_base64_image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision-detection"])

# Dependency injection for model loader
_load_detection_model_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None


def set_detection_loader(
    load_fn: Callable[..., Coroutine[Any, Any, Any]] | None,
) -> None:
    """Set the detection model loader function.

    Called during app initialization to inject the model loading
    dependency from the main server.

    Args:
        load_fn: Async function with signature:
                 async def load_detection(model_id: str) -> YOLOModel
    """
    global _load_detection_model_fn
    _load_detection_model_fn = load_fn


def _get_loader():
    """Get detection loader or raise if not initialized."""
    if _load_detection_model_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Detection model loader not initialized. Server configuration error.",
        )
    return _load_detection_model_fn


@router.post("/v1/vision/detect", response_model=DetectResponse)
@handle_endpoint_errors("vision_detect")
async def detect_objects(request: DetectRequest) -> DetectResponse:
    """Detect objects in an image.

    Uses YOLO models (v8, v11) for real-time object detection.
    Returns bounding boxes with class labels and confidence scores.

    Supported models:
    - yolov8n, yolov8s, yolov8m, yolov8l, yolov8x (YOLOv8 variants)
    - yolov11n, yolov11s, yolov11m (YOLOv11 variants)
    - Custom model paths

    Example request:
    ```json
    {
        "image": "data:image/jpeg;base64,...",
        "model": "yolov8n",
        "confidence_threshold": 0.5,
        "classes": ["person", "car"]
    }
    ```

    Example response:
    ```json
    {
        "detections": [
            {
                "box": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
                "class_name": "person",
                "class_id": 0,
                "confidence": 0.92
            }
        ],
        "model": "yolov8n",
        "inference_time_ms": 45.2
    }
    ```
    """
    start_time = time.perf_counter()

    # Load model
    loader = _get_loader()
    model = await loader(request.model)

    # Decode image from base64
    image_bytes = decode_base64_image(request.image)

    # Run detection
    result = await model.detect(
        image=image_bytes,
        confidence_threshold=request.confidence_threshold,
        classes=request.classes,
    )

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    # Convert to response format
    detections = [
        Detection(
            box=BoundingBox(
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
            ),
            class_name=box.class_name,
            class_id=box.class_id,
            confidence=box.confidence,
        )
        for box in result.boxes
    ]

    return DetectResponse(
        detections=detections,
        model=request.model,
        inference_time_ms=inference_time_ms,
    )


