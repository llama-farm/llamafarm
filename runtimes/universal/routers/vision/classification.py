"""Classification router — POST /v1/vision/classify"""

import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.error_handler import handle_endpoint_errors

from .utils import decode_base64_image

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-classification"])

_load_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None


def set_classification_loader(load_fn: Callable[..., Coroutine[Any, Any, Any]] | None) -> None:
    global _load_fn
    _load_fn = load_fn


class ClassifyRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    model: str = "clip-vit-base"
    classes: list[str] = Field(..., description="Classes for zero-shot classification")
    top_k: int = Field(default=5, ge=1, le=100)

class ClassifyResponse(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    all_scores: dict[str, float]
    model: str
    inference_time_ms: float


@router.post("/v1/vision/classify", response_model=ClassifyResponse)
@handle_endpoint_errors("vision_classify")
async def classify_image(request: ClassifyRequest) -> ClassifyResponse:
    """Classify an image using CLIP (zero-shot)."""
    if _load_fn is None:
        raise HTTPException(status_code=500, detail="Classification loader not initialized")
    if not request.classes:
        raise HTTPException(status_code=400, detail="Classes required for zero-shot classification")

    start = time.perf_counter()
    model = await _load_fn(request.model)
    image_bytes = decode_base64_image(request.image)

    result = await model.classify(image=image_bytes, classes=request.classes, top_k=request.top_k)

    return ClassifyResponse(
        class_name=result.class_name, class_id=result.class_id,
        confidence=result.confidence, all_scores=result.all_scores,
        model=request.model,
        inference_time_ms=(time.perf_counter() - start) * 1000,
    )
