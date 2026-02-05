"""Classification router for image classification endpoints.

Provides CLIP-based image classification with support for:
- Zero-shot classification (text prompts)
- Few-shot learning
- Multiple CLIP variants
"""

import logging
import time
from typing import Any, Callable, Coroutine

from fastapi import APIRouter, HTTPException

from api_types.vision import (
    ImageClassifyRequest,
    ImageClassifyResponse,
)
from services.error_handler import handle_endpoint_errors

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision-classification"])

# Dependency injection for model loader
_load_classification_model_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None


def set_classification_loader(
    load_fn: Callable[..., Coroutine[Any, Any, Any]] | None,
) -> None:
    """Set the classification model loader function.

    Args:
        load_fn: Async function with signature:
                 async def load_classification(model_id: str) -> CLIPClassifier
    """
    global _load_classification_model_fn
    _load_classification_model_fn = load_fn


def _get_loader():
    """Get classification loader or raise if not initialized."""
    if _load_classification_model_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Classification model loader not initialized.",
        )
    return _load_classification_model_fn


@router.post("/v1/vision/classify", response_model=ImageClassifyResponse)
@handle_endpoint_errors("vision_classify")
async def classify_image(request: ImageClassifyRequest) -> ImageClassifyResponse:
    """Classify an image using CLIP.

    Uses CLIP or SigLIP models for zero-shot image classification.
    Provide class names and the model will predict which class
    best matches the image.

    Supported models:
    - clip-vit-base (OpenAI CLIP ViT-B/32)
    - clip-vit-large (OpenAI CLIP ViT-L/14)
    - siglip-base (Google SigLIP)
    - siglip-large (Google SigLIP Large)

    Example request:
    ```json
    {
        "image": "data:image/jpeg;base64,...",
        "model": "clip-vit-base",
        "classes": ["cat", "dog", "bird", "fish"],
        "top_k": 3
    }
    ```

    Example response:
    ```json
    {
        "class_name": "cat",
        "class_id": 0,
        "confidence": 0.87,
        "all_scores": {
            "cat": 0.87,
            "dog": 0.08,
            "bird": 0.05
        },
        "model": "clip-vit-base",
        "inference_time_ms": 125.3
    }
    ```
    """
    if not request.classes:
        raise HTTPException(
            status_code=400,
            detail="Classes must be provided for zero-shot classification",
        )

    start_time = time.perf_counter()

    # Load model
    loader = _get_loader()
    model = await loader(request.model)

    # Decode image
    image_bytes = _decode_base64_image(request.image)

    # Run classification
    result = await model.classify(
        image=image_bytes,
        classes=request.classes,
        top_k=request.top_k,
    )

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    return ImageClassifyResponse(
        class_name=result.class_name,
        class_id=result.class_id,
        confidence=result.confidence,
        all_scores=result.all_scores,
        model=request.model,
        inference_time_ms=inference_time_ms,
    )


def _decode_base64_image(image_str: str) -> bytes:
    """Decode base64 image string to bytes."""
    import base64

    if image_str.startswith("data:"):
        _, base64_data = image_str.split(",", 1)
    else:
        base64_data = image_str

    return base64.b64decode(base64_data)
