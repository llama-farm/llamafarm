"""Embedding router for image/text embedding endpoints.

Provides CLIP-based multimodal embeddings for:
- Image embedding (for similarity search)
- Text embedding (for cross-modal search)
- Image RAG integration
"""

import logging
import time
from typing import Any, Callable, Coroutine

from fastapi import APIRouter, HTTPException

from api_types.vision import (
    ImageEmbedRequest,
    ImageEmbedResponse,
)
from services.error_handler import handle_endpoint_errors

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision-embedding"])

# Dependency injection for model loader
_load_embedding_model_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None


def set_embedding_loader(
    load_fn: Callable[..., Coroutine[Any, Any, Any]] | None,
) -> None:
    """Set the embedding model loader function.

    Args:
        load_fn: Async function with signature:
                 async def load_embedding(model_id: str) -> CLIPEmbedder
    """
    global _load_embedding_model_fn
    _load_embedding_model_fn = load_fn


def _get_loader():
    """Get embedding loader or raise if not initialized."""
    if _load_embedding_model_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Embedding model loader not initialized.",
        )
    return _load_embedding_model_fn


@router.post("/v1/vision/embed", response_model=ImageEmbedResponse)
@handle_endpoint_errors("vision_embed")
async def embed_images_or_texts(request: ImageEmbedRequest) -> ImageEmbedResponse:
    """Generate CLIP embeddings for images and/or texts.

    CLIP embeddings exist in a shared vector space, enabling:
    - Image-to-image similarity search
    - Text-to-image search (find images matching a description)
    - Image-to-text search (find descriptions for images)

    You can provide images, texts, or both. All will be embedded
    into the same vector space.

    Supported models:
    - clip-vit-base (512 dimensions)
    - clip-vit-large (768 dimensions)
    - siglip-base (768 dimensions)

    Example request (images):
    ```json
    {
        "model": "clip-vit-base",
        "images": ["data:image/jpeg;base64,..."]
    }
    ```

    Example request (texts):
    ```json
    {
        "model": "clip-vit-base",
        "texts": ["a photo of a cat", "a photo of a dog"]
    }
    ```

    Example request (mixed):
    ```json
    {
        "model": "clip-vit-base",
        "images": ["data:image/jpeg;base64,..."],
        "texts": ["a photo of a cat"]
    }
    ```

    Example response:
    ```json
    {
        "embeddings": [[0.123, -0.456, ...]],
        "model": "clip-vit-base",
        "dimensions": 512,
        "inference_time_ms": 89.5
    }
    ```
    """
    if not request.images and not request.texts:
        raise HTTPException(
            status_code=400,
            detail="Either images or texts must be provided",
        )

    start_time = time.perf_counter()

    # Load model
    loader = _get_loader()
    model = await loader(request.model)

    all_embeddings = []

    # Embed images if provided
    if request.images:
        image_bytes_list = [_decode_base64_image(img) for img in request.images]
        image_result = await model.embed_images(image_bytes_list)
        all_embeddings.extend(image_result.embeddings)

    # Embed texts if provided
    if request.texts:
        text_result = await model.embed_texts(request.texts)
        all_embeddings.extend(text_result.embeddings)

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    return ImageEmbedResponse(
        embeddings=all_embeddings,
        model=request.model,
        dimensions=model.embedding_dim,
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
