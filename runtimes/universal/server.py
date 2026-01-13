"""
Universal Runtime Server

A FastAPI server that provides OpenAI-compatible endpoints for any HuggingFace
model without restrictions. Supports:
- Text generation (Causal LMs: GPT, Llama, Mistral, etc.)
- Text embeddings & classification (Encoders: BERT, sentence-transformers, etc.)

Key Features:
- Auto-detects hardware (MPS/CUDA/CPU)
- Lazy model loading (load on first request)
- Automatic model unloading (after 5 minutes of inactivity by default)
- Platform-specific optimizations
- OpenAI API compatibility
- No model restrictions (trust_remote_code=True)

Environment Variables:
- MODEL_UNLOAD_TIMEOUT: Seconds of inactivity before unloading models (default: 300)
- CLEANUP_CHECK_INTERVAL: Seconds between cleanup checks (default: 30)
"""

import asyncio
import base64
import os
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    FastAPI,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel as PydanticBaseModel

from core.logging import UniversalRuntimeLogger, setup_logging
from models import (
    AnomalyModel,
    BackgroundRemovalModel,
    BaseModel,
    ClassifierModel,
    CLIPVisionModel,
    DocumentModel,
    EncoderModel,
    GGUFEncoderModel,
    GGUFLanguageModel,
    LanguageDetectionModel,
    LanguageModel,
    ObjectDetectionModel,
    OCRModel,
    PIIModel,
    TimeSeriesModel,
)
from routers.chat_completions import router as chat_completions_router
from utils.device import get_device_info, get_optimal_device
from utils.feature_encoder import FeatureEncoder
from utils.file_handler import (
    delete_file,
    get_file,
    get_file_images,
    list_files,
    store_file,
)
from utils.model_cache import ModelCache
from utils.model_format import detect_model_format

# Configure logging FIRST, before anything else
log_file = os.getenv("LOG_FILE", "")
log_level = os.getenv("LOG_LEVEL", "INFO")
json_logs = os.getenv("LOG_JSON_FORMAT", "false").lower() in ("true", "1", "yes")
setup_logging(json_logs=json_logs, log_level=log_level, log_file=log_file)

logger = UniversalRuntimeLogger("universal-runtime")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle (startup and shutdown)."""
    global _cleanup_task

    # Startup
    logger.info("Starting Universal Runtime")

    # Start model cleanup background task
    _cleanup_task = asyncio.create_task(_cleanup_idle_models())
    logger.info("Model cleanup background task started")

    yield

    # Shutdown
    logger.info("Shutting down Universal Runtime")

    # Stop cleanup task
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await _cleanup_task
        logger.info("Model cleanup task stopped")

    # Unload all remaining models
    if _models:
        logger.info(f"Unloading {len(_models)} remaining model(s)")
        for cache_key, model in list(_models.items()):
            try:
                await model.unload()
                logger.info(f"Unloaded model: {cache_key}")
            except Exception as e:
                logger.error(f"Error unloading model {cache_key}: {e}")
        _models.clear()

    if _classifiers:
        logger.info(f"Unloading {len(_classifiers)} remaining classifier(s)")
        for cache_key, model in list(_classifiers.items()):
            try:
                await model.unload()
                logger.info(f"Unloaded classifier: {cache_key}")
            except Exception as e:
                logger.error(f"Error unloading classifier {cache_key}: {e}")
        _classifiers.clear()

    if _vision_models:
        logger.info(f"Unloading {len(_vision_models)} remaining vision model(s)")
        for cache_key, model in list(_vision_models.items()):
            try:
                await model.unload()
                logger.info(f"Unloaded vision model: {cache_key}")
            except Exception as e:
                logger.error(f"Error unloading vision model {cache_key}: {e}")
        _vision_models.clear()

    logger.info("Shutdown complete")


app = FastAPI(
    title="Universal Runtime",
    description="OpenAI-compatible API for HuggingFace models (transformers, diffusers, embedders)",
    version="2.0.0",
    lifespan=lifespan,
)
app.include_router(chat_completions_router)

# Model unload timeout configuration (in seconds)
# Default: 5 minutes (300 seconds)
MODEL_UNLOAD_TIMEOUT = int(os.getenv("MODEL_UNLOAD_TIMEOUT", "300"))
# Cleanup check interval (in seconds) - how often to check for idle models
# Default: 30 seconds
CLEANUP_CHECK_INTERVAL = int(os.getenv("CLEANUP_CHECK_INTERVAL", "30"))

# Global model caches using TTL-based caching (via cachetools)
# Models are automatically tracked for idle time and cleaned up by background task
_models: ModelCache[BaseModel] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
_classifiers: ModelCache["ClassifierModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
_vision_models: ModelCache["CLIPVisionModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
_lang_detection_models: ModelCache["LanguageDetectionModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
_pii_models: ModelCache["PIIModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
_object_detection_models: ModelCache["ObjectDetectionModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
_background_removal_models: ModelCache["BackgroundRemovalModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
_timeseries_models: ModelCache["TimeSeriesModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
_model_load_lock = asyncio.Lock()
_current_device = None

# Feature encoder cache for anomaly detection with mixed data types
_encoders: dict[str, FeatureEncoder] = {}
_cleanup_task: asyncio.Task | None = None


# ============================================================================
# Helper Functions
# ============================================================================


async def _cleanup_idle_models() -> None:
    """Background task that periodically unloads idle models.

    Uses ModelCache's TTL-based expiration to find and unload models that
    haven't been accessed in MODEL_UNLOAD_TIMEOUT seconds.
    """
    logger.info(
        f"Model cleanup task started (timeout={MODEL_UNLOAD_TIMEOUT}s, "
        f"check_interval={CLEANUP_CHECK_INTERVAL}s)"
    )

    while True:
        try:
            await asyncio.sleep(CLEANUP_CHECK_INTERVAL)

            # Cleanup expired models from all caches
            for cache, cache_name in [
                (_models, "models"),
                (_classifiers, "classifiers"),
                (_vision_models, "vision_models"),
            ]:
                expired_items = cache.pop_expired()
                if expired_items:
                    logger.info(f"Unloading {len(expired_items)} idle {cache_name}")
                    for cache_key, model in expired_items:
                        try:
                            await model.unload()
                            logger.info(f"Successfully unloaded: {cache_key}")
                        except Exception as e:
                            logger.error(
                                f"Error unloading model {cache_key}: {e}", exc_info=True
                            )

        except asyncio.CancelledError:
            logger.info("Model cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}", exc_info=True)
            # Continue running despite errors


def get_device():
    """Get the optimal device for the current platform."""
    global _current_device
    if _current_device is None:
        _current_device = get_optimal_device()
        logger.info(f"Using device: {_current_device}")
    return _current_device


def _make_language_cache_key(
    model_id: str, n_ctx: int | None = None, preferred_quantization: str | None = None
) -> str:
    """Generate a cache key for a causal language model.

    Args:
        model_id: HuggingFace model identifier
        n_ctx: Optional context window size for GGUF models
        preferred_quantization: Optional quantization preference for GGUF models

    Returns:
        A unique cache key string that identifies this specific model configuration
    """
    quant_key = (
        preferred_quantization if preferred_quantization is not None else "default"
    )
    return f"language:{model_id}:ctx{n_ctx if n_ctx is not None else 'auto'}:quant{quant_key}"


async def load_language(
    model_id: str, n_ctx: int | None = None, preferred_quantization: str | None = None
):
    """Load a causal language model (GGUF or transformers format).

    Automatically detects whether the model is in GGUF or transformers format
    and loads it with the appropriate backend. GGUF models use llama-cpp
    for optimized inference, while transformers models use the standard HuggingFace
    transformers library.

    Args:
        model_id: HuggingFace model identifier
        n_ctx: Optional context window size for GGUF models. If None, will be
               computed automatically based on available memory and model defaults.
        preferred_quantization: Optional quantization preference for GGUF models
                                (e.g., "Q4_K_M", "Q8_0"). If None, defaults to Q4_K_M.
                                Only downloads the specified quantization to save disk space.
    """

    # Include n_ctx and quantization in cache key for GGUF models so different configurations are cached separately
    # Use "auto" for None to allow automatic context size computation
    # Use "default" for None quantization to use Q4_K_M default
    # Transformers are obviously not quantized, so just ignore in that case
    cache_key = _make_language_cache_key(model_id, n_ctx, preferred_quantization)
    if cache_key not in _models:
        async with _model_load_lock:
            # Double-check if model was loaded while waiting for the lock
            if cache_key not in _models:
                logger.info(
                    f"Loading causal LM: {model_id} (n_ctx={n_ctx if n_ctx is not None else 'auto'})"
                )
                device = get_device()

                # Detect model format (GGUF vs transformers)
                model_format = detect_model_format(model_id)
                logger.info(f"Detected format: {model_format}")

                # Instantiate appropriate model class based on format
                model: BaseModel
                if model_format == "gguf":
                    model = GGUFLanguageModel(
                        model_id,
                        device,
                        n_ctx=n_ctx,
                        preferred_quantization=preferred_quantization,
                    )
                else:
                    model = LanguageModel(model_id, device)

                await model.load()
                _models[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return _models.get(cache_key)


def _make_encoder_cache_key(
    model_id: str,
    task: str,
    model_format: str,
    preferred_quantization: str | None = None,
    max_length: int | None = None,
) -> str:
    """Generate a cache key for an encoder model.

    Args:
        model_id: HuggingFace model identifier
        task: Model task - "embedding", "classification", "reranking", or "ner"
        model_format: Model format - "gguf" or "transformers"
        preferred_quantization: Optional quantization preference for GGUF models
        max_length: Optional max sequence length override

    Returns:
        A unique cache key string that identifies this specific model configuration
    """
    quant_key = (
        preferred_quantization if preferred_quantization is not None else "default"
    )
    len_key = max_length if max_length is not None else "auto"
    return f"encoder:{task}:{model_format}:{model_id}:quant{quant_key}:len{len_key}"


async def load_encoder(
    model_id: str,
    task: str = "embedding",
    preferred_quantization: str | None = None,
    max_length: int | None = None,
    use_flash_attention: bool = True,
):
    """Load an encoder model for embeddings, classification, reranking, or NER.

    Automatically detects whether the model is in GGUF or transformers format
    and loads it with the appropriate backend. GGUF models use llama-cpp
    for optimized inference, while transformers models use the standard HuggingFace
    transformers library.

    Supports modern encoder features:
    - Configurable max_length (up to 8,192 for ModernBERT)
    - Flash Attention 2 for faster inference on CUDA

    Args:
        model_id: HuggingFace model identifier
        task: Model task - "embedding", "classification", "reranking", or "ner"
        preferred_quantization: Optional quantization preference for GGUF models
                                (e.g., "Q4_K_M", "Q8_0"). If None, defaults to Q4_K_M.
        max_length: Optional max sequence length override (auto-detected if None)
        use_flash_attention: Whether to use Flash Attention 2 if available (default True)
    """
    # Detect model format for proper caching and loading
    model_format = detect_model_format(model_id)
    # Include quantization and max_length in cache key for proper caching
    cache_key = _make_encoder_cache_key(
        model_id, task, model_format, preferred_quantization, max_length
    )

    if cache_key not in _models:
        async with _model_load_lock:
            # Double-check if model was loaded while waiting for the lock
            if cache_key not in _models:
                logger.info(
                    f"Loading encoder ({task}): {model_id} (format: {model_format})"
                )
                device = get_device()

                # Instantiate appropriate model class based on format
                model: BaseModel
                if model_format == "gguf":
                    if task != "embedding":
                        raise ValueError(
                            f"GGUF models only support embedding task, not '{task}'"
                        )
                    model = GGUFEncoderModel(
                        model_id, device, preferred_quantization=preferred_quantization
                    )
                else:
                    model = EncoderModel(
                        model_id,
                        device,
                        task=task,
                        max_length=max_length,
                        use_flash_attention=use_flash_attention,
                    )

                await model.load()
                _models[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return _models.get(cache_key)


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint with device information."""
    device_info = get_device_info()
    return {
        "status": "healthy",
        "device": device_info,
        "loaded_models": list(_models.keys()),
        "timestamp": datetime.utcnow().isoformat(),
        "pid": os.getpid(),
    }


@app.get("/v1/models")
async def list_models():
    """List currently loaded models."""
    models_list = []
    for model_id, model in _models.items():
        models_list.append(
            {
                "id": model_id,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "transformers-runtime",
                "type": model.model_type,
            }
        )

    return {"object": "list", "data": models_list}


# ============================================================================
# File Upload Endpoints
# ============================================================================


# Maximum file upload size (100 MB by default, configurable via env var)
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 100 * 1024 * 1024))


@app.post("/v1/files")
async def upload_file(
    file: UploadFile,
    convert_pdf: bool = Form(default=True),
    pdf_dpi: int = Form(default=150),
):
    """
    Upload a file for use with OCR, document extraction, or image generation.

    Uploaded files are stored temporarily (5 minutes TTL) and can be referenced
    by their file ID in subsequent API calls.

    For PDFs, pages are automatically converted to images for OCR/document processing.

    Args:
        file: The file to upload (images, PDFs supported, max 100MB)
        convert_pdf: If True, convert PDF pages to images (default: True)
        pdf_dpi: DPI for PDF to image conversion (default: 150)

    Returns:
        File metadata including ID for referencing in other endpoints

    Example:
        ```bash
        curl -X POST http://localhost:8000/v1/files \\
            -F "file=@document.pdf" \\
            -F "convert_pdf=true" \\
            -F "pdf_dpi=150"
        ```
    """
    try:
        # Read file with size limit to prevent memory exhaustion
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
            )
        stored = await store_file(
            content=content,
            filename=file.filename or "unknown",
            content_type=file.content_type,
            convert_pdf_to_images=convert_pdf,
            pdf_dpi=pdf_dpi,
        )

        return {
            "id": stored.id,
            "object": "file",
            "filename": stored.filename,
            "content_type": stored.content_type,
            "size": stored.size,
            "created_at": stored.created_at,
            "has_images": stored.page_images is not None
            or stored.filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif")
            ),
            "page_count": len(stored.page_images) if stored.page_images else None,
        }

    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/files")
async def get_uploaded_files():
    """
    List all uploaded files with their metadata.

    Returns:
        List of file metadata
    """
    return {"object": "list", "data": list_files()}


@app.get("/v1/files/{file_id}")
async def get_uploaded_file(file_id: str):
    """
    Get metadata for a specific uploaded file.

    Args:
        file_id: The file ID returned from upload

    Returns:
        File metadata
    """
    stored = get_file(file_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    return {
        "id": stored.id,
        "object": "file",
        "filename": stored.filename,
        "content_type": stored.content_type,
        "size": stored.size,
        "created_at": stored.created_at,
        "has_images": stored.page_images is not None,
        "page_count": len(stored.page_images) if stored.page_images else None,
    }


@app.get("/v1/files/{file_id}/images")
async def get_file_as_images(file_id: str):
    """
    Get base64-encoded images for a file.

    For PDFs, returns one image per page.
    For image files, returns the image itself.

    Args:
        file_id: The file ID returned from upload

    Returns:
        List of base64-encoded images
    """
    stored = get_file(file_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    images = get_file_images(file_id)
    if not images:
        raise HTTPException(
            status_code=400,
            detail=f"File {file_id} cannot be converted to images",
        )

    return {
        "object": "list",
        "file_id": file_id,
        "data": [{"index": i, "base64": img} for i, img in enumerate(images)],
    }


@app.delete("/v1/files/{file_id}")
async def delete_uploaded_file(file_id: str):
    """
    Delete an uploaded file.

    Args:
        file_id: The file ID to delete

    Returns:
        Deletion confirmation
    """
    if delete_file(file_id):
        return {"deleted": True, "id": file_id}
    raise HTTPException(status_code=404, detail=f"File not found: {file_id}")


# ============================================================================
# Embeddings Endpoint
# ============================================================================


class EmbeddingRequest(PydanticBaseModel):
    """OpenAI-compatible embedding request."""

    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = "float"
    user: str | None = None
    extra_body: dict | None = None


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.

    Supports any HuggingFace encoder model for text embeddings.
    Model names can include quantization suffix (e.g., "model:Q4_K_M").
    """
    try:
        # Import parsing utility
        from utils.model_format import parse_model_with_quantization

        # Parse model name to extract quantization if present
        model_id, gguf_quantization = parse_model_with_quantization(request.model)

        model = await load_encoder(
            model_id, task="embedding", preferred_quantization=gguf_quantization
        )

        # Normalize input to list
        texts = [request.input] if isinstance(request.input, str) else request.input

        # Generate embeddings
        embeddings = await model.embed(texts, normalize=True)

        # Format response
        data = []
        for idx, embedding in enumerate(embeddings):
            if request.encoding_format == "base64":
                import struct

                embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
                embedding_data = base64.b64encode(embedding_bytes).decode("utf-8")
            else:
                embedding_data = embedding

            data.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding_data,
                }
            )

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": 0,  # TODO: Implement token counting
                "total_tokens": 0,
            },
        }

    except Exception as e:
        logger.error(f"Error in create_embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Reranking Endpoint (Cross-Encoder)
# ============================================================================


class RerankRequest(PydanticBaseModel):
    """Reranking request for cross-encoder models."""

    model: str
    query: str
    documents: list[str]
    top_k: int | None = None
    return_documents: bool = True


@app.post("/v1/rerank")
async def rerank_documents(request: RerankRequest):
    """
    Cross-encoder reranking endpoint.

    Reranks documents based on relevance to the query using proper
    cross-encoder architecture (query and document jointly encoded).

    This is significantly more accurate than bi-encoder similarity
    and 10-100x faster than LLM-based reranking.
    """
    try:
        model = await load_encoder(request.model, task="reranking")

        # Rerank documents
        results = await model.rerank(
            query=request.query, documents=request.documents, top_k=request.top_k
        )

        # Format response
        data = []
        for result in results:
            item = {
                "index": result["index"],
                "relevance_score": result["relevance_score"],
            }
            if request.return_documents:
                item["document"] = result["document"]
            data.append(item)

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": 0,  # TODO: Implement token counting
                "total_tokens": 0,
            },
        }

    except Exception as e:
        logger.error(f"Error in rerank_documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Classification Endpoint
# ============================================================================


class ClassifyRequest(PydanticBaseModel):
    """Text classification request."""

    model: str  # HuggingFace model ID (e.g., "distilbert-base-uncased-finetuned-sst-2-english")
    texts: list[str]  # Texts to classify
    max_length: int | None = None  # Optional max sequence length


@app.post("/v1/classify")
async def classify_texts(request: ClassifyRequest):
    """
    Text classification endpoint.

    Classify texts using any HuggingFace sequence classification model.
    Supports sentiment analysis, spam detection, intent routing, etc.

    Popular models:
    - distilbert-base-uncased-finetuned-sst-2-english (sentiment)
    - facebook/bart-large-mnli (zero-shot classification)
    - cardiffnlp/twitter-roberta-base-sentiment-latest (social media sentiment)

    Example request:
    ```json
    {
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "texts": ["I love this product!", "This is terrible."]
    }
    ```
    """
    try:
        # Import parsing utility
        from utils.model_format import parse_model_with_quantization

        # Parse model name
        model_id, _ = parse_model_with_quantization(request.model)

        model = await load_encoder(
            model_id,
            task="classification",
            max_length=request.max_length,
        )

        # Run classification
        results = await model.classify(request.texts)

        # Format response
        data = []
        for idx, result in enumerate(results):
            data.append(
                {
                    "index": idx,
                    "label": result["label"],
                    "score": result["score"],
                    "all_scores": result["all_scores"],
                }
            )

        return {
            "object": "list",
            "data": data,
            "total_count": len(data),
            "model": request.model,
            "usage": {
                "texts_processed": len(request.texts),
            },
        }

    except Exception as e:
        logger.error(f"Error in classify_texts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# NER (Named Entity Recognition) Endpoint
# ============================================================================


class NERRequest(PydanticBaseModel):
    """Named entity recognition request."""

    model: str  # HuggingFace model ID (e.g., "dslim/bert-base-NER")
    texts: list[str]  # Texts for entity extraction
    max_length: int | None = None  # Optional max sequence length


@app.post("/v1/ner")
async def extract_entities(request: NERRequest):
    """
    Named Entity Recognition endpoint.

    Extract named entities (people, organizations, locations, etc.) from text
    using HuggingFace token classification models.

    Popular models:
    - dslim/bert-base-NER (English, PERSON/ORG/LOC/MISC)
    - Jean-Baptiste/roberta-large-ner-english (English, high accuracy)
    - xlm-roberta-large-finetuned-conll03-english (multilingual)

    Example request:
    ```json
    {
        "model": "dslim/bert-base-NER",
        "texts": ["John works at Google in San Francisco."]
    }
    ```

    Response entities include:
    - text: The extracted entity text
    - label: Entity type (PERSON, ORG, LOC, etc.)
    - start/end: Character offsets in the original text
    - score: Confidence score
    """
    try:
        # Import parsing utility
        from utils.model_format import parse_model_with_quantization

        # Parse model name
        model_id, _ = parse_model_with_quantization(request.model)

        model = await load_encoder(
            model_id,
            task="ner",
            max_length=request.max_length,
        )

        # Run NER
        results = await model.extract_entities(request.texts)

        # Format response
        data = []
        for idx, entities in enumerate(results):
            data.append(
                {
                    "index": idx,
                    "entities": [
                        {
                            "text": e.text,
                            "label": e.label,
                            "start": e.start,
                            "end": e.end,
                            "score": e.score,
                        }
                        for e in entities
                    ],
                }
            )

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "texts_processed": len(request.texts),
            },
        }

    except Exception as e:
        logger.error(f"Error in extract_entities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Document Understanding Endpoint
# ============================================================================


def _make_document_cache_key(model_id: str, task: str) -> str:
    """Generate a cache key for a document model."""
    return f"document:{task}:{model_id}"


async def load_document(
    model_id: str,
    task: str = "extraction",
):
    """Load a document understanding model.

    Args:
        model_id: HuggingFace model identifier
        task: Model task - "extraction", "vqa", or "classification"

    Returns:
        Loaded DocumentModel instance
    """
    cache_key = _make_document_cache_key(model_id, task)

    if cache_key not in _models:
        async with _model_load_lock:
            if cache_key not in _models:
                logger.info(f"Loading document model ({task}): {model_id}")
                device = get_device()

                model = DocumentModel(
                    model_id=model_id,
                    device=device,
                    task=task,
                )

                await model.load()
                _models[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return _models.get(cache_key)


class DocumentExtractRequest(PydanticBaseModel):
    """Document extraction request."""

    model: str  # HuggingFace model ID (e.g., "naver-clova-ix/donut-base-finetuned-cord-v2")
    images: list[str] | None = None  # Base64-encoded document images
    file_id: str | None = None  # File ID from /v1/files upload
    prompts: list[str] | None = None  # Optional prompts for each image
    task: str = "extraction"  # extraction, vqa, classification


@app.post("/v1/documents/extract")
async def extract_from_documents(request: DocumentExtractRequest):
    """
    Document understanding endpoint.

    Extract structured information from documents using vision-language models.
    Supports forms, invoices, receipts, and other document types.

    Model types:
    - Donut models: End-to-end, no OCR needed (naver-clova-ix/donut-*)
    - LayoutLM models: Uses OCR + layout features (microsoft/layoutlmv3-*)

    Tasks:
    - extraction: Extract key-value pairs from documents
    - vqa: Answer questions about document content
    - classification: Classify document types

    You can provide images either as:
    1. Base64-encoded strings in the `images` field
    2. A file ID from a previous upload via `file_id` field

    Example with base64:
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "images": ["base64_encoded_image..."],
        "task": "extraction"
    }
    ```

    Example with file_id (from /v1/files upload):
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "file_id": "file_abc123_def456",
        "task": "extraction"
    }
    ```

    For VQA, include prompts:
    ```json
    {
        "model": "microsoft/layoutlmv3-base-finetuned-docvqa",
        "file_id": "file_abc123_def456",
        "prompts": ["What is the total amount?"],
        "task": "vqa"
    }
    ```
    """
    try:
        # Resolve images from file_id or direct base64
        images = request.images
        if request.file_id:
            images = get_file_images(request.file_id)
            if not images:
                raise HTTPException(
                    status_code=400,
                    detail=f"No images found for file_id: {request.file_id}",
                )
        elif not images:
            raise HTTPException(
                status_code=400,
                detail="Either 'images' or 'file_id' must be provided",
            )

        # Load document model
        model = await load_document(
            model_id=request.model,
            task=request.task,
        )

        # Extract from documents
        results = await model.extract(
            images=images,
            prompts=request.prompts,
        )

        # Format response
        data = []
        for idx, result in enumerate(results):
            item = {
                "index": idx,
                "confidence": result.confidence,
            }

            if result.text:
                item["text"] = result.text

            if result.fields:
                item["fields"] = [
                    {
                        "key": f.key,
                        "value": f.value,
                        "confidence": f.confidence,
                        "bbox": f.bbox,
                    }
                    for f in result.fields
                ]

            if result.answer:
                item["answer"] = result.answer

            if result.classification:
                item["classification"] = result.classification
                item["classification_scores"] = result.classification_scores

            data.append(item)

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "task": request.task,
            "usage": {
                "documents_processed": len(images),
            },
        }

    except Exception as e:
        logger.error(f"Error in extract_from_documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# OCR Endpoint
# ============================================================================


def _make_ocr_cache_key(backend: str, languages: list[str]) -> str:
    """Generate a cache key for an OCR model.

    Args:
        backend: OCR backend (surya, easyocr, paddleocr, tesseract)
        languages: List of language codes

    Returns:
        A unique cache key string
    """
    lang_key = "_".join(sorted(languages))
    return f"ocr:{backend}:{lang_key}"


async def load_ocr(backend: str = "surya", languages: list[str] | None = None):
    """Load an OCR model with the specified backend.

    Args:
        backend: OCR backend to use (surya, easyocr, paddleocr, tesseract)
        languages: List of language codes (e.g., ['en', 'fr'])

    Returns:
        Loaded OCRModel instance
    """
    langs = languages or ["en"]
    cache_key = _make_ocr_cache_key(backend, langs)

    if cache_key not in _models:
        async with _model_load_lock:
            if cache_key not in _models:
                logger.info(f"Loading OCR model: {backend} (languages: {langs})")
                device = get_device()

                model = OCRModel(
                    model_id=f"ocr-{backend}",
                    device=device,
                    backend=backend,
                    languages=langs,
                )

                await model.load()
                _models[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return _models.get(cache_key)


class OCRRequest(PydanticBaseModel):
    """OCR request for text extraction from images."""

    model: str = "surya"  # Backend: surya, easyocr, paddleocr, tesseract
    images: list[str] | None = None  # Base64-encoded images
    file_id: str | None = None  # File ID from /v1/files upload
    languages: list[str] | None = None  # Language codes (e.g., ['en', 'fr'])
    return_boxes: bool = False  # Return bounding boxes for detected text


@app.post("/v1/ocr")
async def extract_text_from_images(request: OCRRequest):
    """
    OCR endpoint for text extraction from images.

    Supports multiple OCR backends:
    - surya: Best accuracy, transformer-based, layout-aware (recommended)
    - easyocr: Good multilingual support (80+ languages), widely used
    - paddleocr: Fast, optimized for production, excellent for Asian languages
    - tesseract: Classic OCR engine, CPU-only, widely deployed

    You can provide images either as:
    1. Base64-encoded strings in the `images` field
    2. A file ID from a previous upload via `file_id` field

    Example with base64:
    ```json
    {
        "model": "surya",
        "images": ["base64_encoded_image..."],
        "languages": ["en"],
        "return_boxes": false
    }
    ```

    Example with file_id (from /v1/files upload):
    ```json
    {
        "model": "surya",
        "file_id": "file_abc123_def456",
        "languages": ["en"]
    }
    ```
    """
    try:
        # Resolve images from file_id or direct base64
        images = request.images
        if request.file_id:
            images = get_file_images(request.file_id)
            if not images:
                raise HTTPException(
                    status_code=400,
                    detail=f"No images found for file_id: {request.file_id}",
                )
        elif not images:
            raise HTTPException(
                status_code=400,
                detail="Either 'images' or 'file_id' must be provided",
            )

        # Load OCR model
        model = await load_ocr(
            backend=request.model,
            languages=request.languages,
        )

        # Run OCR
        results = await model.recognize(
            images=images,
            languages=request.languages,
            return_boxes=request.return_boxes,
        )

        # Format response
        data = []
        for idx, result in enumerate(results):
            item = {
                "index": idx,
                "text": result.text,
                "confidence": result.confidence,
            }
            if request.return_boxes and result.boxes:
                item["boxes"] = [
                    {
                        "x1": box.x1,
                        "y1": box.y1,
                        "x2": box.x2,
                        "y2": box.y2,
                        "text": box.text,
                        "confidence": box.confidence,
                    }
                    for box in result.boxes
                ]
            data.append(item)

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "images_processed": len(images),
            },
        }

    except ImportError as e:
        logger.error(f"OCR backend not installed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"OCR backend '{request.model}' not installed. {str(e)}",
        ) from e
    except Exception as e:
        logger.error(f"Error in extract_text_from_images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Anomaly Detection Endpoints
# ============================================================================


def _make_anomaly_cache_key(
    model_id: str, backend: str, normalization: str | None = None, scaler_type: str | None = None
) -> str:
    """Generate a cache key for an anomaly model.

    Args:
        model_id: Model identifier or path
        backend: Anomaly detection backend
        normalization: Score normalization method. If provided, it becomes part of
            the cache key to ensure models with different normalization methods
            are cached separately.
        scaler_type: Input data scaler type (robust or standard)

    Returns:
        Cache key string
    """
    parts = ["anomaly", backend]
    if normalization:
        parts.append(normalization)
    if scaler_type:
        parts.append(scaler_type)
    parts.append(model_id)
    return ":".join(parts)


async def load_anomaly(
    model_id: str,
    backend: str = "isolation_forest",
    contamination: float = 0.1,
    threshold: float | None = None,
    normalization: str = "standardization",
    scaler_type: str = "robust",
    validation_split: float = 0.1,
    patience: int = 10,
    min_delta: float = 1e-4,
):
    """Load an anomaly detection model.

    Args:
        model_id: Model identifier or path to pre-trained model
        backend: Anomaly detection backend
        contamination: Expected proportion of anomalies
        threshold: Custom anomaly threshold
        normalization: Score normalization method (standardization, zscore, raw)
        scaler_type: Input data scaler type (robust or standard)
        validation_split: Fraction of data for validation (autoencoder/vae only)
        patience: Epochs without improvement before stopping (autoencoder/vae only)
        min_delta: Minimum change in validation loss for improvement

    Returns:
        Loaded AnomalyModel instance
    """
    cache_key = _make_anomaly_cache_key(model_id, backend, normalization, scaler_type)

    if cache_key not in _models:
        async with _model_load_lock:
            if cache_key not in _models:
                logger.info(f"Loading anomaly model ({backend}): {model_id}")
                device = get_device()

                model = AnomalyModel(
                    model_id=model_id,
                    device=device,
                    backend=backend,
                    contamination=contamination,
                    threshold=threshold,
                    normalization=normalization,
                    scaler_type=scaler_type,
                    validation_split=validation_split,
                    patience=patience,
                    min_delta=min_delta,
                )

                await model.load()
                _models[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return _models.get(cache_key)


def _prepare_anomaly_data(
    data: list[list[float]] | list[dict],
    schema: dict[str, str] | None,
    cache_key: str,
    fit_mode: bool = False,
) -> list[list[float]]:
    """
    Prepare data for anomaly detection by encoding if needed.

    Args:
        data: Raw data (numeric arrays or dicts)
        schema: Feature encoding schema (required for dict data during fit)
        cache_key: Cache key for storing/retrieving encoder
        fit_mode: If True, fit the encoder on the data. If False, use existing encoder.

    Returns:
        Encoded numeric data as list of lists
    """
    # If data is already numeric, return as-is
    if not data:
        return []

    if isinstance(data[0], list):
        # Already numeric arrays
        return data

    # Dict-based data - need to encode
    if fit_mode:
        # Require schema for training
        if schema is None:
            raise HTTPException(
                status_code=400,
                detail="Schema is required when fitting with dict-based data. "
                "Example: schema = {'time_ms': 'numeric', 'user_agent': 'hash'}",
            )
        # Fit encoder on training data
        encoder = FeatureEncoder()
        encoder.fit(data, schema)
        _encoders[cache_key] = encoder
        logger.info(f"Fitted feature encoder for {cache_key} with schema: {schema}")
    else:
        # Use existing encoder (schema already learned during fit)
        if cache_key not in _encoders:
            raise HTTPException(
                status_code=400,
                detail=f"No encoder found for model '{cache_key}'. "
                "Train with /v1/anomaly/fit using dict data first, or pass schema.",
            )
        encoder = _encoders[cache_key]

    # Transform data
    encoded = encoder.transform(data)
    return encoded.tolist()


class AnomalyScoreRequest(PydanticBaseModel):
    """Anomaly scoring request.

    Supports two data formats:
    1. Numeric arrays: data = [[1.0, 2.0], [3.0, 4.0]]
    2. Dict-based with schema: data = [{"time_ms": 100, "user_agent": "curl"}]
       with schema = {"time_ms": "numeric", "user_agent": "hash"}

    Normalization methods:
    - standardization (default): Sigmoid 0-1 range, threshold ~0.5
    - zscore: Standard deviations from mean, threshold ~2.0-3.0
    - raw: Backend-native scores (varies by backend)
    """

    model: str = "default"  # Model identifier
    backend: str = "isolation_forest"  # isolation_forest, one_class_svm, local_outlier_factor, autoencoder
    data: list[list[float]] | list[dict]  # Data points (numeric arrays or dicts)
    schema: dict[str, str] | None = (
        None  # Feature encoding schema (required for dict data)
    )
    threshold: float | None = None  # Override default threshold
    normalization: str = "standardization"  # standardization, zscore, or raw
    scaler_type: str = "robust"  # Input data scaler: robust (default) or standard


class AnomalyFitRequest(PydanticBaseModel):
    """Anomaly model fitting request.

    Supports two data formats:
    1. Numeric arrays: data = [[1.0, 2.0], [3.0, 4.0]]
    2. Dict-based with schema: data = [{"time_ms": 100, "user_agent": "curl"}]
       with schema = {"time_ms": "numeric", "user_agent": "hash"}

    Schema encoding types:
    - numeric: Pass through as-is (int/float)
    - hash: MD5 hash to integer (good for high-cardinality like user_agent)
    - label: Category → integer mapping (learned from training data)
    - onehot: One-hot encoding (for low-cardinality categoricals)
    - binary: Boolean-like values (yes/no, true/false → 0/1)
    - frequency: Encode as occurrence frequency from training data

    Normalization methods:
    - standardization (default): Sigmoid 0-1 range, threshold ~0.5
    - zscore: Standard deviations from mean, threshold ~2.0-3.0
    - raw: Backend-native scores (varies by backend)
    """

    model: str = "default"  # Model identifier (for caching)
    backend: str = "isolation_forest"  # Backend to use
    data: list[list[float]] | list[dict] | None = None  # Training data (numeric arrays or dicts)
    training_file: str | None = None  # File reference ID from upload-training-data endpoint
    schema: dict[str, str] | None = (
        None  # Feature encoding schema (required for dict data)
    )
    contamination: float = 0.1  # Expected proportion of anomalies
    epochs: int = 100  # Training epochs (autoencoder only)
    batch_size: int = 32  # Batch size (autoencoder only)
    normalization: str = "standardization"  # standardization, zscore, or raw
    scaler_type: str = "robust"  # Input data scaler: robust (default) or standard
    # VAE / Early Stopping parameters (Phase 3)
    validation_split: float = 0.1  # Fraction of data for validation (autoencoder/vae)
    patience: int = 10  # Epochs without improvement before stopping
    min_delta: float = 1e-4  # Minimum change in validation loss for improvement


@app.post("/v1/anomaly/score")
async def score_anomalies(request: AnomalyScoreRequest):
    """
    Score data points for anomalies.

    Detects anomalies in data using various algorithms:
    - isolation_forest: Fast tree-based method, good general purpose
    - one_class_svm: Support vector machine for outlier detection
    - local_outlier_factor: Density-based, good for clustering anomalies
    - autoencoder: Neural network, best for complex patterns

    Note: Model must be fitted first via /v1/anomaly/fit or loaded from disk.

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest",
        "data": [[1.0, 2.0], [1.1, 2.1], [100.0, 200.0]],
        "threshold": 0.5
    }
    ```

    Response includes:
    - score: Anomaly score (0-1, higher = more anomalous)
    - is_anomaly: Boolean based on threshold
    - raw_score: Backend-specific raw score
    """
    try:
        cache_key = _make_anomaly_cache_key(
            request.model, request.backend, request.normalization, request.scaler_type
        )

        model = await load_anomaly(
            model_id=request.model,
            backend=request.backend,
            normalization=request.normalization,
            scaler_type=request.scaler_type,
        )

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/anomaly/fit first or load a pre-trained model.",
            )

        # Prepare data (encode if dict-based)
        prepared_data = _prepare_anomaly_data(
            data=request.data,
            schema=request.schema,
            cache_key=cache_key,
            fit_mode=False,  # Use existing encoder
        )

        # Score data
        results = await model.score(
            data=prepared_data,
            threshold=request.threshold,
        )

        # Format response
        data = [
            {
                "index": r.index,
                "score": r.score,
                "is_anomaly": r.is_anomaly,
                "raw_score": r.raw_score,
            }
            for r in results
        ]

        # Summary statistics
        anomaly_count = sum(1 for r in results if r.is_anomaly)

        return {
            "object": "list",
            "data": data,
            "total_count": len(data),
            "model": request.model,
            "backend": request.backend,
            "summary": {
                "total_points": len(data),
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_count / len(data) if data else 0,
                "threshold": request.threshold or model.threshold,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in score_anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# Global streaming data loader for file uploads
_streaming_loader = None


def get_streaming_loader():
    """Get or create the global streaming data loader."""
    global _streaming_loader
    if _streaming_loader is None:
        from utils.streaming_data import StreamingDataLoader
        _streaming_loader = StreamingDataLoader()
    return _streaming_loader


@app.post("/v1/anomaly/upload-training-data")
async def upload_training_data(
    file: UploadFile,
    skip_columns: str = Form(default=""),
):
    """
    Upload a training data file for streaming training.

    Upload CSV, JSON Lines, or Parquet files for memory-efficient training
    on large datasets. Returns a file reference ID that can be used with
    the /v1/anomaly/fit endpoint's training_file parameter.

    Supported formats:
    - CSV (.csv) - First row should be header
    - JSON Lines (.jsonl, .ndjson) - One JSON object per line
    - Parquet (.parquet) - Requires pyarrow

    Args:
        file: The uploaded file
        skip_columns: Comma-separated list of column names to skip (e.g., "timestamp,id")

    Returns:
        file_id: Reference ID for use in fit endpoint
        row_count: Number of rows in the file
        column_count: Number of columns (after skipping)
        columns: List of column names
    """
    import tempfile
    from pathlib import Path

    try:
        # Save uploaded file to temp location
        suffix = Path(file.filename).suffix.lower() if file.filename else ".csv"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Upload to streaming loader
        loader = get_streaming_loader()
        file_ref = await loader.upload_file(tmp_path, copy_to_temp=True)

        # Clean up the initial temp file
        import os
        os.unlink(tmp_path)

        # Handle skip_columns
        columns_to_use = file_ref.column_names
        if skip_columns:
            skip_list = [c.strip() for c in skip_columns.split(",")]
            columns_to_use = [c for c in columns_to_use if c not in skip_list]

        return {
            "object": "file_reference",
            "file_id": file_ref.file_id,
            "filename": file.filename,
            "file_type": file_ref.file_type,
            "row_count": file_ref.row_count,
            "column_count": len(columns_to_use),
            "columns": columns_to_use,
            "status": "ready",
        }

    except Exception as e:
        logger.error(f"Error uploading training data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/anomaly/fit")
async def fit_anomaly_detector(request: AnomalyFitRequest):
    """
    Fit an anomaly detector on training data.

    Train an anomaly detection model on data assumed to be mostly normal.
    The model learns what "normal" looks like and can then detect deviations.

    Backends:
    - isolation_forest: Fast, works well out of the box (recommended)
    - one_class_svm: Good for small datasets
    - local_outlier_factor: Density-based, good for clustering anomalies
    - autoencoder: Best for complex patterns, requires more data
    - vae: Variational Autoencoder with probabilistic scoring (ELBO-based)

    Early Stopping (autoencoder/vae only):
    - validation_split: Fraction of data held out for validation (default: 0.1)
    - patience: Epochs without improvement before stopping (default: 10)
    - min_delta: Minimum change in validation loss for improvement (default: 1e-4)

    Example request (VAE with early stopping):
    ```json
    {
        "model": "sensor-detector",
        "backend": "vae",
        "data": [[1.0, 2.0], [1.1, 2.1], [0.9, 1.9], ...],
        "contamination": 0.1,
        "epochs": 200,
        "patience": 10,
        "validation_split": 0.1
    }
    ```

    After fitting, use /v1/anomaly/score to detect anomalies in new data.
    """
    try:
        cache_key = _make_anomaly_cache_key(
            request.model, request.backend, request.normalization, request.scaler_type
        )

        # Validate that either data or training_file is provided
        if request.data is None and request.training_file is None:
            raise HTTPException(
                status_code=400,
                detail="Either 'data' or 'training_file' must be provided",
            )

        model = await load_anomaly(
            model_id=request.model,
            backend=request.backend,
            contamination=request.contamination,
            normalization=request.normalization,
            scaler_type=request.scaler_type,
            validation_split=request.validation_split,
            patience=request.patience,
            min_delta=request.min_delta,
        )

        # Fit from file or in-memory data
        if request.training_file:
            # Streaming training from file reference
            loader = get_streaming_loader()
            file_ref = loader.get_file_ref(request.training_file)
            if file_ref is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Training file not found: {request.training_file}",
                )

            result = await model.fit_from_file(
                file_ref=file_ref,
                batch_size=request.batch_size,
                epochs=request.epochs,
            )
        else:
            # Prepare data (encode if dict-based, and fit the encoder)
            prepared_data = _prepare_anomaly_data(
                data=request.data,
                schema=request.schema,
                cache_key=cache_key,
                fit_mode=True,  # Fit encoder on training data
            )

            # Fit model
            result = await model.fit(
                data=prepared_data,
                epochs=request.epochs,
                batch_size=request.batch_size,
            )

        # Include encoder info in response if used
        encoder_info = None
        if cache_key in _encoders:
            encoder = _encoders[cache_key]
            encoder_info = {
                "schema": encoder.schema.features if encoder.schema else {},
                "features": list(encoder.schema.features.keys())
                if encoder.schema
                else [],
            }

        # Auto-save model to prevent data loss on restart
        # This is mandatory - models must persist across server restarts
        await _auto_save_anomaly_model(
            model=model,
            model_name=request.model,
            backend=request.backend,
            cache_key=cache_key,
        )

        return {
            "object": "fit_result",
            "model": request.model,
            "backend": request.backend,
            "samples_fitted": result.samples_fitted,
            "training_time_ms": result.training_time_ms,
            "model_params": result.model_params,
            "encoder": encoder_info,
            "status": "fitted",
        }

    except Exception as e:
        logger.error(f"Error in fit_anomaly_detector: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/anomaly/detect")
async def detect_anomalies(request: AnomalyScoreRequest):
    """
    Detect anomalies in data (returns only anomalous points).

    Same as /v1/anomaly/score but filters to return only points
    classified as anomalies.

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest",
        "data": [[1.0, 2.0], [1.1, 2.1], [100.0, 200.0]],
        "threshold": 0.5
    }
    ```
    """
    try:
        cache_key = _make_anomaly_cache_key(
            request.model, request.backend, request.normalization, request.scaler_type
        )

        model = await load_anomaly(
            model_id=request.model,
            backend=request.backend,
            normalization=request.normalization,
            scaler_type=request.scaler_type,
        )

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/anomaly/fit first.",
            )

        # Prepare data (encode if dict-based)
        prepared_data = _prepare_anomaly_data(
            data=request.data,
            schema=request.schema,
            cache_key=cache_key,
            fit_mode=False,  # Use existing encoder
        )

        # Detect anomalies
        results = await model.detect(
            data=prepared_data,
            threshold=request.threshold,
        )

        # Format response
        data = [
            {
                "index": r.index,
                "score": r.score,
                "raw_score": r.raw_score,
            }
            for r in results
        ]

        return {
            "object": "list",
            "data": data,
            "total_count": len(data),
            "model": request.model,
            "backend": request.backend,
            "summary": {
                "anomalies_detected": len(data),
                "threshold": request.threshold or model.threshold,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# Model storage directory - uses standard LlamaFarm data directory structure
# ~/.llamafarm/models/anomaly/ (or LF_DATA_DIR/models/anomaly/)
# This is a controlled directory - users cannot specify arbitrary paths
_LF_DATA_DIR = Path(os.environ.get("LF_DATA_DIR", Path.home() / ".llamafarm"))
ANOMALY_MODELS_DIR = _LF_DATA_DIR / "models" / "anomaly"


class AnomalySaveRequest(PydanticBaseModel):
    """Request to save a fitted anomaly model."""

    model: str  # Model identifier (must be fitted)
    backend: str = "isolation_forest"
    normalization: str = (
        "standardization"  # Must match the normalization used during fit
    )
    # Note: filename is auto-generated from model name, no user control over paths


class AnomalyLoadRequest(PydanticBaseModel):
    """Request to load a pre-trained anomaly model."""

    model: str  # Model identifier to load/cache as
    backend: str = "isolation_forest"
    # Note: filename is derived from model name, no user control over paths


def _sanitize_model_name(name: str) -> str:
    """Sanitize model name to create a safe filename.

    Only allows alphanumeric characters, hyphens, and underscores.
    This prevents path traversal and ensures consistent naming.
    """
    return "".join(c for c in name if c.isalnum() or c in "-_")


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename, preserving extension dots.

    Only allows alphanumeric characters, hyphens, underscores, and dots.
    This prevents path traversal while allowing file extensions like .joblib
    """
    return "".join(c for c in name if c.isalnum() or c in "-_.")


def _validate_path_within_directory(path: Path, safe_dir: Path) -> Path:
    """Validate that a path is within the allowed directory.

    This is a security function to prevent path traversal attacks.
    Returns the resolved (absolute) path if valid.

    Raises:
        ValueError: If path is outside the allowed directory
    """
    resolved = path.resolve()
    safe_resolved = safe_dir.resolve()

    # Use Path.is_relative_to for Python 3.9+ compatibility
    try:
        resolved.relative_to(safe_resolved)
    except ValueError:
        raise ValueError(
            f"Security error: Path '{path}' resolves outside allowed directory"
        ) from None

    return resolved


def _get_model_path(model_name: str, backend: str) -> Path:
    """Get the path for a model file based on name and backend.

    The path is always within ANOMALY_MODELS_DIR - users cannot control it.
    """
    safe_name = _sanitize_model_name(model_name)
    safe_backend = _sanitize_model_name(backend)
    filename = f"{safe_name}_{safe_backend}"
    return ANOMALY_MODELS_DIR / filename


async def _auto_save_anomaly_model(
    model: BaseModel,
    model_name: str,
    backend: str,
    cache_key: str,
) -> None:
    """Auto-save anomaly model after fit to prevent data loss.

    Models are saved immediately after training to ensure they persist
    across server restarts without requiring an explicit /save call.

    Raises:
        Exception: If model save fails. This is intentionally not caught
            because models MUST be persisted - a failed save should fail
            the entire fit operation.
    """
    # Create models directory if needed
    ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate path from model name
    save_path = _get_model_path(model_name, backend)
    await model.save(str(save_path))

    # Determine actual saved file path for logging.
    # The model.save() method appends the appropriate extension based on backend:
    # - autoencoder backend: saves as PyTorch .pt file
    # - sklearn backends (isolation_forest, etc.): save as .joblib (preferred)
    #   or .pkl (legacy fallback for older scikit-learn versions)
    if backend == "autoencoder":
        actual_path = save_path.with_suffix(".pt")
    else:
        # sklearn-based backends prefer joblib for efficient array serialization,
        # but fall back to pickle (.pkl) for compatibility with older models
        actual_path = save_path.with_suffix(".joblib")
        if not actual_path.exists():
            actual_path = save_path.with_suffix(".pkl")

    logger.debug(f"Model saved to {actual_path}")

    # Save encoder if one exists for this model
    if cache_key in _encoders:
        encoder = _encoders[cache_key]
        encoder_save_path = save_path.parent / f"{save_path.name}_encoder.json"
        encoder.save(encoder_save_path)
        logger.debug(f"Feature encoder saved to {encoder_save_path}")


@app.post("/v1/anomaly/save")
async def save_anomaly_model(request: AnomalySaveRequest):
    """
    Save a fitted anomaly model to disk for production use.

    After fitting a model with /v1/anomaly/fit, save it to disk so it
    persists across server restarts.

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest"
    }
    ```

    Models are saved to ~/.llamafarm/models/anomaly/ with auto-generated
    filenames based on the model name and backend.
    """
    try:
        cache_key = _make_anomaly_cache_key(
            request.model, request.backend, request.normalization
        )

        if cache_key not in _models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' with backend '{request.backend}' and "
                f"normalization '{request.normalization}' not found in cache. "
                "Fit the model first with /v1/anomaly/fit",
            )

        model = _models[cache_key]

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/anomaly/fit first.",
            )

        # Create models directory if needed
        ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate path from model name (no user-controlled paths)
        save_path = _get_model_path(request.model, request.backend)
        await model.save(str(save_path))

        # Determine actual saved file
        if request.backend == "autoencoder":
            actual_path = save_path.with_suffix(".pt")
        else:
            actual_path = save_path.with_suffix(".joblib")
            if not actual_path.exists():
                actual_path = save_path.with_suffix(".pkl")

        # Save encoder if one exists for this model
        encoder_path = None
        if cache_key in _encoders:
            encoder = _encoders[cache_key]
            encoder_save_path = save_path.parent / f"{save_path.name}_encoder.json"
            encoder.save(encoder_save_path)
            encoder_path = str(encoder_save_path)
            logger.info(f"Saved feature encoder to {encoder_save_path}")

        return {
            "object": "save_result",
            "model": request.model,
            "backend": request.backend,
            "filename": actual_path.name,
            "path": str(actual_path),
            "encoder_path": encoder_path,
            "status": "saved",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in save_anomaly_model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/anomaly/load")
async def load_anomaly_model(request: AnomalyLoadRequest):
    """
    Load a pre-trained anomaly model from disk.

    Load a previously saved model for production inference without
    re-training. The model path is automatically determined from the
    model name and backend - no user control over file paths.

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest"
    }
    ```

    The model will be loaded from ~/.llamafarm/models/anomaly/ and cached
    for subsequent /v1/anomaly/score and /v1/anomaly/detect calls.
    """
    try:
        # Generate path from model name (no user-controlled paths)
        base_path = _get_model_path(request.model, request.backend)

        # Determine actual file (check for different extensions)
        model_path = None
        for ext in [".joblib", ".pkl", ".pt"]:
            candidate = base_path.with_suffix(ext)
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            available = (
                [f.name for f in ANOMALY_MODELS_DIR.glob("*") if f.is_file()]
                if ANOMALY_MODELS_DIR.exists()
                else []
            )
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' with backend '{request.backend}' not found. "
                f"Available models: {available}",
            )

        async with _model_load_lock:
            logger.info(f"Loading pre-trained anomaly model: {model_path}")
            device = get_device()

            model = AnomalyModel(
                model_id=str(model_path),  # Pass path as model_id for loading
                device=device,
                backend=request.backend,
            )

            await model.load()

            # Use the model's actual normalization (loaded from file) for the cache key
            cache_key = _make_anomaly_cache_key(
                request.model, request.backend, model.normalization
            )

            # Remove existing model from cache if present
            if cache_key in _models:
                await _models[cache_key].unload()
                del _models[cache_key]

            _models[cache_key] = model

        # Try to load encoder if one exists
        encoder_loaded = False
        encoder_schema = None
        # Derive encoder path from base path (same name pattern)
        encoder_path = base_path.parent / f"{base_path.name}_encoder.json"
        if encoder_path.exists():
            encoder = FeatureEncoder.load(encoder_path)
            _encoders[cache_key] = encoder
            encoder_loaded = True
            encoder_schema = encoder.schema
            logger.info(f"Loaded feature encoder from {encoder_path}")

        return {
            "object": "load_result",
            "model": request.model,
            "backend": request.backend,
            "normalization": model.normalization,
            "filename": model_path.name,
            "is_fitted": model.is_fitted,
            "threshold": model.threshold,
            "encoder_loaded": encoder_loaded,
            "encoder_schema": encoder_schema,
            "status": "loaded",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in load_anomaly_model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/anomaly/models")
async def list_anomaly_models():
    """
    List all saved anomaly models available for loading.

    Returns models saved in the ANOMALY_MODELS_DIR directory.

    Response includes:
    - filename: Name of the saved model file
    - size_bytes: File size
    - modified: Last modification timestamp
    - backend: Detected backend type (from file extension)
    """
    try:
        ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        models = []
        for path in ANOMALY_MODELS_DIR.glob("*"):
            if path.is_file() and path.suffix in (".pt", ".pkl", ".joblib"):
                stat = path.stat()

                # Detect backend from extension
                backend = "autoencoder" if path.suffix == ".pt" else "sklearn"

                models.append(
                    {
                        "filename": path.name,
                        "size_bytes": stat.st_size,
                        "modified": stat.st_mtime,
                        "backend": backend,
                    }
                )

        # Sort by modification time (newest first)
        models.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "object": "list",
            "data": models,
            "models_dir": str(ANOMALY_MODELS_DIR),
            "total": len(models),
        }

    except Exception as e:
        logger.error(f"Error in list_anomaly_models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/v1/anomaly/models/{filename}")
async def delete_anomaly_model(filename: str):
    """
    Delete a saved anomaly model.

    Removes the model file from disk. Does not affect cached models.
    """
    try:
        # Sanitize filename to prevent path traversal attacks
        # Use _sanitize_filename to preserve extension dots (.joblib)
        safe_filename = _sanitize_filename(filename)
        if not safe_filename:
            raise HTTPException(
                status_code=400,
                detail="Invalid filename",
            )

        # Also reject any path separators or special directory names
        if (
            "/" in filename
            or "\\" in filename
            or ".." in filename
            or safe_filename == "."
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename: path separators not allowed",
            )

        model_path = ANOMALY_MODELS_DIR / safe_filename

        # Validate the resolved path is still within the safe directory
        try:
            resolved_path = _validate_path_within_directory(
                model_path, ANOMALY_MODELS_DIR
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        if not resolved_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {safe_filename}",
            )

        resolved_path.unlink()

        return {
            "object": "delete_result",
            "filename": safe_filename,
            "deleted": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_anomaly_model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Text Classification Endpoints (SetFit-based few-shot learning)
# ============================================================================

# Classifier model storage directory
CLASSIFIER_MODELS_DIR = _LF_DATA_DIR / "models" / "classifier"


def _make_classifier_cache_key(model_name: str) -> str:
    """Create a cache key for classifier models."""
    return f"classifier:{model_name}"


def _get_classifier_path(model_name: str) -> Path:
    """Get the path for a classifier model directory.

    The path is always within CLASSIFIER_MODELS_DIR - users cannot control it.
    """
    safe_name = _sanitize_model_name(model_name)
    return CLASSIFIER_MODELS_DIR / safe_name


async def _auto_save_classifier_model(
    model: "ClassifierModel",
    model_name: str,
) -> dict[str, str | None]:
    """Auto-save classifier model after fit to prevent data loss.

    Models are saved immediately after training to ensure they persist
    across server restarts without requiring an explicit /save call.

    Returns:
        Dict with saved file path
    """
    try:
        # Create models directory if needed
        CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate path from model name
        save_path = _get_classifier_path(model_name)
        await model.save(str(save_path))

        logger.info(f"Auto-saved classifier model to {save_path}")
        return {"model_path": str(save_path)}

    except Exception as e:
        logger.warning(f"Auto-save failed (model still in memory): {e}")
        return {"model_path": None}


async def load_classifier(
    model_id: str,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> "ClassifierModel":
    """Load or get cached classifier model."""
    cache_key = _make_classifier_cache_key(model_id)

    if cache_key not in _classifiers:
        async with _model_load_lock:
            # Double-check after acquiring lock
            if cache_key not in _classifiers:
                logger.info(f"Loading classifier model: {model_id}")
                device = get_device()

                model = ClassifierModel(
                    model_id=model_id,
                    device=device,
                    base_model=base_model,
                )

                await model.load()
                _classifiers[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return _classifiers.get(cache_key)


class ClassifierFitRequest(PydanticBaseModel):
    """Request to fit a text classifier."""

    model: str  # Model identifier (for caching/saving)
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    training_data: list[dict]  # List of {"text": "...", "label": "..."}
    num_iterations: int = 20
    batch_size: int = 16


class ClassifierPredictRequest(PydanticBaseModel):
    """Request to classify texts."""

    model: str  # Model identifier (must be fitted or loaded)
    texts: list[str]


class ClassifierSaveRequest(PydanticBaseModel):
    """Request to save a fitted classifier."""

    model: str  # Model identifier (must be fitted)


class ClassifierLoadRequest(PydanticBaseModel):
    """Request to load a pre-trained classifier."""

    model: str  # Model identifier to load


@app.post("/v1/classifier/fit")
async def fit_classifier(request: ClassifierFitRequest):
    """
    Fit a text classifier using few-shot learning (SetFit).

    Train a classifier with as few as 8-16 examples per class.
    SetFit uses contrastive learning to fine-tune a sentence-transformer,
    then trains a small classification head.

    Example request:
    ```json
    {
        "model": "intent-classifier",
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "training_data": [
            {"text": "I need to book a flight", "label": "booking"},
            {"text": "Cancel my reservation", "label": "cancellation"},
            {"text": "What's the weather?", "label": "weather"}
        ],
        "num_iterations": 20
    }
    ```

    After fitting, use /v1/classifier/predict to classify new texts.
    """
    try:
        # Extract texts and labels from training data
        texts = [item["text"] for item in request.training_data]
        labels = [item["label"] for item in request.training_data]

        if len(texts) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 training examples required",
            )

        model = await load_classifier(
            model_id=request.model,
            base_model=request.base_model,
        )

        # Fit the classifier
        result = await model.fit(
            texts=texts,
            labels=labels,
            num_iterations=request.num_iterations,
            batch_size=request.batch_size,
        )

        # Auto-save model to prevent data loss on restart
        saved_paths = await _auto_save_classifier_model(
            model=model,
            model_name=request.model,
        )

        return {
            "object": "fit_result",
            "model": request.model,
            "base_model": result.base_model,
            "samples_fitted": result.samples_fitted,
            "num_classes": result.num_classes,
            "labels": result.labels,
            "training_time_ms": result.training_time_ms,
            "status": "fitted",
            "auto_saved": saved_paths["model_path"] is not None,
            "saved_path": saved_paths["model_path"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fit_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/classifier/predict")
async def predict_classifier(request: ClassifierPredictRequest):
    """
    Classify texts using a fitted classifier.

    Example request:
    ```json
    {
        "model": "intent-classifier",
        "texts": ["I want to cancel my trip", "Book me a hotel"]
    }
    ```

    Returns predictions with confidence scores for each text.
    """
    try:
        cache_key = _make_classifier_cache_key(request.model)

        # get() refreshes TTL automatically
        model = _classifiers.get(cache_key)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Classifier '{request.model}' not found. "
                "Fit with /v1/classifier/fit or load with /v1/classifier/load first.",
            )

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/classifier/fit first.",
            )

        results = await model.classify(request.texts)

        return {
            "object": "list",
            "data": [
                {
                    "text": r.text,
                    "label": r.label,
                    "score": r.score,
                    "all_scores": r.all_scores,
                }
                for r in results
            ],
            "total_count": len(results),
            "model": request.model,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/classifier/save")
async def save_classifier(request: ClassifierSaveRequest):
    """
    Save a fitted classifier to disk for production use.

    After fitting a model with /v1/classifier/fit, save it to disk so it
    persists across server restarts.

    Example request:
    ```json
    {
        "model": "intent-classifier"
    }
    ```

    Models are saved to ~/.llamafarm/models/classifier/ with auto-generated
    directory names based on the model name.
    """
    try:
        cache_key = _make_classifier_cache_key(request.model)

        if cache_key not in _classifiers:
            raise HTTPException(
                status_code=404,
                detail=f"Classifier '{request.model}' not found in cache. "
                "Fit the model first with /v1/classifier/fit",
            )

        model = _classifiers[cache_key]

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/classifier/fit first.",
            )

        # Create models directory if needed
        CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate path from model name (no user-controlled paths)
        save_path = _get_classifier_path(request.model)
        await model.save(str(save_path))

        return {
            "object": "save_result",
            "model": request.model,
            "path": str(save_path),
            "labels": model.labels,
            "status": "saved",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in save_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/classifier/load")
async def load_classifier_endpoint(request: ClassifierLoadRequest):
    """
    Load a pre-trained classifier from disk.

    Load a previously saved model for production inference without
    re-training. The model path is automatically determined from the
    model name - no user control over file paths.

    Example request:
    ```json
    {
        "model": "intent-classifier"
    }
    ```

    The model will be loaded from ~/.llamafarm/models/classifier/ and cached
    for subsequent /v1/classifier/predict calls.
    """
    try:
        # Generate path from model name (no user-controlled paths)
        model_path = _get_classifier_path(request.model)

        if not model_path.exists():
            available = (
                [f.name for f in CLASSIFIER_MODELS_DIR.glob("*") if f.is_dir()]
                if CLASSIFIER_MODELS_DIR.exists()
                else []
            )
            raise HTTPException(
                status_code=404,
                detail=f"Classifier '{request.model}' not found. "
                f"Available classifiers: {available}",
            )

        cache_key = _make_classifier_cache_key(request.model)

        # Remove existing model from cache if present
        if cache_key in _classifiers:
            existing = _classifiers.pop(cache_key)
            if existing:
                await existing.unload()

        async with _model_load_lock:
            logger.info(f"Loading pre-trained classifier: {model_path}")
            device = get_device()

            model = ClassifierModel(
                model_id=str(model_path),  # Pass path as model_id for loading
                device=device,
            )

            await model.load()
            _classifiers[cache_key] = model

        return {
            "object": "load_result",
            "model": request.model,
            "path": str(model_path),
            "is_fitted": model.is_fitted,
            "labels": model.labels,
            "num_classes": len(model.labels),
            "status": "loaded",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in load_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/classifier/models")
async def list_classifier_models():
    """
    List all saved classifier models available for loading.

    Returns models saved in the CLASSIFIER_MODELS_DIR directory.

    Response includes:
    - name: Name of the saved model
    - path: Full path to the model directory
    - labels: Class labels (if labels.txt exists)
    """
    try:
        CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        models = []
        for path in CLASSIFIER_MODELS_DIR.glob("*"):
            if path.is_dir():
                # Try to read labels
                labels = []
                labels_file = path / "labels.txt"
                if labels_file.exists():
                    labels = labels_file.read_text().strip().split("\n")

                stat = path.stat()
                models.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "labels": labels,
                        "num_classes": len(labels),
                        "modified": stat.st_mtime,
                    }
                )

        # Sort by modification time (newest first)
        models.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "object": "list",
            "data": models,
            "models_dir": str(CLASSIFIER_MODELS_DIR),
            "total": len(models),
        }

    except Exception as e:
        logger.error(f"Error in list_classifier_models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/v1/classifier/models/{model_name}")
async def delete_classifier_model(model_name: str):
    """
    Delete a saved classifier model.

    Removes the model directory from disk. Does not affect cached models.
    """
    try:
        # Reject any path separators to prevent traversal attempts
        if "/" in model_name or "\\" in model_name or ".." in model_name:
            raise HTTPException(
                status_code=400,
                detail="Invalid model name: path separators not allowed",
            )

        # _get_classifier_path already sanitizes via _sanitize_model_name
        model_path = _get_classifier_path(model_name)

        # Validate the resolved path is still within the safe directory
        try:
            resolved_path = _validate_path_within_directory(
                model_path, CLASSIFIER_MODELS_DIR
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        if not resolved_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Classifier model not found: {model_name}",
            )

        # Remove directory and contents
        import shutil

        shutil.rmtree(resolved_path)

        return {
            "object": "delete_result",
            "model": model_name,
            "deleted": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_classifier_model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Vision Endpoints (CLIP Zero-Shot Classification)
# ============================================================================


def _make_vision_cache_key(
    model_name: str,
) -> str:
    """Generate a cache key for a vision model.

    Args:
        model_name: HuggingFace model name or short name

    Returns:
        A unique cache key string for this vision model configuration
    """
    return f"vision:clip:{model_name}"


async def load_vision_model(
    model_name: str = "openai/clip-vit-base-patch32",
) -> CLIPVisionModel:
    """Load a CLIP vision model.

    Args:
        model_name: HuggingFace model name (default: openai/clip-vit-base-patch32)

    Returns:
        Loaded CLIPVisionModel instance
    """
    cache_key = _make_vision_cache_key(model_name)

    if cache_key not in _vision_models:
        async with _model_load_lock:
            if cache_key not in _vision_models:
                logger.info(f"Loading CLIP vision model: {model_name}")
                device = get_device()

                model = CLIPVisionModel(
                    model_id=cache_key,
                    device=device,
                    hf_model_name=model_name,
                )

                await model.load()
                _vision_models[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return _vision_models.get(cache_key)


class ZeroShotClassifyRequest(PydanticBaseModel):
    """Zero-shot image classification request."""

    image: str  # Base64-encoded image or file path
    labels: list[str]  # Labels to classify against
    model: str = "openai/clip-vit-base-patch32"  # CLIP model to use


class ZeroShotClassifyBatchRequest(PydanticBaseModel):
    """Zero-shot image classification batch request."""

    images: list[str]  # List of base64-encoded images or file paths
    labels: list[str]  # Labels to classify against
    model: str = "openai/clip-vit-base-patch32"  # CLIP model to use


@app.post("/v1/vision/classify-zero-shot")
async def classify_zero_shot(request: ZeroShotClassifyRequest):
    """
    Zero-shot image classification using CLIP.

    Classify images into arbitrary categories without training. Simply provide
    a list of text labels and the model will predict probabilities for each.

    Example request:
    ```json
    {
        "image": "<base64 encoded image>",
        "labels": ["cat", "dog", "bird"],
        "model": "openai/clip-vit-base-patch32"
    }
    ```

    Response:
    ```json
    {
        "object": "classification",
        "label": "cat",
        "score": 0.87,
        "all_scores": {"cat": 0.87, "dog": 0.10, "bird": 0.03},
        "model": "openai/clip-vit-base-patch32"
    }
    ```

    Supported image formats: PNG, JPEG, WebP, GIF, BMP
    Image can be provided as:
    - Base64-encoded string
    - File path (for local files)
    """
    try:
        if not request.labels:
            raise HTTPException(
                status_code=400,
                detail="At least one label is required",
            )

        model = await load_vision_model(request.model)
        result = await model.classify_zero_shot(request.image, request.labels)

        return {
            "object": "classification",
            "label": result["label"],
            "score": result["score"],
            "all_scores": result["all_scores"],
            "model": request.model,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in classify_zero_shot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/vision/classify-zero-shot/batch")
async def classify_zero_shot_batch(request: ZeroShotClassifyBatchRequest):
    """
    Batch zero-shot image classification using CLIP.

    Classify multiple images into arbitrary categories without training.
    More efficient than calling single endpoint multiple times.

    Example request:
    ```json
    {
        "images": ["<base64 image 1>", "<base64 image 2>"],
        "labels": ["cat", "dog", "bird"],
        "model": "openai/clip-vit-base-patch32"
    }
    ```

    Response:
    ```json
    {
        "object": "list",
        "data": [
            {"label": "cat", "score": 0.87, "all_scores": {...}},
            {"label": "dog", "score": 0.91, "all_scores": {...}}
        ],
        "model": "openai/clip-vit-base-patch32"
    }
    ```
    """
    try:
        if not request.labels:
            raise HTTPException(
                status_code=400,
                detail="At least one label is required",
            )

        if not request.images:
            raise HTTPException(
                status_code=400,
                detail="At least one image is required",
            )

        model = await load_vision_model(request.model)
        results = await model.classify_zero_shot_batch(request.images, request.labels)

        return {
            "object": "list",
            "data": results,
            "model": request.model,
            "total_count": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in classify_zero_shot_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Language Detection Endpoints
# ============================================================================


def _make_lang_detection_cache_key(model_name: str) -> str:
    """Create a cache key for language detection models."""
    return f"lang_detect:{model_name}"


async def load_lang_detection_model(
    model_name: str = "papluca/xlm-roberta-base-language-detection",
) -> LanguageDetectionModel:
    """Load a language detection model.

    Args:
        model_name: HuggingFace model name

    Returns:
        Loaded LanguageDetectionModel instance
    """
    cache_key = _make_lang_detection_cache_key(model_name)

    if cache_key not in _lang_detection_models:
        async with _model_load_lock:
            if cache_key not in _lang_detection_models:
                logger.info(f"Loading language detection model: {model_name}")
                device = get_device()

                model = LanguageDetectionModel(
                    model_id=cache_key,
                    device=device,
                    hf_model_name=model_name,
                )

                await model.load()
                _lang_detection_models[cache_key] = model

    return _lang_detection_models.get(cache_key)


class LanguageDetectRequest(PydanticBaseModel):
    """Language detection request."""

    text: str  # Single text to detect
    top_k: int = 5  # Number of top predictions


class LanguageDetectBatchRequest(PydanticBaseModel):
    """Language detection batch request."""

    texts: list[str]  # List of texts to detect
    top_k: int = 1  # Number of top predictions per text


@app.post("/v1/text/language")
async def detect_language(request: LanguageDetectRequest):
    """
    Detect the language of a text.

    Uses XLM-RoBERTa fine-tuned for language detection. Supports 20 languages:
    Arabic, Bulgarian, German, Greek, English, Spanish, French, Hindi, Italian,
    Japanese, Dutch, Polish, Portuguese, Russian, Swahili, Thai, Turkish,
    Urdu, Vietnamese, Chinese.

    Example request:
    ```json
    {
        "text": "Hello, how are you today?",
        "top_k": 5
    }
    ```

    Response:
    ```json
    {
        "object": "language_detection",
        "language": "en",
        "language_name": "English",
        "confidence": 0.99,
        "all_scores": {"en": 0.99, "de": 0.005, ...}
    }
    ```
    """
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty",
            )

        model = await load_lang_detection_model()
        result = await model.detect(request.text, top_k=request.top_k)

        return {
            "object": "language_detection",
            "language": result["language"],
            "language_name": result["language_name"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_language: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/text/language/batch")
async def detect_language_batch(request: LanguageDetectBatchRequest):
    """
    Detect the language of multiple texts.

    More efficient than calling single endpoint multiple times.

    Example request:
    ```json
    {
        "texts": ["Hello world", "Bonjour le monde", "Hallo Welt"],
        "top_k": 1
    }
    ```

    Response:
    ```json
    {
        "object": "list",
        "data": [
            {"language": "en", "language_name": "English", "confidence": 0.99, ...},
            {"language": "fr", "language_name": "French", "confidence": 0.98, ...},
            {"language": "de", "language_name": "German", "confidence": 0.97, ...}
        ],
        "total_count": 3
    }
    ```
    """
    try:
        if not request.texts:
            raise HTTPException(
                status_code=400,
                detail="At least one text is required",
            )

        model = await load_lang_detection_model()
        results = await model.detect_batch(request.texts, top_k=request.top_k)

        return {
            "object": "list",
            "data": [
                {
                    "language": r["language"],
                    "language_name": r["language_name"],
                    "confidence": r["confidence"],
                    "all_scores": r["all_scores"],
                }
                for r in results
            ],
            "total_count": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_language_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Keyword Extraction Endpoints
# ============================================================================


# Cache for keyword extractor (reuses encoder models)
_keyword_extractor = None
_keyword_extractor_lock = asyncio.Lock()


async def get_keyword_extractor():
    """Get or create keyword extractor with encoder model."""
    global _keyword_extractor

    if _keyword_extractor is None:
        async with _keyword_extractor_lock:
            if _keyword_extractor is None:
                from utils.keyword_extractor import KeywordExtractor

                # Try to get an encoder model for better results
                try:
                    # Use a small, fast encoder model
                    encoder = await get_or_load_encoder(
                        "sentence-transformers/all-MiniLM-L6-v2"
                    )
                    _keyword_extractor = KeywordExtractor(encoder_model=encoder)
                    logger.info("Keyword extractor initialized with encoder model")
                except Exception as e:
                    logger.warning(f"Could not load encoder for keywords, using frequency fallback: {e}")
                    _keyword_extractor = KeywordExtractor(encoder_model=None)

    return _keyword_extractor


class KeywordExtractRequest(PydanticBaseModel):
    """Keyword extraction request."""

    text: str  # Text to extract keywords from
    top_k: int = 10  # Number of keywords to return
    diversity: float = 0.5  # Diversity parameter (0-1)
    ngram_range: list[int] = [1, 3]  # Min and max n-gram size


class KeywordExtractBatchRequest(PydanticBaseModel):
    """Keyword extraction batch request."""

    texts: list[str]  # List of texts
    top_k: int = 10  # Keywords per text
    diversity: float = 0.5
    ngram_range: list[int] = [1, 3]


@app.post("/v1/text/keywords")
async def extract_keywords(request: KeywordExtractRequest):
    """
    Extract keywords and keyphrases from text.

    Uses sentence embeddings to find the most relevant n-grams in the text.
    Supports diversity parameter to avoid redundant keywords.

    Example request:
    ```json
    {
        "text": "Machine learning is a subset of artificial intelligence...",
        "top_k": 10,
        "diversity": 0.5,
        "ngram_range": [1, 3]
    }
    ```

    Response:
    ```json
    {
        "object": "keyword_extraction",
        "keywords": [
            {"keyword": "machine learning", "score": 0.87},
            {"keyword": "artificial intelligence", "score": 0.82}
        ],
        "count": 10
    }
    ```
    """
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty",
            )

        extractor = await get_keyword_extractor()

        # Run extraction in executor to not block
        loop = asyncio.get_running_loop()
        keywords = await loop.run_in_executor(
            None,
            lambda: extractor.extract(
                request.text,
                top_k=request.top_k,
                ngram_range=tuple(request.ngram_range),
                diversity=request.diversity,
            )
        )

        return {
            "object": "keyword_extraction",
            "keywords": keywords,
            "count": len(keywords),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_keywords: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/text/keywords/batch")
async def extract_keywords_batch(request: KeywordExtractBatchRequest):
    """
    Extract keywords from multiple texts.

    Example request:
    ```json
    {
        "texts": ["First document...", "Second document..."],
        "top_k": 5
    }
    ```

    Response:
    ```json
    {
        "object": "list",
        "data": [
            {"keywords": [...], "count": 5},
            {"keywords": [...], "count": 5}
        ],
        "total_count": 2
    }
    ```
    """
    try:
        if not request.texts:
            raise HTTPException(
                status_code=400,
                detail="At least one text is required",
            )

        extractor = await get_keyword_extractor()
        loop = asyncio.get_running_loop()

        results = []
        for text in request.texts:
            keywords = await loop.run_in_executor(
                None,
                lambda t=text: extractor.extract(
                    t,
                    top_k=request.top_k,
                    ngram_range=tuple(request.ngram_range),
                    diversity=request.diversity,
                )
            )
            results.append({
                "keywords": keywords,
                "count": len(keywords),
            })

        return {
            "object": "list",
            "data": results,
            "total_count": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_keywords_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# PII Detection and Redaction Endpoints
# ============================================================================


def _make_pii_cache_key(model_name: str) -> str:
    """Create a cache key for PII models."""
    return f"pii:{model_name}"


async def load_pii_model(
    model_name: str = "urchade/gliner_small-v2.1",
) -> PIIModel:
    """Load a PII detection model.

    Args:
        model_name: HuggingFace GLiNER model name

    Returns:
        Loaded PIIModel instance
    """
    cache_key = _make_pii_cache_key(model_name)

    if cache_key not in _pii_models:
        async with _model_load_lock:
            if cache_key not in _pii_models:
                logger.info(f"Loading PII detection model: {model_name}")
                device = get_device()

                model = PIIModel(
                    model_id=cache_key,
                    device=device,
                    hf_model_name=model_name,
                )

                await model.load()
                _pii_models[cache_key] = model

    return _pii_models.get(cache_key)


class PIIDetectRequest(PydanticBaseModel):
    """PII detection request."""

    text: str  # Text to analyze
    entity_types: list[str] | None = None  # Custom entity types (default: standard PII)
    threshold: float = 0.5  # Detection confidence threshold
    use_regex: bool = True  # Also use regex patterns


class PIIRedactRequest(PydanticBaseModel):
    """PII redaction request."""

    text: str  # Text to redact
    entity_types: list[str] | None = None  # Entity types to redact
    replacement: str = "[REDACTED]"  # Default replacement
    replacement_map: dict[str, str] | None = None  # Per-type replacements
    threshold: float = 0.5
    use_regex: bool = True


@app.post("/v1/text/pii-detect")
async def detect_pii(request: PIIDetectRequest):
    """
    Detect PII (Personally Identifiable Information) in text.

    Uses GLiNER for zero-shot entity detection plus regex patterns for
    common PII formats. Supports custom entity types.

    Default entity types detected:
    - person, email, phone number, social security number
    - credit card number, address, date of birth
    - passport number, driver license, bank account, ip address

    Example request:
    ```json
    {
        "text": "Contact John at john@email.com or 555-123-4567",
        "threshold": 0.5
    }
    ```

    Response:
    ```json
    {
        "object": "pii_detection",
        "entities": [
            {"text": "John", "label": "person", "start": 8, "end": 12, "score": 0.95},
            {"text": "john@email.com", "label": "email", "start": 16, "end": 30, "score": 1.0}
        ],
        "entity_count": 2
    }
    ```
    """
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty",
            )

        model = await load_pii_model()
        entities = await model.detect(
            request.text,
            entity_types=request.entity_types,
            threshold=request.threshold,
            use_regex=request.use_regex,
        )

        return {
            "object": "pii_detection",
            "entities": entities,
            "entity_count": len(entities),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_pii: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/text/pii-redact")
async def redact_pii(request: PIIRedactRequest):
    """
    Detect and redact PII from text.

    Returns the text with PII replaced by configurable replacement strings.
    Supports per-entity-type replacement patterns.

    Example request:
    ```json
    {
        "text": "Contact John at john@email.com",
        "replacement": "[REDACTED]",
        "replacement_map": {"email": "[EMAIL]", "person": "[NAME]"}
    }
    ```

    Response:
    ```json
    {
        "object": "pii_redaction",
        "redacted_text": "Contact [NAME] at [EMAIL]",
        "entities": [...],
        "entity_count": 2
    }
    ```
    """
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty",
            )

        model = await load_pii_model()
        result = await model.redact(
            request.text,
            entity_types=request.entity_types,
            replacement=request.replacement,
            replacement_map=request.replacement_map,
            threshold=request.threshold,
            use_regex=request.use_regex,
        )

        return {
            "object": "pii_redaction",
            "redacted_text": result["redacted_text"],
            "entities": result["entities"],
            "entity_count": result["entity_count"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in redact_pii: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Object Detection Endpoint (YOLOS)
# =============================================================================


class ObjectDetectionRequest(PydanticBaseModel):
    """Object detection request."""

    image: str  # Base64-encoded image or file path
    threshold: float = 0.5  # Confidence threshold (0-1)
    labels: list[str] | None = None  # Filter to specific object labels
    model: str = "hustvl/yolos-tiny"  # HuggingFace model name


class ObjectDetectionBatchRequest(PydanticBaseModel):
    """Batch object detection request."""

    images: list[str]  # List of base64-encoded images or file paths
    threshold: float = 0.5
    labels: list[str] | None = None
    model: str = "hustvl/yolos-tiny"


def _make_object_detection_cache_key(model_name: str) -> str:
    """Create a cache key for object detection models."""
    return f"object_detection:{model_name}"


async def load_object_detection_model(
    model_name: str = "hustvl/yolos-tiny",
) -> ObjectDetectionModel:
    """Load or retrieve cached object detection model."""
    cache_key = _make_object_detection_cache_key(model_name)

    if cache_key not in _object_detection_models:
        async with _model_load_lock:
            if cache_key not in _object_detection_models:
                logger.info(f"Loading object detection model: {model_name}")
                device = get_device()

                model = ObjectDetectionModel(
                    model_id=cache_key,
                    device=device,
                    hf_model_name=model_name,
                )

                await model.load()
                _object_detection_models[cache_key] = model

    return _object_detection_models.get(cache_key)


@app.post("/v1/vision/detect-objects")
async def detect_objects(request: ObjectDetectionRequest):
    """
    Detect objects in an image using YOLOS.

    YOLOS (You Only Look at One Sequence) is a Vision Transformer-based
    object detector that identifies objects and their bounding boxes.

    The model detects 80 COCO classes including:
    person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
    traffic light, fire hydrant, stop sign, bench, bird, cat, dog, horse,
    sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag,
    tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat,
    baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass,
    cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli,
    carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed,
    dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
    microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
    teddy bear, hair drier, toothbrush

    Example request:
    ```json
    {
        "image": "<base64-encoded-image>",
        "threshold": 0.5,
        "labels": ["person", "car", "dog"]
    }
    ```

    Response:
    ```json
    {
        "object": "object_detection",
        "objects": [
            {"label": "person", "score": 0.95, "box": {"x1": 10, "y1": 20, "x2": 100, "y2": 200}},
            {"label": "car", "score": 0.88, "box": {"x1": 150, "y1": 50, "x2": 300, "y2": 180}}
        ],
        "count": 2,
        "image_size": {"width": 640, "height": 480}
    }
    ```
    """
    try:
        if not request.image:
            raise HTTPException(
                status_code=400,
                detail="Image data is required",
            )

        model = await load_object_detection_model(request.model)
        result = await model.detect(
            request.image,
            threshold=request.threshold,
            labels=request.labels,
        )

        return {
            "object": "object_detection",
            "objects": result["objects"],
            "count": result["count"],
            "image_size": result["image_size"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_objects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/vision/detect-objects/batch")
async def detect_objects_batch(request: ObjectDetectionBatchRequest):
    """
    Detect objects in multiple images.

    Returns detection results for each image in the batch.
    """
    try:
        if not request.images:
            raise HTTPException(
                status_code=400,
                detail="At least one image is required",
            )

        model = await load_object_detection_model(request.model)
        results = await model.detect_batch(
            request.images,
            threshold=request.threshold,
            labels=request.labels,
        )

        return {
            "object": "object_detection_batch",
            "results": [
                {
                    "objects": r["objects"],
                    "count": r["count"],
                    "image_size": r["image_size"],
                }
                for r in results
            ],
            "total_images": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_objects_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Background Removal Endpoint (RMBG)
# =============================================================================


class BackgroundRemovalRequest(PydanticBaseModel):
    """Background removal request."""

    image: str  # Base64-encoded image or file path
    return_mask: bool = False  # Whether to also return the alpha mask
    model: str = "briaai/RMBG-1.4"  # HuggingFace model name


class BackgroundRemovalBatchRequest(PydanticBaseModel):
    """Batch background removal request."""

    images: list[str]  # List of base64-encoded images or file paths
    return_mask: bool = False
    model: str = "briaai/RMBG-1.4"


def _make_background_removal_cache_key(model_name: str) -> str:
    """Create a cache key for background removal models."""
    return f"background_removal:{model_name}"


async def load_background_removal_model(
    model_name: str = "briaai/RMBG-1.4",
) -> BackgroundRemovalModel:
    """Load or retrieve cached background removal model."""
    cache_key = _make_background_removal_cache_key(model_name)

    if cache_key not in _background_removal_models:
        async with _model_load_lock:
            if cache_key not in _background_removal_models:
                logger.info(f"Loading background removal model: {model_name}")
                device = get_device()

                model = BackgroundRemovalModel(
                    model_id=cache_key,
                    device=device,
                    hf_model_name=model_name,
                )

                await model.load()
                _background_removal_models[cache_key] = model

    return _background_removal_models.get(cache_key)


@app.post("/v1/vision/segment")
async def remove_background(request: BackgroundRemovalRequest):
    """
    Remove background from an image using RMBG.

    RMBG (Remove Background) is a state-of-the-art background removal model
    that produces high-quality alpha masks for separating foreground from background.

    Returns a PNG image with transparent background (alpha channel).

    Example request:
    ```json
    {
        "image": "<base64-encoded-image>",
        "return_mask": false
    }
    ```

    Response:
    ```json
    {
        "object": "background_removal",
        "image": "<base64-encoded-PNG-with-alpha>",
        "width": 640,
        "height": 480
    }
    ```

    With `return_mask: true`:
    ```json
    {
        "object": "background_removal",
        "image": "<base64-encoded-PNG-with-alpha>",
        "mask": "<base64-encoded-grayscale-mask>",
        "width": 640,
        "height": 480
    }
    ```
    """
    try:
        if not request.image:
            raise HTTPException(
                status_code=400,
                detail="Image data is required",
            )

        model = await load_background_removal_model(request.model)
        result = await model.remove_background(
            request.image,
            return_mask=request.return_mask,
        )

        response = {
            "object": "background_removal",
            "image": result["image"],
            "width": result["width"],
            "height": result["height"],
        }

        if request.return_mask and "mask" in result:
            response["mask"] = result["mask"]

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in remove_background: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/vision/segment/batch")
async def remove_background_batch(request: BackgroundRemovalBatchRequest):
    """
    Remove background from multiple images.

    Returns processed images with transparent backgrounds.
    """
    try:
        if not request.images:
            raise HTTPException(
                status_code=400,
                detail="At least one image is required",
            )

        model = await load_background_removal_model(request.model)
        results = await model.remove_background_batch(
            request.images,
            return_mask=request.return_mask,
        )

        return {
            "object": "background_removal_batch",
            "results": [
                {
                    "image": r["image"],
                    "width": r["width"],
                    "height": r["height"],
                    **({"mask": r["mask"]} if request.return_mask and "mask" in r else {}),
                }
                for r in results
            ],
            "total_images": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in remove_background_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Time-Series Forecasting Endpoint (Chronos-Bolt)
# =============================================================================


class TimeSeriesForecastRequest(PydanticBaseModel):
    """Time-series forecast request."""

    values: list[float]  # Historical time-series values
    horizon: int = 7  # Number of future steps to forecast
    quantiles: list[float] | None = None  # Quantile levels (default: [0.1, 0.5, 0.9])
    num_samples: int = 20  # Number of samples for uncertainty
    model: str = "amazon/chronos-t5-small"  # HuggingFace model name


class TimeSeriesForecastBatchRequest(PydanticBaseModel):
    """Batch time-series forecast request."""

    series: list[list[float]]  # List of time-series
    horizon: int = 7
    quantiles: list[float] | None = None
    num_samples: int = 20
    model: str = "amazon/chronos-t5-small"


class ChangePointRequest(PydanticBaseModel):
    """Change point detection request."""

    values: list[float]  # Time-series values
    n_changepoints: int | None = None  # Exact number (if known)
    penalty: float | None = None  # Penalty for regularization (higher = fewer points)
    algorithm: str = "pelt"  # pelt, binseg, window, bottomup
    model: str = "rbf"  # l1, l2, rbf, normal, ar
    min_size: int = 2  # Minimum segment size


class ChangePointBatchRequest(PydanticBaseModel):
    """Batch change point detection request."""

    series: list[list[float]]  # List of time-series
    n_changepoints: int | None = None
    penalty: float | None = None
    algorithm: str = "pelt"
    model: str = "rbf"
    min_size: int = 2


class TableQARequest(PydanticBaseModel):
    """Table question answering request."""

    table: dict[str, Any]  # {"columns": [...], "rows": [[...], ...]}
    question: str  # Natural language question
    model: str = "google/tapas-base-finetuned-wtq"


class TableQABatchRequest(PydanticBaseModel):
    """Batch table question answering request."""

    table: dict[str, Any]  # Same table for all questions
    questions: list[str]  # Multiple questions
    model: str = "google/tapas-base-finetuned-wtq"


def _make_timeseries_cache_key(model_name: str) -> str:
    """Create a cache key for time-series models."""
    return f"timeseries:{model_name}"


async def load_timeseries_model(
    model_name: str = "amazon/chronos-t5-small",
) -> TimeSeriesModel:
    """Load or retrieve cached time-series model."""
    cache_key = _make_timeseries_cache_key(model_name)

    if cache_key not in _timeseries_models:
        async with _model_load_lock:
            if cache_key not in _timeseries_models:
                logger.info(f"Loading time-series model: {model_name}")
                device = get_device()

                model = TimeSeriesModel(
                    model_id=cache_key,
                    device=device,
                    hf_model_name=model_name,
                )

                await model.load()
                _timeseries_models[cache_key] = model

    return _timeseries_models.get(cache_key)


@app.post("/v1/timeseries/forecast")
async def forecast_timeseries(request: TimeSeriesForecastRequest):
    """
    Generate time-series forecasts using Chronos-Bolt.

    Chronos-Bolt is a transformer-based time-series forecasting model that
    produces probabilistic forecasts with confidence intervals.

    Example request:
    ```json
    {
        "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "horizon": 7,
        "quantiles": [0.1, 0.5, 0.9]
    }
    ```

    Response:
    ```json
    {
        "object": "timeseries_forecast",
        "forecasts": [
            {"step": 1, "point": 8.0, "lower": 7.5, "upper": 8.5},
            {"step": 2, "point": 9.0, "lower": 8.3, "upper": 9.7},
            ...
        ],
        "horizon": 7,
        "input_length": 7
    }
    ```
    """
    try:
        if len(request.values) < 3:
            raise HTTPException(
                status_code=400,
                detail="At least 3 historical values are required",
            )

        if request.horizon < 1:
            raise HTTPException(
                status_code=400,
                detail="Horizon must be at least 1",
            )

        model = await load_timeseries_model(request.model)
        result = await model.forecast(
            request.values,
            horizon=request.horizon,
            quantiles=request.quantiles,
            num_samples=request.num_samples,
        )

        return {
            "object": "timeseries_forecast",
            "forecasts": result["forecasts"],
            "horizon": result["horizon"],
            "input_length": result["input_length"],
            "quantiles": result["quantiles"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in forecast_timeseries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/timeseries/forecast/batch")
async def forecast_timeseries_batch(request: TimeSeriesForecastBatchRequest):
    """
    Generate forecasts for multiple time-series.

    Returns forecasts for each series in the batch.
    """
    try:
        if not request.series:
            raise HTTPException(
                status_code=400,
                detail="At least one time-series is required",
            )

        for i, s in enumerate(request.series):
            if len(s) < 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"Series {i} has fewer than 3 values",
                )

        model = await load_timeseries_model(request.model)
        results = await model.forecast_batch(
            request.series,
            horizon=request.horizon,
            quantiles=request.quantiles,
            num_samples=request.num_samples,
        )

        return {
            "object": "timeseries_forecast_batch",
            "results": results,
            "total_series": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in forecast_timeseries_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# CHANGE POINT DETECTION ENDPOINTS
# =============================================================================


@app.post("/v1/timeseries/changepoints")
async def detect_changepoints(request: ChangePointRequest):
    """
    Detect change points in a time-series using ruptures.

    Change points are locations where the statistical properties of the
    time-series (mean, variance, trend) change significantly.

    Algorithms:
    - pelt: Optimal algorithm with linear complexity (default)
    - binseg: Binary segmentation (fast but approximate)
    - window: Sliding window (good for trend changes)
    - bottomup: Bottom-up segmentation

    Models (cost functions):
    - rbf: Radial basis function (default, general purpose)
    - l1: L1 norm (robust to outliers)
    - l2: L2 norm (sensitive to mean shifts)
    - normal: Normal distribution
    - ar: Autoregressive model

    Example request:
    ```json
    {
        "values": [1, 1, 1, 1, 5, 5, 5, 5, 2, 2, 2, 2],
        "algorithm": "pelt",
        "model": "rbf"
    }
    ```

    Response:
    ```json
    {
        "object": "changepoint_detection",
        "change_points": [4, 8],
        "n_segments": 3,
        "segment_boundaries": [
            {"start": 0, "end": 4},
            {"start": 4, "end": 8},
            {"start": 8, "end": 12}
        ]
    }
    ```
    """
    try:
        from utils.changepoint_detector import ChangePointDetector, SUPPORTED_ALGORITHMS, SUPPORTED_MODELS

        if request.algorithm.lower() not in SUPPORTED_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {request.algorithm}. Choose from {SUPPORTED_ALGORITHMS}",
            )

        if request.model.lower() not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {request.model}. Choose from {SUPPORTED_MODELS}",
            )

        if len(request.values) < request.min_size * 2:
            raise HTTPException(
                status_code=400,
                detail=f"Signal too short. Need at least {request.min_size * 2} points.",
            )

        detector = ChangePointDetector(
            algorithm=request.algorithm,
            model=request.model,
            min_size=request.min_size,
        )

        result = detector.detect(
            request.values,
            n_changepoints=request.n_changepoints,
            penalty=request.penalty,
        )

        return {
            "object": "changepoint_detection",
            "change_points": result["change_points"],
            "n_segments": result["n_segments"],
            "segment_boundaries": result["segment_boundaries"],
            "signal_length": result["signal_length"],
            "algorithm": result["algorithm"],
            "model": result["model"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_changepoints: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/timeseries/changepoints/batch")
async def detect_changepoints_batch(request: ChangePointBatchRequest):
    """
    Detect change points in multiple time-series.

    Returns change point results for each series in the batch.
    """
    try:
        from utils.changepoint_detector import ChangePointDetector, SUPPORTED_ALGORITHMS, SUPPORTED_MODELS

        if not request.series:
            raise HTTPException(
                status_code=400,
                detail="At least one time-series is required",
            )

        if request.algorithm.lower() not in SUPPORTED_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {request.algorithm}. Choose from {SUPPORTED_ALGORITHMS}",
            )

        if request.model.lower() not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {request.model}. Choose from {SUPPORTED_MODELS}",
            )

        for i, s in enumerate(request.series):
            if len(s) < request.min_size * 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Series {i} too short. Need at least {request.min_size * 2} points.",
                )

        detector = ChangePointDetector(
            algorithm=request.algorithm,
            model=request.model,
            min_size=request.min_size,
        )

        results = detector.detect_batch(
            request.series,
            n_changepoints=request.n_changepoints,
            penalty=request.penalty,
        )

        return {
            "object": "changepoint_detection_batch",
            "results": results,
            "total_series": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_changepoints_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# TABLE QUESTION ANSWERING ENDPOINTS
# =============================================================================

_table_qa_models: dict[str, Any] = {}


def _make_table_qa_cache_key(model_name: str) -> str:
    """Create a cache key for table QA models."""
    return f"tableqa:{model_name}"


async def load_table_qa_model(
    model_name: str = "google/tapas-base-finetuned-wtq",
) -> Any:
    """Load or retrieve cached table QA model."""
    from models.table_qa_model import TableQAModel

    cache_key = _make_table_qa_cache_key(model_name)

    if cache_key not in _table_qa_models:
        async with _model_load_lock:
            if cache_key not in _table_qa_models:
                logger.info(f"Loading table QA model: {model_name}")
                device = get_device()

                model = TableQAModel(
                    model_id=cache_key,
                    device=device,
                    hf_model_name=model_name,
                )

                await model.load()
                _table_qa_models[cache_key] = model

    return _table_qa_models.get(cache_key)


@app.post("/v1/analysis/table-qa")
async def table_question_answering(request: TableQARequest):
    """
    Answer questions about tabular data using TAPAS.

    TAPAS (Table Parser) can answer natural language questions about tables,
    including questions that require aggregation (sum, average, count).

    Example request:
    ```json
    {
        "table": {
            "columns": ["Name", "Age", "City"],
            "rows": [
                ["Alice", "30", "New York"],
                ["Bob", "25", "Los Angeles"],
                ["Charlie", "35", "Chicago"]
            ]
        },
        "question": "Who is the oldest?"
    }
    ```

    Response:
    ```json
    {
        "object": "table_qa",
        "answer": "Charlie",
        "cells": [{"row": 2, "column": 0}],
        "aggregation": "NONE"
    }
    ```
    """
    try:
        if "columns" not in request.table or "rows" not in request.table:
            raise HTTPException(
                status_code=400,
                detail="Table must have 'columns' and 'rows' keys",
            )

        if not request.table["columns"]:
            raise HTTPException(
                status_code=400,
                detail="Table must have at least one column",
            )

        if not request.table["rows"]:
            raise HTTPException(
                status_code=400,
                detail="Table must have at least one row",
            )

        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty",
            )

        model = await load_table_qa_model(request.model)
        result = await model.answer(request.table, request.question)

        return {
            "object": "table_qa",
            "answer": result["answer"],
            "cells": result["cells"],
            "cell_values": result["cell_values"],
            "aggregation": result["aggregation"],
            "question": result["question"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in table_question_answering: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/analysis/table-qa/batch")
async def table_question_answering_batch(request: TableQABatchRequest):
    """
    Answer multiple questions about the same table.

    Returns answers for each question in the batch.
    """
    try:
        if "columns" not in request.table or "rows" not in request.table:
            raise HTTPException(
                status_code=400,
                detail="Table must have 'columns' and 'rows' keys",
            )

        if not request.questions:
            raise HTTPException(
                status_code=400,
                detail="At least one question is required",
            )

        model = await load_table_qa_model(request.model)
        results = await model.answer_batch(request.table, request.questions)

        return {
            "object": "table_qa_batch",
            "results": results,
            "total_questions": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in table_question_answering_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    from llamafarm_common.pidfile import write_pid

    # Write PID file for service discovery
    write_pid("universal-runtime")

    port = int(os.getenv("LF_RUNTIME_PORT", os.getenv("PORT", "11540")))
    host = os.getenv("LF_RUNTIME_HOST", os.getenv("HOST", "127.0.0.1"))

    logger.info(f"Starting LlamaFarm Universal Runtime on {host}:{port}")
    logger.info(f"Device: {get_device()}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,  # Disable uvicorn's log config (handled in setup_logging)
        access_log=False,  # Disable uvicorn access logs (handled by structlog)
    )
