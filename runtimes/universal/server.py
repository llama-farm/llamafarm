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
from typing import Literal

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
    BaseModel,
    ClassifierModel,
    DocumentModel,
    EncoderModel,
    GGUFEncoderModel,
    GGUFLanguageModel,
    LanguageModel,
    OCRModel,
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
        _model_last_access.clear()

    logger.info("Shutdown complete")


app = FastAPI(
    title="Universal Runtime",
    description="OpenAI-compatible API for HuggingFace models (transformers, diffusers, embedders)",
    version="2.0.0",
    lifespan=lifespan,
)
app.include_router(chat_completions_router)

# Global model cache
_models: dict[str, BaseModel] = {}
_model_last_access: dict[str, datetime] = {}  # Track last access time for each model
_model_load_lock = asyncio.Lock()
_current_device = None

# Feature encoder cache for anomaly detection with mixed data types
_encoders: dict[str, FeatureEncoder] = {}
_cleanup_task: asyncio.Task | None = None

# Model unload timeout configuration (in seconds)
# Default: 5 minutes (300 seconds)
MODEL_UNLOAD_TIMEOUT = int(os.getenv("MODEL_UNLOAD_TIMEOUT", "300"))
# Cleanup check interval (in seconds) - how often to check for idle models
# Default: 30 seconds
CLEANUP_CHECK_INTERVAL = int(os.getenv("CLEANUP_CHECK_INTERVAL", "30"))


# ============================================================================
# Helper Functions
# ============================================================================


def _track_model_access(cache_key: str) -> None:
    """Track that a model was accessed."""
    _model_last_access[cache_key] = datetime.now()


async def _cleanup_idle_models() -> None:
    """Background task that periodically unloads idle models.

    Runs continuously, checking every CLEANUP_CHECK_INTERVAL seconds for models
    that haven't been accessed in MODEL_UNLOAD_TIMEOUT seconds.
    """
    logger.info(
        f"Model cleanup task started (timeout={MODEL_UNLOAD_TIMEOUT}s, "
        f"check_interval={CLEANUP_CHECK_INTERVAL}s)"
    )

    while True:
        try:
            await asyncio.sleep(CLEANUP_CHECK_INTERVAL)

            now = datetime.now()
            models_to_unload = []

            # Find idle models
            for cache_key, last_access in _model_last_access.items():
                idle_time = (now - last_access).total_seconds()
                if idle_time > MODEL_UNLOAD_TIMEOUT:
                    models_to_unload.append(cache_key)

            # Unload idle models
            if models_to_unload:
                logger.info(f"Unloading {len(models_to_unload)} idle model(s)")

                for cache_key in models_to_unload:
                    try:
                        # Re-check idle time immediately before unloading to handle race conditions
                        # A concurrent request could have accessed the model after we built the unload list
                        if cache_key not in _model_last_access:
                            continue  # Model already removed

                        last_access = _model_last_access[cache_key]
                        current_idle_time = (
                            datetime.now() - last_access
                        ).total_seconds()
                        if current_idle_time < MODEL_UNLOAD_TIMEOUT:
                            logger.debug(
                                f"Skipping unload of {cache_key}: accessed during cleanup "
                                f"(idle time now {current_idle_time:.1f}s < {MODEL_UNLOAD_TIMEOUT}s)"
                            )
                            continue

                        model = _models.get(cache_key)
                        if model:
                            logger.info(f"Unloading idle model: {cache_key}")
                            await model.unload()
                            del _models[cache_key]
                            del _model_last_access[cache_key]
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
    and loads it with the appropriate backend. GGUF models use llama-cpp-python
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
                _track_model_access(cache_key)
    else:
        # Model already loaded, track access
        _track_model_access(cache_key)

    return _models[cache_key]


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
    and loads it with the appropriate backend. GGUF models use llama-cpp-python
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
                _track_model_access(cache_key)
    else:
        # Model already loaded, track access
        _track_model_access(cache_key)

    return _models[cache_key]


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
                _track_model_access(cache_key)
    else:
        _track_model_access(cache_key)

    return _models[cache_key]


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
                _track_model_access(cache_key)
    else:
        _track_model_access(cache_key)

    return _models[cache_key]


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


def _make_anomaly_cache_key(model_id: str, backend: str) -> str:
    """Generate a cache key for an anomaly model."""
    return f"anomaly:{backend}:{model_id}"


async def load_anomaly(
    model_id: str,
    backend: str = "isolation_forest",
    contamination: float = 0.1,
    threshold: float | None = None,
):
    """Load an anomaly detection model.

    Args:
        model_id: Model identifier or path to pre-trained model
        backend: Anomaly detection backend
        contamination: Expected proportion of anomalies
        threshold: Custom anomaly threshold

    Returns:
        Loaded AnomalyModel instance
    """
    cache_key = _make_anomaly_cache_key(model_id, backend)

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
                )

                await model.load()
                _models[cache_key] = model
                _track_model_access(cache_key)
    else:
        _track_model_access(cache_key)

    return _models[cache_key]


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
    """

    model: str = "default"  # Model identifier
    backend: str = "isolation_forest"  # isolation_forest, one_class_svm, local_outlier_factor, autoencoder
    data: list[list[float]] | list[dict]  # Data points (numeric arrays or dicts)
    schema: dict[str, str] | None = (
        None  # Feature encoding schema (required for dict data)
    )
    threshold: float | None = None  # Override default threshold


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
    """

    model: str = "default"  # Model identifier (for caching)
    backend: str = "isolation_forest"  # Backend to use
    data: list[list[float]] | list[dict]  # Training data (numeric arrays or dicts)
    schema: dict[str, str] | None = (
        None  # Feature encoding schema (required for dict data)
    )
    contamination: float = 0.1  # Expected proportion of anomalies
    epochs: int = 100  # Training epochs (autoencoder only)
    batch_size: int = 32  # Batch size (autoencoder only)


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
        cache_key = _make_anomaly_cache_key(request.model, request.backend)

        model = await load_anomaly(
            model_id=request.model,
            backend=request.backend,
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

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest",
        "data": [[1.0, 2.0], [1.1, 2.1], [0.9, 1.9], ...],
        "contamination": 0.1
    }
    ```

    After fitting, use /v1/anomaly/score to detect anomalies in new data.
    """
    try:
        cache_key = _make_anomaly_cache_key(request.model, request.backend)

        # Prepare data (encode if dict-based, and fit the encoder)
        prepared_data = _prepare_anomaly_data(
            data=request.data,
            schema=request.schema,
            cache_key=cache_key,
            fit_mode=True,  # Fit encoder on training data
        )

        model = await load_anomaly(
            model_id=request.model,
            backend=request.backend,
            contamination=request.contamination,
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
        cache_key = _make_anomaly_cache_key(request.model, request.backend)

        model = await load_anomaly(
            model_id=request.model,
            backend=request.backend,
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
        cache_key = _make_anomaly_cache_key(request.model, request.backend)

        if cache_key not in _models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' with backend '{request.backend}' not found in cache. "
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

        cache_key = _make_anomaly_cache_key(request.model, request.backend)

        # Remove existing model from cache if present
        if cache_key in _models:
            await _models[cache_key].unload()
            del _models[cache_key]

        async with _model_load_lock:
            logger.info(f"Loading pre-trained anomaly model: {model_path}")
            device = get_device()

            model = AnomalyModel(
                model_id=str(model_path),  # Pass path as model_id for loading
                device=device,
                backend=request.backend,
            )

            await model.load()
            _models[cache_key] = model
            _track_model_access(cache_key)

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
        safe_filename = _sanitize_model_name(filename)
        if not safe_filename:
            raise HTTPException(
                status_code=400,
                detail="Invalid filename",
            )

        # Also reject any path separators that might have survived
        if "/" in filename or "\\" in filename or ".." in filename:
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

# Classifier model cache (separate from _models to avoid conflicts)
_classifiers: dict[str, "ClassifierModel"] = {}


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

    if cache_key in _classifiers:
        _track_model_access(cache_key)
        return _classifiers[cache_key]

    async with _model_load_lock:
        # Double-check after acquiring lock
        if cache_key in _classifiers:
            _track_model_access(cache_key)
            return _classifiers[cache_key]

        logger.info(f"Loading classifier model: {model_id}")
        device = get_device()

        model = ClassifierModel(
            model_id=model_id,
            device=device,
            base_model=base_model,
        )

        await model.load()
        _classifiers[cache_key] = model
        _track_model_access(cache_key)

        return model


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

        if cache_key not in _classifiers:
            raise HTTPException(
                status_code=404,
                detail=f"Classifier '{request.model}' not found. "
                "Fit with /v1/classifier/fit or load with /v1/classifier/load first.",
            )

        model = _classifiers[cache_key]
        _track_model_access(cache_key)

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
            await _classifiers[cache_key].unload()
            del _classifiers[cache_key]

        async with _model_load_lock:
            logger.info(f"Loading pre-trained classifier: {model_path}")
            device = get_device()

            model = ClassifierModel(
                model_id=str(model_path),  # Pass path as model_id for loading
                device=device,
            )

            await model.load()
            _classifiers[cache_key] = model
            _track_model_access(cache_key)

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
