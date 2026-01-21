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

import numpy as np
from fastapi import (
    BackgroundTasks,
    FastAPI,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from core.logging import UniversalRuntimeLogger, setup_logging
from models import (
    AnomalyModel,
    BackgroundRemovalModel,
    BaseModel,
    ClassifierModel,
    CLIPVisionModel,
    DocumentModel,
    EncoderModel,
    FewShotImageClassifier,
    GGUFEncoderModel,
    GGUFLanguageModel,
    LanguageDetectionModel,
    LanguageModel,
    ObjectDetectionModel,
    OCRModel,
    OpenVocabDetectionModel,
    PIIModel,
    SpeechModel,
    TimeSeriesModel,
)
from routers.anomaly import router as anomaly_router
from routers.chat_completions import router as chat_completions_router
from routers.classifier import router as classifier_router
from routers.nlp import router as nlp_router
from routers.timeseries import router as timeseries_router
from routers.vision import router as vision_router
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

    if _few_shot_classifiers:
        logger.info(f"Unloading {len(_few_shot_classifiers)} remaining few-shot classifier(s)")
        for cache_key, model in list(_few_shot_classifiers.items()):
            try:
                await model.unload()
                logger.info(f"Unloaded few-shot classifier: {cache_key}")
            except Exception as e:
                logger.error(f"Error unloading few-shot classifier {cache_key}: {e}")
        _few_shot_classifiers.clear()

    if _open_vocab_detection_models:
        logger.info(f"Unloading {len(_open_vocab_detection_models)} remaining open-vocab detection model(s)")
        for cache_key, model in list(_open_vocab_detection_models.items()):
            try:
                await model.unload()
                logger.info(f"Unloaded open-vocab detection model: {cache_key}")
            except Exception as e:
                logger.error(f"Error unloading open-vocab detection model {cache_key}: {e}")
        _open_vocab_detection_models.clear()

    if _lang_detection_models:
        logger.info(f"Unloading {len(_lang_detection_models)} remaining language detection model(s)")
        for cache_key, model in list(_lang_detection_models.items()):
            try:
                await model.unload()
                logger.info(f"Unloaded language detection model: {cache_key}")
            except Exception as e:
                logger.error(f"Error unloading language detection model {cache_key}: {e}")
        _lang_detection_models.clear()

    if _pii_models:
        logger.info(f"Unloading {len(_pii_models)} remaining PII model(s)")
        for cache_key, model in list(_pii_models.items()):
            try:
                await model.unload()
                logger.info(f"Unloaded PII model: {cache_key}")
            except Exception as e:
                logger.error(f"Error unloading PII model {cache_key}: {e}")
        _pii_models.clear()

    if _object_detection_models:
        logger.info(f"Unloading {len(_object_detection_models)} remaining object detection model(s)")
        for cache_key, model in list(_object_detection_models.items()):
            try:
                await model.unload()
                logger.info(f"Unloaded object detection model: {cache_key}")
            except Exception as e:
                logger.error(f"Error unloading object detection model {cache_key}: {e}")
        _object_detection_models.clear()

    if _background_removal_models:
        logger.info(f"Unloading {len(_background_removal_models)} remaining background removal model(s)")
        for cache_key, model in list(_background_removal_models.items()):
            try:
                await model.unload()
                logger.info(f"Unloaded background removal model: {cache_key}")
            except Exception as e:
                logger.error(f"Error unloading background removal model {cache_key}: {e}")
        _background_removal_models.clear()

    if _timeseries_models:
        logger.info(f"Unloading {len(_timeseries_models)} remaining time series model(s)")
        for cache_key, model in list(_timeseries_models.items()):
            try:
                await model.unload()
                logger.info(f"Unloaded time series model: {cache_key}")
            except Exception as e:
                logger.error(f"Error unloading time series model {cache_key}: {e}")
        _timeseries_models.clear()

    logger.info("Shutdown complete")


app = FastAPI(
    title="Universal Runtime",
    description="OpenAI-compatible API for HuggingFace models (transformers, diffusers, embedders)",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS configuration - use environment variable for allowed origins
# Default allows common local development ports
# Set CORS_ALLOWED_ORIGINS to a comma-separated list of origins in production
_default_origins = "http://localhost:3000,http://localhost:5173,http://localhost:4200,http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:4200"
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", _default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(anomaly_router)
app.include_router(chat_completions_router)
app.include_router(classifier_router)
app.include_router(nlp_router)
app.include_router(timeseries_router)
app.include_router(vision_router)

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
_few_shot_classifiers: ModelCache["FewShotImageClassifier"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
_open_vocab_detection_models: ModelCache["OpenVocabDetectionModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
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
                (_few_shot_classifiers, "few_shot_classifiers"),
                (_open_vocab_detection_models, "open_vocab_detection_models"),
                (_lang_detection_models, "lang_detection_models"),
                (_pii_models, "pii_models"),
                (_object_detection_models, "object_detection_models"),
                (_background_removal_models, "background_removal_models"),
                (_timeseries_models, "timeseries_models"),
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
    model_id: str,
    n_ctx: int | None = None,
    n_batch: int | None = None,
    n_gpu_layers: int | None = None,
    n_threads: int | None = None,
    flash_attn: bool | None = None,
    use_mmap: bool | None = None,
    use_mlock: bool | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    preferred_quantization: str | None = None,
) -> str:
    """Generate a cache key for a causal language model.

    Args:
        model_id: HuggingFace model identifier
        n_ctx: Optional context window size for GGUF models
        n_batch: Optional batch size for GGUF models
        n_gpu_layers: Optional number of GPU layers for GGUF models
        n_threads: Optional thread count for GGUF models
        flash_attn: Optional flash attention flag for GGUF models
        use_mmap: Optional memory-mapping flag for GGUF models
        use_mlock: Optional memory-lock flag for GGUF models
        cache_type_k: Optional KV cache key quantization for GGUF models
        cache_type_v: Optional KV cache value quantization for GGUF models
        preferred_quantization: Optional quantization preference for GGUF models

    Returns:
        A unique cache key string that identifies this specific model configuration
    """
    quant_key = (
        preferred_quantization if preferred_quantization is not None else "default"
    )
    ctx_key = n_ctx if n_ctx is not None else "auto"
    batch_key = n_batch if n_batch is not None else "auto"
    gpu_key = n_gpu_layers if n_gpu_layers is not None else "auto"
    threads_key = n_threads if n_threads is not None else "auto"
    flash_key = flash_attn if flash_attn is not None else "default"
    mmap_key = use_mmap if use_mmap is not None else "default"
    mlock_key = use_mlock if use_mlock is not None else "default"
    cache_k_key = cache_type_k if cache_type_k is not None else "default"
    cache_v_key = cache_type_v if cache_type_v is not None else "default"
    return (
        f"language:{model_id}:ctx{ctx_key}:batch{batch_key}:gpu{gpu_key}:"
        f"threads{threads_key}:flash{flash_key}:mmap{mmap_key}:mlock{mlock_key}:"
        f"cachek{cache_k_key}:cachev{cache_v_key}:quant{quant_key}"
    )


async def load_language(
    model_id: str,
    n_ctx: int | None = None,
    n_batch: int | None = None,
    n_gpu_layers: int | None = None,
    n_threads: int | None = None,
    flash_attn: bool | None = None,
    use_mmap: bool | None = None,
    use_mlock: bool | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    preferred_quantization: str | None = None,
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
        n_batch: Optional batch size for prompt processing. If None, defaults to 2048.
                 Critical for memory: lower values (e.g., 512) reduce compute buffer size.
        n_gpu_layers: Optional number of layers to offload to GPU. If None, will be
                      auto-detected based on device. Use -1 for all layers.
        n_threads: Optional number of CPU threads. If None, auto-detected.
        flash_attn: Optional flag to enable/disable flash attention. If None,
                    defaults to True for faster inference.
        use_mmap: Optional flag for memory-mapped file loading. If None, defaults to True.
        use_mlock: Optional flag to lock model in RAM. If None, defaults to False.
        cache_type_k: Optional KV cache key quantization type (e.g., "q4_0", "q8_0", "f16").
                      Using "q4_0" can reduce KV cache memory by ~4x.
        cache_type_v: Optional KV cache value quantization type. Same options as cache_type_k.
        preferred_quantization: Optional quantization preference for GGUF models
                                (e.g., "Q4_K_M", "Q8_0"). If None, defaults to Q4_K_M.
                                Only downloads the specified quantization to save disk space.
    """

    # Include all parameters in cache key for GGUF models so different configurations are cached separately
    # Use "auto"/"default" for None values to allow automatic detection
    cache_key = _make_language_cache_key(
        model_id, n_ctx, n_batch, n_gpu_layers, n_threads, flash_attn,
        use_mmap, use_mlock, cache_type_k, cache_type_v, preferred_quantization
    )
    if cache_key not in _models:
        async with _model_load_lock:
            # Double-check if model was loaded while waiting for the lock
            if cache_key not in _models:
                logger.info(
                    f"Loading causal LM: {model_id} "
                    f"(n_ctx={n_ctx if n_ctx is not None else 'auto'}, "
                    f"n_batch={n_batch if n_batch is not None else 'auto'}, "
                    f"n_gpu_layers={n_gpu_layers if n_gpu_layers is not None else 'auto'}, "
                    f"flash_attn={flash_attn if flash_attn is not None else 'default'}, "
                    f"cache_type_k={cache_type_k if cache_type_k is not None else 'default'}, "
                    f"cache_type_v={cache_type_v if cache_type_v is not None else 'default'})"
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
                        n_batch=n_batch,
                        n_gpu_layers=n_gpu_layers,
                        n_threads=n_threads,
                        flash_attn=flash_attn,
                        use_mmap=use_mmap,
                        use_mlock=use_mlock,
                        cache_type_k=cache_type_k,
                        cache_type_v=cache_type_v,
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
    threshold: float | None = Field(default=None, ge=0, le=1, description="Anomaly threshold (0-1)")
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"
    scaler_type: Literal["robust", "standard"] = "robust"


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
    contamination: float = Field(
        default=0.1,
        gt=0,
        le=0.5,
        description="Expected proportion of anomalies (0-0.5]",
    )
    epochs: int = Field(default=100, ge=1, description="Training epochs (autoencoder only)")
    batch_size: int = Field(default=32, ge=1, description="Batch size (autoencoder only)")
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"
    scaler_type: Literal["robust", "standard"] = "robust"
    # VAE / Early Stopping parameters
    validation_split: float = Field(
        default=0.1, ge=0, le=0.5, description="Fraction of data for validation"
    )
    patience: int = Field(default=10, ge=1, description="Epochs without improvement before stopping")
    min_delta: float = Field(default=1e-4, ge=0, description="Minimum improvement threshold")


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
    import os
    import tempfile
    from pathlib import Path

    tmp_path = None
    try:
        # Save uploaded file to temp location using streaming to prevent memory exhaustion
        suffix = Path(file.filename).suffix.lower() if file.filename else ".csv"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            # Stream in 1MB chunks to avoid loading entire file into memory
            while chunk := await file.read(1024 * 1024):
                tmp.write(chunk)

        # Upload to streaming loader
        loader = get_streaming_loader()
        file_ref = await loader.upload_file(tmp_path, copy_to_temp=True)

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
    finally:
        # Always clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


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
# Few-Shot Image Classification Endpoints
# ============================================================================


# Few-shot classifier persistence directory
FEW_SHOT_MODELS_DIR = _LF_DATA_DIR / "models" / "few_shot"


def _get_few_shot_path(classifier_id: str) -> Path:
    """Get the path for a few-shot classifier file.

    The path is always within FEW_SHOT_MODELS_DIR - users cannot control it.
    """
    safe_name = _sanitize_model_name(classifier_id)
    return FEW_SHOT_MODELS_DIR / f"{safe_name}.fsc"


async def _auto_save_few_shot_classifier(
    classifier: "FewShotImageClassifier",
    classifier_id: str,
) -> dict[str, str | None]:
    """Auto-save few-shot classifier after training to prevent data loss.

    Models are saved immediately after training to ensure they persist
    across server restarts without requiring an explicit /save call.

    Returns:
        Dict with saved file path
    """
    try:
        FEW_SHOT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        save_path = _get_few_shot_path(classifier_id)
        state_bytes = classifier.save_state()

        # Write atomically using a temp file
        temp_path = save_path.with_suffix(".tmp")
        temp_path.write_bytes(state_bytes)
        temp_path.rename(save_path)

        logger.info(f"Auto-saved few-shot classifier to {save_path}")
        return {"model_path": str(save_path)}

    except Exception as e:
        logger.warning(f"Auto-save failed (model still in memory): {e}")
        return {"model_path": None}


def _make_few_shot_cache_key(classifier_id: str, model_name: str) -> str:
    """Generate a cache key for a few-shot classifier.

    Args:
        classifier_id: User-provided classifier identifier
        model_name: CLIP model name for embeddings

    Returns:
        A unique cache key string
    """
    return f"few_shot:{classifier_id}:{model_name}"


async def load_few_shot_classifier(
    classifier_id: str,
    model_name: str = "openai/clip-vit-base-patch32",
    create_if_missing: bool = True,
) -> FewShotImageClassifier | None:
    """Load or create a few-shot classifier.

    Args:
        classifier_id: Unique identifier for this classifier
        model_name: CLIP model to use for embeddings
        create_if_missing: If True, create a new classifier if not found

    Returns:
        FewShotImageClassifier instance or None if not found and create_if_missing=False
    """
    cache_key = _make_few_shot_cache_key(classifier_id, model_name)

    if cache_key not in _few_shot_classifiers:
        if not create_if_missing:
            return None
        async with _model_load_lock:
            if cache_key not in _few_shot_classifiers:
                logger.info(f"Creating few-shot classifier: {classifier_id} (CLIP: {model_name})")
                device = get_device()

                model = FewShotImageClassifier(
                    model_id=classifier_id,
                    device=device,
                    hf_model_name=model_name,
                )

                await model.load()
                _few_shot_classifiers[cache_key] = model

    return _few_shot_classifiers.get(cache_key)


class FewShotTrainRequest(PydanticBaseModel):
    """Request to train a few-shot classifier."""

    classifier_id: str  # Unique ID for this classifier
    images: list[str]  # Base64-encoded images or file paths
    labels: list[str]  # Labels for each image (same length as images)
    model: str = "openai/clip-vit-base-patch32"  # CLIP model for embeddings
    epochs: int = 100  # Training epochs
    learning_rate: float = 0.001  # Learning rate


class FewShotRefineRequest(PydanticBaseModel):
    """Request to refine an existing few-shot classifier with more data."""

    classifier_id: str  # Classifier ID to refine
    images: list[str]  # Additional images
    labels: list[str]  # Labels for additional images
    model: str = "openai/clip-vit-base-patch32"  # CLIP model (must match original)
    epochs: int = 50  # Refinement epochs
    learning_rate: float = 0.0005  # Lower learning rate for refinement


class FewShotPredictRequest(PydanticBaseModel):
    """Request to classify an image with a trained few-shot classifier."""

    classifier_id: str  # Classifier ID to use
    image: str  # Base64-encoded image or file path
    model: str = "openai/clip-vit-base-patch32"  # CLIP model (must match training)


class FewShotPredictBatchRequest(PydanticBaseModel):
    """Request to classify multiple images with a trained few-shot classifier."""

    classifier_id: str  # Classifier ID to use
    images: list[str]  # Base64-encoded images or file paths
    model: str = "openai/clip-vit-base-patch32"  # CLIP model (must match training)


@app.post("/v1/vision/classify/fit")
async def train_few_shot_classifier(request: FewShotTrainRequest):
    """
    Train a few-shot image classifier using CLIP embeddings with linear probe.

    This creates a custom classifier that can distinguish between your specific
    categories using just 5-50 images per class. The classifier uses frozen
    CLIP features with a trainable linear classifier head.

    Example workflow:
    1. Train: POST /v1/vision/classify/fit with images and labels
    2. Predict: POST /v1/vision/classify/predict with classifier_id
    3. Refine: POST /v1/vision/classify/refine to add more data or classes

    Example request:
    ```json
    {
        "classifier_id": "cat-dog-classifier",
        "images": ["<base64 cat1>", "<base64 cat2>", "<base64 dog1>", "<base64 dog2>"],
        "labels": ["cat", "cat", "dog", "dog"],
        "model": "openai/clip-vit-base-patch32",
        "epochs": 100
    }
    ```

    Response:
    ```json
    {
        "object": "few_shot_classifier",
        "classifier_id": "cat-dog-classifier",
        "success": true,
        "num_samples": 4,
        "num_classes": 2,
        "classes": ["cat", "dog"],
        "final_accuracy": 1.0,
        "training_time_ms": 1234.5
    }
    ```
    """
    try:
        if len(request.images) != len(request.labels):
            raise HTTPException(
                status_code=400,
                detail=f"Number of images ({len(request.images)}) must match labels ({len(request.labels)})",
            )

        if len(request.images) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 images to train a classifier",
            )

        unique_classes = set(request.labels)
        if len(unique_classes) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Training requires at least 2 distinct classes. "
                       f"Found {len(unique_classes)} class(es): {sorted(unique_classes)}",
            )

        classifier = await load_few_shot_classifier(
            classifier_id=request.classifier_id,
            model_name=request.model,
        )

        result = await classifier.fit(
            images=request.images,
            labels=request.labels,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
        )

        # Auto-save after training to persist across restarts
        saved_paths = await _auto_save_few_shot_classifier(
            classifier=classifier,
            classifier_id=request.classifier_id,
        )

        return {
            "object": "few_shot_classifier",
            "classifier_id": request.classifier_id,
            "saved_path": saved_paths.get("model_path"),
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in train_few_shot_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/vision/classify/refine")
async def refine_few_shot_classifier(request: FewShotRefineRequest):
    """
    Refine a few-shot classifier with additional training data.

    Use this to:
    - Add more examples of existing classes
    - Add entirely new classes
    - Correct misclassifications by adding correctly labeled examples

    The classifier will be fine-tuned on the new data while preserving
    knowledge of existing classes.

    Example request:
    ```json
    {
        "classifier_id": "cat-dog-classifier",
        "images": ["<base64 bird1>", "<base64 bird2>"],
        "labels": ["bird", "bird"],
        "epochs": 50
    }
    ```

    Response includes the new classes list:
    ```json
    {
        "object": "few_shot_classifier",
        "classifier_id": "cat-dog-classifier",
        "success": true,
        "refined_samples": 2,
        "num_classes": 3,
        "classes": ["bird", "cat", "dog"],
        "new_classes_added": ["bird"],
        "accuracy_on_new_data": 1.0
    }
    ```
    """
    try:
        if len(request.images) != len(request.labels):
            raise HTTPException(
                status_code=400,
                detail=f"Number of images ({len(request.images)}) must match labels ({len(request.labels)})",
            )

        classifier = await load_few_shot_classifier(
            classifier_id=request.classifier_id,
            model_name=request.model,
            create_if_missing=False,
        )

        if classifier is None:
            raise HTTPException(
                status_code=404,
                detail=f"Classifier '{request.classifier_id}' not found. Train it first with /v1/vision/classify/fit",
            )

        result = await classifier.refine(
            images=request.images,
            labels=request.labels,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
        )

        # Auto-save after refinement to persist changes
        saved_paths = await _auto_save_few_shot_classifier(
            classifier=classifier,
            classifier_id=request.classifier_id,
        )

        return {
            "object": "few_shot_classifier",
            "classifier_id": request.classifier_id,
            "saved_path": saved_paths.get("model_path"),
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in refine_few_shot_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/vision/classify/models")
async def list_few_shot_classifiers():
    """
    List all saved few-shot classifiers available for loading.

    Returns classifiers saved in the FEW_SHOT_MODELS_DIR directory.

    Response includes:
    - classifier_id: Name of the saved classifier
    - path: Full path to the classifier file
    - classes: Class labels (if available in metadata)
    - size_bytes: File size
    """
    try:
        FEW_SHOT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        classifiers = []
        for path in FEW_SHOT_MODELS_DIR.glob("*.fsc"):
            info = {
                "classifier_id": path.stem,
                "path": str(path),
                "size_bytes": path.stat().st_size,
            }

            # Try to extract metadata (classes) from saved state
            try:
                state_bytes = path.read_bytes()
                # Parse metadata length and extract JSON
                metadata_len = int.from_bytes(state_bytes[:4], byteorder="big")
                metadata_bytes = state_bytes[4:4 + metadata_len]
                import json
                metadata = json.loads(metadata_bytes.decode("utf-8"))
                info["classes"] = metadata.get("classes", [])
                info["model"] = metadata.get("model", "unknown")
            except Exception:
                info["classes"] = []
                info["model"] = "unknown"

            classifiers.append(info)

        return {
            "object": "list",
            "classifiers": classifiers,
            "models_dir": str(FEW_SHOT_MODELS_DIR),
            "total": len(classifiers),
        }

    except Exception as e:
        logger.error(f"Error in list_few_shot_classifiers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


class FewShotLoadRequest(PydanticBaseModel):
    """Request to load a saved few-shot classifier."""

    classifier_id: str  # Classifier ID to load
    model: str = "openai/clip-vit-base-patch32"  # CLIP model (must match saved)


@app.post("/v1/vision/classify/load")
async def load_few_shot_classifier_endpoint(request: FewShotLoadRequest):
    """
    Load a previously saved few-shot classifier.

    Example request:
    ```json
    {
        "classifier_id": "cat-dog-classifier",
        "model": "openai/clip-vit-base-patch32"
    }
    ```

    After loading, use /v1/vision/classify/predict to classify images.
    """
    try:
        model_path = _get_few_shot_path(request.classifier_id)

        if not model_path.exists():
            available = (
                [f.stem for f in FEW_SHOT_MODELS_DIR.glob("*.fsc")]
                if FEW_SHOT_MODELS_DIR.exists()
                else []
            )
            raise HTTPException(
                status_code=404,
                detail=f"Classifier '{request.classifier_id}' not found. "
                f"Available classifiers: {available}",
            )

        cache_key = _make_few_shot_cache_key(request.classifier_id, request.model)

        # Remove existing model from cache if present
        if cache_key in _few_shot_classifiers:
            existing = _few_shot_classifiers.pop(cache_key)
            if existing:
                await existing.unload()

        async with _model_load_lock:
            logger.info(f"Loading saved few-shot classifier: {model_path}")
            device = get_device()

            classifier = FewShotImageClassifier(
                model_id=request.classifier_id,
                device=device,
                hf_model_name=request.model,
            )

            await classifier.load()

            # Load saved state
            state_bytes = model_path.read_bytes()
            classifier.load_state(state_bytes)

            _few_shot_classifiers[cache_key] = classifier

        return {
            "object": "load_result",
            "classifier_id": request.classifier_id,
            "path": str(model_path),
            "classes": classifier.classes,
            "loaded": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in load_few_shot_classifier_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/vision/classify/predict")
async def predict_few_shot(request: FewShotPredictRequest):
    """
    Classify an image using a trained few-shot classifier.

    Example request:
    ```json
    {
        "classifier_id": "cat-dog-classifier",
        "image": "<base64 encoded image>"
    }
    ```

    Response:
    ```json
    {
        "object": "classification",
        "classifier_id": "cat-dog-classifier",
        "label": "cat",
        "score": 0.92,
        "all_scores": {"cat": 0.92, "dog": 0.08}
    }
    ```
    """
    try:
        classifier = await load_few_shot_classifier(
            classifier_id=request.classifier_id,
            model_name=request.model,
            create_if_missing=False,
        )

        if classifier is None:
            raise HTTPException(
                status_code=404,
                detail=f"Classifier '{request.classifier_id}' not found. Train it first with /v1/vision/classify/fit",
            )

        if not classifier.is_trained:
            raise HTTPException(
                status_code=400,
                detail=f"Classifier '{request.classifier_id}' exists but is not trained. Call /v1/vision/classify/fit first.",
            )

        result = await classifier.predict(request.image)

        return {
            "object": "classification",
            "classifier_id": request.classifier_id,
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_few_shot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/vision/classify/predict/batch")
async def predict_few_shot_batch(request: FewShotPredictBatchRequest):
    """
    Classify multiple images using a trained few-shot classifier.

    More efficient than calling the single-image endpoint multiple times.

    Example request:
    ```json
    {
        "classifier_id": "cat-dog-classifier",
        "images": ["<base64 image 1>", "<base64 image 2>"]
    }
    ```

    Response:
    ```json
    {
        "object": "list",
        "classifier_id": "cat-dog-classifier",
        "data": [
            {"label": "cat", "score": 0.92, "all_scores": {"cat": 0.92, "dog": 0.08}},
            {"label": "dog", "score": 0.87, "all_scores": {"cat": 0.13, "dog": 0.87}}
        ],
        "total_count": 2
    }
    ```
    """
    try:
        if not request.images:
            raise HTTPException(
                status_code=400,
                detail="At least one image is required",
            )

        classifier = await load_few_shot_classifier(
            classifier_id=request.classifier_id,
            model_name=request.model,
            create_if_missing=False,
        )

        if classifier is None:
            raise HTTPException(
                status_code=404,
                detail=f"Classifier '{request.classifier_id}' not found. Train it first with /v1/vision/classify/fit",
            )

        if not classifier.is_trained:
            raise HTTPException(
                status_code=400,
                detail=f"Classifier '{request.classifier_id}' exists but is not trained. Call /v1/vision/classify/fit first.",
            )

        results = await classifier.predict_batch(request.images)

        return {
            "object": "list",
            "classifier_id": request.classifier_id,
            "data": results,
            "total_count": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_few_shot_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/vision/classify/info/{classifier_id}")
async def get_few_shot_classifier_info(
    classifier_id: str,
    model: str = "openai/clip-vit-base-patch32",
):
    """
    Get information about a few-shot classifier.

    Returns details about the classifier including:
    - Whether it's loaded and trained
    - Classes it can recognize
    - Model configuration

    Example response:
    ```json
    {
        "object": "few_shot_classifier_info",
        "classifier_id": "cat-dog-classifier",
        "is_loaded": true,
        "is_trained": true,
        "classes": ["cat", "dog"],
        "num_classes": 2,
        "model": "openai/clip-vit-base-patch32"
    }
    ```
    """
    try:
        classifier = await load_few_shot_classifier(
            classifier_id=classifier_id,
            model_name=model,
            create_if_missing=False,
        )

        if classifier is None:
            return {
                "object": "few_shot_classifier_info",
                "classifier_id": classifier_id,
                "is_loaded": False,
                "is_trained": False,
                "classes": [],
                "num_classes": 0,
                "model": model,
                "message": "Classifier not found",
            }

        info = classifier.get_model_info()
        return {
            "object": "few_shot_classifier_info",
            "classifier_id": classifier_id,
            **info,
        }

    except Exception as e:
        logger.error(f"Error in get_few_shot_classifier_info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/vision/classify/{classifier_id}/unload")
async def unload_few_shot_classifier(
    classifier_id: str,
    model: str = "openai/clip-vit-base-patch32",
):
    """
    Unload a few-shot classifier from memory and free its resources.

    This does NOT delete the saved model file - use DELETE through
    the main LlamaFarm API for that. This only unloads from the
    in-memory cache to free resources.

    Example response:
    ```json
    {
        "object": "unload",
        "classifier_id": "cat-dog-classifier",
        "unloaded": true
    }
    ```
    """
    try:
        cache_key = _make_few_shot_cache_key(classifier_id, model)

        if cache_key in _few_shot_classifiers:
            classifier = _few_shot_classifiers.pop(cache_key)
            if classifier:
                await classifier.unload()
            return {
                "object": "unload",
                "classifier_id": classifier_id,
                "unloaded": True,
            }

        return {
            "object": "unload",
            "classifier_id": classifier_id,
            "unloaded": False,
            "message": "Classifier not loaded in memory",
        }

    except Exception as e:
        logger.error(f"Error in delete_few_shot_classifier: {e}", exc_info=True)
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


class EncoderAdapter:
    """Adapter to make async EncoderModel compatible with sync KeywordExtractor."""

    def __init__(self, encoder_model):
        self.encoder_model = encoder_model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Sync encode method for KeywordExtractor compatibility."""
        import asyncio

        # Run the async embed method synchronously
        loop = asyncio.new_event_loop()
        try:
            embeddings = loop.run_until_complete(self.encoder_model.embed(texts))
            return np.array(embeddings)
        finally:
            loop.close()


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
                    encoder = await load_encoder(
                        "sentence-transformers/all-MiniLM-L6-v2"
                    )
                    # Wrap encoder to provide sync encode() method
                    adapter = EncoderAdapter(encoder)
                    _keyword_extractor = KeywordExtractor(encoder_model=adapter)
                    logger.info("Keyword extractor initialized with encoder model")
                except Exception as e:
                    logger.warning(f"Could not load encoder for keywords, using frequency fallback: {e}")
                    _keyword_extractor = KeywordExtractor(encoder_model=None)

    return _keyword_extractor


class KeywordExtractRequest(PydanticBaseModel):
    """Keyword extraction request."""

    text: str  # Text to extract keywords from
    top_k: int = Field(default=10, ge=1, le=100, description="Number of keywords to return")
    diversity: float = Field(default=0.5, ge=0, le=1, description="Diversity parameter (0-1)")
    ngram_range: list[int] = [1, 3]  # Min and max n-gram size


class KeywordExtractBatchRequest(PydanticBaseModel):
    """Keyword extraction batch request."""

    texts: list[str]  # List of texts
    top_k: int = Field(default=10, ge=1, le=100, description="Keywords per text")
    diversity: float = Field(default=0.5, ge=0, le=1, description="Diversity parameter (0-1)")
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
    threshold: float = Field(default=0.5, ge=0, le=1, description="Detection confidence threshold")
    use_regex: bool = True  # Also use regex patterns


class PIIRedactRequest(PydanticBaseModel):
    """PII redaction request."""

    text: str  # Text to redact
    entity_types: list[str] | None = None  # Entity types to redact
    replacement: str = "[REDACTED]"  # Default replacement
    replacement_map: dict[str, str] | None = None  # Per-type replacements
    threshold: float = Field(default=0.5, ge=0, le=1, description="Detection confidence threshold")
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
    threshold: float = Field(default=0.5, ge=0, le=1, description="Confidence threshold (0-1)")
    labels: list[str] | None = None  # Filter to specific object labels
    model: str = "hustvl/yolos-tiny"  # HuggingFace model name


class ObjectDetectionBatchRequest(PydanticBaseModel):
    """Batch object detection request."""

    images: list[str]  # List of base64-encoded images or file paths
    threshold: float = Field(default=0.5, ge=0, le=1, description="Confidence threshold (0-1)")
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
# Open-Vocabulary Object Detection (OWL-ViT)
# =============================================================================


def _make_open_vocab_cache_key(model_name: str) -> str:
    """Create a cache key for open-vocab detection models."""
    return f"open_vocab:{model_name}"


async def load_open_vocab_detection_model(
    model_name: str = "google/owlvit-base-patch32",
) -> OpenVocabDetectionModel:
    """Load or retrieve cached open-vocabulary detection model."""
    cache_key = _make_open_vocab_cache_key(model_name)

    if cache_key not in _open_vocab_detection_models:
        async with _model_load_lock:
            if cache_key not in _open_vocab_detection_models:
                logger.info(f"Loading OWL-ViT model: {model_name}")
                device = get_device()

                model = OpenVocabDetectionModel(
                    model_id=cache_key,
                    device=device,
                    hf_model_name=model_name,
                )

                await model.load()
                _open_vocab_detection_models[cache_key] = model

    return _open_vocab_detection_models.get(cache_key)


class OpenVocabDetectTextRequest(PydanticBaseModel):
    """Open-vocabulary detection using text queries."""

    image: str  # Base64-encoded image or file path
    queries: list[str]  # Text queries describing what to find
    threshold: float = 0.1  # Confidence threshold (lower = more detections)
    top_k: int | None = None  # Limit number of detections
    model: str = "google/owlvit-base-patch32"


class OpenVocabDetectTextBatchRequest(PydanticBaseModel):
    """Batch open-vocabulary detection using text queries."""

    images: list[str]  # List of base64-encoded images or file paths
    queries: list[str]  # Text queries (applied to all images)
    threshold: float = 0.1
    top_k: int | None = None
    model: str = "google/owlvit-base-patch32"


class OpenVocabDetectImageRequest(PydanticBaseModel):
    """Open-vocabulary detection using reference images."""

    image: str  # Target image to search in
    query_images: list[str]  # Reference images showing what to find
    threshold: float = 0.9  # Similarity threshold (higher = stricter match)
    top_k: int | None = None
    model: str = "google/owlvit-base-patch32"


@app.post("/v1/vision/detect-open")
async def detect_open_vocabulary(request: OpenVocabDetectTextRequest):
    """
    Detect objects using natural language text queries.

    OWL-ViT enables open-vocabulary object detection - find any object
    described in natural language, without retraining. This is ideal for:
    - Finding specific objects (\"a red fire hydrant\")
    - Species identification (\"a golden retriever\", \"a tabby cat\")
    - Fine-grained detection (\"a person wearing a hat\")
    - Custom domain objects (\"a damaged car door\")

    Tips for better results:
    - Use descriptive queries: \"a photo of a cat\" works better than just \"cat\"
    - Lower threshold (0.05-0.2) for recall, higher (0.3-0.5) for precision
    - Combine with few-shot classification for species/subspecies refinement

    Example request:
    ```json
    {
        \"image\": \"<base64 encoded image>\",
        \"queries\": [\"a golden retriever\", \"a german shepherd\", \"a labrador\"],
        \"threshold\": 0.1,
        \"top_k\": 10
    }
    ```

    Response:
    ```json
    {
        \"object\": \"open_vocab_detection\",
        \"objects\": [
            {
                \"query\": \"a golden retriever\",
                \"label\": \"a golden retriever\",
                \"score\": 0.85,
                \"box\": {\"x1\": 100, \"y1\": 50, \"x2\": 400, \"y2\": 350}
            }
        ],
        \"count\": 1,
        \"queries\": [\"a golden retriever\", \"a german shepherd\", \"a labrador\"],
        \"image_size\": {\"width\": 640, \"height\": 480}
    }
    ```

    Workflow for species identification:
    1. Use detect-open to find animals with broad queries
    2. Crop detected regions
    3. Use /v1/vision/classify/predict with a trained classifier for species
    """
    try:
        if not request.queries:
            raise HTTPException(
                status_code=400,
                detail="At least one text query is required",
            )

        model = await load_open_vocab_detection_model(request.model)
        result = await model.detect_by_text(
            image=request.image,
            queries=request.queries,
            threshold=request.threshold,
            top_k=request.top_k,
        )

        return {
            "object": "open_vocab_detection",
            **result,
            "model": request.model,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_open_vocabulary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/vision/detect-open/batch")
async def detect_open_vocabulary_batch(request: OpenVocabDetectTextBatchRequest):
    """
    Detect objects in multiple images using text queries.

    Applies the same queries to all images. More efficient than
    calling the single-image endpoint multiple times.
    """
    try:
        if not request.images:
            raise HTTPException(
                status_code=400,
                detail="At least one image is required",
            )
        if not request.queries:
            raise HTTPException(
                status_code=400,
                detail="At least one text query is required",
            )

        model = await load_open_vocab_detection_model(request.model)
        results = await model.detect_batch_by_text(
            images=request.images,
            queries=request.queries,
            threshold=request.threshold,
            top_k=request.top_k,
        )

        return {
            "object": "open_vocab_detection_batch",
            "results": results,
            "total_images": len(results),
            "model": request.model,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_open_vocabulary_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/vision/detect-open/by-image")
async def detect_by_reference_image(request: OpenVocabDetectImageRequest):
    """
    Detect objects similar to reference images (few-shot detection).

    Use this when you have example images of what you want to find.
    The model will locate similar objects in the target image.

    Example use cases:
    - Find products matching a reference photo
    - Locate specific landmarks or logos
    - Few-shot object detection for rare categories

    Example request:
    ```json
    {
        \"image\": \"<base64 target image>\",
        \"query_images\": [\"<base64 reference cat image>\"],
        \"threshold\": 0.9,
        \"top_k\": 5
    }
    ```

    Response:
    ```json
    {
        \"object\": \"image_guided_detection\",
        \"objects\": [
            {
                \"query_index\": 0,
                \"score\": 0.95,
                \"box\": {\"x1\": 100, \"y1\": 50, \"x2\": 300, \"y2\": 250}
            }
        ],
        \"count\": 1,
        \"num_queries\": 1,
        \"image_size\": {\"width\": 640, \"height\": 480}
    }
    ```
    """
    try:
        if not request.query_images:
            raise HTTPException(
                status_code=400,
                detail="At least one query image is required",
            )

        model = await load_open_vocab_detection_model(request.model)
        result = await model.detect_by_image(
            image=request.image,
            query_images=request.query_images,
            threshold=request.threshold,
            top_k=request.top_k,
        )

        return {
            "object": "image_guided_detection",
            **result,
            "model": request.model,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_by_reference_image: {e}", exc_info=True)
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


@app.post("/v1/vision/remove-background")
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


@app.post("/v1/vision/remove-background/batch")
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


class DriftDetectionRequest(PydanticBaseModel):
    """Concept drift detection request."""

    values: list[float]  # Data stream values
    algorithm: str = "adwin"  # adwin, page_hinkley, kswin, ddm
    delta: float | None = None  # Sensitivity parameter (for ADWIN, PageHinkley)
    threshold: float | None = None  # Threshold (for PageHinkley)
    alpha: float | None = None  # Significance level (for KSWIN)
    window_size: int | None = None  # Window size (for KSWIN)


class DriftUpdateRequest(PydanticBaseModel):
    """Single value drift update request."""

    value: float  # New data point
    algorithm: str = "adwin"
    detector_id: str | None = None  # Optional ID for persistent detector


class AnomalyExplainRequest(PydanticBaseModel):
    """Anomaly explanation request."""

    model_id: str  # ID of trained anomaly model
    data: list[list[float]]  # Data points to explain
    feature_names: list[str] | None = None  # Optional feature names
    background_samples: int = 100  # Number of background samples
    nsamples: int = 100  # Number of SHAP samples
    backend: str = "isolation_forest"  # Backend used when training
    normalization: str = "standardization"  # Normalization method
    scaler_type: str = "robust"  # Scaler type


class DatasetAuditRequest(PydanticBaseModel):
    """Dataset quality audit request."""

    labels: list[int]  # Ground truth labels (integer indices)
    pred_probs: list[list[float]]  # Prediction probabilities (n_samples, n_classes)
    features: list[list[float]] | None = None  # Feature vectors for duplicate detection
    label_names: list[str] | None = None  # Optional label name mapping
    check_duplicates: bool = True  # Whether to check for near-duplicates
    duplicate_threshold: float = 0.95  # Cosine similarity threshold


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
        from utils.changepoint_detector import (
            SUPPORTED_ALGORITHMS,
            SUPPORTED_MODELS,
            ChangePointDetector,
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
        from utils.changepoint_detector import (
            SUPPORTED_ALGORITHMS,
            SUPPORTED_MODELS,
            ChangePointDetector,
        )

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


# =============================================================================
# CONCEPT DRIFT DETECTION ENDPOINTS
# =============================================================================

# In-memory drift detector storage (for stateful streaming)
_drift_detectors: dict[str, Any] = {}


@app.post("/v1/streaming/drift/detect")
async def detect_drift(request: DriftDetectionRequest):
    """
    Detect concept drift in a data stream using River.

    Concept drift occurs when the statistical properties of data change over time,
    indicating that an ML model may need retraining.

    Algorithms:
    - adwin: ADaptive WINdowing - detects changes in mean (default)
    - page_hinkley: Page-Hinkley test - detects changes in mean
    - kswin: Kolmogorov-Smirnov Windowing - detects distribution changes
    - ddm: Drift Detection Method - monitors error rate

    Example request:
    ```json
    {
        "values": [1.0, 1.1, 1.0, 0.9, 1.0, 5.0, 5.1, 4.9, 5.0, 5.1],
        "algorithm": "adwin"
    }
    ```

    Response:
    ```json
    {
        "object": "drift_detection",
        "drift_detected": true,
        "drift_points": [6],
        "total_processed": 10
    }
    ```
    """
    try:
        from utils.drift_detector import SUPPORTED_ALGORITHMS, DriftDetector

        if request.algorithm.lower() not in SUPPORTED_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {request.algorithm}. Choose from {SUPPORTED_ALGORITHMS}",
            )

        if len(request.values) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 values are required for drift detection",
            )

        # Build kwargs from optional parameters
        kwargs: dict[str, Any] = {}
        if request.delta is not None:
            kwargs["delta"] = request.delta
        if request.threshold is not None:
            kwargs["threshold"] = request.threshold
        if request.alpha is not None:
            kwargs["alpha"] = request.alpha
        if request.window_size is not None:
            kwargs["window_size"] = request.window_size

        detector = DriftDetector(algorithm=request.algorithm, **kwargs)
        result = detector.update_batch(request.values)

        return {
            "object": "drift_detection",
            "drift_detected": result["drift_detected"],
            "drift_points": result["drift_points"],
            "total_processed": result["total_processed"],
            "algorithm": result["algorithm"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_drift: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/streaming/drift/create")
async def create_drift_detector(
    algorithm: str = "adwin",
    detector_id: str | None = None,
    delta: float | None = None,
    threshold: float | None = None,
    alpha: float | None = None,
    window_size: int | None = None,
):
    """
    Create a stateful drift detector for streaming updates.

    Returns a detector_id that can be used for subsequent update calls.
    """
    try:
        import uuid

        from utils.drift_detector import SUPPORTED_ALGORITHMS, DriftDetector

        if algorithm.lower() not in SUPPORTED_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {algorithm}. Choose from {SUPPORTED_ALGORITHMS}",
            )

        # Generate ID if not provided
        if detector_id is None:
            detector_id = str(uuid.uuid4())[:8]

        # Build kwargs
        kwargs: dict[str, Any] = {}
        if delta is not None:
            kwargs["delta"] = delta
        if threshold is not None:
            kwargs["threshold"] = threshold
        if alpha is not None:
            kwargs["alpha"] = alpha
        if window_size is not None:
            kwargs["window_size"] = window_size

        detector = DriftDetector(algorithm=algorithm, **kwargs)
        _drift_detectors[detector_id] = detector

        return {
            "object": "drift_detector",
            "detector_id": detector_id,
            "algorithm": algorithm,
            "status": "created",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create_drift_detector: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/streaming/drift/update/{detector_id}")
async def update_drift_detector(detector_id: str, value: float):
    """
    Update a drift detector with a new value.

    Args:
        detector_id: ID of the detector (from create endpoint)
        value: New data point

    Returns drift detection result.
    """
    try:
        if detector_id not in _drift_detectors:
            raise HTTPException(
                status_code=404,
                detail=f"Detector '{detector_id}' not found",
            )

        detector = _drift_detectors[detector_id]
        result = detector.update(value)

        return {
            "object": "drift_update",
            "detector_id": detector_id,
            "drift_detected": result["drift_detected"],
            "index": result["index"],
            "value": result["value"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in update_drift_detector: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/streaming/drift/state/{detector_id}")
async def get_drift_detector_state(detector_id: str):
    """
    Get the current state of a drift detector.
    """
    try:
        if detector_id not in _drift_detectors:
            raise HTTPException(
                status_code=404,
                detail=f"Detector '{detector_id}' not found",
            )

        detector = _drift_detectors[detector_id]
        state = detector.get_state()

        return {
            "object": "drift_detector_state",
            "detector_id": detector_id,
            **state,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_drift_detector_state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/v1/streaming/drift/{detector_id}")
async def delete_drift_detector(detector_id: str):
    """
    Delete a drift detector.
    """
    try:
        if detector_id not in _drift_detectors:
            raise HTTPException(
                status_code=404,
                detail=f"Detector '{detector_id}' not found",
            )

        del _drift_detectors[detector_id]

        return {
            "object": "drift_detector",
            "detector_id": detector_id,
            "status": "deleted",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_drift_detector: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# ANOMALY EXPLANATION ENDPOINTS
# =============================================================================


@app.post("/v1/anomaly/explain")
async def explain_anomaly(request: AnomalyExplainRequest):
    """
    Explain why data points are flagged as anomalies using SHAP.

    SHAP (SHapley Additive exPlanations) provides feature importance
    showing which features contributed most to the anomaly score.

    Requires a trained anomaly detection model.

    Example request:
    ```json
    {
        "model_id": "my-anomaly-model",
        "data": [[95.0, 20.0, 150.0]],
        "feature_names": ["cpu", "memory", "latency"],
        "background_samples": 100
    }
    ```

    Response:
    ```json
    {
        "object": "anomaly_explanation",
        "explanations": [{
            "features": [
                {"feature": "cpu", "importance": 0.82, "value": 95.0, "direction": "high"},
                {"feature": "latency", "importance": 0.45, "value": 150.0, "direction": "high"},
                {"feature": "memory", "importance": 0.12, "value": 20.0, "direction": "low"}
            ],
            "top_contributors": [...]
        }]
    }
    ```
    """
    try:
        import numpy as np

        from utils.anomaly_explainer import create_explainer_for_sklearn

        # Get the trained model using the same cache key pattern as /v1/anomaly/fit
        cache_key = _make_anomaly_cache_key(
            request.model_id, request.backend, request.normalization, request.scaler_type
        )

        if cache_key not in _models:
            raise HTTPException(
                status_code=404,
                detail=f"Anomaly model '{request.model_id}' not found. Train a model first with /v1/anomaly/fit",
            )

        model = _models[cache_key]

        if model is None or not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_id}' is not trained. Train it first with /v1/anomaly/fit",
            )

        # Get the sklearn model (IsolationForest, etc.)
        sklearn_model = model.get_sklearn_model()
        if sklearn_model is None:
            raise HTTPException(
                status_code=400,
                detail="Model does not support SHAP explanations (no sklearn model)",
            )

        # Convert data to numpy array
        data = np.array(request.data, dtype=np.float64)

        # Get background data from the model's training data
        training_data = model.get_training_data()
        if training_data is None or len(training_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Model has no training data for SHAP background",
            )

        # Create explainer
        explainer = create_explainer_for_sklearn(
            model=sklearn_model,
            background_data=training_data,
            feature_names=request.feature_names,
            n_background_samples=request.background_samples,
        )

        # Get explanations
        explanations = explainer.explain(data, nsamples=request.nsamples)

        return {
            "object": "anomaly_explanation",
            "model_id": request.model_id,
            "explanations": explanations,
            "total_points": len(explanations),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in explain_anomaly: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/dataset/audit")
async def audit_dataset(request: DatasetAuditRequest):
    """
    Audit dataset quality using Cleanlab.

    Identifies potential label errors, near-duplicates, and quality issues
    in classification datasets.

    Requires prediction probabilities from a trained classifier. You can:
    1. Use our /v1/classify endpoint to get predictions
    2. Use your own classifier's predict_proba output

    Example request:
    ```json
    {
        "labels": [0, 1, 0, 1, 0],
        "pred_probs": [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.85, 0.15]],
        "label_names": ["cat", "dog"],
        "check_duplicates": true
    }
    ```

    Response:
    ```json
    {
        "object": "dataset_audit",
        "label_issues": [
            {
                "index": 2,
                "given_label": 0,
                "given_label_name": "cat",
                "suggested_label": 1,
                "suggested_label_name": "dog",
                "given_confidence": 0.3,
                "suggested_confidence": 0.7
            }
        ],
        "duplicates": [...],
        "summary": {
            "total_samples": 5,
            "label_issue_count": 1,
            "label_issue_rate": 0.2
        }
    }
    ```
    """
    try:
        import numpy as np

        from utils.dataset_auditor import DatasetAuditor

        # Convert to numpy array
        pred_probs = np.array(request.pred_probs, dtype=np.float64)
        features = None
        if request.features is not None:
            features = np.array(request.features, dtype=np.float64)

        # Create auditor
        auditor = DatasetAuditor(label_names=request.label_names)

        # Run audit
        result = auditor.audit(
            labels=request.labels,
            pred_probs=pred_probs,
            features=features,
            check_duplicates=request.check_duplicates,
            duplicate_threshold=request.duplicate_threshold,
        )

        return {
            "object": "dataset_audit",
            **result,
        }

    except Exception as e:
        logger.error(f"Error in audit_dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/dataset/quality-scores")
async def get_quality_scores(request: DatasetAuditRequest):
    """
    Get per-sample label quality scores.

    Returns a quality score for each sample (0-1, higher = better quality).
    Low scores indicate potentially mislabeled examples.

    Example request:
    ```json
    {
        "labels": [0, 1, 0, 1, 0],
        "pred_probs": [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.85, 0.15]]
    }
    ```

    Response:
    ```json
    {
        "object": "quality_scores",
        "scores": [0.95, 0.82, 0.31, 0.93, 0.88],
        "mean_quality": 0.78
    }
    ```
    """
    try:
        import numpy as np

        from utils.dataset_auditor import DatasetAuditor

        pred_probs = np.array(request.pred_probs, dtype=np.float64)

        auditor = DatasetAuditor(label_names=request.label_names)
        scores = auditor.get_label_quality_scores(request.labels, pred_probs)

        return {
            "object": "quality_scores",
            "scores": scores.tolist(),
            "mean_quality": float(np.mean(scores)),
            "min_quality": float(np.min(scores)),
            "max_quality": float(np.max(scores)),
        }

    except Exception as e:
        logger.error(f"Error in get_quality_scores: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
# Speech-to-Text Endpoints (Whisper-based transcription)
# ============================================================================

# Safe audio file extensions (whitelist for security)
SAFE_AUDIO_EXTENSIONS = frozenset({
    ".wav", ".mp3", ".m4a", ".webm", ".flac", ".ogg", ".mp4", ".opus",
})

# Silence detection threshold for decoded Opus audio (higher due to noise floor)
SILENCE_THRESHOLD_OPUS = 0.03


def _make_speech_cache_key(model_id: str, compute_type: str | None = None) -> str:
    """Generate a cache key for a speech model.

    Args:
        model_id: Model size/name (e.g., "large-v3", "distil-large-v3")
        compute_type: Compute type for inference

    Returns:
        Cache key string
    """
    ct_key = compute_type if compute_type is not None else "auto"
    return f"speech:{model_id}:{ct_key}"


async def load_speech(
    model_id: str = "distil-large-v3",
    compute_type: str | None = None,
) -> SpeechModel:
    """Load a speech-to-text model.

    Args:
        model_id: Model size/name (e.g., "large-v3", "distil-large-v3", "medium")
        compute_type: Compute type for inference (auto-selected based on device if None)

    Returns:
        Loaded SpeechModel instance
    """
    cache_key = _make_speech_cache_key(model_id, compute_type)

    if cache_key not in _models:
        async with _model_load_lock:
            if cache_key not in _models:
                logger.info(f"Loading speech model: {model_id}")
                device = get_device()

                model = SpeechModel(
                    model_id=model_id,
                    device=device,
                    compute_type=compute_type,
                )

                await model.load()
                _models[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return _models.get(cache_key)


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = None,
    model: str = Form(default="distil-large-v3"),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: str | None = Form(default=None),
    stream: bool = Form(default=False),
):
    """
    OpenAI-compatible audio transcription endpoint.

    Transcribe audio files to text using Whisper models. Supports multiple
    model sizes, languages, and output formats.

    **Supported audio formats:** mp3, wav, m4a, webm, flac, ogg, mp4

    **Model sizes:**
    - tiny, base, small: Fast, lower accuracy
    - medium: Good balance of speed and accuracy
    - large-v3: Best accuracy, slower
    - distil-large-v3: Near large-v3 accuracy, much faster (recommended)

    **Streaming:** Set `stream=true` to receive transcription segments via SSE
    as they're processed, rather than waiting for the complete transcription.

    Example with curl:
    ```bash
    curl -X POST http://localhost:11540/v1/audio/transcriptions \\
        -F "file=@audio.mp3" \\
        -F "model=distil-large-v3" \\
        -F "language=en" \\
        -F "response_format=json"
    ```

    Example streaming:
    ```bash
    curl -X POST http://localhost:11540/v1/audio/transcriptions \\
        -F "file=@audio.mp3" \\
        -F "model=distil-large-v3" \\
        -F "stream=true"
    ```
    """
    import json
    import tempfile
    from pathlib import Path

    from fastapi.responses import StreamingResponse

    try:
        # Get audio content from file upload or file_id
        audio_bytes: bytes | None = None
        file_extension = ".wav"

        if file is not None:
            audio_bytes = await file.read()
            if file.filename:
                # Sanitize file extension against whitelist
                ext = Path(file.filename).suffix.lower()
                file_extension = ext if ext in SAFE_AUDIO_EXTENSIONS else ".wav"
        else:
            raise HTTPException(
                status_code=400,
                detail="Audio file is required. Upload via 'file' field.",
            )

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty audio file",
            )

        # Detect actual audio format from content (don't trust file extension)
        from utils.audio_buffer import (
            decode_audio_bytes,
            detect_audio_format,
            pcm_to_wav,
        )

        format_name, is_compressed = detect_audio_format(audio_bytes)
        logger.debug(f"Detected audio format: {format_name} (compressed={is_compressed})")

        # If audio is compressed, decode to WAV for reliable processing
        if is_compressed:
            try:
                pcm_data = decode_audio_bytes(audio_bytes)
                audio_bytes = pcm_to_wav(pcm_data)
                file_extension = ".wav"
                logger.debug(f"Decoded {format_name} to WAV ({len(audio_bytes)} bytes)")
            except Exception as e:
                logger.warning(f"Failed to decode {format_name}: {e}, using original data")
                # Fall back to original data - faster-whisper might handle it

        # Load speech model
        speech_model = await load_speech(model_id=model)
        if speech_model is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to load speech model",
            )

        # Parse timestamp granularities
        word_timestamps = False
        if timestamp_granularities:
            granularities = [g.strip() for g in timestamp_granularities.split(",")]
            word_timestamps = "word" in granularities

        # Write audio to temp file (faster-whisper requires file path)
        # Assign tmp_path before write to ensure cleanup even if write fails
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=file_extension, delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)
            if stream:
                # Streaming response - yield segments as they're transcribed
                async def generate_sse():
                    async for segment in speech_model.transcribe_stream(
                        audio_path=tmp_path,
                        language=language,
                        word_timestamps=word_timestamps,
                        initial_prompt=prompt,
                    ):
                        segment_data = {
                            "id": segment.id,
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text,
                        }
                        if segment.words:
                            segment_data["words"] = segment.words

                        yield f"data: {json.dumps(segment_data)}\n\n"

                    yield "data: [DONE]\n\n"

                # Use BackgroundTasks to ensure temp file cleanup even on client disconnect
                background_tasks.add_task(Path(tmp_path).unlink, missing_ok=True)

                return StreamingResponse(
                    generate_sse(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                    background=background_tasks,
                )

            # Non-streaming response
            result = await speech_model.transcribe(
                audio_path=tmp_path,
                language=language,
                word_timestamps=word_timestamps,
                initial_prompt=prompt,
                temperature=[temperature] if temperature > 0 else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            )

            # Format response based on requested format
            if response_format == "text":
                return result.text

            if response_format == "srt":
                # Generate SRT subtitle format
                srt_lines = []
                for i, seg in enumerate(result.segments, 1):
                    start_time = _format_timestamp_srt(seg.start)
                    end_time = _format_timestamp_srt(seg.end)
                    srt_lines.append(f"{i}")
                    srt_lines.append(f"{start_time} --> {end_time}")
                    srt_lines.append(seg.text.strip())
                    srt_lines.append("")
                return "\n".join(srt_lines)

            if response_format == "vtt":
                # Generate WebVTT subtitle format
                vtt_lines = ["WEBVTT", ""]
                for seg in result.segments:
                    start_time = _format_timestamp_vtt(seg.start)
                    end_time = _format_timestamp_vtt(seg.end)
                    vtt_lines.append(f"{start_time} --> {end_time}")
                    vtt_lines.append(seg.text.strip())
                    vtt_lines.append("")
                return "\n".join(vtt_lines)

            if response_format == "verbose_json":
                # Detailed JSON with segments
                return {
                    "task": "transcribe",
                    "language": result.language,
                    "duration": result.duration,
                    "text": result.text,
                    "segments": [
                        {
                            "id": seg.id,
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text,
                            "words": seg.words,
                            "avg_logprob": seg.avg_logprob,
                            "no_speech_prob": seg.no_speech_prob,
                        }
                        for seg in result.segments
                    ],
                }

            # Default: simple JSON
            return {
                "text": result.text,
            }

        finally:
            # Clean up temp file (if not streaming)
            if not stream and tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    except ImportError as e:
        logger.error(f"Speech model dependencies not installed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Speech-to-text not available. Install with: uv pip install 'universal-runtime[speech]'. Error: {e}",
        ) from e
    except Exception as e:
        logger.error(f"Error in create_transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


@app.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile,
    model: str = Form(default="distil-large-v3"),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    """
    OpenAI-compatible audio translation endpoint.

    Translate audio to English text. Works the same as transcription but
    always outputs English regardless of the input language.

    Example:
    ```bash
    curl -X POST http://localhost:11540/v1/audio/translations \\
        -F "file=@french_audio.mp3" \\
        -F "model=distil-large-v3"
    ```
    """
    import tempfile
    from pathlib import Path

    from utils.audio_buffer import decode_audio_bytes, detect_audio_format, pcm_to_wav

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        file_extension = Path(file.filename).suffix if file.filename else ".wav"

        # Detect actual audio format from content (don't trust file extension)
        format_name, is_compressed = detect_audio_format(audio_bytes)
        logger.debug(f"Detected audio format: {format_name} (compressed={is_compressed})")

        # If audio is compressed, decode to WAV for reliable processing
        if is_compressed:
            try:
                pcm_data = decode_audio_bytes(audio_bytes)
                audio_bytes = pcm_to_wav(pcm_data)
                file_extension = ".wav"
                logger.debug(f"Decoded {format_name} to WAV ({len(audio_bytes)} bytes)")
            except Exception as e:
                logger.warning(f"Failed to decode {format_name}: {e}, using original data")

        # Load speech model
        speech_model = await load_speech(model_id=model)
        if speech_model is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to load speech model",
            )

        # Write to temp file
        # Assign tmp_path before write to ensure cleanup even if write fails
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=file_extension, delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)

            # Transcribe with translation task
            result = await speech_model.transcribe(
                audio_path=tmp_path,
                task="translate",  # Translate to English
                initial_prompt=prompt,
                temperature=[temperature] if temperature > 0 else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            )

            if response_format == "text":
                return result.text

            return {
                "text": result.text,
            }

        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    except ImportError as e:
        logger.error(f"Speech model dependencies not installed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Speech-to-text not available. Install with: uv pip install 'universal-runtime[speech]'. Error: {e}",
        ) from e
    except Exception as e:
        logger.error(f"Error in create_translation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.websocket("/v1/audio/transcriptions/stream")
async def websocket_transcription(
    websocket: WebSocket,
    model: str = "base",
    language: str | None = None,
    word_timestamps: bool = False,
    chunk_interval: float = 2.0,
):
    """
    WebSocket endpoint for real-time audio streaming transcription.

    Connect via WebSocket and send audio chunks to receive live transcription.
    Audio should be sent as binary messages (raw PCM: 16kHz, 16-bit, mono).

    **IMPORTANT - Model Selection for Real-Time:**
    For real-time on CPU, use small models:
    - "tiny": ~0.5s to process 2s audio - fastest, lower accuracy
    - "base": ~1-2s to process 2s audio - good balance (DEFAULT)
    - "small": ~3-4s to process 2s audio - better accuracy

    Larger models (medium, large-v3, distil-large-v3) require GPU for real-time.

    **Protocol:**
    1. Connect with query params (model, language, chunk_interval)
    2. Send binary audio chunks (raw PCM: 16kHz, 16-bit, mono)
    3. Receive JSON transcription segments as processed
    4. Send text "END" to flush remaining audio and close

    **Query Parameters:**
    - model: Whisper model (default: "base" for CPU real-time)
    - language: ISO language code (auto-detect if not set)
    - word_timestamps: Include word-level timestamps (default: false)
    - chunk_interval: Seconds of audio per chunk (default: 2.0)

    **Response format:**
    ```json
    {"type": "segment", "id": 0, "start": 0.0, "end": 2.0, "text": "Hello", "is_final": false}
    ```

    **Example (JavaScript):**
    ```javascript
    const ws = new WebSocket('ws://localhost:11540/v1/audio/transcriptions/stream');
    ws.onmessage = (e) => console.log(JSON.parse(e.data).text);
    mediaRecorder.ondataavailable = (e) => ws.send(e.data);
    ws.send('END');  // When done
    ```
    """
    import json
    import tempfile
    from pathlib import Path

    from utils.audio_buffer import (
        StreamingAudioBuffer,
        detect_audio_format,
        is_silence,
    )

    await websocket.accept()
    logger.info(
        f"WebSocket connection opened for transcription "
        f"(model={model}, chunk_interval={chunk_interval}s)"
    )

    # Initialize audio buffer with time-based chunking for predictable output
    audio_buffer = StreamingAudioBuffer(
        min_speech_duration=0.5,
        max_speech_duration=30.0,
        chunk_interval=chunk_interval,
    )

    # Track if we've warned about compressed audio (only warn once)
    compressed_audio_warned = False

    # Load speech model
    try:
        speech_model = await load_speech(model_id=model)
        if speech_model is None:
            await websocket.send_json({
                "type": "error",
                "message": "Failed to load speech model",
            })
            await websocket.close(code=1011)
            return
    except ImportError as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Speech dependencies not installed: {e}",
        })
        await websocket.close(code=1011)
        return

    # Warn about CPU performance with large models
    large_models = {"medium", "large", "large-v1", "large-v2", "large-v3", "distil-large-v3"}
    if speech_model.device == "cpu" and model in large_models:
        logger.warning(
            f"Using '{model}' on CPU - real-time transcription not possible. "
            f"Use 'tiny', 'base', or 'small' for real-time on CPU, or use GPU."
        )
        await websocket.send_json({
            "type": "warning",
            "message": f"Model '{model}' is too slow for real-time on CPU. "
            f"Consider using 'base' or 'tiny' instead.",
        })

    segment_id = 0
    cumulative_offset = 0.0  # Track total time offset across chunks

    async def transcribe_audio(wav_bytes: bytes, is_final: bool = False) -> None:
        """Transcribe audio and send results."""
        nonlocal segment_id, cumulative_offset
        import asyncio
        import concurrent.futures

        # Write to temp file
        # Assign tmp_path before write to ensure cleanup even if write fails
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(wav_bytes)

            def sync_transcribe():
                """Run transcription synchronously in thread pool."""
                # Use faster settings for real-time streaming:
                # - vad_filter=False: We handle chunking, don't filter our chunks
                # - beam_size=1: Greedy decoding for speed
                # - best_of=1: No multiple candidates
                segments, info = speech_model._whisper_model.transcribe(
                    tmp_path,
                    language=language,
                    word_timestamps=word_timestamps,
                    vad_filter=False,  # Critical: don't filter our pre-chunked audio
                    beam_size=1,  # Greedy decoding for speed
                    best_of=1,
                    temperature=0.0,  # Deterministic
                    no_speech_threshold=1.0,  # Don't skip segments marked as "no speech"
                    log_prob_threshold=-2.0,  # Lower threshold to avoid skipping low-confidence
                    compression_ratio_threshold=3.0,  # Higher threshold for repetitive text
                )
                return list(segments), info

            # Run transcription in thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                segments, info = await loop.run_in_executor(pool, sync_transcribe)

            # Send segments with adjusted timestamps
            for segment in segments:
                # Skip segments with high probability of being non-speech
                # This helps prevent Whisper hallucinations like "I'm sorry" on silence
                if hasattr(segment, "no_speech_prob") and segment.no_speech_prob > 0.6:
                    logger.debug(
                        f"Skipping segment with high no_speech_prob: "
                        f"{segment.no_speech_prob:.2f} - '{segment.text.strip()}'"
                    )
                    continue

                words = None
                if word_timestamps and segment.words:
                    words = [
                        {
                            "word": w.word,
                            "start": w.start + cumulative_offset,
                            "end": w.end + cumulative_offset,
                            "probability": w.probability,
                        }
                        for w in segment.words
                    ]

                response = {
                    "type": "final" if is_final else "segment",
                    "id": segment_id,
                    "start": segment.start + cumulative_offset,
                    "end": segment.end + cumulative_offset,
                    "text": segment.text.strip(),
                    "is_final": is_final,
                }
                if words:
                    response["words"] = words

                # Only send if there's actual text
                if response["text"]:
                    await websocket.send_json(response)
                    segment_id += 1

            # Update cumulative offset for next chunk
            cumulative_offset += chunk_interval

        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    try:
        while True:
            # Receive message (binary audio or text command)
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "text" in message:
                text = message["text"]
                if text.upper() == "END":
                    # Flush remaining audio
                    remaining = audio_buffer.flush()
                    if remaining:
                        await transcribe_audio(remaining, is_final=True)

                    await websocket.send_json({
                        "type": "done",
                        "message": "Transcription complete",
                    })
                    break

                # Handle JSON config messages
                try:
                    config = json.loads(text)
                    if "language" in config:
                        language = config["language"]
                        logger.info(f"Updated language to: {language}")
                except json.JSONDecodeError:
                    pass

            elif "bytes" in message:
                # Process audio chunk
                audio_data = message["bytes"]

                # Check for compressed audio and warn (once) about performance
                if not compressed_audio_warned:
                    format_name, is_compressed = detect_audio_format(audio_data)
                    if is_compressed:
                        compressed_audio_warned = True
                        logger.warning(
                            f"Receiving compressed audio ({format_name}). "
                            "For better real-time performance, send raw PCM "
                            "(16kHz, 16-bit, mono)."
                        )
                        await websocket.send_json({
                            "type": "warning",
                            "message": f"Compressed audio detected ({format_name}). "
                            "For better real-time performance, configure your client "
                            "to send raw PCM audio (16kHz, 16-bit signed, mono). "
                            "Decoding adds ~50-200ms latency per chunk.",
                        })

                # Add to buffer and check if we should transcribe
                should_transcribe, wav_bytes = audio_buffer.add(audio_data)

                if should_transcribe and wav_bytes:
                    # Skip transcription if audio is silence (prevents Whisper hallucinations)
                    # Extract PCM from WAV for silence check (skip 44-byte header)
                    pcm_data = wav_bytes[44:] if wav_bytes[:4] == b"RIFF" else wav_bytes
                    if is_silence(pcm_data, threshold=SILENCE_THRESHOLD_OPUS):
                        logger.debug("Skipping silent audio chunk")
                        # Still update cumulative offset to maintain timing
                        cumulative_offset += chunk_interval
                        continue

                    await transcribe_audio(wav_bytes)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        # Only suppress connection-related errors when sending error response
        with suppress(WebSocketDisconnect, RuntimeError):
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
    finally:
        # Flush any remaining audio
        try:
            remaining = audio_buffer.flush()
            if remaining:
                await transcribe_audio(remaining, is_final=True)
        except Exception:
            pass

        logger.info("WebSocket connection closed")


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
        ws_ping_interval=30.0,  # Send ping every 30s (default: 20s)
        ws_ping_timeout=60.0,  # Wait 60s for pong (default: 20s) - allows for slow transcription
    )
