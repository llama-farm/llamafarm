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
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    SpeechModel,
)
from routers.anomaly import (
    router as anomaly_router,
)
from routers.anomaly import (
    set_anomaly_loader,
)
from routers.anomaly import (
    set_state as set_anomaly_state,
)
from routers.audio import router as audio_router
from routers.audio import set_speech_loader
from routers.chat_completions import router as chat_completions_router
from routers.classifier import (
    router as classifier_router,
)
from routers.classifier import (
    set_classifier_loader,
)
from routers.classifier import (
    set_models_dir as set_classifier_models_dir,
)
from routers.classifier import (
    set_state as set_classifier_state,
)
from routers.files import router as files_router
from routers.health import (
    router as health_router,
)
from routers.health import (
    set_device_info_getter,
    set_models_cache,
)
from routers.nlp import router as nlp_router
from routers.nlp import set_encoder_loader
from routers.vision import (
    router as vision_router,
)
from routers.vision import (
    set_document_loader,
    set_file_image_getter,
    set_ocr_loader,
)
from utils.device import get_device_info, get_optimal_device
from utils.feature_encoder import FeatureEncoder
from utils.file_handler import get_file_images
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

# Include all routers
app.include_router(anomaly_router)
app.include_router(audio_router)
app.include_router(chat_completions_router)
app.include_router(classifier_router)
app.include_router(files_router)
app.include_router(health_router)
app.include_router(nlp_router)
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
_model_load_lock = asyncio.Lock()
_current_device = None

# Feature encoder cache for anomaly detection with mixed data types
_encoders: dict[str, FeatureEncoder] = {}
_cleanup_task: asyncio.Task | None = None

# Data directories
_LF_DATA_DIR = Path.home() / ".llamafarm"
CLASSIFIER_MODELS_DIR = _LF_DATA_DIR / "models" / "classifier"


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

            # Cleanup expired models from both caches
            for cache, cache_name in [
                (_models, "models"),
                (_classifiers, "classifiers"),
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


# ============================================================================
# Language Model Loading
# ============================================================================


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
    """Generate a cache key for a causal language model."""
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
    """Load a causal language model (GGUF or transformers format)."""
    cache_key = _make_language_cache_key(
        model_id,
        n_ctx,
        n_batch,
        n_gpu_layers,
        n_threads,
        flash_attn,
        use_mmap,
        use_mlock,
        cache_type_k,
        cache_type_v,
        preferred_quantization,
    )
    if cache_key not in _models:
        async with _model_load_lock:
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


# ============================================================================
# Encoder Model Loading
# ============================================================================


def _make_encoder_cache_key(
    model_id: str,
    task: str,
    model_format: str,
    preferred_quantization: str | None = None,
    max_length: int | None = None,
) -> str:
    """Generate a cache key for an encoder model."""
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
    """Load an encoder model for embeddings, classification, reranking, or NER."""
    model_format = detect_model_format(model_id)
    cache_key = _make_encoder_cache_key(
        model_id, task, model_format, preferred_quantization, max_length
    )

    if cache_key not in _models:
        async with _model_load_lock:
            if cache_key not in _models:
                logger.info(
                    f"Loading encoder ({task}): {model_id} (format: {model_format})"
                )
                device = get_device()

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

    return _models.get(cache_key)


# ============================================================================
# Document Model Loading
# ============================================================================


def _make_document_cache_key(model_id: str, task: str) -> str:
    """Generate a cache key for a document model."""
    return f"document:{task}:{model_id}"


async def load_document(
    model_id: str,
    task: str = "extraction",
):
    """Load a document understanding model."""
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

    return _models.get(cache_key)


# ============================================================================
# OCR Model Loading
# ============================================================================


def _make_ocr_cache_key(backend: str, languages: list[str]) -> str:
    """Generate a cache key for an OCR model."""
    lang_key = "_".join(sorted(languages))
    return f"ocr:{backend}:{lang_key}"


async def load_ocr(backend: str = "surya", languages: list[str] | None = None):
    """Load an OCR model with the specified backend."""
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

    return _models.get(cache_key)


# ============================================================================
# Anomaly Model Loading
# ============================================================================


def _make_anomaly_cache_key(
    model_id: str, backend: str, normalization: str | None = None
) -> str:
    """Generate a cache key for an anomaly model."""
    if normalization:
        return f"anomaly:{backend}:{normalization}:{model_id}"
    return f"anomaly:{backend}:{model_id}"


async def load_anomaly(
    model_id: str,
    backend: str = "isolation_forest",
    contamination: float = 0.1,
    threshold: float | None = None,
    normalization: str = "standardization",
):
    """Load an anomaly detection model."""
    cache_key = _make_anomaly_cache_key(model_id, backend, normalization)

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
                )

                await model.load()
                _models[cache_key] = model

    return _models.get(cache_key)


# ============================================================================
# Classifier Model Loading
# ============================================================================


def _make_classifier_cache_key(model_name: str) -> str:
    """Create a cache key for classifier models."""
    return f"classifier:{model_name}"


async def load_classifier(
    model_id: str,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> "ClassifierModel":
    """Load or get cached classifier model."""
    cache_key = _make_classifier_cache_key(model_id)

    if cache_key not in _classifiers:
        async with _model_load_lock:
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

    return _classifiers.get(cache_key)


# ============================================================================
# Speech Model Loading
# ============================================================================


def _make_speech_cache_key(model_id: str, compute_type: str | None = None) -> str:
    """Generate a cache key for a speech model."""
    ct_key = compute_type if compute_type is not None else "auto"
    return f"speech:{model_id}:{ct_key}"


async def load_speech(
    model_id: str = "distil-large-v3",
    compute_type: str | None = None,
) -> SpeechModel:
    """Load a speech-to-text model."""
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

    return _models.get(cache_key)


# ============================================================================
# Router Dependency Injection
# ============================================================================

# Health router
set_models_cache(_models)
set_device_info_getter(get_device_info)

# NLP router
set_encoder_loader(load_encoder)

# Vision router
set_ocr_loader(load_ocr)
set_document_loader(load_document)
set_file_image_getter(get_file_images)

# Anomaly router
set_anomaly_loader(load_anomaly)
set_anomaly_state(_models, _encoders, _model_load_lock)

# Classifier router
set_classifier_loader(load_classifier)
set_classifier_models_dir(CLASSIFIER_MODELS_DIR)
set_classifier_state(_classifiers, _model_load_lock)

# Audio router
set_speech_loader(load_speech)


# ============================================================================
# Server Entry Point
# ============================================================================

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
