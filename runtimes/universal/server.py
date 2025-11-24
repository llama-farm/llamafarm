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
from typing import Literal

from fastapi import (
    FastAPI,
    HTTPException,
)
from pydantic import BaseModel as PydanticBaseModel

from core.logging import UniversalRuntimeLogger, setup_logging
from models import (
    BaseModel,
    EncoderModel,
    GGUFEncoderModel,
    GGUFLanguageModel,
    LanguageModel,
)
from routers.chat_completions import router as chat_completions_router
from utils.device import get_device_info, get_optimal_device
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
) -> str:
    """Generate a cache key for an encoder model.

    Args:
        model_id: HuggingFace model identifier
        task: Model task - "embedding" or "classification"
        model_format: Model format - "gguf" or "transformers"
        preferred_quantization: Optional quantization preference for GGUF models

    Returns:
        A unique cache key string that identifies this specific model configuration
    """
    quant_key = (
        preferred_quantization if preferred_quantization is not None else "default"
    )
    return f"encoder:{task}:{model_format}:{model_id}:quant{quant_key}"


async def load_encoder(
    model_id: str, task: str = "embedding", preferred_quantization: str | None = None
):
    """Load an encoder model for embeddings or classification (GGUF or transformers format).

    Automatically detects whether the model is in GGUF or transformers format
    and loads it with the appropriate backend. GGUF models use llama-cpp-python
    for optimized inference, while transformers models use the standard HuggingFace
    transformers library.

    Args:
        model_id: HuggingFace model identifier
        task: Model task - "embedding" or "classification"
        preferred_quantization: Optional quantization preference for GGUF models
                                (e.g., "Q4_K_M", "Q8_0"). If None, defaults to Q4_K_M.
                                Only downloads the specified quantization to save disk space.
    """
    # Detect model format for proper caching and loading
    model_format = detect_model_format(model_id)
    # Include quantization in cache key for GGUF models so different quantizations are cached separately
    # Use "default" for None quantization to use Q4_K_M default
    cache_key = _make_encoder_cache_key(
        model_id, task, model_format, preferred_quantization
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
                    model = EncoderModel(model_id, device, task=task)

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
