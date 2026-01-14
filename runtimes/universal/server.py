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
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.logging import UniversalRuntimeLogger, setup_logging
from models import BaseModel, GGUFLanguageModel, LanguageModel
from state import (
    cleanup_idle_models,
    get_device,
    get_model_load_lock,
    get_models_cache,
    set_cleanup_task,
    shutdown_models,
)
from utils.model_format import detect_model_format

# Configure logging FIRST, before anything else
log_file = os.getenv("LOG_FILE", "")
log_level = os.getenv("LOG_LEVEL", "INFO")
json_logs = os.getenv("LOG_JSON_FORMAT", "false").lower() in ("true", "1", "yes")
setup_logging(json_logs=json_logs, log_level=log_level, log_file=log_file)

logger = UniversalRuntimeLogger("universal-runtime")


# ============================================================================
# Language Model Loading (for chat_completions router)
# ============================================================================


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
    models_cache = get_models_cache()
    model_load_lock = get_model_load_lock()

    # Include n_ctx and quantization in cache key for GGUF models so different configurations are cached separately
    cache_key = _make_language_cache_key(model_id, n_ctx, preferred_quantization)
    if cache_key not in models_cache:
        async with model_load_lock:
            # Double-check if model was loaded while waiting for the lock
            if cache_key not in models_cache:
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
                models_cache[cache_key] = model

    # Return model (get() refreshes TTL automatically)
    return models_cache.get(cache_key)


# ============================================================================
# Application Lifecycle
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle (startup and shutdown)."""
    # Startup
    logger.info("Starting Universal Runtime")

    # Start model cleanup background task
    cleanup_task = asyncio.create_task(cleanup_idle_models())
    set_cleanup_task(cleanup_task)
    logger.info("Model cleanup background task started")

    yield

    # Shutdown
    logger.info("Shutting down Universal Runtime")
    await shutdown_models()
    logger.info("Shutdown complete")


# ============================================================================
# FastAPI Application
# ============================================================================

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

# ============================================================================
# Register Routers
# ============================================================================

# Import routers (intentionally after app creation to avoid circular imports)
from routers.anomaly import router as anomaly_router  # noqa: E402
from routers.chat_completions import router as chat_completions_router  # noqa: E402
from routers.classifier import router as classifier_router  # noqa: E402
from routers.core import router as core_router  # noqa: E402
from routers.documents import router as documents_router  # noqa: E402
from routers.embeddings import router as embeddings_router  # noqa: E402
from routers.files import router as files_router  # noqa: E402
from routers.ocr import router as ocr_router  # noqa: E402

# Register all routers
app.include_router(core_router)
app.include_router(chat_completions_router)
app.include_router(files_router)
app.include_router(embeddings_router)
app.include_router(documents_router)
app.include_router(ocr_router)
app.include_router(anomaly_router)
app.include_router(classifier_router)


# ============================================================================
# Main Entry Point
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
    )
