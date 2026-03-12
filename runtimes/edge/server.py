"""
LlamaFarm Edge Runtime

A stripped-down FastAPI server for on-device inference.
Designed for constrained hardware (Raspberry Pi, Jetson, etc.)

Supports:
- LLM inference (GGUF via llama.cpp)
- Vision detection (YOLO)
- Health checks

This is the "runtime plane" — no RAG, no UI, no model management.
Models are pre-loaded on device.

Environment Variables:
- MODEL_UNLOAD_TIMEOUT: Seconds of inactivity before unloading models (default: 300)
- CLEANUP_CHECK_INTERVAL: Seconds between cleanup checks (default: 30)
- LF_RUNTIME_PORT: Server port (default: 11540)
- LF_RUNTIME_HOST: Server host (default: 0.0.0.0)
"""

import asyncio
import os
import warnings
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.logging import UniversalRuntimeLogger, setup_logging
from models import (
    BaseModel,
    GGUFLanguageModel,
    LanguageModel,
)
from routers.chat_completions import router as chat_completions_router
from routers.chat_completions.service import ChatCompletionsService
from routers.health import (
    router as health_router,
    set_device_info_getter,
    set_models_cache,
)
from routers.vision import (
    router as vision_router,
    set_detection_loader,
    set_classification_loader,
    set_detect_classify_loaders,
    set_streaming_detection_loader,
    start_session_cleanup,
    stop_session_cleanup,
)
from utils.device import get_device_info, get_optimal_device
from utils.file_handler import get_file_images
from utils.model_cache import ModelCache
from utils.model_format import detect_model_format
from utils.safe_home import get_data_dir

# Suppress spurious warnings
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be \d+ leaked semaphore",
    category=UserWarning,
)

# Configure logging
log_file = os.getenv("LOG_FILE", "")
log_level = os.getenv("LOG_LEVEL", "INFO")
json_logs = os.getenv("LOG_JSON_FORMAT", "false").lower() in ("true", "1", "yes")
setup_logging(json_logs=json_logs, log_level=log_level, log_file=log_file)

logger = UniversalRuntimeLogger("edge-runtime")


def _init_llama_backend():
    """Initialize llama.cpp backend in the main thread.
    Critical for Jetson/Tegra CUDA stability on unified memory architectures.
    """
    try:
        from llamafarm_llama._bindings import ensure_backend
        logger.info("Initializing llama.cpp backend in main thread...")
        ensure_backend()
        logger.info("llama.cpp backend initialized successfully")
    except ImportError:
        logger.debug("llamafarm_llama not installed, skipping backend init")
    except Exception as e:
        logger.warning(f"Failed to initialize llama.cpp backend: {e}")


_init_llama_backend()


# Model unload timeout configuration
MODEL_UNLOAD_TIMEOUT = int(os.getenv("MODEL_UNLOAD_TIMEOUT", "300"))
CLEANUP_CHECK_INTERVAL = int(os.getenv("CLEANUP_CHECK_INTERVAL", "30"))

# Model cache
_models: ModelCache[BaseModel] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
_model_load_lock = asyncio.Lock()
_current_device = None
_cleanup_task: asyncio.Task | None = None

# Data directories
_LF_DATA_DIR = get_data_dir()
VISION_MODELS_DIR = _LF_DATA_DIR / "models" / "vision"


def get_device():
    """Get the optimal device for the current platform."""
    global _current_device
    if _current_device is None:
        _current_device = get_optimal_device()
        logger.info(f"Using device: {_current_device}")
    return _current_device


async def _cleanup_idle_models() -> None:
    """Background task that periodically unloads idle models."""
    logger.info(
        f"Model cleanup task started (timeout={MODEL_UNLOAD_TIMEOUT}s, "
        f"check_interval={CLEANUP_CHECK_INTERVAL}s)"
    )
    while True:
        try:
            await asyncio.sleep(CLEANUP_CHECK_INTERVAL)
            expired_items = _models.pop_expired()
            if expired_items:
                logger.info(f"Unloading {len(expired_items)} idle models")
                for cache_key, model in expired_items:
                    try:
                        await model.unload()
                        logger.info(f"Successfully unloaded: {cache_key}")
                    except Exception as e:
                        logger.error(f"Error unloading model {cache_key}: {e}")
        except asyncio.CancelledError:
            logger.info("Model cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}", exc_info=True)


# ============================================================================
# Language Model Loading
# ============================================================================


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
    quant_key = preferred_quantization or "default"
    cache_key = (
        f"language:{model_id}:ctx{n_ctx or 'auto'}:gpu{n_gpu_layers or 'auto'}:"
        f"quant{quant_key}"
    )

    if cache_key not in _models:
        async with _model_load_lock:
            if cache_key not in _models:
                logger.info(f"Loading causal LM: {model_id}")
                device = get_device()
                model_format = detect_model_format(model_id)
                logger.info(f"Detected format: {model_format}")

                model: BaseModel
                if model_format == "gguf":
                    model = GGUFLanguageModel(
                        model_id, device,
                        n_ctx=n_ctx, n_batch=n_batch,
                        n_gpu_layers=n_gpu_layers, n_threads=n_threads,
                        flash_attn=flash_attn, use_mmap=use_mmap,
                        use_mlock=use_mlock, cache_type_k=cache_type_k,
                        cache_type_v=cache_type_v,
                        preferred_quantization=preferred_quantization,
                    )
                else:
                    model = LanguageModel(model_id, device)

                await model.load()
                _models[cache_key] = model

    return _models.get(cache_key)


# ============================================================================
# Vision Model Loading
# ============================================================================


async def load_detection_model(model_id: str = "yolov8n"):
    """Load a YOLO detection model."""
    cache_key = f"vision:detect:{model_id}"
    if cache_key not in _models:
        async with _model_load_lock:
            if cache_key not in _models:
                from models.yolo_model import YOLOModel
                from pathlib import Path as _Path
                device = get_device()
                safe_id = _Path(model_id).name
                if safe_id != model_id:
                    raise ValueError(f"Invalid model_id: {model_id}")
                custom_path = VISION_MODELS_DIR / safe_id / "current.pt"
                mid = str(custom_path) if custom_path.exists() else model_id
                model = YOLOModel(model_id=mid, device=device)
                await model.load()
                _models[cache_key] = model
    return _models[cache_key]


async def load_classification_model(model_id: str = "clip-vit-base"):
    """Load a CLIP classification model."""
    cache_key = f"vision:classify:{model_id}"
    if cache_key not in _models:
        async with _model_load_lock:
            if cache_key not in _models:
                from models.clip_model import CLIPModel
                device = get_device()
                model = CLIPModel(model_id=model_id, device=device)
                await model.load()
                _models[cache_key] = model
    return _models[cache_key]


# ============================================================================
# Lifecycle
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _cleanup_task

    logger.info("Starting LlamaFarm Edge Runtime")

    _cleanup_task = asyncio.create_task(_cleanup_idle_models())

    # Start KV cache manager
    from utils.kv_cache_manager import (
        KVCacheManager, start_kv_cache_gc, stop_kv_cache_gc,
    )
    global _kv_cache_manager
    _kv_cache_manager = KVCacheManager()
    from routers.cache import set_cache_manager, set_cache_language_loader
    set_cache_manager(_kv_cache_manager)
    set_cache_language_loader(load_language)
    ChatCompletionsService.set_cache_manager(_kv_cache_manager)
    start_kv_cache_gc(_kv_cache_manager)

    start_session_cleanup()

    yield

    # Shutdown
    logger.info("Shutting down Edge Runtime")
    await stop_kv_cache_gc()
    await stop_session_cleanup()

    if _cleanup_task is not None:
        _cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await _cleanup_task

    for cache_key, model in list(_models.items()):
        try:
            await model.unload()
        except Exception as e:
            logger.error(f"Error unloading {cache_key}: {e}")
    _models.clear()

    logger.info("Shutdown complete")


# ============================================================================
# App
# ============================================================================

_kv_cache_manager = None

app = FastAPI(
    title="LlamaFarm Edge Runtime",
    description="Minimal on-device inference API for drones and edge hardware",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Edge device — open CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only the routers the drone needs
app.include_router(health_router)
app.include_router(chat_completions_router)
app.include_router(vision_router)


@app.post("/v1/models/unload", tags=["models"])
async def unload_all_models():
    """Unload all loaded models to free memory."""
    unloaded = []
    for cache_key, model in list(_models.items()):
        try:
            await model.unload()
            unloaded.append(cache_key)
        except Exception as e:
            logger.error(f"Error unloading {cache_key}: {e}")
    _models.clear()
    return {"unloaded": len(unloaded), "models": unloaded}


# ============================================================================
# Router Dependency Injection
# ============================================================================

set_models_cache(_models)
set_device_info_getter(get_device_info)
set_detection_loader(load_detection_model)
set_classification_loader(load_classification_model)
set_detect_classify_loaders(load_detection_model, load_classification_model)
set_streaming_detection_loader(load_detection_model)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    from llamafarm_common.pidfile import write_pid

    write_pid("edge-runtime")

    port = int(os.getenv("LF_RUNTIME_PORT", os.getenv("PORT", "11540")))
    host = os.getenv("LF_RUNTIME_HOST", os.getenv("HOST", "0.0.0.0"))

    logger.info(f"Starting LlamaFarm Edge Runtime on {host}:{port}")
    logger.info(f"Device: {get_device()}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
    )
