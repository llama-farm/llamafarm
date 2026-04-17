"""
LlamaFarm Edge Runtime

A stripped-down FastAPI server for on-device inference.
Designed for constrained hardware (Raspberry Pi, Jetson, etc.)

Supports:
- LLM inference (GGUF via llama.cpp)
- Vision detection (YOLO — Hailo-10H accelerated or CPU fallback)
- Health checks

This is the "runtime plane" — no RAG, no UI, no model management.
Models are pre-loaded on device.

Environment Variables:
- MODEL_UNLOAD_TIMEOUT: Seconds of inactivity before unloading models (default: 300)
- CLEANUP_CHECK_INTERVAL: Seconds between cleanup checks (default: 30)
- LF_RUNTIME_PORT: Server port (default: 11540)
- LF_RUNTIME_HOST: Server host (default: 0.0.0.0)
- HAILO_HEF_DIR: Directory containing .hef model files (default: /models)
- PRELOAD_MODELS: Comma-separated model IDs to load and pin at startup (default: unset)
- PRELOAD_N_CTX: Context size for preloaded models (default: auto-detected)
- FORCE_CPU_VISION: Set to "1" to skip Hailo detection and use CPU (default: unset)
"""

import asyncio
import functools
import os
import re
import subprocess
import sys
import warnings
from contextlib import asynccontextmanager, suppress

# Force UTF-8 on stdout/stderr before any logger is configured. On Windows
# the default console codec is cp1252, and llama.cpp's C→Python log callback
# routes native log output containing byte-level BPE markers (U+0120, U+010A)
# through Python logging, which would otherwise crash the log handler with
# UnicodeEncodeError on any non-latin1 character.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # `reconfigure` is Python 3.7+; ValueError is raised if stdout
        # has been replaced with a non-TextIOWrapper (e.g. pytest capture).
        # Falling back to the default codec is the correct behavior in
        # both cases — there's no safer action we can take here.
        pass

# Import the offline_mode bootstrap BEFORE any module that transitively
# imports huggingface_hub or transformers. The llamafarm_common package's
# __init__ imports offline_mode first, which reads LLAMAFARM_OFFLINE and
# sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE accordingly. If this import
# happened later, the offline env vars would be read by huggingface_hub
# before we had a chance to set them.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llamafarm_common import offline_mode as _offline_mode_bootstrap  # noqa: F401

from core.logging import UniversalRuntimeLogger, setup_logging
from models import (
    BaseModel,
    GGUFLanguageModel,
    LanguageModel,
)
from routers.chat_completions import router as chat_completions_router
from routers.chat_completions.service import ChatCompletionsService
from routers.completions import router as completions_router
from routers.health import (
    router as health_router,
)
from routers.health import (
    set_device_info_getter,
    set_models_cache,
)
from routers.vision import (
    router as vision_router,
)
from routers.vision import (
    set_classification_loader,
    set_detect_classify_loaders,
    set_detection_loader,
    set_streaming_detection_loader,
    start_session_cleanup,
    stop_session_cleanup,
)
from services.zenoh_ipc import ZenohIPC
from utils.device import get_device_info, get_optimal_device
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
_zenoh_ipc: ZenohIPC | None = None

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


# ============================================================================
# Hardware Detection
# ============================================================================

@functools.lru_cache(maxsize=1)
def _detect_hailo() -> bool:
    """Detect if Hailo-10H PCIe device is present.

    Checks for PCI device ID 1e60:45c4 (Hailo-10H) via lspci,
    and verifies hailo_platform is importable.
    """
    if os.getenv("FORCE_CPU_VISION", "").lower() in ("1", "true", "yes"):
        logger.info("Hailo detection skipped (FORCE_CPU_VISION=1)")
        return False

    # Check for hailo_platform package
    try:
        import hailo_platform  # noqa: F401
    except ImportError:
        logger.info("hailo_platform not installed, using CPU backend for vision")
        return False

    # Check for PCIe device
    try:
        result = subprocess.run(
            ["lspci", "-d", "1e60:"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            logger.info("Hailo-10H detected, using Hailo backend for vision")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # lspci not available (macOS) or timed out
        pass

    # Fallback: check for /dev/hailo0
    if os.path.exists("/dev/hailo0"):
        logger.info("Hailo device found at /dev/hailo0, using Hailo backend")
        return True

    logger.info("Hailo not detected, using CPU backend for vision")
    return False


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
    trusted: bool = False,
):
    """Load a causal language model (GGUF or transformers format).

    Args:
        trusted: Skip path traversal validation. Only set True for server-side
                 config (e.g. PRELOAD_MODELS env var), never for HTTP input.
    """
    # Reject path traversal and invalid IDs from untrusted input (HTTP requests)
    if not trusted:
        if (
            ".." in model_id
            or model_id.startswith(("/", "\\"))
            or "\\" in model_id
            or (len(model_id) > 1 and model_id[1] == ":")
        ):
            raise ValueError(f"Invalid model_id: {model_id}")

        # Allow only HuggingFace-style IDs (org/repo, org/repo:quant, repo,
        # repo:quant) and bare GGUF filenames. Blocks arbitrary relative paths.
        if not re.match(r"^[a-zA-Z0-9_.\-]+(/[a-zA-Z0-9_.\-]+)?(:[a-zA-Z0-9_.\-]+)?$", model_id):
            raise ValueError(f"Invalid model_id format: {model_id}")

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

                from utils.alias import derive_alias_from_model_id
                alias = derive_alias_from_model_id(model_id)
                if alias:
                    logger.debug(
                        f"Derived alias {alias!r} from model_id {model_id!r} "
                        f"for LLAMAFARM_MODEL_DIR lookup"
                    )

                # For alias-style model IDs (no org/ namespace), check
                # LLAMAFARM_MODEL_DIR first — avoids HuggingFace API
                # calls that fail in offline mode.  Namespaced IDs like
                # "org/model" always go through detect_model_format so a
                # local alias can't silently override a specific repo.
                model_format: str | None = None
                if alias and "/" not in model_id:
                    from llamafarm_common.model_dir import resolve_from_model_dir
                    if resolve_from_model_dir(alias) is not None:
                        model_format = "gguf"
                if model_format is None:
                    model_format = detect_model_format(model_id, trusted=trusted)
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
                        alias=alias,
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
    """Load a YOLO detection model.

    Auto-selects backend:
    - Hailo-10H: loads .hef model on the AI accelerator
    - CPU fallback: loads .pt model via ultralytics/PyTorch
    """
    backend = "hailo" if _detect_hailo() else "cpu"
    cache_key = f"vision:detect:{backend}:{model_id}"

    if cache_key not in _models:
        async with _model_load_lock:
            if cache_key not in _models:
                from pathlib import Path as _Path

                safe_id = _Path(model_id).name
                if safe_id != model_id or safe_id in (".", ".."):
                    raise ValueError(f"Invalid model_id: {model_id}")
                # Verify resolved path stays within VISION_MODELS_DIR
                vision_root = VISION_MODELS_DIR.resolve()
                resolved = (VISION_MODELS_DIR / safe_id).resolve()
                if not str(resolved).startswith(str(vision_root) + os.sep):
                    raise ValueError(f"Invalid model_id: {model_id}")

                if backend == "hailo":
                    from models.hailo_model import HailoYOLOModel

                    hef_dir = os.getenv("HAILO_HEF_DIR", "/models")
                    model = HailoYOLOModel(
                        model_id=model_id,
                        confidence_threshold=0.5,
                        hef_dir=hef_dir,
                    )
                else:
                    from models.yolo_model import YOLOModel

                    device = get_device()
                    custom_path = resolved / "current.pt"
                    mid = str(custom_path) if custom_path.exists() else model_id
                    model = YOLOModel(model_id=mid, device=device)

                await model.load()
                _models[cache_key] = model

    return _models[cache_key]


async def load_classification_model(model_id: str = "clip-vit-base"):
    """Load a CLIP classification model."""
    # Validate model_id: must be a known variant or a valid HuggingFace repo ID
    # (org/model format). Reject path-like IDs that could reach the filesystem.
    from models.clip_model import CLIP_VARIANTS
    if model_id not in CLIP_VARIANTS and (
        "/" not in model_id
        or model_id.startswith(("/", "\\", "."))
        or ".." in model_id
        or "\\" in model_id
        or ":" in model_id
    ):
        raise ValueError(f"Invalid classification model_id: {model_id}")

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
# Zenoh IPC Inference Bridge
# ============================================================================


async def _zenoh_inference(request: dict) -> str:
    """Bridge between Zenoh request JSON and the model inference path."""
    model_id = request.get("model", "")
    messages = request.get("messages", [])
    max_tokens = request.get("max_tokens", 256)
    temperature = request.get("temperature", 0.7)

    model = await load_language(model_id)
    return await model.generate(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ============================================================================
# Lifecycle
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _cleanup_task, _zenoh_ipc

    logger.info("Starting LlamaFarm Edge Runtime")

    # Emit the single-line offline-mode status so operators can verify
    # LLAMAFARM_OFFLINE / LLAMAFARM_MODEL_DIR are being honored.
    _offline_mode_bootstrap.log_startup_mode()

    _cleanup_task = asyncio.create_task(_cleanup_idle_models())

    # Start KV cache manager
    from utils.kv_cache_manager import (
        KVCacheManager,
        start_kv_cache_gc,
        stop_kv_cache_gc,
    )
    global _kv_cache_manager
    _kv_cache_manager = KVCacheManager()
    from routers.cache import set_cache_language_loader, set_cache_manager
    set_cache_manager(_kv_cache_manager)
    set_cache_language_loader(load_language)
    ChatCompletionsService.set_cache_manager(_kv_cache_manager)
    start_kv_cache_gc(_kv_cache_manager)

    start_session_cleanup()

    # Start Zenoh IPC interface (non-blocking — falls back to HTTP-only on failure)
    _zenoh_ipc = ZenohIPC(inference_fn=_zenoh_inference)
    await _zenoh_ipc.start()

    # Preload and pin models if configured
    preload_csv = os.getenv("PRELOAD_MODELS", "").strip()
    if preload_csv:
        preload_n_ctx_str = os.getenv("PRELOAD_N_CTX", "").strip()
        preload_n_ctx = None
        if preload_n_ctx_str:
            try:
                preload_n_ctx = int(preload_n_ctx_str)
            except ValueError:
                logger.warning(
                    f"Invalid PRELOAD_N_CTX value '{preload_n_ctx_str}', "
                    f"using auto-detected context size"
                )
        for model_id in preload_csv.split(","):
            model_id = model_id.strip()
            if not model_id:
                continue
            try:
                await load_language(model_id, n_ctx=preload_n_ctx, trusted=True)
                # Construct the same cache key load_language() uses
                cache_key = (
                    f"language:{model_id}:ctx{preload_n_ctx or 'auto'}:"
                    f"gpuauto:quantdefault"
                )
                _models.pin(cache_key)
                logger.info(f"Preloaded and pinned model: {model_id} ({cache_key})")
            except Exception as e:
                logger.warning(f"Failed to preload model '{model_id}': {e}")

    yield

    # Shutdown
    logger.info("Shutting down Edge Runtime")

    if _zenoh_ipc is not None:
        await _zenoh_ipc.stop()

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
app.include_router(completions_router)
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
