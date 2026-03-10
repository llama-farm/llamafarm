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
import importlib.util
import os
import warnings
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.logging import UniversalRuntimeLogger, setup_logging
from models import (
    AnomalyModel,
    BaseModel,
    ChatterboxConfig,
    ClassifierModel,
    DocumentModel,
    EncoderModel,
    GGUFEncoderModel,
    GGUFLanguageModel,
    LanguageModel,
    OCRModel,
    SpeechModel,
    TTSModel,
    VoiceProfile,
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
from routers.audio_chat import router as audio_chat_router
from routers.audio_speech import router as audio_speech_router
from routers.cache import router as cache_router
from routers.cache import set_cache_language_loader, set_cache_manager
from routers.chat_completions import router as chat_completions_router
from routers.chat_completions.service import ChatCompletionsService
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

try:
    from routers.explain import router as explain_router
    from routers.explain import set_explain_state, set_model_getter

    _HAS_EXPLAIN = True
except ImportError:
    _HAS_EXPLAIN = False
import sys
from pathlib import Path

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
from routers.polars import router as polars_router
from routers.preload import router as preload_router
from routers.preload import set_preload_function
from routers.vision import (
    router as vision_router,
)
from routers.vision import (
    set_classification_loader,
    set_detect_classify_loaders,
    set_detection_loader,
    set_document_loader,
    set_eval_models_dir,
    set_file_image_getter,
    set_model_export_loader,
    set_ocr_loader,
    set_sample_data_dir,
    set_streaming_detection_loader,
    set_tracking_models_dir,
    set_vision_models_dir,
    start_session_cleanup,
    start_tracking_cleanup,
    stop_session_cleanup,
    stop_tracking_cleanup,
)
from state import set_models_cache as state_set_models_cache
from utils.concurrent_loader import (
    ConcurrentModelLoader,
)
from utils.device import get_device_info, get_optimal_device
from utils.feature_encoder import FeatureEncoder
from utils.file_handler import get_file_images
from utils.model_cache import ModelCache
from utils.model_format import detect_model_format
from utils.resource_detect import (
    get_concurrency_override,
    get_resource_info,
    log_resource_summary,
)
from utils.safe_home import get_data_dir
from vision_training.trainer import set_trainer_model_loader

repo_root = Path(__file__).resolve().parents[2]

# Conditional import for timeseries addon (requires darts package)
_HAS_TIMESERIES = importlib.util.find_spec("darts") is not None
if _HAS_TIMESERIES:
    from models.timeseries_model import TimeseriesModel
    from routers.timeseries import router as timeseries_router
    from routers.timeseries import set_state as set_timeseries_state
    from routers.timeseries import set_timeseries_loader

# Conditional import for ADTK addon (requires adtk package)
_HAS_ADTK = importlib.util.find_spec("adtk") is not None
if _HAS_ADTK:
    from models.adtk_model import ADTKModel
    from routers.adtk import router as adtk_router
    from routers.adtk import set_adtk_loader, set_adtk_state

# Conditional import for Drift Detection addon (requires alibi_detect package)
_HAS_DRIFT = importlib.util.find_spec("alibi_detect") is not None
if _HAS_DRIFT:
    from models.drift_model import DriftModel
    from routers.drift import router as drift_router
    from routers.drift import set_drift_loader, set_drift_state

# Conditional import for CatBoost addon (requires catboost package)
_HAS_CATBOOST = importlib.util.find_spec("catboost") is not None
if _HAS_CATBOOST:
    from models.catboost_model import CatBoostModel
    from routers.catboost import router as catboost_router
    from routers.catboost import set_catboost_state

# Suppress spurious "leaked semaphore" warning from CTranslate2 (used by faster-whisper).
# CTranslate2 creates POSIX semaphores for internal thread pools that aren't explicitly
# released before interpreter shutdown. The OS kernel cleans these up on process exit —
# no resources are actually leaked. See: https://github.com/SYSTRAN/faster-whisper/issues/1057
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be \d+ leaked semaphore",
    category=UserWarning,
)

# Configure logging FIRST, before anything else
log_file = os.getenv("LOG_FILE", "")
log_level = os.getenv("LOG_LEVEL", "INFO")
json_logs = os.getenv("LOG_JSON_FORMAT", "false").lower() in ("true", "1", "yes")
setup_logging(json_logs=json_logs, log_level=log_level, log_file=log_file)

logger = UniversalRuntimeLogger("universal-runtime")


def _init_llama_backend():
    """Initialize llama.cpp backend in the main thread.

    CRITICAL FOR STABILITY: On NVIDIA Jetson/Tegra devices with unified memory,
    the CUDA backend MUST be initialized from the main thread before any worker
    threads attempt to use it. Failure to do so causes a "double free or corruption"
    crash during ggml_backend_load_all() when the CUDA backend tries to initialize
    from a ThreadPoolExecutor worker.

    This is a stability fix, NOT a performance optimization. It prevents crashes
    by ensuring the CUDA context is created in the main thread where GPU state
    management is most reliable on unified memory architectures.

    Affected platforms:
        - NVIDIA Jetson Orin Nano/NX (Tegra, unified memory)
        - NVIDIA Jetson Xavier (Tegra, unified memory)
        - Potentially other unified memory GPU systems

    Technical details:
        - ggml_backend_load_all() discovers and initializes compute backends
        - On Tegra, CUDA initialization from worker threads can corrupt internal state
        - By initializing at module load time (main thread), we avoid this issue
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


# Initialize llama.cpp backend in main thread - REQUIRED for Jetson/Tegra CUDA stability
# See _init_llama_backend() docstring for technical details on why this matters
_init_llama_backend()


def _preload_sklearn():
    """Preload sklearn in the main thread to avoid segfaults on ARM64.

    On Jetson/ARM64 with Python 3.13, importing sklearn's compiled extensions
    concurrently with active llama.cpp CUDA operations can cause segfaults.
    By importing sklearn at startup (before any requests), we avoid this issue.
    """
    try:
        from sklearn.ensemble import IsolationForest  # noqa: F401

        logger.info("sklearn preloaded successfully")
    except ImportError:
        logger.debug("sklearn not installed, skipping preload")
    except Exception as e:
        logger.warning(f"Failed to preload sklearn: {e}")


# Preload sklearn in main thread - prevents segfaults on ARM64/Jetson
_preload_sklearn()


def _preload_async_backends():
    """Preload async backends to avoid segfaults during streaming on ARM64.

    On Jetson/ARM64 with Python 3.13, lazy imports during garbage collection
    can cause segfaults. The anyio library lazily imports its async backend
    (asyncio/trio) on first use (e.g., when StreamingResponse starts).

    By importing these at startup, we ensure they're loaded before any
    concurrent CUDA operations that might trigger GC during import.
    """
    try:
        # Preload anyio's async backend - used by FastAPI StreamingResponse
        import anyio._backends._asyncio  # noqa: F401
        import anyio._core._eventloop  # noqa: F401

        logger.info("anyio async backends preloaded successfully")
    except ImportError:
        logger.debug("anyio not installed, skipping preload")
    except Exception as e:
        logger.warning(f"Failed to preload anyio backends: {e}")


# Preload async backends - prevents segfaults during streaming on ARM64/Jetson
_preload_async_backends()


def _patch_cache_artifact_factory():
    """Make CacheArtifactFactory.register idempotent (PyApp Windows workaround).

    In PyApp-packaged binaries on Windows, importing torch._dynamo fails partway
    through (after package.py registers artifact types but before __init__ completes).
    Python cleans up the failed torch._dynamo.* submodules from sys.modules, but the
    registrations persist in CacheArtifactFactory._artifact_types. On the next import
    attempt, package.py re-runs and @register asserts the type is already registered.

    Patching register() to skip duplicates breaks this cycle.
    """
    try:
        from torch.compiler._cache import CacheArtifactFactory

        if not getattr(CacheArtifactFactory, "_register_patched", False):
            _orig = CacheArtifactFactory.register.__func__

            @classmethod  # type: ignore[misc]
            def _safe_register(cls, artifact_cls):
                if artifact_cls.type() in cls._artifact_types:
                    return artifact_cls
                return _orig(cls, artifact_cls)

            CacheArtifactFactory.register = _safe_register
            CacheArtifactFactory._register_patched = True
    except (ImportError, AttributeError):
        pass  # torch not installed or API changed


_patch_cache_artifact_factory()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle (startup and shutdown)."""
    global _cleanup_task

    # Startup
    logger.info("Starting Universal Runtime")

    # Log addon availability
    if _HAS_TIMESERIES:
        logger.info("Timeseries addon available (darts installed)")
    else:
        logger.info("Timeseries addon unavailable (darts not installed)")

    if _HAS_ADTK:
        logger.info("ADTK addon available (adtk installed)")
    else:
        logger.info("ADTK addon unavailable (adtk not installed)")

    if _HAS_DRIFT:
        logger.info("Drift Detection addon available (alibi_detect installed)")
    else:
        logger.info("Drift Detection addon unavailable (alibi_detect not installed)")

    if _HAS_CATBOOST:
        logger.info("CatBoost addon available (catboost installed)")
    else:
        logger.info("CatBoost addon unavailable (catboost not installed)")

    try:
        preload_results = await preload_models_from_config()
        set_preload_function(preload_models_from_config)
        summary = preload_results.get("summary", {})
        if summary.get("loaded", 0) > 0:
            logger.info(
                f"Preloaded {summary['loaded']} models in "
                f"{summary['total_time_seconds']:.2f}s"
            )
        if summary.get("failed", 0) > 0:
            logger.warning(f"{summary['failed']} models failed to preload")
    except Exception as e:
        logger.error(f"Error during model preload: {e}", exc_info=True)
        # Don't fail startup - models will load on-demand

    # Start model cleanup background task
    _cleanup_task = asyncio.create_task(_cleanup_idle_models())
    logger.info("Model cleanup background task started")

    # Start KV cache manager + GC
    from utils.kv_cache_manager import (
        KVCacheManager,
        start_kv_cache_gc,
        stop_kv_cache_gc,
    )
    global _kv_cache_manager
    _kv_cache_manager = KVCacheManager()
    set_cache_manager(_kv_cache_manager)
    set_cache_language_loader(load_language)
    ChatCompletionsService.set_cache_manager(_kv_cache_manager)
    start_kv_cache_gc(_kv_cache_manager)
    logger.info("KV cache manager started")

    # Start vision streaming session cleanup (needs running event loop)
    start_session_cleanup()
    start_tracking_cleanup()
    logger.info("Vision session cleanup task started")

    yield

    # Shutdown
    logger.info("Shutting down Universal Runtime")

    # Stop KV cache GC task
    await stop_kv_cache_gc()

    # Stop vision cleanup tasks
    await stop_session_cleanup()
    await stop_tracking_cleanup()

    # Stop cleanup task
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await _cleanup_task
        logger.info("Model cleanup task stopped")

    # Unload all remaining models (including addon caches)
    all_caches: list[tuple[ModelCache | None, str]] = [
        (_models, "models"),
        (_classifiers, "classifiers"),
    ]
    if _HAS_TIMESERIES:
        all_caches.append((_timeseries, "timeseries"))
    if _HAS_ADTK:
        all_caches.append((_adtk, "adtk"))
    if _HAS_DRIFT:
        all_caches.append((_drift, "drift"))
    if _HAS_CATBOOST:
        all_caches.append((_catboost, "catboost"))

    for cache, cache_name in all_caches:
        if cache and len(cache) > 0:
            logger.info(f"Unloading {len(cache)} remaining {cache_name}")
            for cache_key, model in list(cache.items()):
                try:
                    await model.unload()
                    logger.info(f"Unloaded {cache_name}: {cache_key}")
                except Exception as e:
                    logger.error(f"Error unloading {cache_name} {cache_key}: {e}")
            cache.clear()

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
if _HAS_EXPLAIN:
    app.include_router(explain_router)
app.include_router(audio_router)
app.include_router(audio_speech_router)
app.include_router(audio_chat_router)
app.include_router(cache_router)
app.include_router(chat_completions_router)
app.include_router(classifier_router)
app.include_router(files_router)
app.include_router(health_router)
app.include_router(nlp_router)
app.include_router(polars_router)
app.include_router(vision_router)
app.include_router(preload_router)

# Conditional addon routers
if _HAS_TIMESERIES:
    app.include_router(timeseries_router)
if _HAS_ADTK:
    app.include_router(adtk_router)
if _HAS_DRIFT:
    app.include_router(drift_router)
if _HAS_CATBOOST:
    app.include_router(catboost_router)



# ── Model management endpoints ──────────────────────────────────────────────

@app.post("/v1/models/unload", tags=["models"])
async def unload_all_models():
    """Unload all loaded models to free memory.

    Useful before loading a large model, or between benchmark runs
    to ensure a clean memory state.
    """
    unloaded = []
    for cache_key, model in list(_models.items()):
        try:
            await model.unload()
            unloaded.append(cache_key)
        except Exception as e:
            logger.error(f"Error unloading {cache_key}: {e}")
    _models.clear()

    # Also clear classifier and addon caches
    addon_caches: list[tuple[ModelCache | None, str]] = [
        (_classifiers, "classifier"),
    ]
    if _HAS_TIMESERIES:
        addon_caches.append((_timeseries, "timeseries"))
    if _HAS_ADTK:
        addon_caches.append((_adtk, "adtk"))
    if _HAS_DRIFT:
        addon_caches.append((_drift, "drift"))
    if _HAS_CATBOOST:
        addon_caches.append((_catboost, "catboost"))

    for cache, cache_name in addon_caches:
        if cache and len(cache) > 0:
            for cache_key, model in list(cache.items()):
                try:
                    await model.unload()
                    unloaded.append(cache_key)
                except Exception as e:
                    logger.error(f"Error unloading {cache_name} {cache_key}: {e}")
            cache.clear()

    logger.info(f"Unloaded {len(unloaded)} models: {unloaded}")
    return {"unloaded": len(unloaded), "models": unloaded}


# Model unload timeout configuration (in seconds)
# Default: 5 minutes (300 seconds)
MODEL_UNLOAD_TIMEOUT = int(os.getenv("MODEL_UNLOAD_TIMEOUT", "300"))
# Cleanup check interval (in seconds) - how often to check for idle models
# Default: 30 seconds
CLEANUP_CHECK_INTERVAL = int(os.getenv("CLEANUP_CHECK_INTERVAL", "30"))

# Global model caches using TTL-based caching (via cachetools)
# Models are automatically tracked for idle time and cleaned up by background task
_models: ModelCache[BaseModel] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)


state_set_models_cache(_models)

# Also set for health router (legacy)
set_models_cache(_models)


_classifiers: ModelCache["ClassifierModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
_model_load_lock = asyncio.Lock()
_current_device = None

# Feature encoder cache for anomaly detection with mixed data types
_encoders: dict[str, FeatureEncoder] = {}
_cleanup_task: asyncio.Task | None = None
_kv_cache_manager = None


_pinned_cache_keys: set[str] = set()

# Flag to ensure we only try to load the pin registry once per worker process.
_pin_registry_loaded: bool = False


def _populate_pin_registry_from_config() -> None:
    """Read llamafarm.yaml and register cache keys for models with pin: true.

    This is called lazily the first time load_language() is invoked in a worker
    process, before any model is loaded. This ensures that even a freshly-spawned
    uvicorn worker that hasn't run the startup preload yet will automatically pin
    models that are configured with pin: true when they are loaded by any caller
    (including /chat/completions).

    Stores fully-parameterized cache keys (not bare model IDs) so only the exact
    configured variant is pinned — other runtime variants of the same model_id
    that arrive via /chat/completions with different params are NOT affected.
    """
    global _pin_registry_loaded
    if _pin_registry_loaded:
        return

    try:
        repo_root = str(Path(__file__).resolve().parents[2])
        path_inserted = False
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
            path_inserted = True
        try:
            from config.helpers.loader import load_config

            config = load_config()
        finally:
            if path_inserted and repo_root in sys.path:
                sys.path.remove(repo_root)

        if hasattr(config, "runtime") and hasattr(config.runtime, "models"):
            for model_cfg in config.runtime.models:
                provider = getattr(model_cfg, "provider", None)
                pin = getattr(model_cfg, "pin", False)
                model_id = getattr(model_cfg, "model", None)
                if provider == "universal" and pin and model_id:
                    # Build cache key using the same extra_body params the
                    # preload would use, so we only pin the configured variant.
                    extra_body = {}
                    if hasattr(model_cfg, "extra_body") and model_cfg.extra_body:
                        raw = model_cfg.extra_body
                        if hasattr(raw, "model_dump"):
                            extra_body = raw.model_dump()
                        elif isinstance(raw, dict):
                            extra_body = raw
                    cache_key = _make_language_cache_key(
                        model_id,
                        extra_body.get("n_ctx"),
                        extra_body.get("n_batch"),
                        extra_body.get("n_gpu_layers"),
                        extra_body.get("n_threads"),
                        extra_body.get("flash_attn"),
                        extra_body.get("use_mmap"),
                        extra_body.get("use_mlock"),
                        extra_body.get("cache_type_k"),
                        extra_body.get("cache_type_v"),
                        extra_body.get("preferred_quantization"),
                    )
                    _pinned_cache_keys.add(cache_key)
                    logger.debug(
                        f"Pin registry (lazy): registered '{model_cfg.name}' key={cache_key}"
                    )

        _pin_registry_loaded = True
    except Exception as e:
        logger.debug(f"Pin registry lazy load skipped: {e}")


# Data directories
_LF_DATA_DIR = get_data_dir()
CLASSIFIER_MODELS_DIR = _LF_DATA_DIR / "models" / "classifier"
CATBOOST_MODELS_DIR = _LF_DATA_DIR / "models" / "catboost"

# Timeseries model cache (conditional on darts availability)
if _HAS_TIMESERIES:
    _timeseries: ModelCache["TimeseriesModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
else:
    _timeseries = None

# ADTK model cache (conditional on adtk availability)
if _HAS_ADTK:
    _adtk: ModelCache["ADTKModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
else:
    _adtk = None

# Drift Detection model cache (conditional on alibi_detect availability)
if _HAS_DRIFT:
    _drift: ModelCache["DriftModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
else:
    _drift = None

# CatBoost model cache (conditional on catboost availability)
if _HAS_CATBOOST:
    _catboost: ModelCache["CatBoostModel"] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
else:
    _catboost = None


# ============================================================================
# Language Model Loading (for chat_completions router)
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
            caches_to_clean = [
                (_models, "models"),
                (_classifiers, "classifiers"),
            ]
            if _HAS_TIMESERIES and _timeseries is not None:
                caches_to_clean.append((_timeseries, "timeseries"))
            if _HAS_ADTK and _adtk is not None:
                caches_to_clean.append((_adtk, "adtk"))
            if _HAS_DRIFT and _drift is not None:
                caches_to_clean.append((_drift, "drift"))
            if _HAS_CATBOOST and _catboost is not None:
                caches_to_clean.append((_catboost, "catboost"))

            for cache, cache_name in caches_to_clean:
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
    pin: bool = False,
):
    """Load a causal language model (GGUF or transformers format)."""

    _populate_pin_registry_from_config()
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

    effective_pin = pin or (cache_key in _pinned_cache_keys)

    if effective_pin and not _models.is_pinned(cache_key):
        _models.pin(cache_key)
        logger.info(f"Pinned model: {cache_key}")

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
    pin: bool = False,
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

                if pin:
                    _models.pin(cache_key)
                    logger.info(f"Pinned encoder model: {cache_key}")

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
# Vision Model Loading (Detection / Classification)
# ============================================================================

VISION_MODELS_DIR = _LF_DATA_DIR / "models" / "vision"


async def load_detection_model(model_id: str = "yolov8n"):
    """Load a YOLO detection model."""
    cache_key = f"vision:detect:{model_id}"
    if cache_key not in _models:
        async with _model_load_lock:
            if cache_key not in _models:
                from models.yolo_model import YOLOModel

                device = get_device()
                # Check for custom model in vision models dir
                from pathlib import Path as _Path

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


async def preload_models_from_config(config_path: str | None = None) -> dict:
    """Preload models marked with preload: true in llamafarm.yaml.

    This function:
    1. Detects system resources (CPU/RAM/VRAM)
    2. Calculates optimal concurrency
    3. Reads the LlamaFarm config
    4. Filters models with preload: true
    5. Loads them concurrently using ConcurrentModelLoader
    6. Pins models with pin: true
    7. Returns detailed per-model status
    Args:
        config_path: Optional path to llamafarm.yaml. If None, searches current directory.

    Returns:
        Dictionary with results:
        {
            "results": {
                "model-name": {
                    "status": "loaded|failed|already_loaded|skipped",
                    "pinned": bool,
                    "load_time_seconds": float,
                    "error_message": str (if failed),
                }
            },
            "summary": {
                "loaded": int,
                "failed": int,
                "already_loaded": int,
                "skipped": int,
                "total_time_seconds": float,
                "concurrency_used": int,
            },
            "resources": {
                "device": str,
                "cpu_count": int,
                "available_ram_gb": float,
                "available_vram_gb": float,
            }
        }
    """

    repo_root = str(Path(__file__).resolve().parents[2])
    path_inserted = False
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        path_inserted = True

    try:
        from config.helpers.loader import load_config
    except ImportError:
        logger.warning("Config loader not available, cannot preload models")
        return {
            "results": {},
            "summary": {
                "loaded": 0,
                "failed": 0,
                "already_loaded": 0,
                "skipped": 0,
                "message": "Config loader not available",
                "total_time_seconds": 0.0,
            },
        }
    finally:
        if path_inserted:
            sys.path.remove(repo_root)

    device = get_device()
    resource_info = get_resource_info(device)
    log_resource_summary(resource_info)

    # Determine concurrency (allow env override)
    concurrency = get_concurrency_override()
    if concurrency is None:
        concurrency = resource_info.optimal_concurrency
    else:
        # Cap user override at max safe concurrency
        if concurrency > resource_info.max_concurrency:
            logger.warning(
                f"Concurrency override {concurrency} exceeds safe maximum "
                f"{resource_info.max_concurrency}, using maximum instead"
            )
            concurrency = resource_info.max_concurrency

    try:
        if config_path:
            config = load_config(config_path=Path(config_path))
        else:
            config = load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {
            "results": {},
            "summary": {
                "loaded": 0,
                "failed": 0,
                "already_loaded": 0,
                "skipped": 0,
                "message": f"Config load failed: {e}",
                "total_time_seconds": 0.0,
            },
        }

    # extract models with preload: true
    models_to_load: list[tuple[str, str, bool, dict]] = []

    if hasattr(config, "runtime") and hasattr(config.runtime, "models"):
        for model_cfg in config.runtime.models:
            # only preload universal provider models with preload: true
            provider = getattr(model_cfg, "provider", None)
            preload = getattr(model_cfg, "preload", False)

            if provider != "universal" or not preload:
                continue

            model_name = getattr(model_cfg, "name", "unknown")
            model_id = getattr(model_cfg, "model", None)
            pin = getattr(model_cfg, "pin", False)

            if not model_id:
                logger.warning(
                    f"Model '{model_name}' has preload: true but no model ID"
                )
                continue

            # Extract extra_body params for GGUF models
            extra_body = {}
            if hasattr(model_cfg, "extra_body") and model_cfg.extra_body:
                extra_body = model_cfg.extra_body
                # Convert Pydantic model to dict if needed
                if hasattr(extra_body, "model_dump"):
                    extra_body = extra_body.model_dump()
                elif not isinstance(extra_body, dict):
                    extra_body = {}

            if pin:
                cache_key = _make_language_cache_key(
                    model_id,
                    extra_body.get("n_ctx"),
                    extra_body.get("n_batch"),
                    extra_body.get("n_gpu_layers"),
                    extra_body.get("n_threads"),
                    extra_body.get("flash_attn"),
                    extra_body.get("use_mmap"),
                    extra_body.get("use_mlock"),
                    extra_body.get("cache_type_k"),
                    extra_body.get("cache_type_v"),
                    extra_body.get("preferred_quantization"),
                )
                _pinned_cache_keys.add(cache_key)
                logger.debug(
                    f"Registered model '{model_name}' key={cache_key} in pin registry"
                )

            models_to_load.append((model_name, model_id, pin, extra_body))

    if not models_to_load:
        logger.info("No models configured for preload")
        return {
            "results": {},
            "summary": {
                "loaded": 0,
                "failed": 0,
                "already_loaded": 0,
                "skipped": 0,
                "message": "No models configured for preload",
                "total_time_seconds": 0.0,
                "concurrency_used": 0,
            },
            "resources": {
                "device": resource_info.device,
                "cpu_count": resource_info.cpu_count,
                "available_ram_gb": resource_info.available_ram_gb,
                "available_vram_gb": resource_info.available_vram_gb,
            },
        }

    logger.info(
        f"Preloading {len(models_to_load)} models with concurrency={concurrency}"
    )

    # create concurrent loader
    loader = ConcurrentModelLoader(concurrency=concurrency)

    async def load_model_wrapper(model_path: str, pin: bool, extra_body: dict) -> None:
        """Wrapper that calls appropriate load function based on model format."""

        # Detect model format to determine if it's a language or encoder model
        # For now, assume language models (CausalLM)
        # TODO for later: add encoder model detection and loading

        # Extract GGUF-specific parameters from extra_body
        n_ctx = extra_body.get("n_ctx")
        n_batch = extra_body.get("n_batch")
        n_gpu_layers = extra_body.get("n_gpu_layers")
        n_threads = extra_body.get("n_threads")
        flash_attn = extra_body.get("flash_attn")
        use_mmap = extra_body.get("use_mmap")
        use_mlock = extra_body.get("use_mlock")
        cache_type_k = extra_body.get("cache_type_k")
        cache_type_v = extra_body.get("cache_type_v")
        preferred_quantization = extra_body.get("preferred_quantization")

        await load_language(
            model_id=model_path,
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
            pin=pin,
        )

    # Check if model is already loaded
    def is_model_loaded(model_path: str) -> bool:
        """Check if model is already in cache."""
        # Generate cache key to check
        cache_key = _make_language_cache_key(model_path)
        return cache_key in _models

    # Prepare models for batch loading
    # Format: [(name, path, pin), ...]

    # Create per-model load functions with extra_body
    # We need to curry the extra_body into each load call
    async def create_load_fn(model_path: str, pin: bool, extra_body: dict):
        """Create a load function for a specific model."""
        return await load_model_wrapper(model_path, pin, extra_body)

    # Since load_many expects a single load_fn, we need to handle this differently
    # Let's load models one by one but with concurrent execution

    tasks = []
    for name, path, pin, extra_body in models_to_load:

        async def load_task(n=name, p=path, pn=pin, eb=extra_body):
            result = await loader.load_one(
                model_name=n,
                model_path=p,
                pin=pn,
                load_fn=lambda mp, pn: load_model_wrapper(mp, pn, eb),
                is_loaded_fn=is_model_loaded,
            )

            from utils.concurrent_loader import LoadStatus

            if pn and result.status == LoadStatus.ALREADY_LOADED:
                already_cache_key = _make_language_cache_key(
                    p,
                    eb.get("n_ctx"),
                    eb.get("n_batch"),
                    eb.get("n_gpu_layers"),
                    eb.get("n_threads"),
                    eb.get("flash_attn"),
                    eb.get("use_mmap"),
                    eb.get("use_mlock"),
                    eb.get("cache_type_k"),
                    eb.get("cache_type_v"),
                    eb.get("preferred_quantization"),
                )
                if already_cache_key in _models and not _models.is_pinned(
                    already_cache_key
                ):
                    _models.pin(already_cache_key)
                    logger.info(
                        f"Applied pin to already-loaded model '{n}' ({already_cache_key})"
                    )
            return result

        tasks.append(load_task())

    # Execute all tasks concurrently
    import time

    start_time = time.perf_counter()
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.perf_counter() - start_time

    # Process results
    results = {}
    loaded_count = 0
    failed_count = 0
    already_loaded_count = 0
    skipped_count = 0

    for i, result in enumerate(results_list):
        model_name = models_to_load[i][0]

        if isinstance(result, Exception):
            logger.error(f"Unexpected exception loading '{model_name}': {result}")
            results[model_name] = {
                "status": "failed",
                "pinned": False,
                "error_message": str(result),
            }
            failed_count += 1
        else:
            # Convert ModelLoadResult to dict
            result_dict = {
                "status": result.status.value,
                "pinned": result.pinned,
            }

            if result.load_time_seconds is not None:
                result_dict["load_time_seconds"] = result.load_time_seconds

            if result.error_message:
                result_dict["error_message"] = result.error_message

            results[model_name] = result_dict

            if result.status.value == "loaded":
                loaded_count += 1
            elif result.status.value == "failed":
                failed_count += 1
            elif result.status.value == "already_loaded":
                already_loaded_count += 1
            elif result.status.value == "skipped":
                skipped_count += 1

    logger.info(
        f"Preload complete: {loaded_count} loaded, {failed_count} failed, "
        f"{already_loaded_count} already loaded, {skipped_count} skipped "
        f"in {total_time:.2f}s"
    )

    return {
        "results": results,
        "summary": {
            "loaded": loaded_count,
            "failed": failed_count,
            "already_loaded": already_loaded_count,
            "skipped": skipped_count,
            "total_time_seconds": round(total_time, 2),
            "concurrency_used": concurrency,
        },
        "resources": {
            "device": resource_info.device,
            "cpu_count": resource_info.cpu_count,
            "available_ram_gb": round(resource_info.available_ram_gb, 1),
            "available_vram_gb": round(resource_info.available_vram_gb, 1),
            "gpu_name": resource_info.gpu_name,
        },
    }


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

    # Evict cached model if base_model changed (prevents returning a model
    # initialized with a different base_model for the same model_id)
    cached = _classifiers.get(cache_key) if cache_key in _classifiers else None
    if cached is not None and getattr(cached, "base_model", None) != base_model:
        logger.info(
            f"Evicting classifier '{model_id}': base_model changed "
            f"({cached.base_model} -> {base_model})"
        )
        _classifiers.pop(cache_key, None)
        await cached.unload()

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
# Timeseries Model Loading
# ============================================================================

if _HAS_TIMESERIES:

    def _make_timeseries_cache_key(model_name: str) -> str:
        """Create a cache key for timeseries models."""
        return f"timeseries:{model_name}"

    async def load_timeseries(
        model_id: str,
        backend: str = "chronos",
    ) -> "TimeseriesModel":
        """Load or get cached timeseries model."""
        cache_key = _make_timeseries_cache_key(model_id)

        # Evict cached model if backend changed
        cached = _timeseries.get(cache_key) if cache_key in _timeseries else None
        if cached is not None and getattr(cached, "backend", None) != backend:
            logger.info(
                f"Evicting timeseries model '{model_id}': backend changed "
                f"({cached.backend} -> {backend})"
            )
            _timeseries.pop(cache_key, None)
            await cached.unload()

        if cache_key not in _timeseries:
            async with _model_load_lock:
                if cache_key not in _timeseries:
                    logger.info(
                        f"Loading timeseries model: {model_id} (backend: {backend})"
                    )
                    device = get_device()

                    model = TimeseriesModel(
                        model_id=model_id,
                        device=device,
                        backend=backend,
                    )

                    await model.load()
                    _timeseries[cache_key] = model

        return _timeseries.get(cache_key)


# ============================================================================
# ADTK Model Loading
# ============================================================================

if _HAS_ADTK:

    def _make_adtk_cache_key(model_name: str) -> str:
        """Create a cache key for ADTK models."""
        return f"adtk:{model_name}"

    async def load_adtk(
        model_id: str,
        detector: str = "level_shift",
        params: dict | None = None,
    ) -> "ADTKModel":
        """Load or get cached ADTK model."""
        cache_key = _make_adtk_cache_key(model_id)

        # Evict cached model if detector changed
        cached = _adtk.get(cache_key) if cache_key in _adtk else None
        if cached is not None and getattr(cached, "detector_type", None) != detector:
            logger.info(
                f"Evicting ADTK model '{model_id}': detector changed "
                f"({cached.detector_type} -> {detector})"
            )
            _adtk.pop(cache_key, None)
            await cached.unload()

        if cache_key not in _adtk:
            async with _model_load_lock:
                if cache_key not in _adtk:
                    logger.info(
                        f"Loading ADTK model: {model_id} (detector: {detector})"
                    )
                    device = get_device()

                    model = ADTKModel(
                        model_id=model_id,
                        device=device,
                        detector=detector,
                        **(params or {}),
                    )

                    await model.load()
                    _adtk[cache_key] = model

        return _adtk.get(cache_key)


# ============================================================================
# Drift Detection Model Loading
# ============================================================================

if _HAS_DRIFT:

    def _make_drift_cache_key(model_name: str) -> str:
        """Create a cache key for drift detection models."""
        return f"drift:{model_name}"

    async def load_drift(
        model_id: str,
        detector: str = "ks",
        params: dict | None = None,
    ) -> "DriftModel":
        """Load or get cached drift detection model."""
        cache_key = _make_drift_cache_key(model_id)

        # Evict cached model if detector changed
        cached = _drift.get(cache_key) if cache_key in _drift else None
        if cached is not None and getattr(cached, "detector_type", None) != detector:
            logger.info(
                f"Evicting Drift model '{model_id}': detector changed "
                f"({cached.detector_type} -> {detector})"
            )
            _drift.pop(cache_key, None)
            await cached.unload()

        if cache_key not in _drift:
            async with _model_load_lock:
                if cache_key not in _drift:
                    logger.info(
                        f"Loading Drift Detection model: {model_id} (detector: {detector})"
                    )
                    device = get_device()

                    model = DriftModel(
                        model_id=model_id,
                        device=device,
                        detector=detector,
                        **(params or {}),
                    )

                    await model.load()
                    _drift[cache_key] = model

        return _drift.get(cache_key)


# ============================================================================
# Speech Model Loading
# ============================================================================

# Safe audio file extensions (whitelist for security)
SAFE_AUDIO_EXTENSIONS = frozenset(
    {
        ".wav",
        ".mp3",
        ".m4a",
        ".webm",
        ".flac",
        ".ogg",
        ".mp4",
        ".opus",
        ".pcm",
    }
)

# Silence detection threshold for decoded Opus audio (higher due to noise floor)
SILENCE_THRESHOLD_OPUS = 0.03


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


# ==============================================================================
# TTS (Text-to-Speech) Model Loading
# ==============================================================================


def _make_tts_cache_key(
    model_id: str,
    voice: str,
    voice_profile_path: str | None = None,
) -> str:
    """Generate cache key for TTS model.

    Args:
        model_id: TTS model identifier
        voice: Default voice for the model
        voice_profile_path: Path to voice profile audio (for Chatterbox)

    Returns:
        Cache key string
    """
    if voice_profile_path:
        # Hash the path to keep key reasonable length
        import hashlib

        path_hash = hashlib.md5(voice_profile_path.encode()).hexdigest()[:8]
        return f"tts:{model_id}:{voice}:{path_hash}"
    return f"tts:{model_id}:{voice}"


async def load_tts(
    model_id: str = "kokoro",
    voice: str = "af_heart",
    voice_profiles: dict[str, dict] | None = None,
    temperature: float = 0.8,
    top_k: int = 1000,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
) -> TTSModel:
    """Load a text-to-speech model.

    Args:
        model_id: TTS model identifier ("kokoro" or "chatterbox-turbo")
        voice: Default voice ID (Kokoro) or profile name (Chatterbox)
        voice_profiles: Dict of {name: {audio_path, description}} for Chatterbox
        temperature: Chatterbox Turbo temperature (0.1-2.0)
        top_k: Chatterbox Turbo top-k sampling (1-5000)
        top_p: Chatterbox Turbo nucleus sampling (0.0-1.0)
        repetition_penalty: Chatterbox Turbo repetition penalty (1.0-2.0)

    Returns:
        Loaded TTSModel instance
    """
    # Convert voice_profiles dict to VoiceProfile objects
    profiles: dict[str, VoiceProfile] | None = None
    voice_profile_path: str | None = None

    if voice_profiles:
        profiles = {
            name: VoiceProfile(
                name=name,
                audio_path=cfg["audio_path"],
                description=cfg.get("description", ""),
            )
            for name, cfg in voice_profiles.items()
        }
        # Get the path for the selected voice for cache key
        if voice in profiles:
            voice_profile_path = profiles[voice].audio_path

    cache_key = _make_tts_cache_key(model_id, voice, voice_profile_path)

    if cache_key not in _models:
        async with _model_load_lock:
            if cache_key not in _models:
                logger.info(f"Loading TTS model: {model_id} (voice={voice})")
                device = get_device()

                # Create Chatterbox config if applicable
                chatterbox_config = None
                if model_id == "chatterbox-turbo":
                    chatterbox_config = ChatterboxConfig(
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )

                model = TTSModel(
                    model_id=model_id,
                    device=device,
                    voice=voice,
                    voice_profiles=profiles,
                    chatterbox_config=chatterbox_config,
                )

                await model.load()
                _models[cache_key] = model

    # Return model (get() refreshes TTL automatically)
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
set_detection_loader(load_detection_model)
set_classification_loader(load_classification_model)
set_detect_classify_loaders(load_detection_model, load_classification_model)
set_streaming_detection_loader(load_detection_model)
set_tracking_models_dir(VISION_MODELS_DIR)
set_vision_models_dir(VISION_MODELS_DIR)
set_sample_data_dir(_LF_DATA_DIR)
set_eval_models_dir(VISION_MODELS_DIR)
set_model_export_loader(load_detection_model)
# NOTE: start_session_cleanup() is called in lifespan() where event loop is running

# Vision training
set_trainer_model_loader(load_detection_model)

# Anomaly router
set_anomaly_loader(load_anomaly)
set_anomaly_state(_models, _encoders, _model_load_lock)

# Classifier router
set_classifier_loader(load_classifier)
set_classifier_models_dir(CLASSIFIER_MODELS_DIR)
set_classifier_state(_classifiers, _model_load_lock)

# Audio router
set_speech_loader(load_speech)

# Timeseries router (conditional)
if _HAS_TIMESERIES:
    set_timeseries_loader(load_timeseries)
    set_timeseries_state(_timeseries, _model_load_lock)

# ADTK router (conditional)
if _HAS_ADTK:
    set_adtk_loader(load_adtk)
    set_adtk_state(_adtk, _model_load_lock)

# Drift Detection router (conditional)
if _HAS_DRIFT:
    set_drift_loader(load_drift)
    set_drift_state(_drift, _model_load_lock)

# CatBoost router (conditional)
if _HAS_CATBOOST:
    set_catboost_state(_catboost, _model_load_lock, CATBOOST_MODELS_DIR)


# ============================================================================
# SHAP Explainer Dependencies
# ============================================================================


async def get_model_for_explain(model_type: str, model_id: str):
    """Get a model by type and ID for SHAP explanation.

    Looks up models from the appropriate cache based on model_type.
    """
    # Look up in the appropriate cache based on model type
    if model_type == "anomaly":
        for key, model in _models.items():
            if key.startswith("anomaly:") and model_id in key:
                return model
    elif model_type == "classifier":
        for key, model in _classifiers.items():
            if model_id in key:
                return model
    elif model_type == "timeseries" and _timeseries is not None:
        for key, model in _timeseries.items():
            if model_id in key:
                return model
    elif model_type == "adtk" and _adtk is not None:
        for key, model in _adtk.items():
            if model_id in key:
                return model
    elif model_type == "drift" and _drift is not None:
        for key, model in _drift.items():
            if model_id in key:
                return model
    elif model_type == "catboost" and _catboost is not None:
        for key, model in _catboost.items():
            if model_id in key:
                return model
    return None


if _HAS_EXPLAIN:
    set_model_getter(get_model_for_explain)
    set_explain_state(_model_load_lock)


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
