"""
Universal Runtime Server

A FastAPI server that provides OpenAI-compatible endpoints for any HuggingFace
model without restrictions. Supports:
- Text generation (Causal LMs: GPT, Llama, Mistral, etc.)
- Text embeddings & classification (Encoders: BERT, sentence-transformers, etc.)

Key Features:
- Auto-detects hardware (MPS/CUDA/CPU)
- Lazy model loading (load on first request)
- Platform-specific optimizations
- OpenAI API compatibility
- No model restrictions (trust_remote_code=True)
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    HTTPException,
)
from pydantic import BaseModel as PydanticBaseModel
from typing import Optional, Literal, List, Union
import os
import base64
from datetime import datetime
from routers.chat_completions import router as chat_completions_router

from models import (
    BaseModel,
    LanguageModel,
    EncoderModel,
)
from utils.device import get_optimal_device, get_device_info
from core.logging import setup_logging, UniversalRuntimeLogger

# Configure logging FIRST, before anything else
log_file = os.getenv("LOG_FILE", "")
log_level = os.getenv("LOG_LEVEL", "INFO")
json_logs = os.getenv("LOG_JSON_FORMAT", "false").lower() in ("true", "1", "yes")
setup_logging(json_logs=json_logs, log_level=log_level, log_file=log_file)

logger = UniversalRuntimeLogger("universal-runtime")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle (startup and shutdown)."""

    # Startup
    logger.info("Starting Universal Runtime")
    yield
    # Shutdown
    logger.info("Shutting down Universal Runtime")
    logger.info("Shutdown complete")


app = FastAPI(
    title="Universal Runtime",
    description="OpenAI-compatible API for HuggingFace models (transformers & diffusers)",
    version="2.0.0",
    lifespan=lifespan,
)
app.include_router(chat_completions_router)

# Global model cache
_models: dict[str, BaseModel] = {}
_model_load_lock = asyncio.Lock()
_current_device = None


# ============================================================================
# Helper Functions
# ============================================================================


def get_device():
    """Get the optimal device for the current platform."""
    global _current_device
    if _current_device is None:
        _current_device = get_optimal_device()
        logger.info(f"Using device: {_current_device}")
    return _current_device


async def load_language(model_id: str):
    """Load a causal language model for text generation."""
    cache_key = f"language:{model_id}"
    if cache_key not in _models:
        async with _model_load_lock:
            # Double-check if model was loaded while waiting for the lock
            if cache_key not in _models:
                logger.info(f"Loading causal LM: {model_id}")
                device = get_device()
                model = LanguageModel(model_id, device)
                await model.load()
                _models[cache_key] = model
    return _models[cache_key]


async def load_encoder(model_id: str, task: str = "embedding"):
    """Load an encoder model for embeddings or classification."""
    cache_key = f"encoder:{task}:{model_id}"
    if cache_key not in _models:
        async with _model_load_lock:
            # Double-check if model was loaded while waiting for the lock
            if cache_key not in _models:
                logger.info(f"Loading encoder ({task}): {model_id}")
                device = get_device()
                model = EncoderModel(model_id, device, task=task)
                await model.load()
                _models[cache_key] = model
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
    input: Union[str, List[str]]
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    user: Optional[str] = None
    extra_body: Optional[dict] = None


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.

    Supports any HuggingFace encoder model for text embeddings.
    """
    try:
        model = await load_encoder(request.model, task="embedding")

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
        raise HTTPException(status_code=500, detail=str(e))


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
