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
    AnomalyModel,
    BaseModel,
    DocumentModel,
    EncoderModel,
    GGUFEncoderModel,
    GGUFLanguageModel,
    LanguageModel,
    OCRModel,
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
    images: list[str]  # Base64-encoded document images
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

    Example request:
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "images": ["base64_encoded_image..."],
        "task": "extraction"
    }
    ```

    For VQA, include prompts:
    ```json
    {
        "model": "microsoft/layoutlmv3-base-finetuned-docvqa",
        "images": ["base64_encoded_image..."],
        "prompts": ["What is the total amount?"],
        "task": "vqa"
    }
    ```
    """
    try:
        # Load document model
        model = await load_document(
            model_id=request.model,
            task=request.task,
        )

        # Extract from documents
        results = await model.extract(
            images=request.images,
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
                "documents_processed": len(request.images),
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
    images: list[str]  # Base64-encoded images
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

    Example request:
    ```json
    {
        "model": "surya",
        "images": ["base64_encoded_image..."],
        "languages": ["en"],
        "return_boxes": false
    }
    ```
    """
    try:
        # Load OCR model
        model = await load_ocr(
            backend=request.model,
            languages=request.languages,
        )

        # Run OCR
        results = await model.recognize(
            images=request.images,
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
                "images_processed": len(request.images),
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


class AnomalyScoreRequest(PydanticBaseModel):
    """Anomaly scoring request."""

    model: str = "default"  # Model identifier
    backend: str = "isolation_forest"  # isolation_forest, one_class_svm, local_outlier_factor, autoencoder
    data: list[list[float]]  # Data points to score
    threshold: float | None = None  # Override default threshold


class AnomalyFitRequest(PydanticBaseModel):
    """Anomaly model fitting request."""

    model: str = "default"  # Model identifier (for caching)
    backend: str = "isolation_forest"  # Backend to use
    data: list[list[float]]  # Training data (assumed mostly normal)
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
        model = await load_anomaly(
            model_id=request.model,
            backend=request.backend,
        )

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/anomaly/fit first or load a pre-trained model.",
            )

        # Score data
        results = await model.score(
            data=request.data,
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
            "model": request.model,
            "backend": request.backend,
            "summary": {
                "total_points": len(results),
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_count / len(results) if results else 0,
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
        model = await load_anomaly(
            model_id=request.model,
            backend=request.backend,
            contamination=request.contamination,
        )

        # Fit model
        result = await model.fit(
            data=request.data,
            epochs=request.epochs,
            batch_size=request.batch_size,
        )

        return {
            "object": "fit_result",
            "model": request.model,
            "backend": request.backend,
            "samples_fitted": result.samples_fitted,
            "training_time_ms": result.training_time_ms,
            "model_params": result.model_params,
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
        model = await load_anomaly(
            model_id=request.model,
            backend=request.backend,
        )

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/anomaly/fit first.",
            )

        # Detect anomalies
        results = await model.detect(
            data=request.data,
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
            "model": request.model,
            "backend": request.backend,
            "summary": {
                "anomalies_detected": len(results),
                "threshold": request.threshold or model.threshold,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {e}", exc_info=True)
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
