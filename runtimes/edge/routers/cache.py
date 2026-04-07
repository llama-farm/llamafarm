"""KV Cache API — prepare, list, evict, stats, and GC endpoints."""

import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.error_handler import handle_endpoint_errors

logger = logging.getLogger(__name__)
router = APIRouter(tags=["cache"])

# ── Dependency injection ────────────────────────────────────────────────────

_cache_manager = None
_load_language_fn = None


def set_cache_manager(manager: Any) -> None:
    global _cache_manager
    _cache_manager = manager


def set_cache_language_loader(fn: Any) -> None:
    global _load_language_fn
    _load_language_fn = fn


def _get_manager():
    if _cache_manager is None:
        raise HTTPException(500, "KV cache manager not initialized")
    return _cache_manager


def _sanitize_model_id(model_id: str) -> str:
    """Validate model ID from HTTP input. Allows HF-style IDs and bare GGUF filenames."""
    if (
        ".." in model_id
        or model_id.startswith(("/", "\\"))
        or "\\" in model_id
        or (len(model_id) > 1 and model_id[1] == ":")
    ):
        raise HTTPException(400, f"Invalid model_id: {model_id}")
    if not re.match(r"^[a-zA-Z0-9_.\-]+(/[a-zA-Z0-9_.\-]+)?(:[a-zA-Z0-9_.\-]+)?$", model_id):
        raise HTTPException(400, f"Invalid model_id format: {model_id}")
    return model_id


# ── Request/Response models ─────────────────────────────────────────────────


MAX_PREPARE_MESSAGES = 200
MAX_PREPARE_TOOLS = 128
MAX_MESSAGE_CONTENT_CHARS = 200_000  # ~50k tokens


class CachePrepareRequest(BaseModel):
    model: str = Field(..., description="Model ID to prepare cache for")
    messages: list[dict] = Field(
        ..., description="Messages to cache (system prompt, etc)"
    )
    tools: list[dict] | None = Field(
        None, description="Tool definitions to include"
    )
    pinned: bool = Field(
        False, description="Pin cache (won't be evicted by GC)"
    )
    ttl: float | None = Field(
        None, description="TTL in seconds (None = use default)"
    )
    warm: bool = Field(
        True,
        description=(
            "If true, loads model and pre-computes KV state "
            "(slower but instant cache hits). "
            "If false, segment-only indexing."
        ),
    )


class CachePrepareResponse(BaseModel):
    cache_key: str
    model: str
    token_count: int
    size_bytes: int
    segments: list[dict]


class CacheValidateRequest(BaseModel):
    cache_key: str
    model: str
    messages: list[dict]
    tools: list[dict] | None = None


class CacheValidateResponse(BaseModel):
    status: str  # hit, partial_hit, miss
    cache_key: str
    reusable_tokens: int
    invalidated_at: str | None
    reason: str


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/v1/cache/prepare", response_model=CachePrepareResponse)
@handle_endpoint_errors("cache_prepare")
async def prepare_cache(request: CachePrepareRequest) -> CachePrepareResponse:
    """Pre-warm KV cache for a message prefix (system prompt, tools, history).

    Loads the model, tokenizes the messages, runs a forward pass to build KV
    state, and serializes it. Returns a cache_key that can be passed to
    /v1/chat/completions to skip all prefix processing.

    Use this to pre-warm system prompts, RAG context, or tool definitions
    at startup so the first user message gets instant TTFT.

    Set warm=false for lightweight segment-only indexing (no model load).
    """
    manager = _get_manager()
    sanitized_model = _sanitize_model_id(request.model)

    # Input validation
    if len(request.messages) > MAX_PREPARE_MESSAGES:
        raise HTTPException(
            400,
            f"Too many messages ({len(request.messages)}), "
            f"max {MAX_PREPARE_MESSAGES}",
        )
    if request.tools and len(request.tools) > MAX_PREPARE_TOOLS:
        raise HTTPException(
            400,
            f"Too many tools ({len(request.tools)}), "
            f"max {MAX_PREPARE_TOOLS}",
        )
    def _content_chars(content: Any) -> int:
        """Count characters in message content, handling multimodal lists."""
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    total += len(part.get("text", ""))
                elif part.get("type") == "image_url":
                    # Count base64 data URI size to prevent bypass
                    image_url = part.get("image_url")
                    if isinstance(image_url, dict):
                        total += len(image_url.get("url", ""))
            return total
        return 0

    total_chars = sum(
        _content_chars(m.get("content")) for m in request.messages
    )
    if total_chars > MAX_MESSAGE_CONTENT_CHARS:
        raise HTTPException(
            400,
            f"Total message content too large ({total_chars} chars), "
            f"max {MAX_MESSAGE_CONTENT_CHARS}",
        )

    model = None
    if request.warm:
        if _load_language_fn is None:
            raise HTTPException(500, "Language model loader not configured")
        try:
            from utils.model_format import parse_model_with_quantization
            model_id, quant = parse_model_with_quantization(sanitized_model)
            model_wrapper = await _load_language_fn(model_id, preferred_quantization=quant)
            # Get the inner Llama instance (not the GGUFLanguageModel wrapper)
            model = getattr(model_wrapper, 'llama', model_wrapper)
        except Exception as e:
            logger.warning(f"Failed to load model for warm prepare: {e}")
            # Fall back to segment-only

    entry = await manager.prepare(
        model_id=sanitized_model,
        messages=request.messages,
        tools=request.tools,
        pinned=request.pinned,
        ttl=request.ttl,
        model=model,
    )

    return CachePrepareResponse(
        cache_key=entry.cache_key,
        model=entry.model_id,
        token_count=entry.token_count,
        size_bytes=entry.size_bytes,
        segments=[{"type": s["type"], "hash": s["hash"]} for s in entry.segments],
    )


@router.post("/v1/cache/validate", response_model=CacheValidateResponse)
@handle_endpoint_errors("cache_validate")
async def validate_cache(request: CacheValidateRequest) -> CacheValidateResponse:
    """Validate a cache key against a payload without using it.

    Useful for checking if a cache would hit before sending a full request.
    """
    manager = _get_manager()
    sanitized_model = _sanitize_model_id(request.model)
    result = manager.validate_and_match(
        cache_key=request.cache_key,
        model_id=sanitized_model,
        messages=request.messages,
        tools=request.tools,
    )
    return CacheValidateResponse(
        status=result["status"],
        cache_key=request.cache_key,
        reusable_tokens=result["reusable_tokens"],
        invalidated_at=result.get("invalidated_at"),
        reason=result["reason"],
    )


@router.get("/v1/cache")
@handle_endpoint_errors("cache_list")
async def list_caches() -> dict[str, Any]:
    """List all cache entries."""
    manager = _get_manager()
    entries = manager.list_entries()
    return {
        "entries": entries,
        "count": len(entries),
    }


@router.get("/v1/cache/stats")
@handle_endpoint_errors("cache_stats")
async def cache_stats() -> dict[str, Any]:
    """Get cache statistics — usage, hit rates, tier breakdown."""
    manager = _get_manager()
    return manager.get_stats()


@router.delete("/v1/cache/{cache_key}")
@handle_endpoint_errors("cache_evict")
async def evict_cache(cache_key: str) -> dict[str, Any]:
    """Evict a specific cache entry."""
    manager = _get_manager()
    if manager.evict(cache_key):
        return {"status": "evicted", "cache_key": cache_key}
    raise HTTPException(404, f"Cache entry not found: {cache_key}")


@router.post("/v1/cache/gc")
@handle_endpoint_errors("cache_gc")
async def force_gc() -> dict[str, Any]:
    """Force garbage collection — removes expired entries."""
    manager = _get_manager()
    removed = manager.gc()
    return {"status": "ok", "removed": removed}
