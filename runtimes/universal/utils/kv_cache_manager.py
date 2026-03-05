"""KV Cache Manager — server-side multi-agent KV cache with tiered storage.

Manages named KV cache slots so multiple agents can share a model without
evicting each other's cached prefixes. Supports segment-level validation
(system prompt, tools, history turns) so partial hits are possible when
only part of the payload has changed.

Tiers: vram (in llama.cpp context) → ram (serialized bytes) → disk → evict
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Segment Hashing ─────────────────────────────────────────────────────────


def hash_segment(content: str) -> str:
    """Deterministic hash of a content segment."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def hash_messages_segments(messages: list[dict], tools: list[dict] | None = None) -> list[dict]:
    """Break messages + tools into hashable segments.

    Returns a list of segment dicts:
      [{"type": "system", "hash": "...", "content": "..."},
       {"type": "tools",  "hash": "...", "content": "..."},
       {"type": "turn",   "hash": "...", "content": "...", "index": 0}, ...]

    The content is the raw string used for hashing (for recomputation on miss).
    """
    segments: list[dict] = []

    # Extract system prompt
    system_parts = []
    non_system: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_parts.append(msg.get("content", ""))
        else:
            non_system.append(msg)

    if system_parts:
        system_content = "\n".join(system_parts)
        segments.append({
            "type": "system",
            "hash": hash_segment(system_content),
            "content": system_content,
        })

    # Tools as a segment (canonical order for deterministic hashing)
    if tools:
        sorted_tools = sorted(
            tools,
            key=lambda t: (
                t.get("type", ""),
                t.get("function", {}).get("name", ""),
            ),
        )
        tools_content = json.dumps(sorted_tools, sort_keys=True, separators=(",", ":"))
        segments.append({
            "type": "tools",
            "hash": hash_segment(tools_content),
            "content": tools_content,
        })

    # Conversation turns (pair user+assistant as one segment)
    turn_idx = 0
    i = 0
    while i < len(non_system):
        turn_parts = []
        # Collect one turn: user + optional assistant response
        msg = non_system[i]
        turn_parts.append(f"{msg.get('role', '')}:{msg.get('content', '')}")
        i += 1
        # If next is assistant, include it in same turn
        if i < len(non_system) and non_system[i].get("role") == "assistant":
            turn_parts.append(f"assistant:{non_system[i].get('content', '')}")
            i += 1
        turn_content = "|".join(turn_parts)
        segments.append({
            "type": "turn",
            "hash": hash_segment(turn_content),
            "content": turn_content,
            "index": turn_idx,
        })
        turn_idx += 1

    return segments


def compare_segments(
    cached_segments: list[dict], incoming_segments: list[dict]
) -> tuple[int, str | None]:
    """Compare cached vs incoming segments. Returns (match_count, invalidated_at).

    match_count: how many leading segments match
    invalidated_at: type of first mismatched segment (None if all match)
    """
    for i, (cached, incoming) in enumerate(zip(cached_segments, incoming_segments, strict=False)):
        if cached["hash"] != incoming["hash"]:
            return i, cached.get("type", "unknown")

    # All compared segments match
    if len(cached_segments) <= len(incoming_segments):
        return len(cached_segments), None
    else:
        # Cached has more segments than incoming (history truncated?)
        return len(incoming_segments), "truncated"


# ── Cache Entry ──────────────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    """A cached KV state with segment metadata."""
    cache_key: str
    model_id: str
    segments: list[dict]  # segment hashes for validation
    content_hash: str  # hash of all segments combined
    token_count: int  # number of tokens in the cached prefix
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    hit_count: int = 0
    pinned: bool = False
    ttl: float | None = None  # seconds, None = no expiry
    tier: str = "ram"  # "vram" | "ram" | "disk"
    seq_id: int = -1  # llama.cpp sequence ID if in vram
    # Serialized KV state (when in ram tier)
    kv_data: bytes = b""
    # Disk path (when in disk tier)
    disk_path: str | None = None
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.last_used > self.ttl

    def touch(self) -> None:
        self.last_used = time.time()
        self.hit_count += 1

    def to_dict(self) -> dict:
        return {
            "cache_key": self.cache_key,
            "model_id": self.model_id,
            "segments": [{"type": s["type"], "hash": s["hash"]} for s in self.segments],
            "content_hash": self.content_hash,
            "token_count": self.token_count,
            "tier": self.tier,
            "size_bytes": self.size_bytes,
            "hit_count": self.hit_count,
            "pinned": self.pinned,
            "last_used": self.last_used,
            "created_at": self.created_at,
            "is_expired": self.is_expired,
        }


# ── KV Cache Manager ────────────────────────────────────────────────────────


def _generate_cache_key() -> str:
    """Generate a unique cache key (24 hex chars = 96 bits of entropy)."""
    return hashlib.sha256(os.urandom(32)).hexdigest()[:24]


@dataclass
class CacheBudget:
    """Budget limits for each tier."""
    max_vram_entries: int = 8  # max sequences in llama.cpp context
    max_ram_bytes: int = 2 * 1024 * 1024 * 1024  # 2GB
    max_disk_bytes: int = 10 * 1024 * 1024 * 1024  # 10GB
    default_ttl: float = 1800.0  # 30 minutes


class KVCacheManager:
    """Manages KV cache entries with tiered storage and GC.

    Lifecycle:
    1. prepare() — tokenize + forward pass a prefix, save KV state
    2. lookup() — find cache entry by key, validate segments
    3. restore() — load KV state back into model context
    4. save_after_generation() — update cache with new conversation state
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        budget: CacheBudget | None = None,
    ):
        self._entries: dict[str, CacheEntry] = {}  # cache_key → entry
        self._content_index: dict[str, str] = {}  # content_hash → cache_key (dedup)
        self._budget = budget or CacheBudget()
        self._cache_dir = cache_dir or Path.home() / ".llamafarm" / "cache" / "kv"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        # Stats
        self._total_hits = 0
        self._total_misses = 0
        self._total_partial_hits = 0

    # ── Core Operations ──────────────────────────────────────────────────

    async def prepare(
        self,
        model_id: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        pinned: bool = False,
        ttl: float | None = None,
        model: Any = None,  # Llama instance — if provided, does real KV serialization
    ) -> CacheEntry:
        """Pre-compute and serialize KV cache for a message prefix.

        If `model` is provided: tokenizes the messages through the model's chat
        template, runs a forward pass to build KV state, and serializes it.
        Future requests with this cache_key skip all prompt processing.

        If `model` is None: indexes segments for validation only. Real KV state
        is serialized lazily after the first completion via save_after_generation().
        """
        segments = hash_messages_segments(messages, tools)
        content_hash = hash_segment(json.dumps([s["hash"] for s in segments]))

        # Quick dedup check (under lock)
        async with self._lock:
            if content_hash in self._content_index:
                existing_key = self._content_index[content_hash]
                if existing_key in self._entries:
                    entry = self._entries[existing_key]
                    entry.touch()
                    logger.info(f"Cache dedup hit: {entry.cache_key[:8]}… (content_hash={content_hash[:8]})")
                    return entry

        kv_data = b""
        size_bytes = 0
        token_count = 0

        if model is not None:
            # Real KV serialization: tokenize → decode → serialize
            # Run blocking model ops in a thread to avoid blocking the event loop
            try:
                import time as _time
                t0 = _time.perf_counter()

                def _prepare_kv():
                    prompt = model._apply_chat_template(messages, add_generation_prompt=True)
                    tokens = model.tokenize(prompt, add_special=False, parse_special=True)
                    tc = len(tokens)
                    model._lib.llama_memory_clear(model._memory, True)
                    if not model._decode_batch(tokens):
                        raise RuntimeError(f"Failed to decode {tc} prefix tokens")
                    kv = model.state_seq_save(0)
                    return kv, tc

                kv_data, token_count = await asyncio.to_thread(_prepare_kv)
                size_bytes = len(kv_data)

                t1 = _time.perf_counter()
                logger.info(
                    f"Cache prepare (warm): {token_count} tokens, "
                    f"{size_bytes / 1024:.1f}KB KV state, "
                    f"{(t1 - t0) * 1000:.1f}ms"
                )
            except Exception as e:
                logger.error(f"KV serialization failed during prepare: {e}")
                # Fall back to segment-only
                kv_data = b""
                size_bytes = 0
                total_chars = sum(len(s.get("content", "")) for s in segments)
                token_count = max(1, total_chars // 4)
                logger.info(f"Falling back to segment-only prepare: ~{token_count} tokens")
        else:
            # Segment-only: estimate tokens, real KV saved after first completion
            total_chars = sum(len(s.get("content", "")) for s in segments)
            token_count = max(1, total_chars // 4)
            logger.info(f"Cache prepare (segment-only): ~{token_count} tokens indexed")

        cache_key = _generate_cache_key()
        entry = CacheEntry(
            cache_key=cache_key,
            model_id=model_id,
            segments=segments,
            content_hash=content_hash,
            token_count=token_count,
            pinned=pinned,
            ttl=ttl if ttl is not None else (None if pinned else self._budget.default_ttl),
            tier="ram",
            kv_data=kv_data,
            size_bytes=size_bytes,
        )

        async with self._lock:
            # Re-check dedup inside lock to prevent TOCTOU race
            if content_hash in self._content_index:
                existing_key = self._content_index[content_hash]
                if existing_key in self._entries:
                    existing = self._entries[existing_key]
                    existing.touch()
                    logger.info(f"Cache dedup hit (re-check): {existing.cache_key[:8]}…")
                    return existing
            self._entries[cache_key] = entry
            self._content_index[content_hash] = cache_key
            self._enforce_budget()

        logger.info(
            f"Prepared cache {cache_key[:8]}…: {token_count} tokens, "
            f"{size_bytes / 1024:.1f}KB, warm={'yes' if kv_data else 'no'}, "
            f"segments={[s['type'] for s in segments]}"
        )
        return entry

    def lookup(self, cache_key: str) -> CacheEntry | None:
        """Look up a cache entry by key. Returns None if not found or expired."""
        entry = self._entries.get(cache_key)
        if entry is None:
            return None
        if entry.is_expired:
            logger.debug(f"Cache {cache_key[:8]}… expired")
            return None
        return entry

    def validate_and_match(
        self,
        cache_key: str,
        model_id: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Validate a cache key against incoming payload.

        Returns a dict with:
          status: "hit" | "partial_hit" | "miss"
          entry: CacheEntry or None
          reusable_tokens: number of tokens that can be reused
          invalidated_at: segment type where mismatch occurred
          reason: human-readable reason
        """
        entry = self.lookup(cache_key)
        if entry is None:
            self._total_misses += 1
            return {
                "status": "miss",
                "entry": None,
                "reusable_tokens": 0,
                "invalidated_at": None,
                "reason": "cache_key_not_found",
            }

        # Model must match
        if entry.model_id != model_id:
            self._total_misses += 1
            return {
                "status": "miss",
                "entry": None,
                "reusable_tokens": 0,
                "invalidated_at": "model",
                "reason": f"model_mismatch: cached={entry.model_id}, requested={model_id}",
            }

        # Compare segments
        incoming_segments = hash_messages_segments(messages, tools)
        match_count, invalidated_at = compare_segments(entry.segments, incoming_segments)

        if invalidated_at is None and match_count == len(entry.segments):
            # Full hit
            entry.touch()
            self._total_hits += 1
            return {
                "status": "hit",
                "entry": entry,
                "reusable_tokens": entry.token_count,
                "invalidated_at": None,
                "reason": "full_match",
            }

        if match_count > 0:
            # Partial hit — some leading segments match
            entry.touch()
            self._total_partial_hits += 1
            # Estimate reusable tokens (proportional to matched segments)
            ratio = match_count / max(len(entry.segments), 1)
            reusable_tokens = int(entry.token_count * ratio)
            return {
                "status": "partial_hit",
                "entry": entry,
                "reusable_tokens": reusable_tokens,
                "invalidated_at": invalidated_at,
                "reason": f"{invalidated_at}_changed",
            }

        # Complete miss
        self._total_misses += 1
        return {
            "status": "miss",
            "entry": None,
            "reusable_tokens": 0,
            "invalidated_at": invalidated_at,
            "reason": f"{invalidated_at}_changed" if invalidated_at else "no_match",
        }

    async def restore(self, entry: CacheEntry, model: Any, seq_id: int = 0) -> bool:
        """Restore a cache entry's KV state into the model.

        If the entry has serialized KV data, loads it into the model context.
        If no KV data (segment-only validation), returns True as a signal
        that the prefix is validated — the caller can optimize accordingly.

        Returns True on success, False on failure.
        """
        try:
            # Segment-only entry (from prepare without serialization)
            if not entry.kv_data and not entry.disk_path:
                entry.touch()
                logger.info(
                    f"Cache validated (segment-only): {entry.cache_key[:8]}…, "
                    f"{entry.token_count} prefix tokens confirmed unchanged"
                )
                return True

            if entry.tier == "disk":
                if entry.disk_path and Path(entry.disk_path).exists():
                    # Use thread pool to avoid blocking event loop on large files
                    entry.kv_data = await asyncio.to_thread(
                        Path(entry.disk_path).read_bytes
                    )
                    entry.tier = "ram"
                    async with self._lock:
                        self._enforce_budget()
                else:
                    logger.warning(f"Cache {entry.cache_key[:8]}… disk path missing")
                    return False

            if not entry.kv_data:
                logger.warning(f"Cache {entry.cache_key[:8]}… has no KV data")
                return False

            # Run blocking model ops in a thread to avoid blocking the event loop
            kv_data = entry.kv_data

            def _restore_kv():
                model.memory_seq_rm(seq_id)
                return model.state_seq_load(kv_data, seq_id)

            consumed = await asyncio.to_thread(_restore_kv)
            if consumed == 0:
                logger.error(f"Failed to restore cache {entry.cache_key[:8]}…")
                return False

            entry.touch()
            logger.info(
                f"Restored cache {entry.cache_key[:8]}…: {entry.token_count} tokens, "
                f"{consumed} bytes into seq_id={seq_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to restore cache {entry.cache_key[:8]}…: {e}")
            return False

    async def save_after_generation(
        self,
        model: Any,
        model_id: str,
        parent_key: str | None,
        messages: list[dict],
        tools: list[dict] | None = None,
        seq_id: int = 0,
        prompt_tokens: int = 0,
    ) -> CacheEntry:
        """Save the current KV state after generation as a new cache entry.

        This creates a new cache_key that includes the full conversation
        (system + tools + all turns including the latest response).
        The parent_key is informational only.

        Args:
            prompt_tokens: Exact prompt token count from the model (for KV restore).
        """
        segments = hash_messages_segments(messages, tools)
        content_hash = hash_segment(json.dumps([s["hash"] for s in segments]))

        # Quick dedup check
        async with self._lock:
            if content_hash in self._content_index:
                existing = self._entries.get(self._content_index[content_hash])
                if existing:
                    existing.touch()
                    return existing

        # Serialize current KV state (blocking model op → run in thread)
        def _serialize_kv():
            kv = model.state_seq_save(seq_id)
            tc = prompt_tokens
            if tc <= 0:
                try:
                    prompt_text = model._apply_chat_template(
                        [dict(m) if not isinstance(m, dict) else m for m in messages],
                        add_generation_prompt=True,
                    )
                    toks = model.tokenize(prompt_text, add_special=False, parse_special=True)
                    tc = len(toks)
                except Exception as e:
                    logger.warning(f"Failed to get exact token count: {e}, using estimate")
                    tc = 0
                    for seg in segments:
                        tc += max(1, len(seg.get("content", "")) // 4)
            return kv, tc

        kv_data, token_count = await asyncio.to_thread(_serialize_kv)

        cache_key = _generate_cache_key()
        entry = CacheEntry(
            cache_key=cache_key,
            model_id=model_id,
            segments=segments,
            content_hash=content_hash,
            token_count=token_count,
            ttl=self._budget.default_ttl,
            tier="ram",
            kv_data=kv_data,
            size_bytes=len(kv_data),
        )

        async with self._lock:
            # Re-check dedup inside lock to prevent TOCTOU race
            if content_hash in self._content_index:
                existing = self._entries.get(self._content_index[content_hash])
                if existing:
                    existing.touch()
                    return existing
            self._entries[cache_key] = entry
            self._content_index[content_hash] = cache_key
            self._enforce_budget()

        logger.info(f"Saved post-generation cache {cache_key[:8]}…: ~{token_count} tokens, {len(kv_data) / 1024:.1f}KB")
        return entry

    # ── Cache Management ─────────────────────────────────────────────────

    def list_entries(self) -> list[dict]:
        """List all cache entries."""
        return [e.to_dict() for e in self._entries.values()]

    def get_stats(self) -> dict:
        """Get cache statistics."""
        entries = list(self._entries.values())
        ram_bytes = sum(e.size_bytes for e in entries if e.tier == "ram")
        disk_bytes = sum(e.size_bytes for e in entries if e.tier == "disk")
        total_requests = self._total_hits + self._total_misses + self._total_partial_hits
        return {
            "total_entries": len(entries),
            "by_tier": {
                "ram": len([e for e in entries if e.tier == "ram"]),
                "disk": len([e for e in entries if e.tier == "disk"]),
            },
            "ram_bytes": ram_bytes,
            "disk_bytes": disk_bytes,
            "total_hits": self._total_hits,
            "total_partial_hits": self._total_partial_hits,
            "total_misses": self._total_misses,
            "hit_rate": self._total_hits / max(total_requests, 1),
            "pinned_entries": len([e for e in entries if e.pinned]),
        }

    def evict(self, cache_key: str) -> bool:
        """Evict a specific cache entry.

        Note: Callers should hold self._lock when calling from async context,
        or use evict_async() instead.
        """
        entry = self._entries.pop(cache_key, None)
        if entry is None:
            return False
        # Clean up content index
        if entry.content_hash in self._content_index and self._content_index[entry.content_hash] == cache_key:
            del self._content_index[entry.content_hash]
        # Clean up disk file
        if entry.disk_path:
            with contextlib.suppress(Exception):
                Path(entry.disk_path).unlink(missing_ok=True)
        # Clear kv_data to free memory even if other references exist
        entry.kv_data = b""
        logger.info(f"Evicted cache {cache_key[:8]}…")
        return True

    async def evict_async(self, cache_key: str) -> bool:
        """Thread-safe eviction of a cache entry."""
        async with self._lock:
            return self.evict(cache_key)

    def gc(self) -> int:
        """Run garbage collection. Returns number of entries removed.

        Note: Called from the GC background task. Uses dict snapshot
        to avoid mutation during iteration.
        """
        removed = 0
        expired_keys = [
            k for k, e in list(self._entries.items())
            if e.is_expired and not e.pinned
        ]
        for key in expired_keys:
            self.evict(key)
            removed += 1
        if removed:
            logger.info(f"GC removed {removed} expired cache entries")
        return removed

    def _enforce_budget(self) -> None:
        """Enforce budget limits by demoting/evicting entries."""
        # Demote ram entries to disk if over budget
        ram_entries = [e for e in self._entries.values() if e.tier == "ram" and not e.pinned]
        ram_bytes = sum(e.size_bytes for e in self._entries.values() if e.tier == "ram")

        if ram_bytes > self._budget.max_ram_bytes:
            # Sort by last_used (oldest first)
            ram_entries.sort(key=lambda e: e.last_used)
            for entry in ram_entries:
                if ram_bytes <= self._budget.max_ram_bytes:
                    break
                self._demote_to_disk(entry)
                ram_bytes -= entry.size_bytes

        # Evict disk entries if over budget
        disk_entries = [e for e in self._entries.values() if e.tier == "disk" and not e.pinned]
        disk_bytes = sum(e.size_bytes for e in self._entries.values() if e.tier == "disk")

        if disk_bytes > self._budget.max_disk_bytes:
            disk_entries.sort(key=lambda e: e.last_used)
            for entry in disk_entries:
                if disk_bytes <= self._budget.max_disk_bytes:
                    break
                self.evict(entry.cache_key)
                disk_bytes -= entry.size_bytes

    def _demote_to_disk(self, entry: CacheEntry) -> None:
        """Move a ram entry to disk.

        Note: This performs synchronous disk I/O. When called from _enforce_budget()
        under the async lock, it blocks the event loop briefly.
        """
        if not entry.kv_data:
            return
        disk_path = self._cache_dir / f"{entry.cache_key}.kvstate"
        try:
            disk_path.write_bytes(entry.kv_data)
            entry.disk_path = str(disk_path)
            entry.kv_data = b""  # Free RAM
            entry.tier = "disk"
            logger.debug(f"Demoted cache {entry.cache_key[:8]}… to disk: {disk_path}")
        except Exception as e:
            logger.error(f"Failed to demote cache {entry.cache_key[:8]}… to disk: {e}")


# ── Background GC Task ──────────────────────────────────────────────────────

_gc_task: asyncio.Task | None = None


async def _gc_loop(manager: KVCacheManager, interval: float = 60.0) -> None:
    """Periodic GC sweep."""
    while True:
        await asyncio.sleep(interval)
        try:
            manager.gc()
        except Exception as e:
            logger.error(f"KV cache GC error: {e}")


def start_kv_cache_gc(manager: KVCacheManager) -> None:
    """Start background GC task."""
    global _gc_task
    if _gc_task is None or _gc_task.done():
        _gc_task = asyncio.create_task(_gc_loop(manager))


async def stop_kv_cache_gc() -> None:
    """Cancel the background GC task (call during shutdown)."""
    global _gc_task
    if _gc_task is not None and not _gc_task.done():
        _gc_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _gc_task
        logger.info("KV cache GC task stopped")
    _gc_task = None
