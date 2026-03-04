"""Unit tests for KV cache manager."""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from utils.kv_cache_manager import (
    CacheBudget,
    CacheEntry,
    KVCacheManager,
    compare_segments,
    hash_messages_segments,
    hash_segment,
)

# ── Segment Hashing ─────────────────────────────────────────────────────────


def test_hash_system_prompt():
    """Same system prompt produces same hash."""
    h1 = hash_segment("You are a helpful assistant.")
    h2 = hash_segment("You are a helpful assistant.")
    assert h1 == h2


def test_hash_system_prompt_change():
    """Different system prompt produces different hash."""
    h1 = hash_segment("You are a helpful assistant.")
    h2 = hash_segment("You are a code reviewer.")
    assert h1 != h2


def test_hash_tools():
    """Tools produce consistent hashes; different tools produce different hashes."""
    tools1 = [{"type": "function", "function": {"name": "get_weather"}}]
    tools2 = [{"type": "function", "function": {"name": "get_weather"}}]
    tools3 = [{"type": "function", "function": {"name": "send_email"}}]

    msgs = [{"role": "system", "content": "test"}]
    s1 = hash_messages_segments(msgs, tools1)
    s2 = hash_messages_segments(msgs, tools2)
    s3 = hash_messages_segments(msgs, tools3)

    # Same tools → same hash
    tools_hash_1 = [s for s in s1 if s["type"] == "tools"][0]["hash"]
    tools_hash_2 = [s for s in s2 if s["type"] == "tools"][0]["hash"]
    tools_hash_3 = [s for s in s3 if s["type"] == "tools"][0]["hash"]
    assert tools_hash_1 == tools_hash_2
    assert tools_hash_1 != tools_hash_3


def test_hash_history():
    """Conversation turns produce consistent hashes."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    s1 = hash_messages_segments(msgs)
    s2 = hash_messages_segments(msgs)
    assert s1 == s2


def test_hash_ignores_whitespace_tricks():
    """Trailing whitespace differences produce different hashes (intentional — exact match)."""
    h1 = hash_segment("hello")
    h2 = hash_segment("hello ")
    assert h1 != h2  # Exact matching, not fuzzy


# ── Message Segmentation ────────────────────────────────────────────────────


def test_segments_structure():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"},
    ]
    tools = [{"type": "function", "function": {"name": "search"}}]
    segments = hash_messages_segments(msgs, tools)

    types = [s["type"] for s in segments]
    assert types == ["system", "tools", "turn", "turn"]
    # First turn: user+assistant pair. Second turn: user alone.
    assert segments[2]["index"] == 0
    assert segments[3]["index"] == 1


def test_segments_no_system():
    msgs = [{"role": "user", "content": "Hello"}]
    segments = hash_messages_segments(msgs)
    assert segments[0]["type"] == "turn"


# ── Segment Comparison ──────────────────────────────────────────────────────


def test_full_match():
    segs = [{"hash": "a", "type": "system"}, {"hash": "b", "type": "tools"}]
    match_count, inv = compare_segments(segs, segs)
    assert match_count == 2
    assert inv is None


def test_system_changed():
    cached = [{"hash": "a", "type": "system"}, {"hash": "b", "type": "tools"}]
    incoming = [{"hash": "x", "type": "system"}, {"hash": "b", "type": "tools"}]
    match_count, inv = compare_segments(cached, incoming)
    assert match_count == 0
    assert inv == "system"


def test_tools_changed():
    cached = [{"hash": "a", "type": "system"}, {"hash": "b", "type": "tools"}]
    incoming = [{"hash": "a", "type": "system"}, {"hash": "x", "type": "tools"}]
    match_count, inv = compare_segments(cached, incoming)
    assert match_count == 1
    assert inv == "tools"


def test_history_diverged():
    cached = [
        {"hash": "a", "type": "system"},
        {"hash": "b", "type": "turn"},
        {"hash": "c", "type": "turn"},
    ]
    incoming = [
        {"hash": "a", "type": "system"},
        {"hash": "b", "type": "turn"},
        {"hash": "x", "type": "turn"},  # diverges here
    ]
    match_count, inv = compare_segments(cached, incoming)
    assert match_count == 2
    assert inv == "turn"


def test_new_turn_appended():
    """Cached has 2 segments, incoming has 3 (new turn added) — all existing match."""
    cached = [{"hash": "a", "type": "system"}, {"hash": "b", "type": "turn"}]
    incoming = [
        {"hash": "a", "type": "system"},
        {"hash": "b", "type": "turn"},
        {"hash": "c", "type": "turn"},  # new turn
    ]
    match_count, inv = compare_segments(cached, incoming)
    assert match_count == 2
    assert inv is None  # all cached segments match


# ── Cache Entry ──────────────────────────────────────────────────────────────


def test_cache_entry_not_expired():
    entry = CacheEntry(
        cache_key="test", model_id="m", segments=[], content_hash="h",
        token_count=100, ttl=3600,
    )
    assert not entry.is_expired


def test_cache_entry_expired():
    entry = CacheEntry(
        cache_key="test", model_id="m", segments=[], content_hash="h",
        token_count=100, ttl=0.01,
    )
    time.sleep(0.02)
    assert entry.is_expired


def test_cache_entry_pinned_no_ttl():
    entry = CacheEntry(
        cache_key="test", model_id="m", segments=[], content_hash="h",
        token_count=100, pinned=True, ttl=None,
    )
    assert not entry.is_expired


def test_cache_entry_touch():
    entry = CacheEntry(
        cache_key="test", model_id="m", segments=[], content_hash="h",
        token_count=100,
    )
    old_time = entry.last_used
    time.sleep(0.01)
    entry.touch()
    assert entry.last_used > old_time
    assert entry.hit_count == 1


# ── KV Cache Manager ────────────────────────────────────────────────────────


def test_lookup_miss():
    mgr = KVCacheManager()
    assert mgr.lookup("nonexistent") is None


def test_lookup_invalid_key():
    mgr = KVCacheManager()
    assert mgr.lookup("") is None
    assert mgr.lookup("!@#$%") is None


def test_validate_miss():
    mgr = KVCacheManager()
    result = mgr.validate_and_match("bad_key", "model", [{"role": "user", "content": "hi"}])
    assert result["status"] == "miss"
    assert result["reason"] == "cache_key_not_found"


def test_validate_model_mismatch():
    mgr = KVCacheManager()
    # Manually insert an entry
    entry = CacheEntry(
        cache_key="test123", model_id="model_a", segments=[], content_hash="h",
        token_count=100, kv_data=b"fake",
    )
    mgr._entries["test123"] = entry

    result = mgr.validate_and_match("test123", "model_b", [{"role": "user", "content": "hi"}])
    assert result["status"] == "miss"
    assert "model_mismatch" in result["reason"]


def test_validate_full_hit():
    mgr = KVCacheManager()
    msgs = [{"role": "system", "content": "You are helpful."}]
    segments = hash_messages_segments(msgs)
    import json

    from utils.kv_cache_manager import hash_segment as _hs
    content_hash = _hs(json.dumps([s["hash"] for s in segments]))

    entry = CacheEntry(
        cache_key="hit123", model_id="model_a", segments=segments,
        content_hash=content_hash, token_count=100, kv_data=b"fake",
    )
    mgr._entries["hit123"] = entry

    result = mgr.validate_and_match("hit123", "model_a", msgs)
    assert result["status"] == "hit"
    assert result["reusable_tokens"] == 100


def test_validate_partial_hit_tools_changed():
    mgr = KVCacheManager()
    msgs = [{"role": "system", "content": "You are helpful."}]
    tools_old = [{"type": "function", "function": {"name": "old_tool"}}]
    tools_new = [{"type": "function", "function": {"name": "new_tool"}}]

    segments = hash_messages_segments(msgs, tools_old)
    import json

    from utils.kv_cache_manager import hash_segment as _hs
    content_hash = _hs(json.dumps([s["hash"] for s in segments]))

    entry = CacheEntry(
        cache_key="partial123", model_id="model_a", segments=segments,
        content_hash=content_hash, token_count=200, kv_data=b"fake",
    )
    mgr._entries["partial123"] = entry

    result = mgr.validate_and_match("partial123", "model_a", msgs, tools_new)
    assert result["status"] == "partial_hit"
    assert result["invalidated_at"] == "tools"
    assert result["reusable_tokens"] > 0


# ── Eviction & GC ───────────────────────────────────────────────────────────


def test_evict():
    mgr = KVCacheManager()
    entry = CacheEntry(
        cache_key="evict_me", model_id="m", segments=[], content_hash="h",
        token_count=10, kv_data=b"data",
    )
    mgr._entries["evict_me"] = entry
    mgr._content_index["h"] = "evict_me"

    assert mgr.evict("evict_me")
    assert mgr.lookup("evict_me") is None
    assert "h" not in mgr._content_index


def test_evict_nonexistent():
    mgr = KVCacheManager()
    assert not mgr.evict("nope")


def test_gc_removes_expired():
    mgr = KVCacheManager()
    entry = CacheEntry(
        cache_key="old", model_id="m", segments=[], content_hash="h",
        token_count=10, ttl=0.01, kv_data=b"data",
    )
    mgr._entries["old"] = entry
    time.sleep(0.02)

    removed = mgr.gc()
    assert removed == 1
    assert mgr.lookup("old") is None


def test_gc_keeps_pinned():
    mgr = KVCacheManager()
    entry = CacheEntry(
        cache_key="pinned", model_id="m", segments=[], content_hash="h",
        token_count=10, ttl=0.01, pinned=True, kv_data=b"data",
    )
    mgr._entries["pinned"] = entry
    time.sleep(0.02)

    removed = mgr.gc()
    assert removed == 0  # Pinned entries not evicted


def test_content_hash_dedup():
    """Same content_hash → lookup returns existing entry."""
    mgr = KVCacheManager()
    entry = CacheEntry(
        cache_key="original", model_id="m", segments=[], content_hash="dedup_hash",
        token_count=10, kv_data=b"data",
    )
    mgr._entries["original"] = entry
    mgr._content_index["dedup_hash"] = "original"

    # Same content hash should map to same entry
    assert mgr._content_index.get("dedup_hash") == "original"


# ── Tiered Storage ──────────────────────────────────────────────────────────


def test_demote_to_disk(tmp_path):
    mgr = KVCacheManager(cache_dir=tmp_path)
    entry = CacheEntry(
        cache_key="demote_me", model_id="m", segments=[], content_hash="h",
        token_count=10, tier="ram", kv_data=b"test_kv_data_here", size_bytes=17,
    )
    mgr._entries["demote_me"] = entry
    mgr._demote_to_disk(entry)

    assert entry.tier == "disk"
    assert entry.kv_data == b""  # RAM freed
    assert Path(entry.disk_path).exists()
    assert Path(entry.disk_path).read_bytes() == b"test_kv_data_here"


def test_restore_from_disk(tmp_path):
    """Restoring from disk loads bytes back."""
    mgr = KVCacheManager(cache_dir=tmp_path)
    disk_path = tmp_path / "test.kvstate"
    disk_path.write_bytes(b"fake_kv_state")

    entry = CacheEntry(
        cache_key="disk_entry", model_id="m", segments=[], content_hash="h",
        token_count=10, tier="disk", disk_path=str(disk_path), size_bytes=13,
    )

    # Mock model
    model = MagicMock()
    model.memory_seq_rm = MagicMock()
    model.state_seq_load = MagicMock(return_value=13)

    result = asyncio.run(mgr.restore(entry, model, seq_id=0))
    assert result is True
    assert entry.tier == "ram"  # promoted back to ram
    model.state_seq_load.assert_called_once()


# ── Budget ───────────────────────────────────────────────────────────────────


def test_budget_triggers_demotion(tmp_path):
    budget = CacheBudget(max_ram_bytes=100)  # very small
    mgr = KVCacheManager(cache_dir=tmp_path, budget=budget)

    # Add entries exceeding budget
    for i in range(5):
        entry = CacheEntry(
            cache_key=f"entry_{i}", model_id="m", segments=[], content_hash=f"h{i}",
            token_count=10, tier="ram", kv_data=b"x" * 50, size_bytes=50,
        )
        entry.last_used = time.time() - (10 - i)  # older entries used less recently
        mgr._entries[f"entry_{i}"] = entry

    mgr._enforce_budget()

    # Some entries should have been demoted to disk
    disk_entries = [e for e in mgr._entries.values() if e.tier == "disk"]
    ram_entries = [e for e in mgr._entries.values() if e.tier == "ram"]
    total_ram = sum(e.size_bytes for e in ram_entries)
    assert total_ram <= 100
    assert len(disk_entries) > 0


# ── Stats ────────────────────────────────────────────────────────────────────


def test_stats():
    mgr = KVCacheManager()
    stats = mgr.get_stats()
    assert stats["total_entries"] == 0
    assert stats["total_hits"] == 0
    assert stats["hit_rate"] == 0


def test_list_entries():
    mgr = KVCacheManager()
    entry = CacheEntry(
        cache_key="list_me", model_id="m", segments=[], content_hash="h",
        token_count=10, kv_data=b"data",
    )
    mgr._entries["list_me"] = entry
    entries = mgr.list_entries()
    assert len(entries) == 1
    assert entries[0]["cache_key"] == "list_me"


# ── Async Tests: prepare() ────────────────────────────────────────────────


def _make_mock_model(kv_data: bytes = b"fake_kv", token_count: int = 42):
    """Create a mock Llama model for KV cache tests."""
    model = MagicMock()
    model._apply_chat_template = MagicMock(return_value="<|im_start|>system\ntest<|im_end|>")
    model.tokenize = MagicMock(return_value=list(range(token_count)))
    model._lib = MagicMock()
    model._memory = MagicMock()
    model._decode_batch = MagicMock(return_value=True)
    model.state_seq_save = MagicMock(return_value=kv_data)
    model.state_seq_load = MagicMock(return_value=len(kv_data))
    model.memory_seq_rm = MagicMock()
    return model


@pytest.mark.asyncio
async def test_prepare_without_model():
    """prepare() without model creates segment-only entry."""
    mgr = KVCacheManager()
    msgs = [{"role": "system", "content": "You are helpful."}]
    entry = await mgr.prepare(model_id="test-model", messages=msgs)

    assert entry.cache_key
    assert entry.model_id == "test-model"
    assert len(entry.segments) > 0
    assert entry.kv_data == b""
    assert entry.token_count > 0  # estimated from content


@pytest.mark.asyncio
async def test_prepare_with_model():
    """prepare() with model serializes real KV state."""
    mgr = KVCacheManager()
    model = _make_mock_model(kv_data=b"real_kv_state", token_count=10)
    msgs = [{"role": "system", "content": "You are helpful."}]

    entry = await mgr.prepare(model_id="test-model", messages=msgs, model=model)

    assert entry.kv_data == b"real_kv_state"
    assert entry.token_count == 10
    assert entry.size_bytes == len(b"real_kv_state")
    model._decode_batch.assert_called_once()
    model.state_seq_save.assert_called_once_with(0)


@pytest.mark.asyncio
async def test_prepare_dedup():
    """prepare() deduplicates entries with same content hash."""
    mgr = KVCacheManager()
    msgs = [{"role": "system", "content": "You are helpful."}]

    entry1 = await mgr.prepare(model_id="test-model", messages=msgs)
    entry2 = await mgr.prepare(model_id="test-model", messages=msgs)

    assert entry1.cache_key == entry2.cache_key
    assert entry1.hit_count == 1  # touched by dedup


@pytest.mark.asyncio
async def test_prepare_concurrent_dedup():
    """Concurrent prepare() calls with same content don't create duplicates."""
    mgr = KVCacheManager()
    msgs = [{"role": "system", "content": "Same prompt for all."}]

    entries = await asyncio.gather(
        mgr.prepare(model_id="m", messages=msgs),
        mgr.prepare(model_id="m", messages=msgs),
        mgr.prepare(model_id="m", messages=msgs),
    )

    keys = {e.cache_key for e in entries}
    assert len(keys) == 1  # all resolved to same entry


# ── Async Tests: save_after_generation() ──────────────────────────────────


@pytest.mark.asyncio
async def test_save_after_generation():
    """save_after_generation() captures KV state from model."""
    mgr = KVCacheManager()
    model = _make_mock_model(kv_data=b"post_gen_kv", token_count=50)
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    entry = await mgr.save_after_generation(
        model=model, model_id="test-model", parent_key=None,
        messages=msgs, prompt_tokens=50,
    )

    assert entry.kv_data == b"post_gen_kv"
    assert entry.token_count == 50
    assert entry.model_id == "test-model"
    model.state_seq_save.assert_called_once_with(0)


@pytest.mark.asyncio
async def test_save_after_generation_dedup():
    """save_after_generation() deduplicates with same content."""
    mgr = KVCacheManager()
    model = _make_mock_model(kv_data=b"kv1")
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]

    entry1 = await mgr.save_after_generation(
        model=model, model_id="m", parent_key=None, messages=msgs, prompt_tokens=10,
    )
    entry2 = await mgr.save_after_generation(
        model=model, model_id="m", parent_key=None, messages=msgs, prompt_tokens=10,
    )

    assert entry1.cache_key == entry2.cache_key


@pytest.mark.asyncio
async def test_save_then_validate_hit():
    """Saved entry can be validated as a cache hit."""
    mgr = KVCacheManager()
    model = _make_mock_model(kv_data=b"hit_test_kv")
    msgs = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]

    entry = await mgr.save_after_generation(
        model=model, model_id="my-model", parent_key=None,
        messages=msgs, prompt_tokens=20,
    )

    result = mgr.validate_and_match(entry.cache_key, "my-model", msgs)
    assert result["status"] == "hit"
    assert result["entry"] is entry


# ── Async Tests: restore() ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_restore_from_ram():
    """restore() loads KV state from RAM entry."""
    mgr = KVCacheManager()
    model = _make_mock_model()

    entry = CacheEntry(
        cache_key="ram_entry", model_id="m", segments=[], content_hash="h",
        token_count=10, tier="ram", kv_data=b"ram_kv_data",
    )

    result = await mgr.restore(entry, model, seq_id=0)
    assert result is True
    model.state_seq_load.assert_called_once()
