# KV Cache Test Plan

## Test Model
`unsloth/Qwen3-0.6B-GGUF` — small, fast, GGUF with chat template.

## Unit Tests (test_kv_cache.py)

### Segment Hashing
1. `test_hash_system_prompt` — same system prompt → same hash
2. `test_hash_system_prompt_change` — different system prompt → different hash
3. `test_hash_tools` — same tools → same hash, different tools → different hash
4. `test_hash_history` — same conversation turns → same hash
5. `test_hash_ignores_whitespace_tricks` — trailing spaces don't create false misses

### Cache Entry Lifecycle
6. `test_create_cache_entry` — create entry, verify fields (segments, tokens, size)
7. `test_lookup_by_key` — created entry findable by cache_key
8. `test_lookup_miss` — unknown key returns None
9. `test_lookup_invalid_key` — garbage key returns None, no crash

### Segment Comparison
10. `test_full_match` — all segments match → full hit
11. `test_system_changed` — system prompt differs → miss from position 0
12. `test_tools_changed` — tools differ → partial hit (system reused)
13. `test_history_diverged` — turn 3 of 5 changed → partial hit (reuse 0-2)
14. `test_new_turn_appended` — all existing match + new turn → full hit + extend

### Eviction & GC
15. `test_lru_eviction` — exceed budget → least recently used evicted
16. `test_pinned_not_evicted` — pinned entries survive eviction
17. `test_ttl_expiration` — entry with TTL expires after deadline
18. `test_gc_cleans_expired` — GC sweep removes expired entries
19. `test_content_hash_dedup` — same content → shared entry, not duplicated

### Tiered Storage
20. `test_demote_to_ram` — entry moves from vram tier to ram tier
21. `test_promote_from_ram` — accessing ram entry promotes to vram
22. `test_demote_to_disk` — ram entry demotes to disk (tmp_path)
23. `test_promote_from_disk` — disk entry loads back

### Budget
24. `test_budget_enforcement` — can't exceed configured max per tier
25. `test_budget_triggers_demotion` — exceeding vram budget auto-demotes oldest

## Integration Tests (test_kv_cache_integration.py) — LIVE SERVER

All tests use `unsloth/Qwen3-0.6B-GGUF` on `:11540` runtime.

### Basic Cache Flow
1. `test_prepare_cache` — POST /v1/cache/prepare with system prompt, get cache_key back
2. `test_chat_with_cache_key` — POST /v1/chat/completions with cache_key, verify response + x_cache.hit=true
3. `test_chat_without_cache_key` — normal chat, no cache_key, verify it still works (baseline)
4. `test_cache_key_speeds_up` — compare latency with/without cache_key on same prompt (cache should be faster)

### Multi-Turn Cache Chaining
5. `test_multi_turn_chain` — 
   Turn 1: send system+user → get cache_key_1
   Turn 2: send cache_key_1 + new user → get cache_key_2
   Turn 3: send cache_key_2 + new user → get cache_key_3
   Verify each turn processes fewer tokens than full recompute
6. `test_multi_turn_response_coherent` — verify the model's responses make sense across cached turns (not hallucinating from wrong context)

### Partial Hit Scenarios
7. `test_system_same_tools_changed` — cache_key with changed tools → x_cache.status=partial_hit, invalidated_at=tools
8. `test_system_changed` — cache_key with changed system prompt → x_cache.status=miss (full recompute)
9. `test_history_edited` — cache_key but one middle turn changed → partial hit up to divergence
10. `test_new_tools_added` — same tools + one more → partial hit

### Graceful Fallback
11. `test_invalid_cache_key` — send garbage cache_key + full messages → works normally, x_cache.hit=false
12. `test_expired_cache_key` — prepare, wait for eviction, send expired key → graceful fallback
13. `test_cache_key_without_messages` — send cache_key but empty messages → still works (uses cached state only + generates)

### Cache Management API
14. `test_list_caches` — GET /v1/cache → lists prepared caches
15. `test_cache_stats` — GET /v1/cache/stats → shows vram/ram/disk usage, hit rates
16. `test_evict_cache` — DELETE /v1/cache/{cache_id} → removes it, subsequent use falls back
17. `test_gc_endpoint` — POST /v1/cache/gc → force garbage collection

### Multi-Agent Simulation
18. `test_two_agents_different_prompts` — prepare 2 caches with different system prompts, alternate requests, verify no cross-contamination
19. `test_two_agents_same_prefix` — prepare 2 caches with same system prompt → verify dedup (same content hash)
20. `test_concurrent_cache_use` — 3 agents sending requests concurrently with different cache keys → all get correct responses

### Edge Cases
21. `test_very_long_system_prompt` — 2000 token system prompt, cache and reuse
22. `test_empty_system_prompt` — no system prompt, just user message, cache works
23. `test_cache_with_tools` — system prompt + 5 tool definitions cached together
24. `test_prepare_then_model_change` — prepare with model A, chat with model B → cache miss (model mismatch)

## Test Execution Order
1. Run unit tests first (no server needed) — must all pass
2. Ensure runtime is up on :11540 with Qwen3-0.6B loaded
3. Run integration tests against live server
4. Check runtime logs for errors, warnings, unexpected behavior
