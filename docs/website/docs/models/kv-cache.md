# KV Cache — Server-Side Prompt Caching

LlamaFarm's KV Cache eliminates redundant prompt processing by serializing and restoring the model's key-value state across requests. In multi-turn conversations and multi-agent setups, this reduces Time To First Token (TTFT) by **10-20x**.

## How It Works

Large language models process every token in the prompt through a forward pass before generating the first output token. In a typical agent conversation, this means reprocessing the system prompt, tool definitions, RAG context, and full conversation history on every turn — even though most of it hasn't changed.

LlamaFarm's KV Cache solves this by:

1. **Serializing** the model's KV state after each completion
2. **Returning a `cache_key`** in the response
3. **Restoring** the KV state on the next request when the client sends the key back
4. **Validating** via segment hashing that the conversation prefix is unchanged
5. **Decoding only the new tokens** (the latest user message)

```
Turn 1: [system + RAG + user1] → process 2000 tokens → response + cache_key_1
Turn 2: [system + RAG + user1 + assistant1 + user2] + cache_key_1
         ├── restore KV from cache_key_1 (60ms)
         ├── decode only user2 (~25 new tokens)
         └── response + cache_key_2
Turn 3: [full history + user3] + cache_key_2
         ├── restore KV from cache_key_2 (60ms)
         ├── decode only user3 (~30 new tokens)
         └── response + cache_key_3
```

## Quick Start

### Multi-Turn Cache Chaining

```python
import json
import httpx

base = "http://localhost:11540"

# Turn 1: get a cache key
r1 = httpx.post(f"{base}/v1/chat/completions", json={
    "model": "Qwen/Qwen3-8B-GGUF",
    "messages": [
        {"role": "system", "content": "You are a financial analyst..."},  # large system prompt
        {"role": "user", "content": "What are our top risks?"}
    ],
    "return_cache_key": True,  # ask for a cache key in the response
    "stream": True,
})
# Parse SSE stream to extract cache key and response content
cache_key_1 = None
r1_content = ""
for line in r1.iter_lines():
    if line.startswith("event: x_cache"):
        continue  # next line has the cache data
    if line.startswith("data: ") and line != "data: [DONE]":
        payload = json.loads(line[6:])
        # Check for cache event (named SSE event)
        if "new_cache_key" in payload:
            cache_key_1 = payload["new_cache_key"]
        # Collect response content
        choices = payload.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            r1_content += delta.get("content", "")

# Turn 2: send the cache key — only the new message gets processed
r2 = httpx.post(f"{base}/v1/chat/completions", json={
    "model": "Qwen/Qwen3-8B-GGUF",
    "messages": [
        {"role": "system", "content": "You are a financial analyst..."},
        {"role": "user", "content": "What are our top risks?"},
        {"role": "assistant", "content": r1_content},  # full Turn 1 response
        {"role": "user", "content": "What about NVDA specifically?"}  # only this gets decoded
    ],
    "cache_key": cache_key_1,       # restore KV from Turn 1
    "return_cache_key": True,        # get cache_key_2 for next turn
})
# Non-streaming response — cache info is in x_cache field
cache_key_2 = r2.json()["x_cache"]["new_cache_key"]
```

### Pre-Warming System Prompts

Pre-compute KV state at startup so the first user message gets instant TTFT:

```python
# At startup: pre-warm your system prompt + tools
prep = httpx.post(f"{base}/v1/cache/prepare", json={
    "model": "Qwen/Qwen3-8B-GGUF",
    "messages": [
        {"role": "system", "content": "You are a financial analyst..."}
    ],
    "tools": [{"type": "function", "function": {"name": "get_price", "parameters": {}}}],
    "warm": True,    # actually loads model and pre-computes KV
    "pinned": True,  # won't be evicted by GC
})
system_cache_key = prep.json()["cache_key"]

# Later: first user message — no cold start
r = httpx.post(f"{base}/v1/chat/completions", json={
    "model": "Qwen/Qwen3-8B-GGUF",
    "messages": [
        {"role": "system", "content": "You are a financial analyst..."},
        {"role": "user", "content": "What's our portfolio value?"}
    ],
    "cache_key": system_cache_key,
    "return_cache_key": True,
    "stream": True,
})
```

## API Reference

### Chat Completions — Cache Parameters

Added to `POST /v1/chat/completions`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `cache_key` | `string` | Cache key from a previous response or `/v1/cache/prepare`. Server restores KV state and only processes new tokens. |
| `return_cache_key` | `bool` | If true, saves KV state after generation and returns a `new_cache_key` in the response. |

**Response field** (`x_cache` in response body or SSE event):

```json
{
  "x_cache": {
    "hit": true,
    "status": "hit",
    "cache_key": "abc123",
    "reused_tokens": 1851,
    "new_cache_key": "def456",
    "cached_tokens": 1958
  }
}
```

| Field | Description |
|-------|-------------|
| `hit` | Whether the cache was used |
| `status` | `"hit"`, `"partial_hit"`, or `"miss"` |
| `reused_tokens` | Number of tokens restored from cache |
| `new_cache_key` | Cache key for this turn's state (use for next request) |
| `cached_tokens` | Total tokens in the new cache entry |

For streaming responses, cache info is emitted as a named SSE event (`event: x_cache`) before `[DONE]`. The OpenAI SDK ignores named events, so this is fully compatible:
```
event: x_cache
data: {"hit": true, "new_cache_key": "def456", "cached_tokens": 1958}

data: [DONE]
```

### POST /v1/cache/prepare

Pre-compute KV state for a message prefix.

**Request:**

```json
{
  "model": "Qwen/Qwen3-8B-GGUF",
  "messages": [{"role": "system", "content": "..."}],
  "tools": [{"type": "function", "function": {...}}],
  "warm": true,
  "pinned": true,
  "ttl": 3600
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | required | Model ID |
| `messages` | `list` | required | Messages to pre-warm (typically system prompt) |
| `tools` | `list` | `null` | Tool definitions to include |
| `warm` | `bool` | `true` | If true, loads model and runs forward pass. If false, segment-only indexing. |
| `pinned` | `bool` | `false` | Pinned entries are never evicted by GC |
| `ttl` | `float` | `1800` | Time-to-live in seconds (null = no expiry when pinned) |

**Response:**

```json
{
  "cache_key": "704e05061389",
  "model": "Qwen/Qwen3-8B-GGUF",
  "token_count": 1730,
  "size_bytes": 255120384,
  "segments": [{"type": "system", "hash": "a1b2c3d4"}]
}
```

### GET /v1/cache/stats

```json
{
  "total_entries": 5,
  "by_tier": {"ram": 4, "disk": 1},
  "ram_bytes": 1020000000,
  "total_hits": 12,
  "total_misses": 2,
  "hit_rate": 0.86,
  "pinned_entries": 1
}
```

### Other Cache Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /v1/cache` | GET | List all cache entries |
| `POST /v1/cache/validate` | POST | Check if a cache_key would hit without using it |
| `DELETE /v1/cache/{key}` | DELETE | Evict a specific entry |
| `POST /v1/cache/gc` | POST | Force garbage collection of expired entries |

## Segment-Based Validation

The cache validates requests segment-by-segment:

1. **System prompt** — hash of all system messages
2. **Tools** — hash of tool definitions (sorted for determinism)
3. **Conversation turns** — hash of each user+assistant pair

If the system prompt changes → full miss. If a mid-conversation turn changes → partial hit (reuse up to the changed point). If only new turns are appended → full hit.

This means:
- Changing your system prompt invalidates the cache (correct behavior)
- Adding a new user message to an existing conversation → cache hit
- Editing a previous message → miss from that point forward

## Tiered Storage

KV state is managed across tiers:

| Tier | Storage | Latency | Budget |
|------|---------|---------|--------|
| RAM | In-process bytes | ~60ms restore | 2GB default |
| Disk | `~/.llamafarm/cache/kv/` | ~200ms restore | 10GB default |

When RAM budget is exceeded, least-recently-used entries are demoted to disk. Expired entries are cleaned by background GC (runs every 60s).

## Integration Patterns

### Agentic Workflows

For multi-agent systems where agents share a model:

```python
# Agent A: financial analyst
agent_a_cache = prepare_cache(model, system_prompt_a, tools_a, pinned=True)

# Agent B: code assistant  
agent_b_cache = prepare_cache(model, system_prompt_b, tools_b, pinned=True)

# Each agent sends its cache_key — no cross-contamination,
# instant TTFT even after the other agent just used the model
```

### LlamaFarm Project Config

Future: auto-warm caches from project config at startup:

```yaml
# llamafarm.yaml
projects:
  financial-agent:
    provider: universal
    model: Qwen/Qwen3-8B-GGUF
    system_prompt: "You are a financial analyst..."
    tools:
      - get_portfolio
      - get_market_data
    cache:
      warm_on_startup: true
      pinned: true
```

### OpenAI SDK Compatible

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11540/v1", api_key="unused")

# Pass cache params via extra_body
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B-GGUF",
    messages=[...],
    stream=True,
    extra_body={
        "cache_key": "abc123",
        "return_cache_key": True,
    },
)
```

## Benchmarking

Run the included benchmark to measure TTFT savings on your hardware:

```bash
cd runtimes/universal
uv run python benchmarks/kv_cache_benchmark.py

# Against a specific server:
uv run python benchmarks/kv_cache_benchmark.py --base-url http://localhost:14345
```

### Reference Results (Qwen3-8B Q4_K_M, Apple M1 Max)

| Turn | TTFT (no cache) | TTFT (Llama-cache) | Speedup |
|------|----------------:|-------------------:|--------:|
| Turn 1 (cold) | 5,548ms | 5,548ms | 1x |
| Turn 1 (pre-warmed) | 5,341ms | 392ms | **14x** |
| Turn 2 (cached) | 5,548ms | 246ms | **22x** |
| Turn 3 (cached) | 5,887ms | 329ms | **18x** |

## Limitations

- **Single-model cache**: Each cache entry is tied to one model. Loading a different model invalidates all entries.
- **Thinking tokens**: Models that generate `<think>...</think>` tokens include them in the cached state. The cache handles this transparently.
- **Memory usage**: KV state for Qwen3-8B is ~250MB per entry. Budget defaults (2GB RAM) allow ~8 concurrent cache entries.
- **No cross-request KV stitching**: Due to position-dependent RoPE embeddings, you can't concatenate KV states from different requests.
