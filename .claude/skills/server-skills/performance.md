# Performance Patterns for LlamaFarm Server

Best practices and code review checklist for server-specific optimizations, async patterns, and resource management.

## Overview

Performance considerations for the LlamaFarm Server:

- **Async I/O** - Non-blocking operations for HTTP, file, and database access
- **Connection Pooling** - Reuse HTTP clients and database connections
- **Caching** - In-memory caching for frequently accessed data
- **Resource Cleanup** - Proper lifecycle management with context managers

---

## Ideal Patterns

### HTTP Client Reuse

```python
import httpx
from contextlib import asynccontextmanager

# Module-level client for reuse
_http_client: httpx.AsyncClient | None = None

async def get_http_client() -> httpx.AsyncClient:
    """Get or create a shared HTTP client."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
    return _http_client

async def cleanup_http_client():
    """Cleanup on shutdown."""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None
```

### Async File Operations

```python
import aiofiles
from pathlib import Path

async def read_file_async(path: Path) -> str:
    """Read file without blocking the event loop."""
    async with aiofiles.open(path, mode='r') as f:
        return await f.read()

async def write_file_async(path: Path, content: str) -> None:
    """Write file without blocking the event loop."""
    async with aiofiles.open(path, mode='w') as f:
        await f.write(content)
```

### In-Memory Session Cache with TTL

```python
import threading
import time
from dataclasses import dataclass

SESSION_TTL_SECONDS = 30 * 60  # 30 minutes

@dataclass
class SessionRecord:
    agent: ChatOrchestratorAgent
    created_at: float
    last_used: float
    request_count: int

agent_sessions: dict[str, SessionRecord] = {}
_agent_sessions_lock = threading.RLock()

def _cleanup_expired_sessions(now: float | None = None) -> None:
    """Remove expired sessions from cache."""
    timestamp = now or time.time()
    to_delete = [
        key for key, record in agent_sessions.items()
        if timestamp - record.last_used > SESSION_TTL_SECONDS
    ]
    for key in to_delete:
        agent_sessions.pop(key, None)
```

### Streaming Response for Large Data

```python
from starlette.responses import StreamingResponse
from typing import AsyncIterator

async def stream_chat_response(
    messages: list,
) -> AsyncIterator[str]:
    """Stream chat responses as SSE events."""
    async for chunk in agent.stream_chat(messages):
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        stream_chat_response(request.messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

### Lifespan for Resource Management

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle resources."""
    # Startup
    logger.info("Starting LlamaFarm API")
    await initialize_connection_pools()
    yield
    # Shutdown
    logger.info("Shutting down LlamaFarm API")
    await cleanup_all_mcp_services()
    await cleanup_http_client()
    logger.info("Shutdown complete")

app = FastAPI(lifespan=lifespan)
```

---

## Checklist

### 1. No Blocking Calls in Async Functions

**Description:** Async route handlers must not use blocking I/O.

**Search Pattern:**
```bash
grep -rn "async def" server/api/routers/ -A20 | grep -E "time\.sleep|requests\.|open\(|os\.read"
```

**Pass Criteria:** No `time.sleep()`, `requests.*`, or synchronous file I/O in async functions.

**Severity:** Critical

**Recommendation:**
```python
# BAD
async def handler():
    time.sleep(1)  # Blocks!
    data = requests.get(url)  # Blocks!

# GOOD
async def handler():
    await asyncio.sleep(1)
    async with httpx.AsyncClient() as client:
        data = await client.get(url)
```

---

### 2. HTTP Client Reused

**Description:** Create HTTP clients once and reuse them across requests.

**Search Pattern:**
```bash
grep -rn "httpx\.\(Client\|AsyncClient\)()" server/ | grep -v "_client\|global"
```

**Pass Criteria:** No inline client creation per request.

**Severity:** Medium

**Recommendation:** Use module-level client or dependency injection:
```python
# Module level
_client = httpx.AsyncClient(timeout=30.0)

# Or as dependency
async def get_client() -> httpx.AsyncClient:
    return shared_client
```

---

### 3. Session Cache Has TTL Cleanup

**Description:** In-memory caches must implement TTL-based cleanup.

**Search Pattern:**
```bash
grep -rn "dict\[str.*\].*=" server/ | grep -v "cleanup\|TTL\|expire"
```

**Pass Criteria:** Dict-based caches have cleanup mechanism.

**Severity:** Medium

**Recommendation:** Implement periodic cleanup:
```python
def _cleanup_expired_sessions(now: float | None = None) -> None:
    timestamp = now or time.time()
    to_delete = [
        key for key, record in sessions.items()
        if timestamp - record.last_used > TTL_SECONDS
    ]
    for key in to_delete:
        sessions.pop(key, None)
```

---

### 4. Thread Lock for Shared State

**Description:** Shared mutable state requires thread synchronization.

**Search Pattern:**
```bash
grep -rn "^[a-z_]*_sessions\|^[a-z_]*_cache" server/ | grep "dict\|{}"
```

**Pass Criteria:** Module-level mutable state has associated lock.

**Severity:** High

**Recommendation:**
```python
agent_sessions: dict[str, SessionRecord] = {}
_agent_sessions_lock = threading.RLock()

with _agent_sessions_lock:
    agent_sessions[key] = record
```

---

### 5. Streaming for Large Responses

**Description:** Large responses should use streaming to reduce memory.

**Search Pattern:**
```bash
grep -rn "return.*\[.*for.*in" server/api/routers/ | grep -v "StreamingResponse"
```

**Pass Criteria:** List comprehensions for large data use streaming.

**Severity:** Medium

**Recommendation:**
```python
# BAD - loads all into memory
return [process(item) for item in large_list]

# GOOD - streams results
async def stream_results():
    for item in large_list:
        yield process(item)
return StreamingResponse(stream_results())
```

---

### 6. Context Managers for Resources

**Description:** External resources should use context managers for cleanup.

**Search Pattern:**
```bash
grep -rn "open(" server/ | grep -v "with\|async with" | grep -v ".venv"
```

**Pass Criteria:** All `open()` calls use `with` or `async with`.

**Severity:** Medium

**Recommendation:**
```python
# BAD - may leak file handle
f = open(path)
data = f.read()

# GOOD - automatic cleanup
with open(path) as f:
    data = f.read()
```

---

### 7. Lifespan Used for Startup/Shutdown

**Description:** Use lifespan context manager instead of deprecated events.

**Search Pattern:**
```bash
grep -rn "on_event" server/
```

**Pass Criteria:** No `@app.on_event("startup")` or `@app.on_event("shutdown")`.

**Severity:** Medium

**Recommendation:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await setup()
    yield
    # Shutdown
    await cleanup()

app = FastAPI(lifespan=lifespan)
```

---

### 8. Efficient JSON Serialization

**Description:** Use efficient JSON serialization for responses.

**Search Pattern:**
```bash
grep -rn "json\.dumps" server/api/ | grep -v "orjson\|ujson"
```

**Pass Criteria:** Standard `json.dumps` used appropriately or faster alternatives for hot paths.

**Severity:** Low

**Recommendation:** For high-throughput endpoints, consider orjson:
```python
import orjson
from fastapi.responses import ORJSONResponse

app = FastAPI(default_response_class=ORJSONResponse)
```

---

### 9. Pagination for List Endpoints

**Description:** List endpoints should support pagination for large datasets.

**Search Pattern:**
```bash
grep -rn "def list_\|async def list_" server/api/routers/ | grep -v "limit\|offset\|page"
```

**Pass Criteria:** List endpoints accept pagination parameters.

**Severity:** Medium

**Recommendation:**
```python
@router.get("/items")
async def list_items(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> ListResponse:
    items = service.list_items(limit=limit, offset=offset)
    return ListResponse(items=items, total=total)
```

---

### 10. Logging Level Checks

**Description:** Expensive log message construction should check level first.

**Search Pattern:**
```bash
grep -rn "logger\.debug(" server/ | grep -E "\+|format|%|f\""
```

**Pass Criteria:** Debug logs with expensive formatting check level or use lazy evaluation.

**Severity:** Low

**Recommendation:**
```python
# BAD - always constructs string
logger.debug(f"Processing {expensive_operation()}")

# GOOD - structlog handles this efficiently
logger.debug("Processing", result=value)

# Or check level
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Processing {expensive_operation()}")
```

---

## Anti-Patterns to Avoid

### 1. Creating Clients Per Request

```python
# BAD - creates new connection each time
async def handler():
    async with httpx.AsyncClient() as client:
        return await client.get(url)

# GOOD - reuse client
async def handler():
    client = await get_shared_client()
    return await client.get(url)
```

### 2. Unbounded Cache Growth

```python
# BAD - never cleans up
cache = {}
def get_cached(key):
    if key not in cache:
        cache[key] = compute(key)
    return cache[key]

# GOOD - with TTL cleanup
def get_cached(key):
    cleanup_expired()
    if key not in cache or is_expired(cache[key]):
        cache[key] = CacheEntry(value=compute(key), expires=time.time() + TTL)
    return cache[key].value
```

### 3. Blocking File I/O in Async

```python
# BAD - blocks event loop
async def read_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

# GOOD - use aiofiles or run_in_executor
async def read_config():
    async with aiofiles.open("config.yaml") as f:
        content = await f.read()
    return yaml.safe_load(content)
```

### 4. Missing Response Streaming

```python
# BAD - loads all in memory
@router.get("/export")
async def export_data():
    data = await get_all_records()  # Could be millions
    return {"records": data}

# GOOD - stream response
@router.get("/export")
async def export_data():
    async def generate():
        async for record in get_records_stream():
            yield record.model_dump_json() + "\n"
    return StreamingResponse(generate(), media_type="application/x-ndjson")
```
