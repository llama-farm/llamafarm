# Async/Await Best Practices

Async patterns for LlamaFarm's FastAPI and Celery-based services.

---

## Category: Async Function Basics

### No Blocking Calls in Async Functions

**What to check**: Async functions must not call blocking I/O

**Bad pattern**:
```python
async def load_model(model_id: str):
    with open(file_path, "rb") as f:  # BLOCKING
        data = f.read()
    time.sleep(1)  # BLOCKING
    requests.get(url)  # BLOCKING
```

**Good pattern**:
```python
async def load_model(model_id: str):
    async with aiofiles.open(file_path, "rb") as f:
        data = await f.read()
    await asyncio.sleep(1)
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
```

**Search pattern**:
```bash
rg "async def" -A 20 --type py | rg "time\.sleep|requests\.(get|post)|open\("
```

**Pass criteria**: No blocking calls inside async functions

**Severity**: Critical

**Recommendation**: Use `asyncio.to_thread()` if blocking is unavoidable

---

### Use asyncio.to_thread for CPU-Bound Work

**What to check**: Offload CPU-bound operations to thread pool

**Good pattern** (from runtimes/universal):
```python
async def load(self) -> None:
    # Model loading is CPU-bound, run in thread pool
    self.model = await asyncio.to_thread(
        AutoModelForCausalLM.from_pretrained,
        self.model_id,
        trust_remote_code=True
    )
```

**Search pattern**:
```bash
rg "asyncio\.to_thread|run_in_executor" --type py
```

**Severity**: High

---

### Async Context Managers

**What to check**: Use async context managers for async resources

**Good pattern**:
```python
async with httpx.AsyncClient() as client:
    response = await client.get(url)

async with aiofiles.open(path) as f:
    content = await f.read()
```

**Search pattern**:
```bash
rg "async with" --type py
```

**Severity**: Medium

---

## Category: Concurrency Control

### asyncio.Lock for Shared State

**What to check**: Use locks to prevent race conditions

**Good pattern** (from runtimes/universal/server.py):
```python
_model_load_lock = asyncio.Lock()

async def load_language(model_id: str):
    if cache_key not in _models:
        async with _model_load_lock:
            # Double-check after acquiring lock
            if cache_key not in _models:
                model = LanguageModel(model_id, device)
                await model.load()
                _models[cache_key] = model
    return _models.get(cache_key)
```

**Pass criteria**: Double-checked locking pattern for lazy initialization

**Severity**: High

---

### Avoid Global Lock Contention

**What to check**: Use fine-grained locks or lock-free structures

**Bad pattern**:
```python
global_lock = asyncio.Lock()

async def any_operation():
    async with global_lock:  # Bottleneck for all operations
        ...
```

**Good pattern**:
```python
# Per-resource locks
_resource_locks: dict[str, asyncio.Lock] = {}

async def get_resource_lock(resource_id: str) -> asyncio.Lock:
    if resource_id not in _resource_locks:
        _resource_locks[resource_id] = asyncio.Lock()
    return _resource_locks[resource_id]
```

**Severity**: Medium

---

## Category: Task Management

### Background Tasks with asyncio.create_task

**What to check**: Long-running background work should use create_task

**Good pattern** (from runtimes/universal/server.py):
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_cleanup_idle_models())
    yield
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await _cleanup_task
```

**Severity**: Medium

---

### Handle Task Cancellation

**What to check**: Background tasks must handle CancelledError

**Good pattern**:
```python
async def _cleanup_idle_models() -> None:
    while True:
        try:
            await asyncio.sleep(CLEANUP_CHECK_INTERVAL)
            # ... cleanup logic
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            break  # Exit cleanly
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
            # Continue running despite errors
```

**Search pattern**:
```bash
rg "asyncio\.CancelledError|CancelledError" --type py
```

**Pass criteria**: CancelledError caught and handled gracefully

**Severity**: High

---

### Use suppress for Expected Cancellation

**What to check**: Use contextlib.suppress for expected cancellation

**Good pattern**:
```python
from contextlib import suppress

# When stopping a task
task.cancel()
with suppress(asyncio.CancelledError):
    await task
```

**Severity**: Low

---

## Category: Async Generators

### AsyncGenerator Return Type

**What to check**: Async generators should have proper type hints

**Good pattern** (from server/agents/base/agent.py):
```python
from collections.abc import AsyncGenerator

async def run_async_stream(
    self,
    messages: list[LFChatCompletionMessageParam] | None = None,
) -> AsyncGenerator[LFChatCompletionChunk]:
    async for chunk in self._client.stream_chat(messages=messages):
        yield chunk
```

**Search pattern**:
```bash
rg "AsyncGenerator\[" --type py
```

**Severity**: Medium

---

### Async For Loops

**What to check**: Use `async for` with async iterables

**Good pattern**:
```python
async for chunk in self._client.stream_chat(messages=messages):
    yield chunk
```

**Bad pattern**:
```python
# Collecting all items defeats streaming
chunks = [chunk async for chunk in stream]  # Avoid if streaming is intended
```

**Severity**: Medium

---

## Category: Concurrent Execution

### asyncio.gather for Parallel Operations

**What to check**: Use gather for independent async operations

**Good pattern**:
```python
results = await asyncio.gather(
    fetch_data(url1),
    fetch_data(url2),
    fetch_data(url3),
    return_exceptions=True  # Don't fail fast on single error
)

# Handle results
for result in results:
    if isinstance(result, Exception):
        logger.error(f"Request failed: {result}")
    else:
        process(result)
```

**Search pattern**:
```bash
rg "asyncio\.gather" --type py
```

**Severity**: Medium

---

### TaskGroup for Structured Concurrency (Python 3.11+)

**What to check**: Consider TaskGroup for better error handling

**Good pattern**:
```python
async def process_batch(items: list[str]):
    async with asyncio.TaskGroup() as tg:
        for item in items:
            tg.create_task(process_item(item))
    # All tasks completed successfully if we reach here
```

**Note**: TaskGroup raises ExceptionGroup on failure

**Severity**: Low

---

## Category: FastAPI Specific

### Async Route Handlers

**What to check**: Use async def for I/O-bound route handlers

**Good pattern**:
```python
@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    model = await load_encoder(request.model)
    embeddings = await model.embed(texts)
    return {"data": embeddings}
```

**Severity**: Medium

---

### Lifespan Context Manager

**What to check**: Use lifespan for startup/shutdown logic

**Good pattern** (from runtimes/universal/server.py):
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting server")
    _cleanup_task = asyncio.create_task(_cleanup_idle_models())

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down")
    _cleanup_task.cancel()
    with suppress(asyncio.CancelledError):
        await _cleanup_task

app = FastAPI(lifespan=lifespan)
```

**Search pattern**:
```bash
rg "@asynccontextmanager" --type py -A 10 | rg "lifespan"
```

**Severity**: Medium

---

### Dependency Injection with Async

**What to check**: FastAPI dependencies can be async

**Good pattern**:
```python
async def get_model(model_id: str = Query(...)) -> BaseModel:
    return await load_model(model_id)

@app.post("/predict")
async def predict(model: BaseModel = Depends(get_model)):
    return await model.predict()
```

**Severity**: Low

---

## Category: Celery Integration

### Async in Celery Tasks

**What to check**: Celery tasks are synchronous by default

**Pattern** (from rag/tasks):
```python
@app.task(bind=True, base=IngestTask)
def ingest_file_with_rag_task(self, project_dir: str, ...):
    # Celery tasks run in worker process, not async
    # Use synchronous code here
    handler = IngestHandler(config_path=str(config_path))
    result = handler.ingest_file(file_data=file_data)
    return result
```

**Note**: For async operations in Celery, use `asyncio.run()` inside the task

**Severity**: Medium

---

### Calling Celery from Async Code

**What to check**: Use `.delay()` or `.apply_async()` (they don't block)

**Good pattern**:
```python
async def start_ingestion(files: list[str]):
    # apply_async is non-blocking
    result = ingest_file_task.apply_async(args=[file_path])
    return result.id
```

**Severity**: Medium

---

## Category: Error Handling

### Async Exception Handling

**What to check**: Handle exceptions in async code appropriately

**Good pattern**:
```python
async def safe_operation():
    try:
        result = await risky_operation()
    except SomeAsyncError as e:
        logger.error("Operation failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e)) from e
```

**Severity**: High

---

### Timeout for Async Operations

**What to check**: Use timeouts for external calls

**Good pattern**:
```python
try:
    result = await asyncio.wait_for(
        external_api_call(),
        timeout=30.0
    )
except asyncio.TimeoutError:
    logger.error("External API timed out")
    raise HTTPException(status_code=504, detail="Gateway timeout")
```

**Search pattern**:
```bash
rg "asyncio\.wait_for|asyncio\.timeout" --type py
```

**Severity**: High
