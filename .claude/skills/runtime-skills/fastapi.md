# FastAPI for Inference Serving Checklist

FastAPI patterns for ML inference endpoints, streaming, and lifecycle management.

---

## Category: Application Lifecycle

### Use Lifespan Context Manager

**What to check**: Manage startup/shutdown with lifespan

**Search pattern**:
```bash
rg "@asynccontextmanager" --type py runtimes/universal/server.py -A 15
```

**Pass criteria**:
- Lifespan function handles startup and shutdown
- Background tasks started at startup
- Resources cleaned up at shutdown
- Models unloaded gracefully

**Severity**: High

**Good pattern** (from server.py):
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cleanup_task

    # Startup
    logger.info("Starting Universal Runtime")
    _cleanup_task = asyncio.create_task(_cleanup_idle_models())

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down Universal Runtime")
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await _cleanup_task

    # Unload all models
    for cache_key, model in list(_models.items()):
        await model.unload()
    _models.clear()

app = FastAPI(lifespan=lifespan)
```

---

### Background Task Cleanup

**What to check**: Properly cancel and await background tasks

**Search pattern**:
```bash
rg "\.cancel\(\)|CancelledError" --type py runtimes/universal/
```

**Pass criteria**:
- Call task.cancel() on shutdown
- Use suppress(CancelledError) to await cleanly
- Background tasks have try/except CancelledError

**Severity**: High

**Good pattern**:
```python
# In shutdown:
_cleanup_task.cancel()
with suppress(asyncio.CancelledError):
    await _cleanup_task

# In background task:
async def _cleanup_idle_models():
    while True:
        try:
            await asyncio.sleep(CLEANUP_CHECK_INTERVAL)
            # ... cleanup logic
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            break
```

---

## Category: Endpoint Design

### Use Async Route Handlers

**What to check**: Define routes as async for I/O-bound operations

**Search pattern**:
```bash
rg "@app\.(get|post|put|delete)" --type py runtimes/universal/server.py -A 2
```

**Pass criteria**:
- All routes use `async def`
- Await async model operations
- No blocking calls in route handlers

**Severity**: High

**Good pattern**:
```python
@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    model = await load_encoder(request.model, task="embedding")
    embeddings = await model.embed(texts, normalize=True)
    return {"data": embeddings}
```

---

### Pydantic Request/Response Models

**What to check**: Use Pydantic models for request validation

**Search pattern**:
```bash
rg "class.*Request\(.*BaseModel\)" --type py runtimes/universal/
```

**Pass criteria**:
- Request models inherit from pydantic.BaseModel
- Use Literal types for enum-like fields
- Document fields with descriptions

**Severity**: Medium

**Good pattern**:
```python
class EmbeddingRequest(PydanticBaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = "float"
    user: str | None = None
    extra_body: dict | None = None
```

---

### HTTPException with Proper Status Codes

**What to check**: Use appropriate HTTP status codes

**Search pattern**:
```bash
rg "HTTPException\(status_code=" --type py runtimes/universal/
```

**Pass criteria**:
- 400 for bad request / validation errors
- 404 for resource not found
- 413 for payload too large
- 500 for internal errors
- Chain exceptions with `from e`

**Severity**: Medium

**Good pattern**:
```python
if len(content) > MAX_UPLOAD_SIZE:
    raise HTTPException(
        status_code=413,
        detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
    )

try:
    result = await model.process()
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e)) from e
```

---

## Category: Streaming Responses

### Server-Sent Events (SSE) for Streaming

**What to check**: Use StreamingResponse with proper media type

**Search pattern**:
```bash
rg "StreamingResponse|text/event-stream" --type py runtimes/universal/
```

**Pass criteria**:
- media_type="text/event-stream"
- Headers to disable buffering
- Proper SSE format: "data: {...}\n\n"

**Severity**: High

**Good pattern**:
```python
return StreamingResponse(
    generate_sse(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
    },
)
```

---

### SSE Format with JSON Data

**What to check**: Format SSE messages correctly

**Search pattern**:
```bash
rg "data:.*\\\\n\\\\n" --type py runtimes/universal/
```

**Pass criteria**:
- Format: `data: {json}\n\n`
- End with `data: [DONE]\n\n`
- Use .model_dump_json() for Pydantic models

**Severity**: High

**Good pattern**:
```python
async def generate_sse():
    # Initial chunk
    yield f"data: {initial_chunk.model_dump_json(exclude_none=True)}\n\n".encode()

    # Content chunks
    async for token in model.generate_stream(...):
        chunk = ChatCompletionChunk(...)
        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode()

    # Final chunk
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n".encode()
    yield b"data: [DONE]\n\n"
```

---

### Force Event Loop Yield for Real-Time Streaming

**What to check**: Use asyncio.sleep(0) to flush stream buffers

**Search pattern**:
```bash
rg "asyncio\.sleep\(0\)" --type py runtimes/universal/
```

**Pass criteria**:
- Call `await asyncio.sleep(0)` after each token yield
- Ensures immediate delivery of tokens
- Without this, tokens buffer and arrive in batches

**Severity**: High

**Good pattern**:
```python
async for token in token_stream:
    chunk = ChatCompletionChunk(...)
    yield f"data: {chunk.model_dump_json()}\n\n".encode()

    # CRITICAL: Force event loop to yield for immediate delivery
    await asyncio.sleep(0)
```

---

## Category: File Uploads

### Validate File Size Before Processing

**What to check**: Check file size to prevent memory exhaustion

**Search pattern**:
```bash
rg "MAX_UPLOAD_SIZE|\.read\(\)" --type py runtimes/universal/server.py
```

**Pass criteria**:
- Read file content first
- Check size against limit
- Return 413 if too large

**Severity**: High

**Good pattern**:
```python
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 100 * 1024 * 1024))

@app.post("/v1/files")
async def upload_file(file: UploadFile):
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
        )
```

---

### Use Form Parameters for File Metadata

**What to check**: Accept file options via Form parameters

**Search pattern**:
```bash
rg "Form\(default=" --type py runtimes/universal/
```

**Pass criteria**:
- Use Form() for multipart form data
- Provide sensible defaults
- Document parameters in docstring

**Severity**: Low

**Good pattern**:
```python
@app.post("/v1/files")
async def upload_file(
    file: UploadFile,
    convert_pdf: bool = Form(default=True),
    pdf_dpi: int = Form(default=150),
):
    """Upload a file for use with OCR or document extraction."""
```

---

## Category: OpenAI API Compatibility

### Follow OpenAI Response Schema

**What to check**: Match OpenAI API response structure

**Search pattern**:
```bash
rg "\"object\":|\"id\":|\"created\":" --type py runtimes/universal/
```

**Pass criteria**:
- Include "object" field (e.g., "list", "embedding")
- Include "id" with unique identifier
- Include "created" timestamp
- Include "model" field

**Severity**: High

**Good pattern**:
```python
return {
    "id": f"chatcmpl-{os.urandom(16).hex()}",
    "object": "chat.completion",
    "created": int(datetime.now().timestamp()),
    "model": request.model,
    "choices": [...],
    "usage": {...},
}
```

---

### Support OpenAI Types from SDK

**What to check**: Use openai SDK types for responses

**Search pattern**:
```bash
rg "from openai\.types" --type py runtimes/universal/
```

**Pass criteria**:
- Import types from openai SDK
- Use for streaming chunk types
- Ensures compatibility with OpenAI client

**Severity**: Medium

**Good pattern**:
```python
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    Choice as ChoiceChunk,
)

chunk = ChatCompletionChunk(
    id=completion_id,
    object="chat.completion.chunk",
    created=created_time,
    model=model_name,
    choices=[ChoiceChunk(index=0, delta=ChoiceDelta(content=token), finish_reason=None)],
)
```

---

## Category: Router Organization

### Use APIRouter for Endpoint Groups

**What to check**: Organize related endpoints in routers

**Search pattern**:
```bash
rg "APIRouter|include_router" --type py runtimes/universal/
```

**Pass criteria**:
- Related endpoints in separate router module
- Router included in main app
- Prefix for route grouping if needed

**Severity**: Low

**Good pattern**:
```python
# routers/chat_completions/router.py
from fastapi import APIRouter
router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(...):
    ...

# server.py
from routers.chat_completions import router as chat_completions_router
app.include_router(chat_completions_router)
```

---

## Category: Health and Monitoring

### Health Check Endpoint

**What to check**: Provide health check with device info

**Search pattern**:
```bash
rg "@app\.get\(\"/health" --type py runtimes/universal/
```

**Pass criteria**:
- Return device information
- List loaded models
- Include timestamp and PID

**Severity**: Medium

**Good pattern**:
```python
@app.get("/health")
async def health_check():
    device_info = get_device_info()
    return {
        "status": "healthy",
        "device": device_info,
        "loaded_models": list(_models.keys()),
        "timestamp": datetime.utcnow().isoformat(),
        "pid": os.getpid(),
    }
```

---

### List Loaded Models Endpoint

**What to check**: Provide endpoint to list loaded models

**Search pattern**:
```bash
rg "@app\.get\(\"/v1/models" --type py runtimes/universal/
```

**Pass criteria**:
- Return model IDs and types
- Follow OpenAI list format
- Include metadata (created, owner)

**Severity**: Low

**Good pattern**:
```python
@app.get("/v1/models")
async def list_models():
    models_list = [
        {
            "id": model_id,
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": "transformers-runtime",
            "type": model.model_type,
        }
        for model_id, model in _models.items()
    ]
    return {"object": "list", "data": models_list}
```
