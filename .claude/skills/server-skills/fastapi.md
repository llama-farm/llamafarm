# FastAPI Patterns for LlamaFarm Server

Best practices and code review checklist for FastAPI routes, middleware, and dependencies.

## Route Definition Patterns

### Ideal Route Structure

```python
from fastapi import APIRouter, HTTPException, Response
from fastapi import Path as FastAPIPath
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
)

class GetProjectResponse(BaseModel):
    project: Project = Field(..., description="The project details")

@router.get(
    "/{namespace}/{project_id}",
    operation_id="project_get",
    summary="Get a project",
    tags=["projects", "mcp"],
    response_model=GetProjectResponse,
    responses={
        200: {"model": GetProjectResponse},
        404: {"model": ErrorResponse},
    },
)
async def get_project(
    namespace: str = FastAPIPath(..., description="The namespace"),
    project_id: str = FastAPIPath(..., description="The project ID"),
) -> GetProjectResponse:
    """Get a project by namespace and ID."""
    project = ProjectService.get_project(namespace, project_id)
    return GetProjectResponse(project=project)
```

### Key Elements

1. **Router with prefix and tags** - Groups related endpoints
2. **Operation ID** - Unique identifier for OpenAPI/code generation
3. **Response models** - Pydantic models for type safety
4. **Path parameter descriptions** - Use FastAPIPath for documentation
5. **Docstring** - Appears in OpenAPI documentation

---

## Checklist

### 1. Route Has Response Model

**Description:** All routes should define a response model for type safety and documentation.

**Search Pattern:**
```bash
grep -rn "@router\.\(get\|post\|put\|delete\|patch\)" server/api/routers/ | grep -v "response_model"
```

**Pass Criteria:** Every route decorator includes `response_model=` or is documented in `responses={}`.

**Severity:** Medium

**Recommendation:** Add `response_model=ResponseClass` to route decorators or define response in `responses={}` dict.

---

### 2. Route Has Operation ID

**Description:** Routes exposed via MCP or used for code generation need unique operation_id.

**Search Pattern:**
```bash
grep -rn "@router\.\(get\|post\|put\|delete\)" server/api/routers/ -A5 | grep -E "tags=.*mcp" | grep -v "operation_id"
```

**Pass Criteria:** All routes with `tags=["mcp"]` have an `operation_id`.

**Severity:** Medium

**Recommendation:** Add `operation_id="resource_action"` (e.g., `project_get`, `dataset_create`).

---

### 3. Async Functions for I/O Operations

**Description:** Routes performing I/O (database, file, network) should be async.

**Search Pattern:**
```bash
grep -rn "^def " server/api/routers/ | grep -v "__init__" | grep -v "test_"
```

**Pass Criteria:** Route handlers performing I/O use `async def`.

**Severity:** High

**Recommendation:** Convert to `async def` and use `await` for I/O operations.

---

### 4. HTTPException with Proper Status Codes

**Description:** Use HTTPException with appropriate status codes, not bare exceptions.

**Search Pattern:**
```bash
grep -rn "raise Exception" server/api/routers/
```

**Pass Criteria:** No bare `raise Exception` in route handlers. Use HTTPException or custom exceptions.

**Severity:** High

**Recommendation:** Replace with `raise HTTPException(status_code=4xx/5xx, detail="message")`.

---

### 5. Path Traversal Prevention

**Description:** User-provided paths must be validated to prevent directory traversal.

**Search Pattern:**
```bash
grep -rn "os.path.join.*namespace\|os.path.join.*project" server/ | grep -v "normpath\|resolve"
```

**Pass Criteria:** All path constructions validate against base path after normalization.

**Severity:** Critical

**Recommendation:** Use pattern:
```python
norm_path = os.path.normpath(raw_path)
if not norm_path.startswith(os.path.abspath(base_path) + os.sep):
    raise NamespaceNotFoundError("Invalid: path traversal detected")
```

---

### 6. Exception Handlers Registered

**Description:** Custom exceptions should have registered handlers for consistent API responses.

**Search Pattern:**
```bash
grep -rn "class.*Error.*Exception" server/api/errors.py
```

**Pass Criteria:** Each custom exception has a corresponding handler in `register_exception_handlers()`.

**Severity:** Medium

**Recommendation:** Add handler in `api/errors.py`:
```python
async def _handle_my_error(request: Request, exc: Exception) -> Response:
    payload = ErrorResponse(error="MyError", message=str(exc))
    return JSONResponse(status_code=4xx, content=payload.model_dump())

app.add_exception_handler(MyError, _handle_my_error)
```

---

### 7. Response Headers Set Correctly

**Description:** Custom headers (e.g., session IDs) should be set on Response object.

**Search Pattern:**
```bash
grep -rn "response.headers\[" server/api/routers/
```

**Pass Criteria:** Headers set via `response.headers["X-Header"] = value`.

**Severity:** Low

**Recommendation:** Inject `response: Response` parameter and set headers:
```python
async def endpoint(response: Response):
    response.headers["X-Session-ID"] = session_id
```

---

### 8. Middleware Order Correct

**Description:** Middleware is applied in reverse order - last added runs first on request.

**Search Pattern:**
```bash
grep -rn "add_middleware" server/api/main.py
```

**Pass Criteria:** Error handling middleware added after logging middleware.

**Severity:** Medium

**Recommendation:** Order in `api/main.py`:
```python
app.add_middleware(ErrorHandlerMiddleware)  # Added last, runs first
app.add_middleware(StructLogMiddleware)
app.add_middleware(CorrelationIdMiddleware)  # Added first, runs last
```

---

### 9. Lifespan Context Manager Used

**Description:** Use lifespan for startup/shutdown instead of deprecated events.

**Search Pattern:**
```bash
grep -rn "@app.on_event" server/
```

**Pass Criteria:** No `@app.on_event("startup")` or `@app.on_event("shutdown")`.

**Severity:** Medium

**Recommendation:** Use lifespan context manager:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting API")
    yield
    # Shutdown
    await cleanup_resources()

app = FastAPI(lifespan=lifespan)
```

---

### 10. Streaming Responses Handled Correctly

**Description:** SSE/streaming responses should use StreamingResponse with proper content type.

**Search Pattern:**
```bash
grep -rn "StreamingResponse\|EventSourceResponse" server/api/routers/
```

**Pass Criteria:** Streaming endpoints return proper response type with correct media_type.

**Severity:** Medium

**Recommendation:**
```python
from starlette.responses import StreamingResponse

return StreamingResponse(
    generator(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache"}
)
```

---

## Anti-Patterns to Avoid

### 1. Blocking Calls in Async Routes

```python
# BAD - blocks event loop
@router.get("/data")
async def get_data():
    time.sleep(5)  # Blocks!
    data = requests.get(url)  # Blocks!

# GOOD - non-blocking
@router.get("/data")
async def get_data():
    await asyncio.sleep(5)
    async with httpx.AsyncClient() as client:
        data = await client.get(url)
```

### 2. Missing Error Context

```python
# BAD - loses original error
except Exception:
    raise HTTPException(status_code=500)

# GOOD - chains exceptions
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e)) from e
```

### 3. Hardcoded Status Codes Without Model

```python
# BAD - no type safety
return {"status": "ok"}

# GOOD - typed response
class StatusResponse(BaseModel):
    status: str

@router.get("/", response_model=StatusResponse)
async def health():
    return StatusResponse(status="ok")
```
