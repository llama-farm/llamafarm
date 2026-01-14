# Error Handling Checklist

Exception handling and logging patterns for LlamaFarm Python components.

---

## Category: Custom Exceptions

### Domain-Specific Exception Classes

**What to check**: Define custom exceptions for domain errors

**Good pattern** (from config/helpers/loader.py):
```python
class ConfigError(Exception):
    """Raised when there's an error loading or validating configuration."""
    pass

class ValidationError(ConfigError):
    """Raised when configuration validation fails."""
    def __init__(self, message: str, path: str = ""):
        self.path = path
        super().__init__(f"{message}" + (f" at path {path}" if path else ""))
```

**Good pattern** (from rag/utils/embedding_safety.py):
```python
class EmbedderUnavailableError(Exception):
    """Raised when an embedder cannot complete requests."""
    pass

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker prevents requests."""
    def __init__(self, message: str, failures: int = 0, reset_time: float = 0):
        self.failures = failures
        self.reset_time = reset_time
        super().__init__(message)
```

**Search pattern**:
```bash
rg "class \w+Error\(|class \w+Exception\(" --type py
```

**Pass criteria**: Domain errors have specific exception classes with context

**Severity**: Medium

---

### Exception Hierarchy

**What to check**: Exceptions follow a clear hierarchy

**Good pattern**:
```python
class LlamaFarmError(Exception):
    """Base exception for all LlamaFarm errors."""
    pass

class ConfigError(LlamaFarmError):
    """Configuration-related errors."""
    pass

class ProcessingError(LlamaFarmError):
    """Processing pipeline errors."""
    pass

class EmbeddingError(ProcessingError):
    """Embedding generation errors."""
    pass
```

**Pass criteria**: Custom exceptions inherit from a base application exception

**Severity**: Low

---

## Category: Exception Handling

### Catch Specific Exceptions

**What to check**: Catch specific exceptions, not bare Exception

**Bad pattern**:
```python
try:
    process()
except Exception:  # Too broad
    pass

try:
    process()
except:  # Even worse - catches SystemExit, KeyboardInterrupt
    pass
```

**Good pattern** (from server/services/dataset_service.py):
```python
try:
    with open(file_path) as f:
        metadata = MetadataFileContent.model_validate_json(f.read())
        files.append(metadata)
except OSError as e:
    logger.warning(
        "Failed to read metadata file",
        namespace=namespace,
        file=file,
        error=str(e),
    )
except ValueError as e:
    # Pydantic validation errors
    logger.warning(
        "Failed to parse metadata file",
        file=file,
        error=str(e),
    )
```

**Search pattern**:
```bash
rg "except Exception:|except:" --type py
```

**Pass criteria**: Specific exceptions caught; broad catches only at top level

**Severity**: High

---

### Exception Chaining with `from`

**What to check**: Chain exceptions to preserve context

**Good pattern** (from config/helpers/loader.py):
```python
try:
    jsonschema.validate(config, schema)
except jsonschema.ValidationError as e:
    path_str = ".".join(str(p) for p in e.path)
    raise ConfigError(
        f"Configuration validation error: {e.message}"
        + (f" at path {path_str}" if path_str else "")
    ) from e
except Exception as e:
    raise ConfigError(f"Error during validation: {e}") from e
```

**Search pattern**:
```bash
rg "raise \w+ from " --type py
```

**Pass criteria**: Use `from e` when re-raising to preserve traceback

**Severity**: Medium

---

### No Silent Failures

**What to check**: Exceptions must not be silently swallowed

**Bad pattern**:
```python
try:
    risky_operation()
except Exception:
    pass  # Silent failure - data loss, debugging nightmare
```

**Good pattern**:
```python
try:
    risky_operation()
except SpecificError as e:
    logger.warning(
        "Operation failed, using fallback",
        extra={"error": str(e), "fallback": fallback_value}
    )
    return fallback_value
```

**Search pattern**:
```bash
rg "except.*:\s*$" -A 1 --type py | rg "pass$"
```

**Pass criteria**: No `except: pass` without logging or explicit justification

**Severity**: High

---

### Use contextlib.suppress for Expected Exceptions

**What to check**: Use suppress() when exception is truly expected and ignorable

**Good pattern** (from server/services/dataset_service.py):
```python
from contextlib import suppress

with suppress(FileNotFoundError):
    existing_file = DataService.get_data_file_metadata_by_hash(
        namespace, project, dataset, file_hash
    )
```

**When to use**:
- File may not exist and that's OK
- Optional cleanup that shouldn't fail the operation
- Checking for presence of optional features

**Severity**: Low

---

## Category: Structured Logging

### Log with Extra Dict

**What to check**: Use structured logging with extra dict, not f-strings

**Good pattern** (from server/services/dataset_service.py):
```python
logger.info(
    "Deleted chunks from vector store",
    namespace=namespace,
    project=project,
    dataset=dataset,
    file_hash=file_hash[:16] + "...",
    deleted_chunks=result.get("deleted_count", 0),
)

logger.warning(
    "Dataset metadata directory not found",
    namespace=namespace,
    project=project,
    dataset=dataset,
    path=dataset_meta_dir,
)
```

**Bad pattern**:
```python
logger.info(f"Deleted {count} chunks from {dataset}")  # Hard to parse/search
```

**Search pattern**:
```bash
rg 'logger\.(info|error|warning|debug)\(f"' --type py
```

**Pass criteria**: Structured data in extra dict, not embedded in message

**Severity**: Medium

---

### Log Before Raising

**What to check**: Log errors with context before raising

**Good pattern**:
```python
try:
    result = process(data)
except ProcessingError as e:
    logger.error(
        "Processing failed",
        extra={
            "error": str(e),
            "error_type": type(e).__name__,
            "data_id": data.id,
            "operation": "process",
        },
        exc_info=True,
    )
    raise
```

**Severity**: Medium

---

### Use exc_info for Stack Traces

**What to check**: Include stack traces for unexpected errors

**Good pattern**:
```python
try:
    complex_operation()
except Exception as e:
    logger.error(
        "Unexpected error in complex operation",
        extra={"error": str(e)},
        exc_info=True,  # Includes full traceback
    )
    raise
```

**Severity**: Medium

---

### Error Context in Logs

**What to check**: Include relevant context for debugging

**Good pattern**:
```python
logger.error(
    "Document processing failed",
    extra={
        "document_id": doc.id,
        "document_source": doc.source,
        "error_type": type(e).__name__,
        "error_message": str(e),
        "stage": "parsing",
        "retry_count": retry_count,
    }
)
```

**Essential context**:
- Entity IDs (document_id, user_id, task_id)
- Operation being performed
- Error type and message
- Current state (stage, retry count)

**Severity**: Medium

---

## Category: FastAPI Error Handling

### HTTPException for API Errors

**What to check**: Use HTTPException for HTTP error responses

**Good pattern**:
```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def get_item(item_id: str):
    item = await fetch_item(item_id)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail=f"Item {item_id} not found"
        )
    return item

@app.post("/process")
async def process_data(request: ProcessRequest):
    try:
        result = await process(request.data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=500, detail="Processing failed")
    return result
```

**Severity**: High

---

### Custom Exception Handlers

**What to check**: Register handlers for custom exceptions

**Good pattern**:
```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(ConfigError)
async def config_error_handler(request: Request, exc: ConfigError):
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
            "error_code": "CONFIG_ERROR",
            "path": getattr(exc, "path", None),
        }
    )

@app.exception_handler(EmbedderUnavailableError)
async def embedder_error_handler(request: Request, exc: EmbedderUnavailableError):
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Embedding service temporarily unavailable",
            "error_code": "EMBEDDER_UNAVAILABLE",
            "retry_after": 60,
        }
    )
```

**Severity**: Medium

---

### Error Response Models

**What to check**: Define consistent error response schemas

**Good pattern**:
```python
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
    path: str | None = None

class ValidationErrorResponse(BaseModel):
    detail: list[dict[str, Any]]
    error_code: str = "VALIDATION_ERROR"
```

**Severity**: Low

---

## Category: Resource Cleanup

### Finally for Cleanup

**What to check**: Use finally for guaranteed cleanup

**Good pattern**:
```python
resource = acquire_resource()
try:
    result = process(resource)
    return result
finally:
    resource.release()
```

**Severity**: High

---

### Context Managers for Resources

**What to check**: Prefer context managers over try/finally

**Good pattern** (from config/helpers/loader.py):
```python
with open(file_path, encoding="utf-8") as f:
    return yaml_instance.load(f)

async with httpx.AsyncClient() as client:
    response = await client.get(url)
```

**Pass criteria**: Use context managers for file handles, connections, locks

**Severity**: Medium

---

### Async Context Managers

**What to check**: Use async context managers for async cleanup

**Good pattern**:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_model(model_id: str):
    model = await load_model(model_id)
    try:
        yield model
    finally:
        await model.unload()

# Usage
async with managed_model("gpt-2") as model:
    result = await model.generate(prompt)
```

**Severity**: Medium

---

## Category: Resilience Patterns

### Circuit Breaker

**What to check**: Use circuit breaker for external dependencies

**Good pattern** (from rag/core/base.py):
```python
class Embedder(Component):
    DEFAULT_FAILURE_THRESHOLD = 5
    DEFAULT_RESET_TIMEOUT = 60.0

    def __init__(self, ...):
        circuit_config = (config or {}).get("circuit_breaker", {})
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_config.get(
                "failure_threshold", self.DEFAULT_FAILURE_THRESHOLD
            ),
            reset_timeout=circuit_config.get(
                "reset_timeout", self.DEFAULT_RESET_TIMEOUT
            ),
        )

    def check_circuit_breaker(self) -> None:
        if not self._circuit_breaker.can_execute():
            state_info = self._circuit_breaker.get_state_info()
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open for {self.name}. "
                f"Too many consecutive failures."
            )
```

**Severity**: Medium (for external service calls)

---

### Retry with Backoff

**What to check**: Use exponential backoff for transient failures

**Good pattern**:
```python
import backoff

@backoff.on_exception(
    backoff.expo,
    (ConnectionError, TimeoutError),
    max_tries=3,
    max_time=30,
)
async def call_external_api():
    async with httpx.AsyncClient() as client:
        return await client.get(url)
```

**Or manual implementation**:
```python
async def with_retry(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return await func()
        except TransientError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s")
            await asyncio.sleep(delay)
```

**Severity**: Medium

---

### Graceful Degradation

**What to check**: Provide fallbacks when services fail

**Good pattern**:
```python
async def get_embeddings(texts: list[str]) -> list[list[float]]:
    try:
        return await primary_embedder.embed(texts)
    except EmbedderUnavailableError:
        logger.warning("Primary embedder unavailable, using fallback")
        try:
            return await fallback_embedder.embed(texts)
        except EmbedderUnavailableError:
            logger.error("All embedders unavailable")
            raise
```

**Severity**: Medium

---

## Category: Validation Errors

### Early Validation

**What to check**: Validate input at API boundaries

**Good pattern**:
```python
def process_file(path: str) -> ProcessingResult:
    # Validate early
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"Not a file: {path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Empty file: {path}")

    # Process validated input
    return _do_process(file_path)
```

**Severity**: Medium

---

### Pydantic Validation

**What to check**: Use Pydantic for input validation

**Good pattern**:
```python
from pydantic import BaseModel, Field, field_validator

class ProcessRequest(BaseModel):
    content: str = Field(min_length=1, max_length=100000)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Content cannot be blank")
        return v
```

**Severity**: High
