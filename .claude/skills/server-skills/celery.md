# Celery Patterns for LlamaFarm Server

Best practices and code review checklist for Celery task patterns, error handling, and cross-service communication.

## Architecture Overview

The Server uses Celery to dispatch tasks to the RAG worker. Key concepts:

- **Signatures** - Build task calls without executing them
- **Groups** - Parallel execution of multiple tasks
- **Polling** - Server polls for task completion (no direct result access)
- **Filesystem Broker** - Default broker for local development

```
Server (FastAPI) --> Celery Signature --> Filesystem Broker --> RAG Worker
                                                            <-- Result Backend
```

---

## Ideal Patterns

### Building Task Signatures

```python
from celery import signature
from core.celery import app

def build_ingest_signature(
    project_dir: str,
    database_name: str,
    source_path: str,
) -> signature:
    """Build a Celery signature for the rag.ingest_file task."""
    return signature(
        "rag.ingest_file",
        args=[project_dir, database_name, source_path],
        app=app,
    )
```

### Async Polling for Task Completion

```python
import asyncio
from typing import Any

async def ingest_file_with_rag(
    project_dir: str,
    database_name: str,
    source_path: str,
    timeout: int = 300,
    poll_interval: float = 2.0,
) -> tuple[bool, dict[str, Any]]:
    """Dispatch task and poll for completion asynchronously."""
    sig = build_ingest_signature(project_dir, database_name, source_path)
    result = sig.apply_async()

    waited = 0.0
    while waited < timeout:
        try:
            status = result.status
            if status not in ("PENDING", "STARTED"):
                break
        except Exception:
            await asyncio.sleep(poll_interval)
            waited += poll_interval
            continue

        await asyncio.sleep(poll_interval)
        waited += poll_interval

    if result.status == "SUCCESS":
        return True, result.result
    elif result.status == "FAILURE":
        if hasattr(result, "traceback") and result.traceback:
            raise Exception(f"Task failed: {result.traceback}")
        raise Exception("Task failed")

    return False, {"error": f"Task timed out with status: {result.status}"}
```

### Synchronous Polling Helper

```python
import time

def _run_sync_task_with_polling(
    task_signature,
    timeout: float,
    poll_interval: float
) -> Any:
    """Helper for synchronous contexts to poll a Celery AsyncResult."""
    result = task_signature.apply_async()
    waited = 0.0

    while waited < timeout:
        try:
            status = result.status
            if status not in ("PENDING", "STARTED"):
                break
        except Exception:
            time.sleep(poll_interval)
            waited += poll_interval
            continue
        time.sleep(poll_interval)
        waited += poll_interval

    if result.status == "SUCCESS":
        return result.result
    elif result.status == "FAILURE":
        if hasattr(result, "traceback") and result.traceback:
            raise Exception(f"Task failed: {result.traceback}")
        raise Exception("Task failed")

    return None
```

### Group Task with Metadata Storage

```python
from celery import group
from core.celery import app

def dispatch_parallel_ingestion(
    namespace: str,
    project_id: str,
    file_paths: list[str],
) -> str:
    """Dispatch multiple files for parallel processing."""
    # Build signatures for each file
    signatures = [
        build_ingest_signature(project_dir, database, path)
        for path in file_paths
    ]

    # Create and dispatch group
    task_group = group(signatures)
    group_result = task_group.apply_async()

    # Store metadata for status tracking
    # (GroupResult.restore() doesn't work well with filesystem backend)
    task_id = str(uuid.uuid4())
    metadata = {
        "type": "group",
        "namespace": namespace,
        "project": project_id,
        "children": [child.id for child in group_result.results],
        "file_hashes": [hash_file(p) for p in file_paths],
    }
    app.backend.store_result(task_id, metadata, "PENDING")

    return task_id
```

---

## Checklist

### 1. Use Signatures for Cross-Service Calls

**Description:** Never import tasks directly from other services. Use signatures.

**Search Pattern:**
```bash
grep -rn "from rag\." server/ | grep -v "\.pyc"
```

**Pass Criteria:** No direct imports from `rag` package. Use `signature("rag.task_name")`.

**Severity:** High

**Recommendation:** Use signature pattern:
```python
task = signature("rag.ingest_file", args=[...], app=app)
result = task.apply_async()
```

---

### 2. Async Polling Uses asyncio.sleep

**Description:** Async functions must use `asyncio.sleep()`, not `time.sleep()`.

**Search Pattern:**
```bash
grep -rn "time\.sleep" server/ | grep "async def" -B10 | grep "time\.sleep"
```

**Pass Criteria:** No `time.sleep()` in async functions.

**Severity:** High

**Recommendation:** Replace with `await asyncio.sleep(interval)`.

---

### 3. Task Timeout Implemented

**Description:** All task polling must have a timeout to prevent infinite waits.

**Search Pattern:**
```bash
grep -rn "while.*result\." server/core/celery/ | grep -v "timeout\|waited"
```

**Pass Criteria:** Every polling loop has a timeout condition.

**Severity:** High

**Recommendation:** Implement timeout pattern:
```python
waited = 0.0
while waited < timeout:
    # ... poll logic
    waited += poll_interval
```

---

### 4. Task Failure Handling

**Description:** Handle task failures gracefully with proper error messages.

**Search Pattern:**
```bash
grep -rn "result\.status" server/ | grep -v "FAILURE"
```

**Pass Criteria:** Every status check handles FAILURE state.

**Severity:** High

**Recommendation:**
```python
if result.status == "FAILURE":
    if hasattr(result, "traceback") and result.traceback:
        raise Exception(f"Task failed: {result.traceback}")
    raise Exception("Task failed")
```

---

### 5. Group Metadata Stored for Tracking

**Description:** Parallel tasks need metadata storage for status tracking.

**Search Pattern:**
```bash
grep -rn "group(" server/ | grep -v "store_result"
```

**Pass Criteria:** Group tasks store metadata with children IDs.

**Severity:** Medium

**Recommendation:** Store group metadata:
```python
metadata = {
    "type": "group",
    "children": [child.id for child in group_result.results],
}
app.backend.store_result(task_id, metadata, "PENDING")
```

---

### 6. Celery App Configuration Complete

**Description:** Celery app must be configured with proper serialization and routing.

**Search Pattern:**
```bash
grep -rn "app.conf.update" server/core/celery/celery.py
```

**Pass Criteria:** Configuration includes serializer, timezone, and task routes.

**Severity:** Medium

**Recommendation:**
```python
app.conf.update({
    "task_serializer": "json",
    "accept_content": ["json"],
    "result_serializer": "json",
    "timezone": "UTC",
    "enable_utc": True,
    "task_routes": {
        "rag.*": {"queue": "rag"},
        "core.celery.tasks.*": {"queue": "server"},
    },
})
```

---

### 7. Result Backend Path Handles Windows

**Description:** File paths for filesystem backend must handle Windows paths.

**Search Pattern:**
```bash
grep -rn "file://" server/core/celery/
```

**Pass Criteria:** Windows paths use `file:///C:/...` format.

**Severity:** Medium

**Recommendation:**
```python
result_backend_path = path.replace("\\", "/")
if sys.platform == "win32" and path[1] == ":":
    result_backend_url = f"file:///{result_backend_path}"
else:
    result_backend_url = f"file://{result_backend_path}"
```

---

### 8. Prevent Celery Logger Override

**Description:** Prevent Celery from overriding structlog configuration.

**Search Pattern:**
```bash
grep -rn "setup_logging.connect" server/core/celery/
```

**Pass Criteria:** Empty `setup_celery_logging` signal handler exists.

**Severity:** Low

**Recommendation:**
```python
from celery import signals

@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    pass  # Prevent Celery from overriding root logger
```

---

### 9. Task Cancellation Implemented

**Description:** Long-running tasks should support cancellation.

**Search Pattern:**
```bash
grep -rn "revoke" server/api/routers/
```

**Pass Criteria:** Cancellation endpoint uses `app.control.revoke()`.

**Severity:** Medium

**Recommendation:**
```python
for child_id in child_task_ids:
    child_result = celery_app.AsyncResult(child_id)
    if child_result.state == "PENDING":
        celery_app.control.revoke(child_id, terminate=False)
```

---

### 10. Default Return Values for Timeouts

**Description:** Provide sensible defaults when tasks timeout.

**Search Pattern:**
```bash
grep -rn "_run_sync_task_with_polling" server/core/celery/rag_client.py -A2 | grep "or {"
```

**Pass Criteria:** Functions return default values on timeout, not None.

**Severity:** Low

**Recommendation:**
```python
return _run_sync_task_with_polling(task, timeout=30) or {
    "status": "degraded",
    "message": "Task timed out",
}
```

---

## Anti-Patterns to Avoid

### 1. Direct Task Import

```python
# BAD - creates import dependency
from rag.tasks import ingest_file
ingest_file.delay(...)

# GOOD - loose coupling
task = signature("rag.ingest_file", args=[...], app=app)
task.apply_async()
```

### 2. Blocking Sleep in Async

```python
# BAD - blocks event loop
async def poll_task():
    while True:
        time.sleep(1)  # Blocks!

# GOOD - non-blocking
async def poll_task():
    while True:
        await asyncio.sleep(1)
```

### 3. Missing Timeout

```python
# BAD - infinite loop possible
while result.status == "PENDING":
    time.sleep(1)

# GOOD - bounded wait
waited = 0.0
while waited < timeout and result.status == "PENDING":
    time.sleep(poll_interval)
    waited += poll_interval
```

### 4. Ignoring Task Traceback

```python
# BAD - loses error context
if result.status == "FAILURE":
    raise Exception("Task failed")

# GOOD - includes traceback
if result.status == "FAILURE":
    if hasattr(result, "traceback") and result.traceback:
        raise Exception(f"Task failed: {result.traceback}")
    raise Exception("Task failed")
```
