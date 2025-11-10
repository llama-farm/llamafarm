# Windows Endpoint Celery Result Access Crash Fix

## Problem
The Windows e2e test was failing with a 500 Internal Server Error during dataset processing, with **no server logs** after the request was received. The server logs would stop completely, suggesting a crash before any exception handling could run.

## Root Cause
While we fixed Celery task code in `task_process_dataset.py`, we missed that the **FastAPI endpoint itself** (`datasets.py`) also directly accesses `task.status` and `task.result` when polling for task completion.

On Windows with Celery's filesystem backend, accessing these properties can throw exceptions due to:
- File locking issues
- Path encoding problems
- Permission errors
- Race conditions with file writes

**Timeline showing the crash:**
```
19:25:14 - Last server log: "POST /v1/projects/default HTTP/1.1" 200
19:26:30 - Dataset processing request sent
19:26:29 - 500 error returned (no server log for this request!)
```

The request ID from the 500 error (`f47efbc78b9e4c5a9a40e4cf0bdaf790`) **never appears in server logs**, proving the crash happened before any middleware could execute.

## Solution
Added comprehensive error handling around **all** `task.status` and `task.result` accesses in the FastAPI endpoint polling loop:

1. **Polling loop** - wrap status checks
2. **Final status retrieval** - wrap with HTTPException on failure  
3. **SUCCESS branch** - wrap result access
4. **FAILURE branch** - wrap failure details access

### Key Changes in `server/api/routers/datasets/datasets.py`:

**Before (line ~493):**
```python
while waited < timeout:
    if task.status not in ("PENDING", "STARTED"):  # ⚠️ Can crash on Windows!
        break
    await asyncio.sleep(poll_interval)
    waited += poll_interval

if task.status == "SUCCESS":  # ⚠️ Can crash here too!
    result = task.result  # ⚠️ And here!
```

**After:**
```python
while waited < timeout:
    try:
        status = task.status
        if status not in ("PENDING", "STARTED"):
            break
    except Exception as e:
        logger.error(
            f"Error checking task status for file {file_hash}: {e}",
            exc_info=True
        )
        await asyncio.sleep(poll_interval)
        waited += poll_interval
        continue
    
    await asyncio.sleep(poll_interval)
    waited += poll_interval

# Get final status with error handling
try:
    final_status = task.status
except Exception as e:
    logger.error(
        f"Error getting final task status for file {file_hash}: {e}",
        exc_info=True
    )
    raise HTTPException(
        status_code=500,
        detail=f"Failed to get task status for file {file_hash}: {str(e)}"
    )

if final_status == "SUCCESS":
    try:
        result = task.result
        ok = result["success"]
        file_details = result["details"]
    except Exception as e:
        logger.error(...)
        raise HTTPException(...)
```

## Why This Was Hard to Find
1. **The crash was silent** - no logs, no stack traces, because the exception happened in the FastAPI request handler before middleware could catch it
2. **We fixed the wrong place first** - we protected the Celery worker code, but the endpoint code also accesses the same problematic properties
3. **Windows-specific** - works fine on Linux/macOS where Celery uses different backends
4. **Merge commits in CI** - made it hard to verify which code version was actually running

## Testing
After this fix, the endpoint will:
1. **Log errors** when file access fails
2. **Return proper 500 errors** with details instead of crashing silently
3. **Continue processing** if status checks fail temporarily
4. **Raise HTTPExceptions** that middleware can catch and log

## Related Files
- `server/api/routers/datasets/datasets.py` - Fixed FastAPI endpoint polling
- `server/core/celery/tasks/task_process_dataset.py` - Previously fixed Celery task code
- `server/api/middleware/structlog.py` - Middleware that now properly logs exceptions
- `server/api/errors.py` - Global exception handler that now logs all uncaught errors

## Prevention
When working with Celery AsyncResult on Windows filesystem backend:
- **Always** wrap `result.status` access in try/except
- **Always** wrap `result.result` access in try/except  
- **Always** log errors with `exc_info=True` for debugging
- **Never** assume file operations will succeed on Windows

