# Windows Celery File URL Fix

## Problem

On Windows, the Celery worker was failing to start with the following error:

```
ValueError: Port could not be cast to integer value as '\\Users\\runneradmin\\.llamafarm'
```

This occurred when Celery tried to parse the result backend URL during worker startup.

## Root Cause

The issue was in how `file://` URLs were being constructed for Celery's result backend configuration in both:
- `rag/celery_app.py`
- `server/core/celery/celery.py`

The code was directly concatenating Windows paths (containing backslashes) into `file://` URLs:

```python
result_backend=f"file://{lf_data_dir}/broker/results"
```

On Windows, `lf_data_dir` would be something like `C:\Users\runneradmin\.llamafarm`, resulting in:

```
file://C:\Users\runneradmin\.llamafarm/broker/results
```

This is not a valid file:// URL. When Kombu (Celery's messaging library) tried to parse this URL, it incorrectly interpreted the `:` after `C` as a port separator, and tried to parse `\Users\runneradmin\.llamafarm` as a port number, leading to the error.

## Solution

The fix converts Windows paths to proper file:// URLs by:

1. **Replacing backslashes with forward slashes**: Windows paths are normalized to use forward slashes
2. **Using proper file:// URL format**: For Windows absolute paths (e.g., `C:/Users/...`), we use `file:///` (three slashes) format

### Implementation

Both `rag/celery_app.py` and `server/core/celery/celery.py` now use:

```python
# Convert Windows backslashes to forward slashes for file:// URL
result_backend_path = f"{lf_data_dir}/broker/results".replace("\\", "/")
# Ensure proper file:// URL format (file:/// for absolute paths on Windows)
if sys.platform == "win32" and len(result_backend_path) > 1 and result_backend_path[1] == ":":
    # Windows absolute path (e.g., C:/Users/...) needs file:///C:/...
    result_backend_url = f"file:///{result_backend_path}"
else:
    # Unix absolute path needs file:///path or relative path needs file://path
    result_backend_url = f"file://{result_backend_path}"
```

This produces:
- **Windows**: `file:///C:/Users/runneradmin/.llamafarm/broker/results`
- **Unix**: `file:///home/user/.llamafarm/broker/results`
- **Relative**: `file://.llamafarm/broker/results`

## Files Changed

- `rag/celery_app.py`: Updated result backend URL construction (line ~66-103)
- `server/core/celery/celery.py`: Updated result backend URL construction (line ~43-73), added `sys` import

## Tests Added

Two new test files were created to verify the fix:

1. **`rag/tests/test_celery_windows_url.py`**
2. **`server/tests/test_celery_windows_url.py`**

Both test suites verify:
- Windows path URL construction
- Unix path URL construction
- Relative path URL construction
- URL parsing with Kombu's URL parser
- Module loading on Windows (platform-specific test)

All tests pass on Unix platforms (macOS/Linux), and the Windows-specific tests are skipped on non-Windows platforms using `@pytest.mark.skipif`.

## Impact

This fix enables LlamaFarm to run successfully on Windows without requiring an external message broker (Redis/RabbitMQ). The filesystem-based broker now works correctly on all platforms.

## Related Issues

- E2E tests failing on Windows runners
- Windows build failures in CI/CD pipeline

## Verification

To verify the fix on Windows:

```bash
# Run the specific tests
cd rag
uv run pytest tests/test_celery_windows_url.py -v

cd ../server
uv run --group test pytest tests/test_celery_windows_url.py -v

# Run full E2E tests
cd ..
# Follow E2E test instructions
```

## Notes

- The fix maintains backward compatibility with Unix systems
- No changes to external broker configurations (Redis/RabbitMQ) are needed
- The same URL construction logic is used in both server and rag workers for consistency

