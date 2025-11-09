# Windows Schema Compilation Fix

## Issue
When running LlamaFarm CLI v0.0.11 on Windows, the setup process failed during schema compilation with the following error:

```
Failed to start server: failed to ensure source code: failed to generate datamodel: datamodel generation failed: exit status 1
Output: Error running command: uv run python compile_schema.py
error during schema compilation: Error while resolving 'file:///C:/Users/jh7st/.llamafarm/src/rag/schema.yaml': OSError: [Errno 22] Invalid argument: '\\C:\\Users\\jh7st\\.llamafarm\\src\\rag\\schema.yaml'
```

### Root Cause
The `compile_schema.py` script was not properly handling Windows file URIs. When a Windows path like `C:\Users\...` is converted to a file URI using `Path.as_uri()`, it becomes `file:///C:/Users/...`. When this URI is parsed by `urlparse()`, the `parsed.path` component becomes `/C:/Users/...` (with a leading slash).

On Windows, attempting to use `Path('/C:/Users/...')` fails because the leading slash is invalid for Windows paths. The Python `pathlib.Path` class doesn't automatically handle the conversion from file URI paths to filesystem paths.

## Solution
Updated `config/compile_schema.py` to use `urllib.request.url2pathname()` which properly converts file URI paths to filesystem paths on all platforms:

```python
from urllib.request import url2pathname

def load_text_from_uri(uri: str) -> str:
    """Read local file:// or plain path URIs into text (UTF-8)."""
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        # Use url2pathname to properly convert file URIs to filesystem paths
        # This handles Windows paths correctly (e.g., file:///C:/Users/... -> C:\Users\...)
        if parsed.scheme == "file":
            # For file:// URIs, convert the path component to a filesystem path
            path = Path(url2pathname(parsed.path))
        else:
            # For plain paths without a scheme, use them directly
            path = Path(uri)
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported URI scheme in $ref: {uri}")
```

### How `url2pathname` Works
- **Unix/macOS**: Converts `/path/to/file` → `/path/to/file` (no change)
- **Windows**: Converts `/C:/Users/...` → `C:\Users\...` (removes leading slash, adds backslashes)

This ensures that file URI paths are correctly converted to valid filesystem paths on all platforms.

## Testing
Created comprehensive test suite in `config/tests/test_compile_schema.py` covering:

1. **Plain path loading** - Loading from filesystem paths without URI scheme
2. **Unix-style file URIs** - Testing `file:///path/to/file` format
3. **Windows drive letters** - Testing `file:///C:/Users/...` format
4. **Paths with spaces** - Testing URL-encoded paths (e.g., `%20`)
5. **Invalid schemes** - Ensuring non-file URIs are rejected
6. **Windows path conversion** - Verifying correct conversion on Windows
7. **Schema compilation** - Full integration tests

All tests pass on macOS (Windows-specific test skipped on non-Windows platforms).

## Verification
1. ✅ All new tests pass (9 passed, 1 skipped on macOS)
2. ✅ Schema compilation succeeds: `uv run python compile_schema.py`
3. ✅ No linter errors introduced
4. ✅ Existing config tests continue to pass

## Impact
- **Fixes**: Windows users can now successfully run `lf start` and initialize projects
- **Compatibility**: No impact on Unix/macOS behavior
- **Risk**: Low - changes are isolated to URI-to-path conversion logic

## Files Changed
- `config/compile_schema.py` - Added `url2pathname` import and proper URI handling
- `config/tests/test_compile_schema.py` - New test file with comprehensive coverage

## Related Issues
- Reported by user attempting to use v0.0.11 on Windows
- Issue affects all Windows users trying to initialize or start LlamaFarm
- Problem exists in all versions prior to this fix

## Recommendations
1. Include this fix in the next patch release
2. Consider adding Windows to CI/CD testing matrix
3. Test installer on Windows to verify complete setup flow

## Version Targeted
- Fix applied to: `main` branch
- Should be included in: Next release after v0.0.11

