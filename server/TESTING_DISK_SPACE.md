# Testing Disk Space Feature

## Quick Test

### 1. Start the server
```bash
cd server
uv run uvicorn server.main:app --reload
```

### 2. Check disk space endpoint
```bash
curl http://localhost:8000/v1/system/disk
```

Expected response:
```json
{
  "cache": {
    "total_bytes": 500000000000,
    "used_bytes": 100000000000,
    "free_bytes": 400000000000,
    "path": "/Users/username/.cache/huggingface/hub",
    "percent_free": 80.0
  },
  "system": {
    "total_bytes": 1000000000000,
    "used_bytes": 500000000000,
    "free_bytes": 500000000000,
    "path": "/",
    "percent_free": 50.0
  }
}
```

### 3. Test model download with disk space check

The disk space check happens automatically when you download a model. Try downloading a model:

```bash
curl -X POST http://localhost:8000/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{"model_name": "sentence-transformers/all-MiniLM-L6-v2", "provider": "universal"}'
```

**What happens:**
- If disk space < 100MB: Download is **blocked** with 400 error
- If disk space < 10% free: Download **proceeds** but emits a warning event in the stream
- If disk space is sufficient: Download proceeds normally

### 4. Watch for warning events

If space is low (< 10%), you'll see a warning event in the stream:
```
data: {"event": "warning", "message": "Nearing disk space max - you have X GB available, it could alter LF capabilities. Do you want to continue anyway?"}
```

## Automated Tests

Run the test suite:

```bash
cd server
uv run pytest tests/test_disk_space_service.py -v
uv run pytest tests/test_models_endpoint.py::test_download_model_insufficient_space -v
uv run pytest tests/test_models_endpoint.py::test_download_model_sufficient_space -v
uv run pytest tests/test_models_endpoint.py::test_download_model_low_space_warning -v
```

## What Gets Checked

1. **HuggingFace cache directory** - Where models are stored
2. **System disk** - Overall system health
3. **Model size** - Estimated from HuggingFace API (if available)
4. **Warning threshold** - 10% free space
5. **Critical threshold** - 100MB absolute minimum

## Key Features

- ✅ Checks both cache and system disk
- ✅ Blocks downloads if < 100MB free
- ✅ Warns (but allows) if < 10% free
- ✅ Gracefully handles API failures (proceeds with download)
- ✅ Works on macOS, Linux, and Windows

