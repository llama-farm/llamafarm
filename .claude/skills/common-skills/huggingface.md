# HuggingFace Hub Checklist

Patterns for HuggingFace Hub integration in LlamaFarm common utilities.

---

## Category: API Usage

### Use HfApi for Repository Operations

**What to check**: Use the HuggingFace Hub API client for repository operations

**Good pattern**:
```python
from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files(repo_id=model_id, token=token)
```

**Bad pattern**:
```python
import requests

# Manual API calls - don't do this
response = requests.get(f"https://huggingface.co/api/models/{model_id}")
```

**Search pattern**:
```bash
rg "huggingface_hub|HfApi" --type py common/
```

**Pass criteria**: All HuggingFace operations use official SDK

**Severity**: Medium

**Recommendation**: The official SDK handles authentication, retries, and API changes automatically

---

### Use snapshot_download for File Downloads

**What to check**: Use snapshot_download with allow_patterns for selective downloads

**Good pattern**:
```python
from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id=model_id,
    token=token,
    allow_patterns=[selected_filename],  # Only download specific file
)
```

**Bad pattern**:
```python
from huggingface_hub import hf_hub_download

# Downloads entire file without pattern filtering
path = hf_hub_download(repo_id=model_id, filename=filename)
```

**Pass criteria**: Use snapshot_download with allow_patterns for selective downloads

**Severity**: Medium

**Why it matters**: GGUF repos often contain multiple quantization variants; downloading all wastes bandwidth and storage

---

### Enable High-Speed Transfers

**What to check**: Enable HF_XET_HIGH_PERFORMANCE for faster downloads

**Good pattern**:
```python
import os

# Enable high-speed HuggingFace transfers by default
if "HF_XET_HIGH_PERFORMANCE" not in os.environ:
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from huggingface_hub import snapshot_download
```

**Search pattern**:
```bash
rg "HF_XET_HIGH_PERFORMANCE|HF_HUB_ENABLE_HF_TRANSFER" --type py
```

**Pass criteria**: High-speed transfer enabled for large model downloads

**Severity**: Low

**Recommendation**: Xet high-performance transfer significantly speeds up large file downloads

---

## Category: Authentication

### Pass Token to All API Calls

**What to check**: Authentication token passed consistently to all HuggingFace API calls

**Good pattern**:
```python
def list_gguf_files(model_id: str, token: str | None = None) -> list[str]:
    api = HfApi()
    all_files = api.list_repo_files(repo_id=model_id, token=token)
    return [f for f in all_files if f.endswith(".gguf")]
```

**Bad pattern**:
```python
def list_gguf_files(model_id: str) -> list[str]:
    api = HfApi()
    # Missing token - will fail for gated models
    all_files = api.list_repo_files(repo_id=model_id)
    return [f for f in all_files if f.endswith(".gguf")]
```

**Search pattern**:
```bash
rg "list_repo_files|snapshot_download|hf_hub_download" --type py -A5 | rg -v "token="
```

**Pass criteria**: All HuggingFace API calls accept and pass optional token parameter

**Severity**: High

**Why it matters**: Many models (Llama, Mistral, etc.) are gated and require authentication

---

### Never Log Authentication Tokens

**What to check**: HuggingFace tokens never appear in logs

**Bad pattern**:
```python
logger.info(f"Downloading with token: {token}")
logger.debug(f"API call: token={token}")
```

**Good pattern**:
```python
logger.info("Downloading model", extra={"model_id": model_id})
logger.debug("API call", extra={"has_token": token is not None})
```

**Search pattern**:
```bash
rg "logger.*token" --type py common/
```

**Pass criteria**: No token values in log statements

**Severity**: Critical

---

### Use Environment Variable for Default Token

**What to check**: Support HF_TOKEN environment variable

**Good pattern**:
```python
import os

def get_token(token: str | None = None) -> str | None:
    """Get HuggingFace token from parameter or environment."""
    return token or os.environ.get("HF_TOKEN")
```

**Pass criteria**: Functions accept token parameter and fall back to HF_TOKEN env var

**Severity**: Medium

---

## Category: Model ID Parsing

### Strip Quantization Suffix Before API Calls

**What to check**: Parse and remove quantization suffix from model IDs before HuggingFace API calls

**Good pattern**:
```python
def list_gguf_files(model_id: str, token: str | None = None) -> list[str]:
    # Parse model ID to remove quantization suffix if present
    base_model_id, _ = parse_model_with_quantization(model_id)

    api = HfApi()
    all_files = api.list_repo_files(repo_id=base_model_id, token=token)
    return [f for f in all_files if f.endswith(".gguf")]
```

**Bad pattern**:
```python
def list_gguf_files(model_id: str) -> list[str]:
    # Will fail if model_id contains :Q4_K_M suffix
    api = HfApi()
    all_files = api.list_repo_files(repo_id=model_id)
    return [f for f in all_files if f.endswith(".gguf")]
```

**Search pattern**:
```bash
rg "parse_model_with_quantization" --type py
```

**Pass criteria**: Model ID parsed before HuggingFace API calls

**Severity**: High

**Why it matters**: HuggingFace API rejects model IDs with colon-separated suffixes

---

### Case-Insensitive Quantization Matching

**What to check**: Quantization comparisons are case-insensitive

**Good pattern**:
```python
if quant and quant.upper() == preferred_quantization.upper():
    return filename
```

**Bad pattern**:
```python
if quant == preferred_quantization:  # Case-sensitive - will miss matches
    return filename
```

**Search pattern**:
```bash
rg "\.upper\(\)" --type py common/
```

**Pass criteria**: Quantization strings normalized to uppercase before comparison

**Severity**: Medium

---

## Category: GGUF File Selection

### Maintain Quantization Preference Order

**What to check**: Use consistent preference order for GGUF quantization selection

**Good pattern**:
```python
GGUF_QUANTIZATION_PREFERENCE_ORDER = [
    "Q4_K_M",  # Best default: good balance of size and quality
    "Q4_K",    # Generic Q4_K
    "Q5_K_M",  # Slightly higher quality, larger size
    "Q5_K",    # Generic Q5_K
    "Q8_0",    # High quality, larger size
    "Q6_K",    # Between Q5 and Q8
    # ... more variants
]

def select_gguf_file(files: list[str], preferred: str | None = None) -> str | None:
    # Try preferred first
    if preferred:
        for filename, quant in file_quantizations:
            if quant and quant.upper() == preferred.upper():
                return filename

    # Fall back to preference order
    for pref in GGUF_QUANTIZATION_PREFERENCE_ORDER:
        for filename, quant in file_quantizations:
            if quant and quant.upper() == pref:
                return filename
```

**Pass criteria**: Q4_K_M is default preference; consistent fallback order

**Severity**: Medium

**Recommendation**: Q4_K_M offers best balance of model quality and file size for most use cases

---

### Handle Missing Quantization Gracefully

**What to check**: Fall back gracefully when requested quantization not available

**Good pattern**:
```python
def select_gguf_file(files: list[str], preferred: str | None = None) -> str | None:
    if not files:
        return None

    if len(files) == 1:
        return files[0]

    # Try preferred, then defaults, then first file
    ...
    return files[0]  # Last resort fallback
```

**Pass criteria**: Never raise exception for missing quantization; fall back to available option

**Severity**: Medium

---

### Parse All Common Quantization Formats

**What to check**: Support all common GGUF quantization naming patterns

**Good pattern**:
```python
patterns = [
    r"[\._-](I?Q[2-8]_K_[SML])",  # Q4_K_M, IQ3_K_S
    r"[\._-](I?Q[2-8]_[01])",     # Q4_0, Q8_0
    r"[\._-](I?Q[2-8]_K)",        # Q4_K, Q6_K
    r"[\._-](I?Q[2-8]_XS)",       # IQ4_XS (imatrix)
    r"[\._-](F16|F32|FP16|FP32)", # Full precision
]
```

**Pass criteria**: Support Q2-Q8, imatrix (IQ), and full precision (F16/F32) variants

**Severity**: Medium

**Why it matters**: Different model providers use different naming conventions

---

## Category: Error Handling

### Raise Descriptive Errors for Missing Files

**What to check**: Provide helpful error messages when GGUF files not found

**Good pattern**:
```python
if not available_gguf_files:
    raise FileNotFoundError(
        f"No GGUF files found in model repository: {base_model_id}"
    )
```

**Pass criteria**: Error messages include model ID and context

**Severity**: Medium

---

### Handle Network Errors Gracefully

**What to check**: Catch and wrap HuggingFace API errors

**Good pattern**:
```python
try:
    api = HfApi()
    all_files = api.list_repo_files(repo_id=model_id, token=token)
except Exception as e:
    logger.error(f"Error listing files in {model_id}: {e}")
    raise
```

**Pass criteria**: Network errors logged with context before re-raising

**Severity**: High

---

### Verify Downloaded Files Exist

**What to check**: Verify file exists after download before returning path

**Good pattern**:
```python
local_path = snapshot_download(
    repo_id=model_id,
    token=token,
    allow_patterns=[selected_filename],
)

gguf_path = os.path.join(local_path, selected_filename)

# Verify the file exists
if not os.path.exists(gguf_path):
    raise FileNotFoundError(f"GGUF file not found after download: {gguf_path}")

return gguf_path
```

**Pass criteria**: File existence verified before returning path

**Severity**: High

---

## Category: Caching

### Leverage HuggingFace Cache

**What to check**: Don't implement custom caching; let huggingface_hub handle it

**Good pattern**:
```python
# snapshot_download automatically uses HF cache (~/.cache/huggingface/hub)
local_path = snapshot_download(repo_id=model_id)
```

**Bad pattern**:
```python
# Custom caching - unnecessary complexity
cache_dir = Path.home() / ".my_cache"
if not (cache_dir / model_id).exists():
    download_model(model_id, cache_dir)
```

**Pass criteria**: Use HuggingFace's built-in caching mechanism

**Severity**: Low

**Why it matters**: HF cache handles deduplication, partial downloads, and cleanup

---

### Allow Cache Directory Override

**What to check**: Support HF_HOME environment variable for cache location

**Good pattern**:
```python
# huggingface_hub respects HF_HOME automatically
# Users can set HF_HOME=/path/to/cache to change location
```

**Pass criteria**: Document that HF_HOME can override default cache location

**Severity**: Low

---

## Category: Testing

### Mock HuggingFace API Calls

**What to check**: Tests should mock HuggingFace API to avoid network calls

**Good pattern**:
```python
from unittest.mock import Mock, patch

@patch("llamafarm_common.model_utils.HfApi")
def test_list_gguf_files(mock_hf_api_class):
    mock_api = Mock()
    mock_api.list_repo_files.return_value = [
        "model.Q4_K_M.gguf",
        "model.Q8_0.gguf",
        "README.md",
    ]
    mock_hf_api_class.return_value = mock_api

    result = list_gguf_files("test/model")

    assert len(result) == 2
    mock_api.list_repo_files.assert_called_once_with(
        repo_id="test/model", token=None
    )
```

**Search pattern**:
```bash
rg "@patch.*HfApi|@patch.*snapshot_download" --type py common/tests/
```

**Pass criteria**: All HuggingFace API calls mocked in tests

**Severity**: High

**Why it matters**: Tests must be deterministic and not depend on network

---

### Test Edge Cases

**What to check**: Cover edge cases in model selection tests

**Pass criteria**:
- Empty file list returns None
- Single file returns that file
- Missing preferred quantization falls back correctly
- Case-insensitive matching works
- Quantization suffix in model ID handled

**Severity**: Medium

---

### Use Temporary Directories for File Tests

**What to check**: Use tempfile for tests involving file paths

**Good pattern**:
```python
import tempfile

def test_get_gguf_file_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_file = os.path.join(tmpdir, "model.Q4_K_M.gguf")
        with open(gguf_file, "w") as f:
            f.write("fake gguf")

        mock_snapshot_download.return_value = tmpdir

        result = get_gguf_file_path("test/model")
        assert result == gguf_file
```

**Pass criteria**: Tests use tempfile for any file system operations

**Severity**: Medium
