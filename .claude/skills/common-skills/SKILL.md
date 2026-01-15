---
name: common-skills
description: Best practices for the Common utilities package in LlamaFarm. Covers HuggingFace Hub integration, GGUF model management, and shared utilities.
allowed-tools: Read, Grep, Glob
user-invocable: false
---

# Common Skills for LlamaFarm

Best practices and code review checklists for the `common/` package - shared Python utilities used across all LlamaFarm services.

## Component Overview

| Attribute | Value |
|-----------|-------|
| Path | `common/` |
| Package | `llamafarm-common` |
| Python | 3.10+ |
| Key Dependencies | huggingface_hub, hf-transfer |

## Purpose

The `common/` package provides shared functionality that needs to be consistent across multiple Python services:
- Model file utilities (GGUF selection, quantization parsing)
- HuggingFace Hub integration (listing, downloading)
- Process management (PID files)

## Shared Python Skills

This skill inherits all patterns from the shared Python skills:

| Topic | File | Relevance |
|-------|------|-----------|
| Patterns | [../python-skills/patterns.md](../python-skills/patterns.md) | Dataclasses, type hints, comprehensions |
| Typing | [../python-skills/typing.md](../python-skills/typing.md) | Type annotations, modern syntax |
| Testing | [../python-skills/testing.md](../python-skills/testing.md) | Pytest fixtures, mocking HuggingFace APIs |
| Errors | [../python-skills/error-handling.md](../python-skills/error-handling.md) | Custom exceptions, logging |
| Security | [../python-skills/security.md](../python-skills/security.md) | Path validation, safe file handling |

## Framework-Specific Checklists

| Topic | File | Key Points |
|-------|------|------------|
| HuggingFace | [huggingface.md](huggingface.md) | Hub API, model download, caching, authentication |

## Module Structure

```
common/
├── pyproject.toml           # UV-managed dependencies
├── llamafarm_common/
│   ├── __init__.py          # Public API exports
│   ├── model_utils.py       # GGUF file utilities
│   └── pidfile.py           # PID file management
└── tests/
    └── test_model_utils.py  # Unit tests with mocking
```

## Public API

### Model Utilities

```python
from llamafarm_common import (
    # Parse model:quantization syntax
    parse_model_with_quantization,
    # Extract quantization from filename
    parse_quantization_from_filename,
    # Select best GGUF file from list
    select_gguf_file,
    select_gguf_file_with_logging,
    # List GGUF files in HF repo
    list_gguf_files,
    # Download and get path to GGUF file
    get_gguf_file_path,
    # Default quantization preference order
    GGUF_QUANTIZATION_PREFERENCE_ORDER,
)
```

### PID File Management

```python
from llamafarm_common.pidfile import write_pid, get_pid_file
```

## Review Checklist Summary

When reviewing code in `common/`:

1. **HuggingFace Integration** (High priority)
   - Proper error handling for network failures
   - Authentication token passed correctly
   - High-speed transfer enabled appropriately

2. **Model Selection** (Medium priority)
   - Quantization preference order maintained
   - Case-insensitive matching
   - Graceful fallback when preferred not available

3. **Testing** (High priority)
   - HuggingFace API calls mocked
   - Network isolation in tests
   - Edge cases covered (empty lists, missing files)

4. **Security** (Medium priority)
   - No token exposure in logs
   - Safe file path handling
   - Environment variable protection

See [huggingface.md](huggingface.md) for detailed HuggingFace-specific checklists.
