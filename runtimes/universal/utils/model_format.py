"""Model format detection utilities for Universal Runtime.

Detects whether a HuggingFace model repository contains GGUF or transformers format files.

Note: Core GGUF utilities (list_gguf_files, select_gguf_file, get_gguf_file_path, etc.)
are provided by llamafarm_common and re-exported here for backward compatibility.

Performance optimizations:
- Results are cached to avoid repeated API calls within a session
- Checks local HuggingFace cache before making network requests
"""

import logging

from huggingface_hub import HfApi, scan_cache_dir
from huggingface_hub.utils import HFCacheInfo
from llamafarm_common import (
    GGUF_QUANTIZATION_PREFERENCE_ORDER,
    get_gguf_file_path,
    list_gguf_files,
    parse_model_with_quantization,
    parse_quantization_from_filename,
    select_gguf_file,
    select_gguf_file_with_logging,
)

logger = logging.getLogger(__name__)

# Cache detection results to avoid repeated filesystem checks
_format_cache: dict[str, str] = {}

# Cache for local repo info to avoid repeated cache scans
_local_cache_info: HFCacheInfo | None = None

# Re-export commonly used functions for backward compatibility
__all__ = [
    "GGUF_QUANTIZATION_PREFERENCE_ORDER",
    "parse_model_with_quantization",
    "parse_quantization_from_filename",
    "select_gguf_file",
    "select_gguf_file_with_logging",
    "detect_model_format",
    "list_gguf_files",
    "get_gguf_file_path",
    "clear_format_cache",
]


def _check_local_cache_for_model(model_id: str) -> list[str] | None:
    """Check if model files are available in local HuggingFace cache.

    This avoids making network requests when we can determine format locally.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        List of cached filenames if model is cached, None otherwise
    """
    global _local_cache_info

    try:
        # Scan cache once and reuse (scanning is ~10-50ms)
        if _local_cache_info is None:
            _local_cache_info = scan_cache_dir()

        # Look for this model in cache
        for repo in _local_cache_info.repos:
            if repo.repo_id == model_id and repo.repo_type == "model":
                # Found cached repo - collect all filenames across revisions
                filenames = set()
                for revision in repo.revisions:
                    for file in revision.files:
                        filenames.add(file.file_name)
                if filenames:
                    logger.debug(
                        f"Found {len(filenames)} files in local cache for {model_id}"
                    )
                    return list(filenames)

        return None

    except Exception as e:
        logger.debug(f"Could not scan local cache: {e}")
        return None


def detect_model_format(model_id: str, token: str | None = None) -> str:
    """
    Detect if a HuggingFace model is GGUF or transformers format.

    This function first checks if the model is in the local HuggingFace cache,
    and only makes API calls if not cached locally. Results are cached in memory
    to avoid repeated checks within a session.

    Args:
        model_id: HuggingFace model identifier (e.g., "unsloth/Qwen3-0.6B-GGUF" or "unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
        token: Optional HuggingFace authentication token for gated models

    Returns:
        "gguf" if model contains .gguf files, "transformers" otherwise

    Raises:
        Exception: If model cannot be accessed

    Examples:
        >>> detect_model_format("unsloth/Qwen3-0.6B-GGUF")
        "gguf"
        >>> detect_model_format("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
        "gguf"
        >>> detect_model_format("google/gemma-3-1b-it")
        "transformers"
    """
    # Parse model ID to remove quantization suffix if present
    base_model_id, _ = parse_model_with_quantization(model_id)

    # Check memory cache first (fastest)
    if base_model_id in _format_cache:
        logger.debug(
            f"Using cached format for {base_model_id}: {_format_cache[base_model_id]}"
        )
        return _format_cache[base_model_id]

    logger.info(f"Detecting format for model: {base_model_id}")

    # Try local cache first to avoid API call
    local_files = _check_local_cache_for_model(base_model_id)
    if local_files is not None:
        has_gguf = any(f.endswith(".gguf") for f in local_files)
        if has_gguf:
            logger.info("Detected GGUF format from local cache (found .gguf files)")
            _format_cache[base_model_id] = "gguf"
            return "gguf"
        else:
            logger.info(
                "Detected transformers format from local cache (no .gguf files)"
            )
            _format_cache[base_model_id] = "transformers"
            return "transformers"

    # Not in local cache - must query API
    try:
        api = HfApi()
        all_files = api.list_repo_files(repo_id=base_model_id, token=token)

        # Check if any .gguf files exist
        has_gguf = any(f.endswith(".gguf") for f in all_files)

        if has_gguf:
            logger.info("Detected GGUF format (found .gguf files)")
            _format_cache[base_model_id] = "gguf"
            return "gguf"

        # No GGUF files found - assume transformers format
        logger.info("Detected transformers format (no .gguf files found)")
        _format_cache[base_model_id] = "transformers"
        return "transformers"

    except Exception as e:
        logger.error(f"Error detecting model format for {base_model_id}: {e}")
        raise


def clear_format_cache():
    """Clear the format detection cache.

    Useful for testing or when model repositories are updated.
    """
    global _format_cache, _local_cache_info
    _format_cache = {}
    _local_cache_info = None
    logger.debug("Format detection cache cleared")
