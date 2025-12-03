"""Model format detection utilities for Universal Runtime.

Detects whether a HuggingFace model repository contains GGUF or transformers format files.

Note: Core GGUF utilities (list_gguf_files, select_gguf_file, get_gguf_file_path, etc.)
are provided by llamafarm_common and re-exported here for backward compatibility.
"""

import logging

from huggingface_hub import HfApi
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


def detect_model_format(model_id: str, token: str | None = None) -> str:
    """
    Detect if a HuggingFace model is GGUF or transformers format.

    This function uses the HuggingFace Hub API to list files in the repository
    and checks for .gguf files to determine the format, without downloading anything.
    Results are cached to avoid repeated API calls.

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

    # Check cache first (use base model ID for caching)
    if base_model_id in _format_cache:
        logger.debug(
            f"Using cached format for {base_model_id}: {_format_cache[base_model_id]}"
        )
        return _format_cache[base_model_id]

    logger.info(f"Detecting format for model: {base_model_id}")

    try:
        # Use API to list files without downloading anything
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
    global _format_cache
    _format_cache = {}
    logger.debug("Format detection cache cleared")
