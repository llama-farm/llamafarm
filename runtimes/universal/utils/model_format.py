"""Model format detection utilities for Universal Runtime.

Detects whether a HuggingFace model repository contains GGUF or transformers format files.
"""

import logging
import os

from huggingface_hub import HfApi, snapshot_download
from llamafarm_common import (
    GGUF_QUANTIZATION_PREFERENCE_ORDER,
    parse_model_with_quantization,
    parse_quantization_from_filename,
    select_gguf_file,
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
    "detect_model_format",
    "list_gguf_files",
    "get_gguf_file_path",
]


def detect_model_format(model_id: str, token: str | None = None) -> str:
    """
    Detect if a HuggingFace model is GGUF or transformers format.

    This function uses the HuggingFace Hub API to list files in the repository
    and checks for .gguf files to determine the format, without downloading anything.
    Results are cached to avoid repeated API calls.

    Args:
        model_id: HuggingFace model identifier (e.g., "unsloth/Qwen3-0.6B-GGUF")
        token: Optional HuggingFace authentication token for gated models

    Returns:
        "gguf" if model contains .gguf files, "transformers" otherwise

    Raises:
        Exception: If model cannot be accessed

    Examples:
        >>> detect_model_format("unsloth/Qwen3-0.6B-GGUF")
        "gguf"
        >>> detect_model_format("google/gemma-3-1b-it")
        "transformers"
    """
    # Check cache first
    if model_id in _format_cache:
        logger.debug(f"Using cached format for {model_id}: {_format_cache[model_id]}")
        return _format_cache[model_id]

    logger.info(f"Detecting format for model: {model_id}")

    try:
        # Use API to list files without downloading anything
        api = HfApi()
        all_files = api.list_repo_files(repo_id=model_id, token=token)

        # Check if any .gguf files exist
        has_gguf = any(f.endswith(".gguf") for f in all_files)

        if has_gguf:
            logger.info("Detected GGUF format (found .gguf files)")
            _format_cache[model_id] = "gguf"
            return "gguf"

        # No GGUF files found - assume transformers format
        logger.info("Detected transformers format (no .gguf files found)")
        _format_cache[model_id] = "transformers"
        return "transformers"

    except Exception as e:
        logger.error(f"Error detecting model format for {model_id}: {e}")
        raise


def list_gguf_files(model_id: str, token: str | None = None) -> list[str]:
    """
    List all GGUF files available in a HuggingFace model repository.

    This function uses the HuggingFace Hub API to list all files in the repository
    and returns only the .gguf files without downloading them.

    Args:
        model_id: HuggingFace model identifier (e.g., "unsloth/Qwen3-1.7B-GGUF")
        token: Optional HuggingFace authentication token for gated models

    Returns:
        List of .gguf filenames available in the repository

    Examples:
        >>> files = list_gguf_files("unsloth/Qwen3-1.7B-GGUF")
        >>> files
        ['qwen3-1.7b.Q4_K_M.gguf', 'qwen3-1.7b.Q8_0.gguf', 'qwen3-1.7b.F16.gguf']
    """
    try:
        api = HfApi()
        all_files = api.list_repo_files(repo_id=model_id, token=token)
        gguf_files = [f for f in all_files if f.endswith(".gguf")]
        logger.debug(f"Found {len(gguf_files)} GGUF files in {model_id}: {gguf_files}")
        return gguf_files
    except Exception as e:
        logger.error(f"Error listing files in {model_id}: {e}")
        raise


# Note: parse_quantization_from_filename and select_gguf_file are imported from llamafarm_common
# and re-exported above for backward compatibility. We wrap select_gguf_file to add logging.


def select_gguf_file_with_logging(
    gguf_files: list[str], preferred_quantization: str | None = None
) -> str:
    """
    Select the best GGUF file from a list based on quantization preference, with logging.

    This is a wrapper around llamafarm_common.select_gguf_file that adds logging.

    Args:
        gguf_files: List of .gguf filenames from the repository
        preferred_quantization: Optional preferred quantization type (e.g., "Q4_K_M", "Q8_0")

    Returns:
        Selected GGUF filename

    Raises:
        ValueError: If no GGUF files provided
    """
    if not gguf_files:
        raise ValueError("No GGUF files provided")

    # If only one file, return it
    if len(gguf_files) == 1:
        logger.info(f"Only one GGUF file available: {gguf_files[0]}")
        return gguf_files[0]

    # Check if preferred quantization exists and log appropriately
    found = False
    if preferred_quantization:
        file_quantizations = [
            (filename, parse_quantization_from_filename(filename))
            for filename in gguf_files
        ]

        preferred_upper = preferred_quantization.upper()
        found = any(
            quant and quant.upper() == preferred_upper
            for _, quant in file_quantizations
        )

        if found:
            logger.info(
                f"Selected GGUF file with preferred quantization '{preferred_quantization}'"
            )
        else:
            available = [q for _, q in file_quantizations if q]
            logger.warning(
                f"Preferred quantization '{preferred_quantization}' not found. "
                f"Available quantizations: {available}. Falling back to default selection."
            )

    # Use common selection logic
    result = select_gguf_file(gguf_files, preferred_quantization)

    if not preferred_quantization or not found:
        # Log which default was selected
        quant = parse_quantization_from_filename(result)
        if quant:
            logger.info(
                f"Selected GGUF file with default quantization '{quant}': {result}"
            )
        else:
            logger.warning(
                f"No preferred quantization found. Using first file: {result}"
            )

    return result


def get_gguf_file_path(
    model_id: str,
    token: str | None = None,
    preferred_quantization: str | None = None,
) -> str:
    """
    Get the full path to a GGUF file in the HuggingFace cache.

    This function intelligently selects a GGUF file based on quantization preference,
    downloads only that specific file, and returns its path.

    Args:
        model_id: HuggingFace model identifier (e.g., "unsloth/Qwen3-1.7B-GGUF")
        token: Optional HuggingFace authentication token for gated models
        preferred_quantization: Optional quantization preference (e.g., "Q4_K_M", "Q8_0")
                                If not specified, defaults to Q4_K_M

    Returns:
        Full absolute path to the selected .gguf file

    Raises:
        FileNotFoundError: If no GGUF file found in the model repository

    Examples:
        >>> path = get_gguf_file_path("unsloth/Qwen3-0.6B-GGUF")
        >>> path.endswith('.gguf')
        True
        >>> path = get_gguf_file_path("unsloth/Qwen3-1.7B-GGUF", preferred_quantization="Q8_0")
        >>> "Q8_0" in path
        True
    """
    logger.info(f"Locating GGUF file for model: {model_id}")

    # Step 1: List all GGUF files in the repository (without downloading)
    available_gguf_files = list_gguf_files(model_id, token)

    if not available_gguf_files:
        raise FileNotFoundError(f"No GGUF files found in model repository: {model_id}")

    # Step 2: Select the best GGUF file based on preference
    selected_filename = select_gguf_file_with_logging(
        available_gguf_files, preferred_quantization
    )

    logger.info(
        f"Selected GGUF file: {selected_filename} "
        f"(from {len(available_gguf_files)} available files)"
    )

    # Step 3: Download only the selected file using allow_patterns
    local_path = snapshot_download(
        repo_id=model_id,
        token=token,
        allow_patterns=[selected_filename],  # Only download this specific file
    )

    # Step 4: Construct full path to the downloaded file
    gguf_path = os.path.join(local_path, selected_filename)

    # Verify the file exists
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF file not found after download: {gguf_path}")

    logger.info(f"GGUF file ready: {gguf_path}")
    return gguf_path


def clear_format_cache():
    """Clear the format detection cache.

    Useful for testing or when model repositories are updated.
    """
    global _format_cache
    _format_cache = {}
    logger.debug("Format detection cache cleared")
