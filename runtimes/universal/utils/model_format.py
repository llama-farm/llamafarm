"""Model format detection utilities for Universal Runtime.

Detects whether a HuggingFace model repository contains GGUF or transformers format files.
"""

import logging
import os
import re

from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger(__name__)

# Cache detection results to avoid repeated filesystem checks
_format_cache: dict[str, str] = {}


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


def parse_quantization_from_filename(filename: str) -> str | None:
    """
    Extract quantization type from a GGUF filename.

    Quantization types follow patterns like Q4_K_M, Q8_0, F16, etc.
    This function uses regex to extract these patterns from filenames.

    Args:
        filename: GGUF filename (e.g., "qwen3-1.7b.Q4_K_M.gguf")

    Returns:
        Quantization type (e.g., "Q4_K_M") or None if not found

    Examples:
        >>> parse_quantization_from_filename("qwen3-1.7b.Q4_K_M.gguf")
        'Q4_K_M'
        >>> parse_quantization_from_filename("model.Q8_0.gguf")
        'Q8_0'
        >>> parse_quantization_from_filename("model.F16.gguf")
        'F16'
    """
    # Common GGUF quantization patterns:
    # - Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q5_0, Q5_1, Q5_K_S, Q5_K_M
    # - Q6_K, Q8_0, F16, F32
    pattern = r"[\.-](Q[2-8]_(?:K_[SML]|K|[01])|(F(?:16|32)))\."
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def select_gguf_file(
    gguf_files: list[str], preferred_quantization: str | None = None
) -> str:
    """
    Select the best GGUF file from a list based on quantization preference.

    Selection logic:
    1. If preferred_quantization is specified and found, use it
    2. Otherwise, use default preference order: Q4_K_M > Q4_K > Q5_K_M > Q5_K > Q8_0 > others
    3. Fall back to first file if no quantized versions found

    Args:
        gguf_files: List of .gguf filenames from the repository
        preferred_quantization: Optional preferred quantization type (e.g., "Q4_K_M", "Q8_0")

    Returns:
        Selected GGUF filename

    Raises:
        ValueError: If no GGUF files provided or preferred quantization not found

    Examples:
        >>> files = ["model.Q4_K_M.gguf", "model.Q8_0.gguf", "model.F16.gguf"]
        >>> select_gguf_file(files)
        'model.Q4_K_M.gguf'
        >>> select_gguf_file(files, preferred_quantization="Q8_0")
        'model.Q8_0.gguf'
    """
    if not gguf_files:
        raise ValueError("No GGUF files provided")

    # If only one file, return it
    if len(gguf_files) == 1:
        logger.info(f"Only one GGUF file available: {gguf_files[0]}")
        return gguf_files[0]

    # Parse quantization types for all files
    file_quantizations = [
        (filename, parse_quantization_from_filename(filename))
        for filename in gguf_files
    ]

    # If preferred quantization specified, try to find exact match
    if preferred_quantization:
        preferred_upper = preferred_quantization.upper()
        for filename, quant in file_quantizations:
            if quant and quant.upper() == preferred_upper:
                logger.info(
                    f"Selected GGUF file with preferred quantization '{preferred_quantization}': {filename}"
                )
                return filename

        # Preferred not found - log warning and fall through to default selection
        available = [q for _, q in file_quantizations if q]
        logger.warning(
            f"Preferred quantization '{preferred_quantization}' not found. "
            f"Available quantizations: {available}. Falling back to default selection."
        )

    # Default preference order (good balance of size/quality)
    preference_order = [
        "Q4_K_M",  # Best default: good balance of size and quality
        "Q4_K",  # Generic Q4_K
        "Q5_K_M",  # Slightly higher quality, larger size
        "Q5_K",  # Generic Q5_K
        "Q8_0",  # High quality, larger size
        "Q6_K",  # Between Q5 and Q8
        "Q4_K_S",  # Smaller Q4 variant
        "Q5_K_S",  # Smaller Q5 variant
        "Q3_K_M",  # Smaller, lower quality
        "Q2_K",  # Very small, lower quality
        "F16",  # Full precision, very large
    ]

    # Try to find best match from preference order
    for preferred in preference_order:
        for filename, quant in file_quantizations:
            if quant and quant.upper() == preferred:
                logger.info(
                    f"Selected GGUF file with default quantization '{quant}': {filename}"
                )
                return filename

    # No quantized version found in preference order - use first file
    logger.warning(
        f"No preferred quantization found in {gguf_files}. Using first file: {gguf_files[0]}"
    )
    return gguf_files[0]


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
    selected_filename = select_gguf_file(available_gguf_files, preferred_quantization)

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
