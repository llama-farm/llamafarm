"""
Model utility functions shared across LlamaFarm services.

This module provides common utilities for:
- Parsing model names with quantization suffixes
- Parsing quantization from GGUF filenames
- Selecting optimal GGUF quantization variants
- Listing GGUF files from HuggingFace repositories
- Downloading GGUF files with intelligent quantization selection
"""

import logging
import os
import re

# Enable high-speed HuggingFace transfers by default (uses Xet high-performance transfer)
# This provides significantly faster downloads for large model files
# Can be disabled by setting HF_XET_HIGH_PERFORMANCE=0
if "HF_XET_HIGH_PERFORMANCE" not in os.environ:
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger(__name__)

# Default preference order for GGUF quantization (best balance of size/quality)
GGUF_QUANTIZATION_PREFERENCE_ORDER = [
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


def parse_model_with_quantization(model_name: str) -> tuple[str, str | None]:
    """
    Parse a model name that may include a quantization suffix.

    Format: "model_id:quantization" or just "model_id"
    Examples:
        - "unsloth/Qwen3-4B-GGUF:Q4_K_M" -> ("unsloth/Qwen3-4B-GGUF", "Q4_K_M")
        - "unsloth/Qwen3-4B-GGUF:q8_0" -> ("unsloth/Qwen3-4B-GGUF", "Q8_0")
        - "unsloth/Qwen3-4B-GGUF" -> ("unsloth/Qwen3-4B-GGUF", None)

    Args:
        model_name: Model identifier, optionally with :quantization suffix

    Returns:
        Tuple of (model_id, quantization_or_none)
        Quantization is normalized to uppercase if present
    """
    if ":" in model_name:
        parts = model_name.rsplit(":", 1)
        model_id = parts[0]
        quantization = parts[1].upper() if parts[1] else None
        return model_id, quantization
    return model_name, None


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
    pattern = r"[\.-](I?Q[2-8]_(?:K_[SML]|K|[01])|(F(?:16|32)))\."
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def select_gguf_file(
    gguf_files: list[str], preferred_quantization: str | None = None
) -> str | None:
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
        Selected GGUF filename, or None if no files provided

    Examples:
        >>> files = ["model.Q4_K_M.gguf", "model.Q8_0.gguf", "model.F16.gguf"]
        >>> select_gguf_file(files)
        'model.Q4_K_M.gguf'
        >>> select_gguf_file(files, preferred_quantization="Q8_0")
        'model.Q8_0.gguf'
    """
    if not gguf_files:
        return None

    # If only one file, return it
    if len(gguf_files) == 1:
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
                return filename

    # Use default preference order
    for preferred in GGUF_QUANTIZATION_PREFERENCE_ORDER:
        for filename, quant in file_quantizations:
            if quant and quant.upper() == preferred:
                return filename

    # No quantized version found in preference order - use first file
    return gguf_files[0]


def list_gguf_files(model_id: str, token: str | None = None) -> list[str]:
    """
    List all GGUF files available in a HuggingFace model repository.

    This function uses the HuggingFace Hub API to list all files in the repository
    and returns only the .gguf files without downloading them.

    Args:
        model_id: HuggingFace model identifier (e.g., "unsloth/Qwen3-1.7B-GGUF" or "unsloth/Qwen3-1.7B-GGUF:Q4_K_M")
        token: Optional HuggingFace authentication token for gated models

    Returns:
        List of .gguf filenames available in the repository

    Examples:
        >>> files = list_gguf_files("unsloth/Qwen3-1.7B-GGUF")
        >>> files
        ['qwen3-1.7b.Q4_K_M.gguf', 'qwen3-1.7b.Q8_0.gguf', 'qwen3-1.7b.F16.gguf']
        >>> files = list_gguf_files("unsloth/Qwen3-1.7B-GGUF:Q8_0")
        >>> files
        ['qwen3-1.7b.Q4_K_M.gguf', 'qwen3-1.7b.Q8_0.gguf', 'qwen3-1.7b.F16.gguf']
    """
    # Parse model ID to remove quantization suffix if present
    base_model_id, _ = parse_model_with_quantization(model_id)

    try:
        api = HfApi()
        all_files = api.list_repo_files(repo_id=base_model_id, token=token)
        gguf_files = [f for f in all_files if f.endswith(".gguf")]
        logger.debug(
            f"Found {len(gguf_files)} GGUF files in {base_model_id}: {gguf_files}"
        )
        return gguf_files
    except Exception as e:
        logger.error(f"Error listing files in {base_model_id}: {e}")
        raise


def select_gguf_file_with_logging(
    gguf_files: list[str], preferred_quantization: str | None = None
) -> str:
    """
    Select the best GGUF file from a list based on quantization preference, with logging.

    This is a wrapper around select_gguf_file that adds logging for better observability.

    Args:
        gguf_files: List of .gguf filenames from the repository
        preferred_quantization: Optional preferred quantization type (e.g., "Q4_K_M", "Q8_0")

    Returns:
        Selected GGUF filename

    Raises:
        ValueError: If no GGUF files provided or no file could be selected

    Examples:
        >>> files = ["model.Q4_K_M.gguf", "model.Q8_0.gguf"]
        >>> select_gguf_file_with_logging(files)
        'model.Q4_K_M.gguf'
        >>> select_gguf_file_with_logging(files, preferred_quantization="Q8_0")
        'model.Q8_0.gguf'
    """
    if not gguf_files:
        raise ValueError("No GGUF files provided")

    # Use common selection logic
    result = select_gguf_file(gguf_files, preferred_quantization)

    if not result:
        raise ValueError("No GGUF file selected")

    # Log the selection
    quant = parse_quantization_from_filename(result)
    if len(gguf_files) == 1:
        logger.info(f"Only one GGUF file available: {result}")
    elif (
        preferred_quantization
        and quant
        and quant.upper() == preferred_quantization.upper()
    ):
        logger.info(
            f"Selected GGUF file with preferred quantization '{preferred_quantization}': {result}"
        )
    elif quant:
        logger.info(f"Selected GGUF file with quantization '{quant}': {result}")
    else:
        logger.info(f"Selected GGUF file: {result}")

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

    High-speed downloads via Xet are enabled by default.
    To disable, set HF_XET_HIGH_PERFORMANCE=0.

    Args:
        model_id: HuggingFace model identifier (e.g., "unsloth/Qwen3-1.7B-GGUF" or "unsloth/Qwen3-1.7B-GGUF:Q4_K_M")
        token: Optional HuggingFace authentication token for gated models
        preferred_quantization: Optional quantization preference (e.g., "Q4_K_M", "Q8_0")
                                If not specified, will use quantization from model_id if present, otherwise defaults to Q4_K_M

    Returns:
        Full absolute path to the selected .gguf file

    Raises:
        FileNotFoundError: If no GGUF file found in the model repository

    Examples:
        >>> path = get_gguf_file_path("unsloth/Qwen3-0.6B-GGUF")
        >>> path.endswith('.gguf')
        True
        >>> path = get_gguf_file_path("unsloth/Qwen3-1.7B-GGUF:Q8_0")
        >>> "Q8_0" in path
        True
        >>> path = get_gguf_file_path("unsloth/Qwen3-1.7B-GGUF", preferred_quantization="Q8_0")
        >>> "Q8_0" in path
        True
    """
    # Parse model ID to extract base model and quantization suffix if present
    base_model_id, model_quantization = parse_model_with_quantization(model_id)

    # Use quantization from model_id if no explicit preference provided
    if preferred_quantization is None and model_quantization is not None:
        preferred_quantization = model_quantization
        logger.info(f"Using quantization from model ID: {preferred_quantization}")

    logger.info(f"Locating GGUF file for model: {base_model_id}")

    # Step 1: List all GGUF files in the repository (without downloading)
    available_gguf_files = list_gguf_files(base_model_id, token)

    if not available_gguf_files:
        raise FileNotFoundError(
            f"No GGUF files found in model repository: {base_model_id}"
        )

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
        repo_id=base_model_id,
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
