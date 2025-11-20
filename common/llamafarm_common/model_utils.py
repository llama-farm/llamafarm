"""
Model utility functions shared across LlamaFarm services.

This module provides common utilities for:
- Parsing model names with quantization suffixes
- Parsing quantization from GGUF filenames
- Selecting optimal GGUF quantization variants
"""

import re

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
