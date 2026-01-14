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
from huggingface_hub.constants import HF_HUB_CACHE

logger = logging.getLogger(__name__)


def _validate_model_id(model_id: str) -> str:
    """
    Validate and sanitize a HuggingFace model ID to prevent path traversal.

    Args:
        model_id: HuggingFace model identifier (e.g., "unsloth/Qwen3-1.7B-GGUF")

    Returns:
        Sanitized model_id safe for use in file paths

    Raises:
        ValueError: If model_id contains path traversal attempts or invalid characters
    """
    # Check for path traversal attempts
    if ".." in model_id or model_id.startswith("/") or model_id.startswith("\\"):
        raise ValueError(f"Invalid model_id: path traversal not allowed: {model_id}")

    # Valid HuggingFace model IDs: org/repo or just repo
    # Allow alphanumeric, hyphens, underscores, periods, and single forward slash
    if not re.match(r"^[a-zA-Z0-9_.\-]+(/[a-zA-Z0-9_.\-]+)?$", model_id):
        raise ValueError(f"Invalid model_id format: {model_id}")

    return model_id


def _get_cached_gguf_files(model_id: str) -> list[str]:
    """
    Check local HuggingFace cache for GGUF files.

    Args:
        model_id: HuggingFace model identifier (e.g., "unsloth/Qwen3-1.7B-GGUF")

    Returns:
        List of .gguf filenames found in local cache, or empty list if not cached

    Raises:
        ValueError: If model_id contains path traversal attempts
    """
    # Validate model_id to prevent path traversal
    _validate_model_id(model_id)

    # HuggingFace cache structure: ~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/{hash}/
    cache_dir = os.path.join(
        HF_HUB_CACHE, f"models--{model_id.replace('/', '--')}"
    )

    if not os.path.exists(cache_dir):
        return []

    snapshots_dir = os.path.join(cache_dir, "snapshots")
    if not os.path.exists(snapshots_dir):
        return []

    # Find GGUF files in any snapshot
    gguf_files: set[str] = set()  # Use set to avoid duplicates efficiently
    try:
        for snapshot_hash in os.listdir(snapshots_dir):
            snapshot_path = os.path.join(snapshots_dir, snapshot_hash)
            if os.path.isdir(snapshot_path):
                for filename in os.listdir(snapshot_path):
                    if filename.endswith(".gguf") and filename not in gguf_files:
                        # Verify it's a real file with content
                        # Use single getsize() call which fails on broken symlinks
                        file_path = os.path.join(snapshot_path, filename)
                        try:
                            if os.path.getsize(file_path) > 0:
                                gguf_files.add(filename)
                        except (OSError, FileNotFoundError):
                            # Broken symlink or file removed - skip silently
                            pass
    except OSError as e:
        logger.debug(f"Error scanning cache directory {cache_dir}: {e}")
        return []

    return sorted(gguf_files)


def _get_cached_gguf_path(model_id: str, filename: str) -> str | None:
    """
    Get the full path to a cached GGUF file.

    Args:
        model_id: HuggingFace model identifier
        filename: GGUF filename to find

    Returns:
        Full path to the cached file, or None if not found

    Raises:
        ValueError: If model_id or filename contains path traversal attempts
    """
    # Validate inputs to prevent path traversal
    _validate_model_id(model_id)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError(f"Invalid filename: path traversal not allowed: {filename}")

    cache_dir = os.path.join(
        HF_HUB_CACHE, f"models--{model_id.replace('/', '--')}"
    )
    snapshots_dir = os.path.join(cache_dir, "snapshots")

    if not os.path.exists(snapshots_dir):
        return None

    try:
        for snapshot_hash in os.listdir(snapshots_dir):
            file_path = os.path.join(snapshots_dir, snapshot_hash, filename)
            try:
                # Single getsize() call - fails on broken symlinks or missing files
                if os.path.getsize(file_path) > 0:
                    return file_path
            except (OSError, FileNotFoundError):
                # File doesn't exist or is broken symlink - continue to next snapshot
                continue
    except OSError:
        pass

    return None

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
    # Also handle patterns like Q2_K, Q3_K_L, Q4_K_M, etc.

    # Normalize filename for matching (handle case insensitivity)
    filename_lower = filename.lower()

    # Try multiple patterns to catch all variations
    # Pattern order matters - more specific first
    # Note: I? prefix supports imatrix-based quantization types (IQ2_K, IQ3_K, IQ4_XS, etc.)
    patterns = [
        # Patterns with separators (most common)
        r"[\._-](I?Q[2-8]_K_[SML])",  # Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, IQ2_K, IQ3_K, etc.
        r"[\._-](I?Q[2-8]_[01])",  # Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_0, etc.
        r"[\._-](I?Q[2-8]_K)(?![_\.])",  # Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_K, IQ3_K, etc. (not followed by _ or .)
        r"[\._-](I?Q[2-8]_XS)",  # IQ4_XS, IQ3_XS, etc. (imatrix extra small variants)
        r"[\._-](F16|F32|FP16|FP32)",  # F16, F32, FP16, FP32
        # Patterns without separators (less common but possible)
        r"(I?Q[2-8]_K_[SML])",  # Without separator prefix
        r"(I?Q[2-8]_[01])",  # Without separator prefix
        r"(I?Q[2-8]_K)(?![_\.])",  # Without separator prefix
        r"(I?Q[2-8]_XS)",  # Without separator prefix
        r"(F16|F32|FP16|FP32)",  # Without separator prefix
        # Handle edge cases with stricter boundary checks
        r"(I?Q[2-8]_K)(?![_\.A-Za-z0-9])",  # Q2_K, Q3_K, etc. not followed by alphanumeric (stricter than above)
    ]

    for pattern in patterns:
        match = re.search(pattern, filename_lower, re.IGNORECASE)
        if match:
            quant = match.group(1).upper()
            # Normalize FP16/FP32 to F16/F32
            if quant == "FP16":
                quant = "F16"
            elif quant == "FP32":
                quant = "F32"
            # Validate it's a known quantization type
            # Supports both regular (Q) and imatrix-based (IQ) quantization types
            if re.match(
                r"^(I?Q[2-8](?:_K(?:_[SML])?|_[01]|_K|_XS)|F(?:16|32))$", quant, re.IGNORECASE
            ):
                return quant

    # Check for unquantized files (large files without quantization markers)
    # These might be full precision models
    # We'll return None for these - they should be filtered out or handled separately
    return None


def is_split_gguf_file(filename: str) -> bool:
    """Check if a GGUF file is a split file (part of a multi-file model).

    Split files have patterns like:
    - model-00001-of-00002.gguf
    - model-00001-of-00002.Q4_K_M.gguf
    - qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf
    """
    return bool(re.search(r"-\d{5}-of-\d{5}[.\-]", filename, re.IGNORECASE))


def select_gguf_file(
    gguf_files: list[str], preferred_quantization: str | None = None
) -> str | None:
    """
    Select the best GGUF file from a list based on quantization preference.

    Selection logic:
    1. Filter out split files when a non-split version with same quantization exists
    2. If preferred_quantization is specified and found, use it
    3. Otherwise, use default preference order: Q4_K_M > Q4_K > Q5_K_M > Q5_K > Q8_0 > others
    4. Fall back to first file if no quantized versions found

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

    # Separate split and non-split files
    non_split_files = [f for f in gguf_files if not is_split_gguf_file(f)]
    split_files = [f for f in gguf_files if is_split_gguf_file(f)]

    # Prefer non-split files; only use split files if no non-split version exists
    # for the desired quantization
    working_files = non_split_files if non_split_files else split_files

    # Parse quantization types for working files
    file_quantizations = [
        (filename, parse_quantization_from_filename(filename))
        for filename in working_files
    ]

    # If preferred quantization specified, try to find exact match
    if preferred_quantization:
        preferred_upper = preferred_quantization.upper()
        for filename, quant in file_quantizations:
            if quant and quant.upper() == preferred_upper:
                return filename
        # If not found in non-split, check split files
        if non_split_files and split_files:
            for filename in split_files:
                quant = parse_quantization_from_filename(filename)
                if quant and quant.upper() == preferred_upper:
                    return filename

    # Use default preference order
    for preferred in GGUF_QUANTIZATION_PREFERENCE_ORDER:
        for filename, quant in file_quantizations:
            if quant and quant.upper() == preferred:
                return filename

    # No quantized version found in preference order - use first file
    return working_files[0] if working_files else gguf_files[0]


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

    # Step 1: Check local cache first (enables offline operation)
    cached_gguf_files = _get_cached_gguf_files(base_model_id)
    if cached_gguf_files:
        logger.info(f"Found {len(cached_gguf_files)} GGUF files in local cache")
        # Try to select from cached files
        selected_filename = select_gguf_file(cached_gguf_files, preferred_quantization)
        if selected_filename:
            cached_path = _get_cached_gguf_path(base_model_id, selected_filename)
            if cached_path:
                quant = parse_quantization_from_filename(selected_filename)
                logger.info(
                    f"Using cached GGUF file: {selected_filename} "
                    f"(quantization: {quant or 'unknown'})"
                )
                return cached_path

    # Step 2: List all GGUF files in the repository (requires network)
    try:
        available_gguf_files = list_gguf_files(base_model_id, token)
    except Exception as e:
        # If we have cached files but network failed, use cached version
        if cached_gguf_files:
            logger.warning(
                f"Network error listing files, falling back to cache: {e}"
            )
            selected_filename = select_gguf_file(
                cached_gguf_files, preferred_quantization
            )
            if selected_filename:
                cached_path = _get_cached_gguf_path(base_model_id, selected_filename)
                if cached_path:
                    logger.info(f"Using cached GGUF file (offline): {cached_path}")
                    return cached_path
        raise

    if not available_gguf_files:
        raise FileNotFoundError(
            f"No GGUF files found in model repository: {base_model_id}"
        )

    # Step 3: Select the best GGUF file based on preference
    selected_filename = select_gguf_file_with_logging(
        available_gguf_files, preferred_quantization
    )

    logger.info(
        f"Selected GGUF file: {selected_filename} "
        f"(from {len(available_gguf_files)} available files)"
    )

    # Step 4: Download only the selected file using allow_patterns
    local_path = snapshot_download(
        repo_id=base_model_id,
        token=token,
        allow_patterns=[selected_filename],  # Only download this specific file
    )

    # Step 5: Construct full path to the downloaded file
    gguf_path = os.path.join(local_path, selected_filename)

    # Verify the file exists
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF file not found after download: {gguf_path}")

    logger.info(f"GGUF file ready: {gguf_path}")
    return gguf_path
