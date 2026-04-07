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

# Import offline_mode BEFORE huggingface_hub so that, when LLAMAFARM_OFFLINE=1,
# the HF_HUB_OFFLINE env var has already been propagated and huggingface_hub
# picks it up at import time. offline_mode runs its propagation at module-load.
from . import offline_mode  # noqa: F401

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


# Aliases are used to construct filesystem subdirectory names under
# `$LLAMAFARM_MODEL_DIR/<alias>/`. They come from project config
# (`runtime.models[].name`) which is not direct HTTP input, but because
# an alias is turned into a filesystem path we still sanitize it to
# prevent path traversal — both for defense in depth and to satisfy
# static analysis tools (CodeQL, cubic, qodo) that rightly flag any
# unsanitized identifier used in a path expression.
_ALIAS_PATTERN = re.compile(r"^[a-zA-Z0-9._][a-zA-Z0-9._\-]*$")


def validate_alias(alias: str) -> str:
    """
    Validate a model alias for safe use as a filesystem subdirectory name.

    Aliases must:
      - Not be empty
      - Not contain path separators (``/``, ``\\``)
      - Not contain ``..``
      - Not be an absolute path
      - Start with alphanumeric, period, or underscore
      - Contain only alphanumerics, periods, underscores, and hyphens

    Args:
        alias: The model alias from ``runtime.models[].name``.

    Returns:
        The validated alias string, unchanged.

    Raises:
        ValueError: If the alias fails validation.
    """
    if not isinstance(alias, str) or alias == "":
        raise ValueError("Invalid alias: must be a non-empty string")
    # Explicit ".." rejection — the pattern recognized by CodeQL's
    # py/path-injection as a traversal sanitizer.
    if ".." in alias:
        raise ValueError(f"Invalid alias: path traversal not allowed: {alias!r}")
    if "/" in alias or "\\" in alias:
        raise ValueError(f"Invalid alias: path separator not allowed: {alias!r}")
    if os.path.isabs(alias):
        raise ValueError(f"Invalid alias: absolute path not allowed: {alias!r}")
    if not _ALIAS_PATTERN.match(alias):
        raise ValueError(
            f"Invalid alias: must match {_ALIAS_PATTERN.pattern!r}: {alias!r}"
        )
    return alias


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

    # Strict offline mode: never call the HF API.
    if offline_mode.is_offline():
        raise FileNotFoundError(
            f"list_gguf_files({base_model_id!r}) refused in offline mode "
            f"(LLAMAFARM_OFFLINE=1). The caller should use the local cache or "
            f"$LLAMAFARM_MODEL_DIR instead."
        )

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
    # For HuggingFace-style IDs (non-local filenames), enforce model ID validation.
    # Parse quantization suffix first so "org/repo:Q8_0" validates correctly.
    if not model_id.endswith(".gguf"):
        base_id, _ = parse_model_with_quantization(model_id)
        _validate_model_id(base_id)

    # Local file resolution — check safe directories before any HuggingFace calls.
    # Uses os.path.basename to strip any directory components from user input,
    # then only checks within known safe root directories.
    if model_id.endswith(".gguf"):
        basename = os.path.basename(model_id)

        # Strict filename validation — only safe characters allowed.
        # Cuts the CodeQL taint chain before any filesystem operations.
        if not re.match(r"^[a-zA-Z0-9_.\-]+\.gguf$", basename):
            raise ValueError(f"Invalid GGUF filename: {basename}")
        # Reassign to a new variable so static analysis tools (CodeQL) treat
        # the value as validated/untainted from this point forward.
        safe_name = basename

        # 1. Standard model directory (~/.llamafarm/models/)
        from .safe_home import get_data_dir
        models_root = os.path.abspath(os.path.join(str(get_data_dir()), "models"))
        candidate = os.path.realpath(os.path.join(models_root, safe_name))
        if (
            os.path.commonpath([models_root, candidate]) == models_root
            and os.path.isfile(candidate)
        ):
            logger.info(f"Using GGUF file from data dir: {candidate}")
            return candidate
        # 2. Custom directory via GGUF_MODELS_DIR env var
        gguf_dir = os.environ.get("GGUF_MODELS_DIR")
        if gguf_dir:
            gguf_root = os.path.abspath(gguf_dir)
            candidate = os.path.realpath(os.path.join(gguf_root, safe_name))
            if (
                os.path.commonpath([gguf_root, candidate]) == gguf_root
                and os.path.isfile(candidate)
            ):
                logger.info(f"Using GGUF file from GGUF_MODELS_DIR: {candidate}")
                return candidate

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

    # Strict offline mode: do not attempt any network calls.
    if offline_mode.is_offline():
        cache_dir = os.path.join(
            HF_HUB_CACHE, f"models--{base_model_id.replace('/', '--')}"
        )
        offline_mode.raise_offline_error(
            alias=base_model_id,
            tried_paths=[cache_dir],
            fix_command=(
                f"run 'lf models pull {base_model_id}' on a host with internet, "
                f"then sync the files"
            ),
        )

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


def resolve_gguf_path(
    model_id: str,
    alias: str,
    token: str | None = None,
    preferred_quantization: str | None = None,
) -> str:
    """
    Resolve a GGUF weights file for a runtime model using the three-tier
    precedence order:

      1. ``$LLAMAFARM_MODEL_DIR/<alias>/`` via the alias-directory resolver
      2. HuggingFace cache via the existing ``get_gguf_file_path`` logic
      3. Network download (only when NOT in strict offline mode)

    When strict offline mode is active (``LLAMAFARM_OFFLINE=1``) and tiers 1
    and 2 both miss, this function raises ``FileNotFoundError`` with a
    structured multi-line message naming the alias, the paths that were
    tried, and the ``lf`` CLI command that would make the missing file
    available.

    Note: absolute-path-in-model-spec is deliberately NOT a tier here.
    Users who want to load a hand-placed GGUF file should set
    ``LLAMAFARM_MODEL_DIR`` to the parent directory and reference the file
    by alias. This avoids a CodeQL py/path-injection finding on an
    unsanitized ``os.path.*`` call with a caller-controlled path, and
    centralizes the "where do my models live" decision on a single
    well-understood env var.

    Args:
        model_id: HuggingFace model identifier, optionally with ``:quant``
            suffix.
        alias: The model's ``runtime.models[].name`` from llamafarm.yaml.
            Used as the subdirectory name under ``LLAMAFARM_MODEL_DIR`` and
            as the user-facing identifier in error messages.
        token: Optional HuggingFace token for gated models.
        preferred_quantization: Optional quantization preference. If absent,
            parsed from ``model_id``'s suffix.

    Returns:
        Absolute filesystem path to the selected GGUF weights file.

    Raises:
        ValueError: If `alias` contains path traversal characters or is
            otherwise unsafe for use as a filesystem subdirectory name.
        FileNotFoundError: when no matching file can be found in any tier
            and offline mode (or a network failure) prevents fallback to a
            network download.
    """
    # Validate the alias before any path manipulation. The alias comes from
    # project config (runtime.models[].name) and gets turned into a
    # filesystem subdirectory under LLAMAFARM_MODEL_DIR — path traversal
    # here would allow loading arbitrary .gguf files from the host.
    validate_alias(alias)

    # Reject absolute paths up front with a clear remediation message. The
    # old behavior flowed absolute paths into os.path.* which triggered a
    # CodeQL py/path-injection finding; users who want to reference a
    # hand-placed GGUF should instead set LLAMAFARM_MODEL_DIR to the
    # parent directory and reference the file by alias.
    if os.path.isabs(model_id):
        raise ValueError(
            f"Absolute model paths are not supported (got {model_id!r}). "
            f"Set LLAMAFARM_MODEL_DIR to the parent directory and reference "
            f"the file by alias (runtime.models[].name) instead."
        )

    # Tier 1: alias-directory lookup via $LLAMAFARM_MODEL_DIR.
    # Lazy import to avoid a circular dependency at module-load time.
    from . import model_dir as _model_dir

    md_result = _model_dir.resolve_from_model_dir(alias)
    if md_result is not None:
        logger.info(
            f"Resolved alias {alias!r} from LLAMAFARM_MODEL_DIR: {md_result.weights_path}"
        )
        return md_result.weights_path

    # Tier 2+3: fall through to the existing HF-cache-first logic.
    # get_gguf_file_path already enforces strict offline mode internally when
    # the cache is empty and LLAMAFARM_OFFLINE=1 is set. If offline mode is
    # active and it raises, we catch and re-raise with alias context so the
    # operator sees the alias name rather than just the repo id.
    try:
        return get_gguf_file_path(
            model_id,
            token=token,
            preferred_quantization=preferred_quantization,
        )
    except FileNotFoundError as e:
        # Enrich the error with alias + LLAMAFARM_MODEL_DIR context if we're
        # in offline mode. Other FileNotFoundError messages pass through.
        if offline_mode.is_offline():
            tried: list[str] = []
            md_root = offline_mode.model_dir()
            if md_root is not None:
                tried.append(os.path.join(md_root, alias))
            base_repo_id, _ = parse_model_with_quantization(model_id)
            tried.append(
                os.path.join(
                    HF_HUB_CACHE,
                    f"models--{base_repo_id.replace('/', '--')}",
                )
            )
            offline_mode.raise_offline_error(
                alias=alias,
                tried_paths=tried,
                fix_command=(
                    f"run 'lf models pull {model_id}' on a host with internet, "
                    f"then sync the files to this host"
                ),
                extra=(
                    "If the build host has internet, you can also use "
                    "'lf models path --ensure' to pull before emitting a plan."
                ),
            )
        # Not in offline mode — re-raise original.
        raise


def resolve_mmproj_path(
    model_id: str,
    alias: str,
    token: str | None = None,
) -> str | None:
    """
    Resolve a multimodal projector file for a runtime model, applying the
    same precedence order as ``resolve_gguf_path``. Returns None when no
    mmproj is available (mmproj is optional, so missing is not an error).

    Tier 1 (absolute path) is not applicable — mmproj is always derived
    from either the alias directory or the HF cache.

    Raises:
        ValueError: If `alias` contains path traversal characters.
    """
    # Same alias validation as resolve_gguf_path — prevents path traversal
    # through LLAMAFARM_MODEL_DIR.
    validate_alias(alias)

    # Tier 2: alias-directory lookup.
    from . import model_dir as _model_dir

    md_result = _model_dir.resolve_from_model_dir(alias)
    if md_result is not None and md_result.mmproj_path is not None:
        logger.info(
            f"Resolved mmproj for alias {alias!r} from LLAMAFARM_MODEL_DIR: "
            f"{md_result.mmproj_path}"
        )
        return md_result.mmproj_path

    # Tier 3+4: HF cache + network.
    return get_mmproj_file_path(model_id, token=token)


def get_mmproj_file_path(
    model_id: str,
    token: str | None = None,
) -> str | None:
    """
    Get the full path to a multimodal projector (mmproj) GGUF file.

    Multimodal projector files enable audio/vision input for models like Qwen2.5-Omni.
    These files are typically named with patterns like:
    - mmproj-{model}-f16.gguf
    - mmproj-{model}-f32.gguf
    - {model}-mmproj-f16.gguf

    Args:
        model_id: HuggingFace model identifier (e.g., "ggml-org/Qwen2.5-Omni-7B-GGUF")
        token: Optional HuggingFace authentication token for gated models

    Returns:
        Full absolute path to the mmproj .gguf file, or None if not found

    Examples:
        >>> path = get_mmproj_file_path("ggml-org/Qwen2.5-Omni-7B-GGUF")
        >>> path  # Could be None or a path like ".../mmproj-Qwen2.5-Omni-7B-f16.gguf"
    """
    # Parse model ID to extract base model
    base_model_id, _ = parse_model_with_quantization(model_id)

    logger.debug(f"Looking for mmproj file in: {base_model_id}")

    # Step 1: Check local cache first
    cached_gguf_files = _get_cached_gguf_files(base_model_id)
    mmproj_files = [f for f in cached_gguf_files if _is_mmproj_file(f)]

    if mmproj_files:
        # Prefer f16 precision for mmproj
        selected = _select_mmproj_file(mmproj_files)
        cached_path = _get_cached_gguf_path(base_model_id, selected)
        if cached_path:
            logger.info(f"Using cached mmproj file: {selected}")
            return cached_path

    # Strict offline mode: do not attempt any network calls. mmproj is
    # optional (a chat model may or may not have one), so silently returning
    # None is the correct behavior here — callers handle None gracefully.
    if offline_mode.is_offline():
        logger.debug(
            "Offline mode: no cached mmproj for %s; returning None", base_model_id
        )
        return None

    # Step 2: List all GGUF files in the repository (requires network)
    try:
        available_gguf_files = list_gguf_files(base_model_id, token)
    except Exception as e:
        logger.debug(f"Could not list files for mmproj detection: {e}")
        return None

    # Find mmproj files
    mmproj_files = [f for f in available_gguf_files if _is_mmproj_file(f)]

    if not mmproj_files:
        logger.debug(f"No mmproj files found in {base_model_id}")
        return None

    # Select best mmproj file (prefer f16)
    selected_filename = _select_mmproj_file(mmproj_files)
    logger.info(f"Found mmproj file: {selected_filename}")

    # Step 3: Download the mmproj file
    local_path = snapshot_download(
        repo_id=base_model_id,
        token=token,
        allow_patterns=[selected_filename],
    )

    mmproj_path = os.path.join(local_path, selected_filename)

    if not os.path.exists(mmproj_path):
        logger.warning(f"mmproj file not found after download: {mmproj_path}")
        return None

    logger.info(f"mmproj file ready: {mmproj_path}")
    return mmproj_path


def _is_mmproj_file(filename: str) -> bool:
    """Check if a filename is a multimodal projector file."""
    filename_lower = filename.lower()
    # Common patterns for mmproj files
    return (
        filename_lower.endswith(".gguf")
        and ("mmproj" in filename_lower or "multimodal" in filename_lower)
        # Exclude main model files that might have "multimodal" in the name
        and not any(q in filename_lower for q in ["q4_", "q5_", "q8_", "q2_", "q3_", "q6_"])
    )


def _select_mmproj_file(mmproj_files: list[str]) -> str:
    """Select the best mmproj file, preferring f16 precision.

    On Mac, F16 is the safest choice as BF16 has limited hardware support.
    F32 always works but is 2x larger.

    Preference order: F16 > BF16 > FP16 > F32 > FP32
    """
    if not mmproj_files:
        raise ValueError("No mmproj files provided")

    if len(mmproj_files) == 1:
        return mmproj_files[0]

    # Prefer f16 (best Mac compatibility), then bf16, then f32 (larger)
    # Use precise matching with delimiters to avoid "f16" matching "bf16"
    for precision in ["f16", "bf16", "fp16", "f32", "fp32"]:
        for f in mmproj_files:
            f_lower = f.lower()
            # Match with common delimiters: -f16., _f16., -f16-, _f16_
            if (
                f"-{precision}." in f_lower
                or f"_{precision}." in f_lower
                or f"-{precision}-" in f_lower
                or f"_{precision}_" in f_lower
                or f_lower.endswith(f"-{precision}.gguf")
                or f_lower.endswith(f"_{precision}.gguf")
            ):
                return f

    # Fall back to first file
    return mmproj_files[0]
