"""Context size calculator for GGUF models.

Determines optimal context window size based on:
1. User configuration (highest priority)
2. Available memory and model size
3. Model family defaults from config file
4. Fallback defaults
"""

import fnmatch
import logging
import os
from pathlib import Path

import psutil
import torch
import yaml
from gguf import GGUFReader

logger = logging.getLogger(__name__)

# Cache for config file
_config_cache: dict | None = None


def get_gguf_metadata(gguf_path: str) -> dict:
    """Read GGUF file metadata without loading the full model.

    Reads model metadata including the training context size (n_ctx_train)
    which indicates the maximum context the model was trained to handle.

    Args:
        gguf_path: Path to .gguf file

    Returns:
        dict with metadata including:
        - file_size_bytes: Size of the GGUF file in bytes
        - file_size_mb: Size in megabytes (for logging)
        - n_ctx_train: Training context size (if available)

    Raises:
        FileNotFoundError: If GGUF file doesn't exist
    """
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    file_size = os.path.getsize(gguf_path)

    metadata = {
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024),
        "n_ctx_train": None,
    }

    # Try to read GGUF metadata using gguf library
    try:
        # Read GGUF file metadata without loading model weights
        reader = GGUFReader(gguf_path)

        max_key_length = max(len(key) for key in reader.fields)
        for key, field in reader.fields.items():
            if field.data:
                value = field.parts[field.data[0]]
                logger.debug(f"{key:<{max_key_length}}: {value}")

        # Try to get context length from metadata fields
        # Common field names or substrings for context length
        context_field_names = [
            "context_length",
            "n_ctx_train",
            "n_ctx",
        ]

        # Check if any field key contains one of our target strings
        for key, field in reader.fields.items():
            if any(target in key for target in context_field_names) and field.data:
                # Extract the value from the field parts
                n_ctx_train = field.parts[field.data[0]]
                if n_ctx_train:
                    metadata["n_ctx_train"] = int(n_ctx_train)
                    logger.debug(f"Found context size in field '{key}': {n_ctx_train}")
                    break

    except Exception as e:
        logger.debug(f"Could not read GGUF metadata details: {e}")
        # Not a fatal error - we can still use file size

    logger.debug(f"GGUF metadata: {metadata}")
    return metadata


def get_available_memory(device: str) -> int:
    """Get available memory in bytes for the device.

    Args:
        device: Target device ("cuda", "mps", or "cpu")

    Returns:
        Available memory in bytes

    Notes:
        - For CUDA: Returns total GPU memory (not available, as model will allocate what it needs)
        - For MPS/CPU: Returns available system RAM
    """
    try:
        if device == "cuda" and torch.cuda.is_available():
            # Get total GPU memory (in bytes)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            logger.debug(f"CUDA total memory: {gpu_memory / (1024**3):.2f} GB")
            return gpu_memory
        else:
            # For CPU and MPS, use system RAM
            # Get available (not total) to be conservative
            vm = psutil.virtual_memory()
            available_memory = vm.available
            logger.debug(
                f"System memory - Total: {vm.total / (1024**3):.2f} GB, "
                f"Available: {available_memory / (1024**3):.2f} GB"
            )
            return available_memory
    except Exception as e:
        logger.warning(f"Error detecting memory for device {device}: {e}")
        # Fallback to conservative estimate (4GB)
        return 4 * 1024 * 1024 * 1024


def compute_max_context(
    model_size_bytes: int,
    available_memory_bytes: int,
    memory_factor: float = 0.8,
    max_context_cap: int = 131072,
) -> int:
    """Compute maximum safe context size based on available memory.

    Uses an aggressive formula that allocates most available memory
    to context, accounting for:
    - Model weight storage
    - KV cache (primary memory consumer for context)
    - Overhead for activation tensors and buffers

    Args:
        model_size_bytes: Size of model file in bytes
        available_memory_bytes: Available memory on target device
        memory_factor: Fraction of available memory to use (default 0.8)
        max_context_cap: Hard upper limit for context size (default 131072/128K).
            Most models don't support more than 128K context even with
            sufficient memory.

    Returns:
        Maximum safe context size (number of tokens)

    Formula:
        usable_memory = (available_memory * factor) - model_size
        bytes_per_token ≈ 2-4 bytes for quantized models (Q4/Q8)
        context_overhead = 4 (conservative multiplier for KV cache + buffers)
        n_ctx_max = usable_memory / (bytes_per_token * context_overhead)
    """
    # Calculate usable memory after loading model
    usable_memory = (available_memory_bytes * memory_factor) - model_size_bytes

    if usable_memory <= 0:
        logger.warning(
            f"Model size ({model_size_bytes / (1024**3):.2f} GB) exceeds "
            f"available memory budget. Using minimal context size."
        )
        return 512  # Minimal context

    # Estimate bytes per token for quantized GGUF models
    # Q4: ~0.5 bytes per param, Q8: ~1 byte per param
    # For KV cache: typically 2-4 bytes per token depending on precision
    bytes_per_token = 4  # Conservative estimate for KV cache
    context_overhead = 4  # Account for both K and V caches plus buffers

    max_context = int(usable_memory / (bytes_per_token * context_overhead))

    # Apply hard cap - most models don't support extremely large contexts
    # even if memory would allow it
    if max_context > max_context_cap:
        logger.debug(
            f"Computed context {max_context} exceeds cap {max_context_cap}, capping"
        )
        max_context = max_context_cap

    # Round down to nearest power of 2 for better memory alignment
    # Common sizes: 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
    power_of_2 = 1
    while power_of_2 * 2 <= max_context:
        power_of_2 *= 2

    logger.debug(
        f"Memory calculation: available={available_memory_bytes / (1024**3):.2f}GB, "
        f"model={model_size_bytes / (1024**3):.2f}GB, "
        f"usable={usable_memory / (1024**3):.2f}GB, "
        f"max_ctx_computed={max_context}, "
        f"max_ctx_aligned={power_of_2}"
    )

    return power_of_2


def load_model_context_config() -> dict:
    """Load model_context_defaults.yaml configuration.

    Caches the configuration to avoid repeated file I/O.

    Returns:
        dict with 'memory_usage_factor' and 'model_defaults' keys

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    # Find config file relative to this module
    config_path = (
        Path(__file__).parent.parent / "config" / "model_context_defaults.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Context config file not found: {config_path}. "
            "Create runtimes/universal/config/model_context_defaults.yaml"
        )

    logger.debug(f"Loading context config from: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate config structure
    if "model_defaults" not in config or not isinstance(config["model_defaults"], list):
        raise ValueError(
            "Invalid config: 'model_defaults' must be a list of pattern entries"
        )

    _config_cache = config
    logger.debug(f"Loaded {len(config['model_defaults'])} model patterns from config")
    return config


def match_model_pattern(model_id: str, config: dict) -> int | None:
    """Match model_id against patterns in config using fnmatch.

    Patterns are checked in order, with more specific patterns
    listed first. Returns the n_ctx for the first matching pattern.

    Args:
        model_id: HuggingFace model identifier (e.g., "unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF")
        config: Configuration dict from load_model_context_config()

    Returns:
        n_ctx value for first matching pattern, or None if no match

    Examples:
        >>> config = load_model_context_config()
        >>> match_model_pattern("unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF", config)
        32768
        >>> match_model_pattern("*/Llama-3-8B-GGUF", config)
        8192
    """
    model_defaults = config.get("model_defaults", [])

    for entry in model_defaults:
        pattern = entry.get("pattern")
        n_ctx = entry.get("n_ctx")

        if not pattern or n_ctx is None:
            logger.warning(f"Invalid config entry: {entry}")
            continue

        if fnmatch.fnmatch(model_id, pattern):
            notes = entry.get("notes", "")
            logger.info(
                f"Matched model '{model_id}' to pattern '{pattern}': "
                f"n_ctx={n_ctx} ({notes})"
            )
            return n_ctx

    logger.warning(f"No pattern match found for model: {model_id}")
    return None


def get_default_context_size(
    model_id: str,
    gguf_path: str,
    device: str,
    config_n_ctx: int | None = None,
) -> tuple[int, list[str]]:
    """Determine context size with four-tier priority system.

    Priority order (highest to lowest):
    1. config_n_ctx (from llamafarm.yaml via API) - user's explicit choice
    2. Model's n_ctx_train (training context) - what the model was designed for
    3. Pattern match from model_context_defaults.yaml - known model defaults
    4. Computed max from memory constraints - hardware limitation
    5. Fallback default (2048) - safe conservative value

    All choices are capped by available memory to prevent OOM errors.

    Args:
        model_id: HuggingFace model identifier
        gguf_path: Path to GGUF file
        device: Target device ("cuda", "mps", "cpu")
        config_n_ctx: Optional explicit context size from config

    Returns:
        tuple of (final_n_ctx, warnings_list)
        - final_n_ctx: Determined context size to use
        - warnings_list: List of warning messages (empty if none)

    Examples:
        >>> n_ctx, warnings = get_default_context_size(
        ...     "unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        ...     "/path/to/model.gguf",
        ...     "mps",
        ...     config_n_ctx=32768
        ... )
        >>> n_ctx
        32768  # or lower if memory constrained
    """
    warnings = []

    try:
        # Load configuration
        config = load_model_context_config()
        memory_factor = config.get("memory_usage_factor", 0.8)

        # Get model metadata and compute memory constraints
        metadata = get_gguf_metadata(gguf_path)
        available_memory = get_available_memory(device)
        max_context_from_memory = compute_max_context(
            metadata["file_size_bytes"], available_memory, memory_factor
        )

        logger.info(
            f"Memory-based max context for {model_id}: {max_context_from_memory} "
            f"(model size: {metadata['file_size_mb']:.1f} MB, "
            f"available memory: {available_memory / (1024**3):.2f} GB)"
        )

        # Get model's training context size if available
        n_ctx_train = metadata.get("n_ctx_train")
        if n_ctx_train:
            logger.info(f"Model trained with context size: {n_ctx_train}")

        # Get pattern-based default
        pattern_n_ctx = match_model_pattern(model_id, config)

        # Determine final context size based on priority
        if config_n_ctx is not None:
            # Priority 1: User specified a value - use it but check against memory limit
            if config_n_ctx > max_context_from_memory:
                warning_msg = (
                    f"Requested context size {config_n_ctx} exceeds computed maximum "
                    f"{max_context_from_memory} based on available memory "
                    f"({available_memory / (1024**3):.2f} GB). "
                    f"Using {max_context_from_memory} instead."
                )
                warnings.append(warning_msg)
                final_n_ctx = max_context_from_memory
            else:
                final_n_ctx = config_n_ctx
                logger.info(f"Using configured context size: {final_n_ctx}")

        elif n_ctx_train is not None:
            # Priority 2: Use model's training context, but respect memory limit
            if n_ctx_train > max_context_from_memory:
                warning_msg = (
                    f"Model training context {n_ctx_train} exceeds computed maximum "
                    f"{max_context_from_memory} based on available memory. "
                    f"Using {max_context_from_memory} to prevent OOM."
                )
                warnings.append(warning_msg)
                final_n_ctx = max_context_from_memory
            else:
                final_n_ctx = n_ctx_train
                logger.info(f"Using model's training context size: {final_n_ctx}")

        elif pattern_n_ctx is not None:
            # Priority 3: Use pattern match, but respect memory limit
            if pattern_n_ctx > max_context_from_memory:
                warning_msg = (
                    f"Pattern default context size {pattern_n_ctx} exceeds computed maximum "
                    f"{max_context_from_memory} based on available memory. "
                    f"Using {max_context_from_memory} instead."
                )
                warnings.append(warning_msg)
                final_n_ctx = max_context_from_memory
            else:
                final_n_ctx = pattern_n_ctx
                logger.info(f"Using pattern-matched context size: {final_n_ctx}")

        else:
            # Priority 4: No other source - use computed max or fallback
            if max_context_from_memory >= 2048:
                final_n_ctx = max_context_from_memory
                logger.info(f"Using computed max context: {final_n_ctx}")
            else:
                final_n_ctx = 2048
                warning_msg = (
                    f"Low memory detected. Using fallback context size: {final_n_ctx}"
                )
                warnings.append(warning_msg)

        # Final sanity check - ensure we have at least 512 tokens
        if final_n_ctx < 512:
            warning_msg = (
                f"Computed context size {final_n_ctx} is very low. "
                "Using minimum of 512 tokens."
            )
            warnings.append(warning_msg)
            final_n_ctx = 512

        return final_n_ctx, warnings

    except Exception as e:
        # If anything fails, use safe fallback
        error_msg = f"Error computing context size: {e}. Using fallback of 2048."
        logger.error(error_msg, exc_info=True)
        warnings.append(error_msg)
        return 2048, warnings


def clear_config_cache():
    """Clear the configuration cache.

    Useful for testing or when config file is modified at runtime.
    """
    global _config_cache
    _config_cache = None
    logger.debug("Context config cache cleared")
