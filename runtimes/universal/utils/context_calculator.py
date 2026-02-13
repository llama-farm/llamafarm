"""Context size calculator for GGUF models.

Determines optimal context window size based on:
1. User configuration (highest priority)
2. Available memory and model size
3. Model family defaults from config file
4. Fallback defaults
"""

import fnmatch
import logging
from pathlib import Path

import psutil
import torch
import yaml

from utils.gguf_metadata_cache import get_gguf_metadata_cached

logger = logging.getLogger(__name__)

# Cache for config file
_config_cache: dict | None = None


def get_gguf_metadata(gguf_path: str) -> dict:
    """Read GGUF file metadata without loading the full model.

    Uses the shared GGUF metadata cache to avoid redundant file reads.
    The cache is populated once per file and reused by context_calculator,
    jinja_tools, and other modules.

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
    # Use shared cache - single read for all metadata needs
    cached = get_gguf_metadata_cached(gguf_path)

    # Return in legacy format for backward compatibility
    return {
        "file_size_bytes": cached.file_size_bytes,
        "file_size_mb": cached.file_size_mb,
        "n_ctx_train": cached.n_ctx_train,
        "n_layer": cached.n_layer,
        "n_head_kv": cached.n_head_kv,
        "head_k_size": cached.head_k_size,
        "head_v_size": cached.head_v_size,
    }


def get_available_memory(device: str, gpu_index: int | None = None) -> int:
    """Get available memory in bytes for the device.

    Args:
        device: Target device ("cuda", "mps", or "cpu")
        gpu_index: Specific CUDA GPU index. If None, uses GPU 0.

    Returns:
        Available memory in bytes

    Notes:
        - For CUDA: Returns free GPU memory on the specified device
        - For MPS/CPU: Returns available system RAM
    """
    try:
        if device == "cuda" and torch.cuda.is_available():
            idx = gpu_index if gpu_index is not None else 0
            free, total = torch.cuda.mem_get_info(idx)
            logger.debug(
                f"CUDA GPU {idx} memory: {free / (1024**3):.2f} GB free / "
                f"{total / (1024**3):.2f} GB total"
            )
            return free
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


def compute_kv_bytes_per_token(
    n_layer: int,
    n_head_kv: int,
    head_k_size: int,
    head_v_size: int,
) -> int:
    """Compute exact KV cache bytes per token from model architecture.

    The KV cache stores key and value tensors for every layer, using f16 precision
    (2 bytes per element). This is the dominant memory cost that scales with context.

    Args:
        n_layer: Number of transformer layers (block_count)
        n_head_kv: Number of key-value attention heads
        head_k_size: Dimension of each key head
        head_v_size: Dimension of each value head

    Returns:
        Bytes of KV cache needed per token of context
    """
    # K cache per token: n_layer * n_head_kv * head_k_size * sizeof(f16)
    # V cache per token: n_layer * n_head_kv * head_v_size * sizeof(f16)
    return n_layer * n_head_kv * (head_k_size + head_v_size) * 2


# Fallback estimate when GGUF architecture metadata isn't available.
# Deliberately conservative (overestimates cost) to prevent OOM.
# Actual KV cache costs range from ~18 KB/token (1.5B) to ~320 KB/token (70B).
# 256 KB covers most 7B+ models safely; smaller models just get less context than
# they could handle, which is preferable to OOM.
_FALLBACK_BYTES_PER_TOKEN = 256 * 1024  # 256 KB


def compute_max_context(
    model_size_bytes: int,
    available_memory_bytes: int,
    memory_factor: float = 0.8,
    max_context_cap: int = 131072,
    n_layer: int | None = None,
    n_head_kv: int | None = None,
    head_k_size: int | None = None,
    head_v_size: int | None = None,
) -> int:
    """Compute maximum safe context size based on available memory.

    Uses model architecture metadata (when available) to compute the exact
    KV cache cost per token, rather than relying on a fixed estimate.

    Args:
        model_size_bytes: Size of model file in bytes
        available_memory_bytes: Available memory on target device
        memory_factor: Fraction of available memory to use (default 0.8)
        max_context_cap: Hard upper limit for context size (default 131072/128K).
            Most models don't support more than 128K context even with
            sufficient memory.
        n_layer: Number of transformer layers (from GGUF metadata)
        n_head_kv: Number of key-value attention heads (from GGUF metadata)
        head_k_size: Dimension of each key head (from GGUF metadata)
        head_v_size: Dimension of each value head (from GGUF metadata)

    Returns:
        Maximum safe context size (number of tokens)
    """
    # Calculate usable memory after loading model
    usable_memory = (available_memory_bytes * memory_factor) - model_size_bytes

    if usable_memory <= 0:
        logger.warning(
            f"Model size ({model_size_bytes / (1024**3):.2f} GB) exceeds "
            f"available memory budget. Using minimal context size."
        )
        return 512  # Minimal context

    # Compute per-token memory cost
    has_arch_params = all(
        v is not None for v in [n_layer, n_head_kv, head_k_size, head_v_size]
    )
    if has_arch_params:
        kv_bytes = compute_kv_bytes_per_token(
            n_layer, n_head_kv, head_k_size, head_v_size
        )
        # Add 30% overhead for compute buffers and activation tensors
        bytes_per_token = int(kv_bytes * 1.3)
        logger.debug(
            f"KV cache from architecture: {kv_bytes} bytes/token "
            f"(n_layer={n_layer}, n_head_kv={n_head_kv}, "
            f"head_k={head_k_size}, head_v={head_v_size}), "
            f"with overhead: {bytes_per_token} bytes/token"
        )
    else:
        bytes_per_token = _FALLBACK_BYTES_PER_TOKEN
        logger.debug(
            f"Architecture metadata unavailable, using fallback: "
            f"{bytes_per_token} bytes/token"
        )

    max_context = int(usable_memory / bytes_per_token)

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
        f"bytes_per_token={bytes_per_token}, "
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
    gpu_index: int | None = None,
    available_memory_override: int | None = None,
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
        gpu_index: Specific CUDA GPU index for memory queries. If None, uses GPU 0.
        available_memory_override: Pre-computed available memory in bytes.
            When provided, skips the ``get_available_memory()`` query.  Used
            for multi-GPU splits where the effective memory is the combined
            free VRAM across all participating devices.

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
        if available_memory_override is not None:
            available_memory = available_memory_override
        else:
            available_memory = get_available_memory(device, gpu_index=gpu_index)
        max_context_from_memory = compute_max_context(
            metadata["file_size_bytes"],
            available_memory,
            memory_factor,
            n_layer=metadata.get("n_layer"),
            n_head_kv=metadata.get("n_head_kv"),
            head_k_size=metadata.get("head_k_size"),
            head_v_size=metadata.get("head_v_size"),
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
