"""GPU allocation for multi-model, multi-GPU GGUF inference.

Selects the optimal GPU for each model load, preferring single-GPU placement
(split_mode=NONE) over multi-GPU splitting. Prevents OOM crashes by estimating
VRAM requirements before loading.

llama.cpp's default split_mode=LAYER distributes every model across ALL visible
Vulkan/CUDA devices proportionally. This is problematic for multi-model scenarios:
a second model may OOM on a weaker GPU that's already partially filled.

This module queries actual free VRAM per device via torch.cuda.mem_get_info()
and routes each model to the single GPU with the most headroom. Multi-GPU split
is only used as a fallback when no single GPU can fit the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from utils.context_calculator import compute_kv_bytes_per_token

logger = logging.getLogger(__name__)

# Split mode constants (match llama.cpp enum llama_split_mode)
SPLIT_MODE_NONE = 0  # Entire model on main_gpu
SPLIT_MODE_LAYER = 1  # Split layers across GPUs
SPLIT_MODE_ROW = 2  # Split rows within layers


class InsufficientVRAMError(RuntimeError):
    """Raised when no GPU configuration has enough VRAM for the model.

    The user-facing message is intentionally generic to avoid exposing
    internal GPU inventory details.  Detailed diagnostics are stored in
    ``self.gpu_details`` for server-side logging.
    """

    def __init__(self, message: str, gpu_details: str = ""):
        super().__init__(message)
        self.gpu_details = gpu_details


@dataclass
class GPUDevice:
    """Information about a single GPU device."""

    index: int
    name: str
    total_vram: int  # bytes
    free_vram: int  # bytes


@dataclass
class GPUAllocation:
    """Result of GPU allocation for a model load."""

    gpu_index: int  # Primary GPU (-1 if CPU)
    split_mode: int  # SPLIT_MODE_* constant
    main_gpu: int  # main_gpu param for llama.cpp
    tensor_split: list[float] | None  # Proportions for multi-GPU split
    estimated_vram: int  # Estimated VRAM usage in bytes
    total_free_vram: int  # Combined free VRAM across viable GPUs (bytes)


def enumerate_gpus() -> list[GPUDevice]:
    """Enumerate available CUDA/Vulkan GPUs with free VRAM.

    Uses torch.cuda APIs which reflect the same physical devices as Vulkan
    (Vulkan0 = cuda:0, Vulkan1 = cuda:1, etc.).

    Returns:
        List of GPUDevice, empty if no CUDA GPUs or torch unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        devices = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            name = torch.cuda.get_device_name(i)
            devices.append(
                GPUDevice(index=i, name=name, total_vram=total, free_vram=free)
            )
            logger.debug(
                f"GPU {i} ({name}): {free / (1024**3):.2f} GiB free / "
                f"{total / (1024**3):.2f} GiB total"
            )

        return devices
    except ImportError:
        logger.debug("PyTorch not available, skipping GPU enumeration")
        return []
    except Exception as e:
        logger.warning(f"Error enumerating GPUs: {e}")
        return []


def estimate_model_vram(
    model_size_bytes: int,
    n_ctx: int,
    n_gpu_layers: int,
    total_layers: int | None = None,
    n_layer: int | None = None,
    n_head_kv: int | None = None,
    head_k_size: int | None = None,
    head_v_size: int | None = None,
) -> int:
    """Estimate total VRAM needed for a GGUF model.

    Args:
        model_size_bytes: GGUF file size in bytes (approximates GPU weight size).
        n_ctx: Context window size (tokens).
        n_gpu_layers: Number of layers to offload (-1 or 999 = all).
        total_layers: Total layer count in the model (for partial offload).
        n_layer: Number of transformer layers (from GGUF metadata).
        n_head_kv: Number of KV attention heads (from GGUF metadata).
        head_k_size: Key head dimension (from GGUF metadata).
        head_v_size: Value head dimension (from GGUF metadata).

    Returns:
        Estimated VRAM in bytes.
    """
    # Model weights on GPU
    if n_gpu_layers == 0:
        # CPU only
        return 0

    if total_layers and n_gpu_layers > 0 and n_gpu_layers < 999:
        # Partial offload: scale weight size proportionally
        gpu_weight_bytes = int(model_size_bytes * (n_gpu_layers / total_layers))
    else:
        # Full offload
        gpu_weight_bytes = model_size_bytes

    # KV cache
    has_arch = all(
        v is not None for v in [n_layer, n_head_kv, head_k_size, head_v_size]
    )
    if has_arch:
        kv_bytes_per_token = compute_kv_bytes_per_token(
            n_layer, n_head_kv, head_k_size, head_v_size
        )
    else:
        # Conservative fallback: 256 KB/token (matches context_calculator)
        kv_bytes_per_token = 256 * 1024

    kv_cache_bytes = kv_bytes_per_token * n_ctx

    # Total with 20% overhead for compute buffers and scratch space
    total = int((gpu_weight_bytes + kv_cache_bytes) * 1.2)

    logger.debug(
        f"VRAM estimate: weights={gpu_weight_bytes / (1024**3):.2f} GiB, "
        f"KV cache={kv_cache_bytes / (1024**3):.2f} GiB, "
        f"total (with overhead)={total / (1024**3):.2f} GiB"
    )

    return total


def allocate_gpu(estimated_vram: int, gpus: list[GPUDevice]) -> GPUAllocation:
    """Select the optimal GPU(s) for a model load.

    Strategy:
    1. Try single-GPU placement on the GPU with most free VRAM (split_mode=NONE).
    2. Fall back to multi-GPU layer splitting if no single GPU fits.
    3. Raise InsufficientVRAMError if even combined VRAM is insufficient.

    Args:
        estimated_vram: Estimated VRAM needed (from estimate_model_vram).
        gpus: Available GPUs (from enumerate_gpus).

    Returns:
        GPUAllocation with parameters to pass to llama.cpp.

    Raises:
        InsufficientVRAMError: If no GPU configuration can fit the model.
    """
    if not gpus:
        raise InsufficientVRAMError("No GPUs available")

    # 10% safety margin for driver overhead and estimation error
    required = int(estimated_vram * 1.1)

    # Sort by free VRAM descending
    sorted_gpus = sorted(gpus, key=lambda g: g.free_vram, reverse=True)

    # Strategy 1: Single GPU placement
    best = sorted_gpus[0]
    if best.free_vram >= required:
        logger.info(
            f"Allocating model to GPU {best.index} ({best.name}): "
            f"{estimated_vram / (1024**3):.2f} GiB needed, "
            f"{best.free_vram / (1024**3):.2f} GiB free"
        )
        return GPUAllocation(
            gpu_index=best.index,
            split_mode=SPLIT_MODE_NONE,
            main_gpu=best.index,
            tensor_split=None,
            estimated_vram=estimated_vram,
            total_free_vram=best.free_vram,
        )

    # Strategy 2: Multi-GPU split
    # Exclude GPUs with too little free VRAM to carry their share of
    # non-splittable overhead (compute buffers, scratch space). A GPU
    # needs at least 512 MiB free to participate usefully in a split.
    min_participation = 512 * 1024**2  # 512 MiB
    # Per-GPU fixed overhead for compute buffers and scratch space that
    # llama.cpp allocates on each device regardless of split fraction.
    per_gpu_overhead = 256 * 1024**2  # 256 MiB
    viable_gpus = [g for g in gpus if g.free_vram >= min_participation]

    # Iteratively prune GPUs whose free VRAM cannot cover their
    # proportional share of the model plus per-device fixed overhead.
    # Use estimated_vram (not required) for per-device checks — the 10%
    # safety margin is already enforced globally via total_free >= required.
    pruned = True
    while pruned and len(viable_gpus) > 1:
        pruned = False
        total_free = sum(g.free_vram for g in viable_gpus)
        if total_free < required:
            break
        for g in viable_gpus:
            share = (g.free_vram / total_free) * estimated_vram
            if share + per_gpu_overhead > g.free_vram:
                viable_gpus = [v for v in viable_gpus if v.index != g.index]
                pruned = True
                break

    total_free = sum(g.free_vram for g in viable_gpus)

    if total_free >= required and len(viable_gpus) > 1:
        # Build split proportions only for viable GPUs, zero out excluded ones
        by_index = sorted(gpus, key=lambda g: g.index)
        viable_indices = {g.index for g in viable_gpus}
        raw_split = [
            float(g.free_vram) if g.index in viable_indices else 0.0 for g in by_index
        ]
        total = sum(raw_split)
        tensor_split = [v / total for v in raw_split]

        gpu_desc = ", ".join(
            f"GPU {g.index} ({g.name}): {g.free_vram / (1024**3):.2f} GiB free"
            for g in viable_gpus
        )
        logger.info(
            f"Model requires multi-GPU split: "
            f"{estimated_vram / (1024**3):.2f} GiB needed, "
            f"no single GPU has enough. Splitting across: {gpu_desc}"
        )
        return GPUAllocation(
            gpu_index=sorted_gpus[0].index,
            split_mode=SPLIT_MODE_LAYER,
            main_gpu=sorted_gpus[0].index,
            tensor_split=tensor_split,
            estimated_vram=estimated_vram,
            total_free_vram=total_free,
        )

    # Strategy 3: Insufficient VRAM
    gpu_desc = "\n".join(
        f"  GPU {g.index} ({g.name}): {g.free_vram / (1024**3):.2f} GiB free / "
        f"{g.total_vram / (1024**3):.2f} GiB total"
        for g in sorted_gpus
    )
    details = (
        f"Estimated VRAM needed: {estimated_vram / (1024**3):.2f} GiB\n"
        f"Available GPUs:\n{gpu_desc}\n"
        f"Combined free: {total_free / (1024**3):.2f} GiB"
    )
    logger.error(f"Insufficient GPU memory to load model.\n{details}")
    raise InsufficientVRAMError(
        "Insufficient GPU memory to load model. "
        "Consider unloading other models, reducing context size, "
        "or using a smaller quantization.",
        gpu_details=details,
    )


def get_llama_gpu_params(
    model_size_bytes: int,
    n_ctx: int,
    n_gpu_layers: int,
    total_layers: int | None = None,
    n_layer: int | None = None,
    n_head_kv: int | None = None,
    head_k_size: int | None = None,
    head_v_size: int | None = None,
) -> dict:
    """Get GPU parameters for a Llama() constructor call.

    Convenience function that enumerates GPUs, estimates VRAM, and allocates.
    Returns a dict of keyword arguments to pass to Llama().

    Args:
        model_size_bytes: GGUF file size in bytes.
        n_ctx: Context window size.
        n_gpu_layers: Number of layers to offload.
        total_layers: Total layers in the model.
        n_layer: Transformer layer count (GGUF metadata).
        n_head_kv: KV head count (GGUF metadata).
        head_k_size: Key head dimension (GGUF metadata).
        head_v_size: Value head dimension (GGUF metadata).

    Returns:
        Dict with keys: main_gpu, split_mode, tensor_split, gpu_index.
        Empty dict if CUDA is not available (caller should use defaults).

    Raises:
        InsufficientVRAMError: If no GPU can fit the model.
    """
    if n_gpu_layers == 0:
        # CPU-only, no GPU allocation needed
        return {}

    gpus = enumerate_gpus()
    if not gpus:
        # No CUDA GPUs - llama.cpp will use Vulkan/Metal/CPU on its own
        return {}

    estimated_vram = estimate_model_vram(
        model_size_bytes=model_size_bytes,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        total_layers=total_layers,
        n_layer=n_layer,
        n_head_kv=n_head_kv,
        head_k_size=head_k_size,
        head_v_size=head_v_size,
    )

    allocation = allocate_gpu(estimated_vram, gpus)

    result = {
        "main_gpu": allocation.main_gpu,
        "split_mode": allocation.split_mode,
        "gpu_index": allocation.gpu_index,
        "total_free_vram": allocation.total_free_vram,
    }
    if allocation.tensor_split is not None:
        result["tensor_split"] = allocation.tensor_split

    return result
