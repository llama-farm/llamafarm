"""
Device detection and optimization utilities.

PyTorch is optional - this module provides fallback behavior for GGUF-only
deployments where torch is not installed. llama.cpp has its own GPU detection
independent of PyTorch.
"""

from __future__ import annotations

import logging
import platform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch as torch_type

logger = logging.getLogger(__name__)

# Cached torch module reference (lazy loaded)
_torch: torch_type | None = None
_torch_available: bool | None = None


def _get_torch() -> torch_type | None:
    """Lazy-load torch module. Returns None if not installed."""
    global _torch, _torch_available

    if _torch_available is None:
        try:
            import torch

            _torch = torch
            _torch_available = True
            logger.debug(f"PyTorch {torch.__version__} loaded successfully")
        except ImportError:
            _torch = None
            _torch_available = False
            logger.info("PyTorch not installed - encoder models will not be available")

    return _torch


def is_torch_available() -> bool:
    """Check if PyTorch is available without importing it."""
    _get_torch()
    return _torch_available or False


def get_optimal_device() -> str:
    """
    Detect the optimal device for the current platform.

    Returns:
        str: Device name ("cuda", "mps", or "cpu")

    Note:
        If PyTorch is not installed, always returns "cpu".
        This allows GGUF models to still use GPU via llama.cpp's own detection.
    """
    import os

    # Allow forcing CPU via environment variable
    force_cpu = os.environ.get("TRANSFORMERS_FORCE_CPU", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if force_cpu:
        logger.info("Forcing CPU device (TRANSFORMERS_FORCE_CPU=1)")
        return "cpu"

    # Try to use PyTorch for device detection
    torch = _get_torch()
    if torch is None:
        logger.info("PyTorch not available - using CPU for encoder models")
        return "cpu"

    # Check for CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return "cuda"

    # Check for MPS (Apple Silicon)
    # Note: MPS has a 4GB temporary buffer limit which can cause issues with some models
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Check if user wants to skip MPS due to known limitations
        skip_mps = os.environ.get("TRANSFORMERS_SKIP_MPS", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if skip_mps:
            logger.info("Skipping MPS (TRANSFORMERS_SKIP_MPS=1), using CPU")
            return "cpu"
        logger.info("MPS (Apple Silicon) available")
        logger.warning(
            "MPS has a 4GB temporary buffer limit. Set TRANSFORMERS_SKIP_MPS=1 to use CPU if you encounter errors."
        )
        return "mps"

    # Fallback to CPU
    logger.info("Using CPU (no GPU acceleration)")
    return "cpu"


def get_device_info() -> dict:
    """
    Get detailed device information.

    Returns:
        dict: Device information including platform, acceleration, memory
    """
    device = get_optimal_device()
    torch = _get_torch()

    info = {
        "device": device,
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__ if torch else "not installed",
        "torch_available": torch is not None,
    }

    if torch is not None:
        if device == "cuda":
            info.update(
                {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                    "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                }
            )
        elif device == "mps":
            info.update(
                {
                    "gpu_name": "Apple Silicon (MPS)",
                    "architecture": platform.machine(),
                }
            )

    return info


def get_gguf_gpu_layers() -> int:
    """
    Get the number of GPU layers to use for GGUF models.

    IMPORTANT: llama.cpp has its own GPU detection (CUDA, Metal, Vulkan, etc.)
    that is independent of PyTorch. We should always try to use GPU layers (-1)
    and let llama.cpp fall back to CPU if no GPU backend is available.
    This allows users with CPU-only PyTorch but GPU llama.cpp to get acceleration.

    Returns:
        int: Number of GPU layers (-1 for all layers on GPU, 0 for CPU only)
    """
    import os

    force_cpu = os.environ.get("LLAMAFARM_GGUF_FORCE_CPU", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if force_cpu:
        logger.info("Configuring for CPU-only inference (LLAMAFARM_GGUF_FORCE_CPU=1)")
        return 0

    # Use all layers on GPU - llama.cpp will use whatever backend is available
    # (CUDA, Metal, Vulkan, etc.) and fall back to CPU if none are available
    logger.info(
        "Configuring for GPU acceleration (all layers on GPU, llama.cpp will "
        "auto-detect available backends)"
    )
    return -1
