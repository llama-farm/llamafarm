"""
Device detection and optimization utilities.
"""

import torch
import platform
import logging

logger = logging.getLogger(__name__)


def get_optimal_device() -> str:
    """
    Detect the optimal device for the current platform.

    Returns:
        str: Device name ("cuda", "mps", or "cpu")
    """
    import os

    # Allow forcing CPU via environment variable
    force_cpu = os.environ.get("TRANSFORMERS_FORCE_CPU", "").lower() in ("1", "true", "yes")
    if force_cpu:
        logger.info("Forcing CPU device (TRANSFORMERS_FORCE_CPU=1)")
        return "cpu"

    # Check for CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return "cuda"

    # Check for MPS (Apple Silicon)
    # Note: MPS has a 4GB temporary buffer limit which can cause issues with some models
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Check if user wants to skip MPS due to known limitations
        skip_mps = os.environ.get("TRANSFORMERS_SKIP_MPS", "").lower() in ("1", "true", "yes")
        if skip_mps:
            logger.info("Skipping MPS (TRANSFORMERS_SKIP_MPS=1), using CPU")
            return "cpu"
        logger.info("MPS (Apple Silicon) available")
        logger.warning("MPS has a 4GB temporary buffer limit. Set TRANSFORMERS_SKIP_MPS=1 to use CPU if you encounter errors.")
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

    info = {
        "device": device,
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    if device == "cuda":
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
            "gpu_memory_allocated": torch.cuda.memory_allocated(0),
        })
    elif device == "mps":
        info.update({
            "gpu_name": "Apple Silicon (MPS)",
            "architecture": platform.machine(),
        })

    return info
