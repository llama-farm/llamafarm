"""
Resource detection utilities for optimal model preloading.

Detects available CPU cores, RAM, and GPU VRAM to determine safe concurrency
levels for parallel model loading. Prevents OOM errors by computing memory
constraints before attempting to load multiple models simultaneously.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ResourceInfo:
    """System resource information for concurrency planning."""

    cpu_count: int
    """Total CPU cores available"""

    available_ram_gb: float
    """Available system RAM in GB"""

    total_ram_gb: float
    """Total system RAM in GB"""

    device: str
    """Primary device (cuda, mps, or cpu)"""

    gpu_count: int
    """Number of GPUs available (0 for CPU/MPS)"""

    available_vram_gb: float
    """Available GPU VRAM in GB (0 for CPU)"""

    total_vram_gb: float
    """Total GPU VRAM in GB (0 for CPU)"""

    gpu_name: str | None
    """GPU name (None for CPU)"""

    optimal_concurrency: int
    """Recommended parallel loading concurrency"""

    max_concurrency: int
    """Maximum safe concurrency (hard limit)"""


def get_available_ram_gb() -> tuple[float, float]:
    """Get available and total system RAM in GB.

    Returns:
        Tuple of (available_gb, total_gb)
    """
    try:
        # Try Linux /proc/meminfo first (works on most Linux systems)
        with open("/proc/meminfo") as f:
            mem_info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    value_kb = int(parts[1])
                    mem_info[key] = value_kb

            # MemAvailable is more accurate than MemFree (includes cached/buffered memory)
            available_kb = mem_info.get("MemAvailable", mem_info.get("MemFree", 0))
            total_kb = mem_info.get("MemTotal", 0)

            return available_kb / (1024 * 1024), total_kb / (1024 * 1024)
    except (FileNotFoundError, PermissionError, OSError):
        pass

    # Fallback: try psutil if available
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.available / (1024**3), mem.total / (1024**3)
    except ImportError:
        pass

    # Last resort: return conservative estimates
    logger.warning("Could not detect system memory, using conservative estimates")
    return 4.0, 8.0  # Assume 4GB available out of 8GB total


def get_gpu_memory_info(device: str) -> tuple[float, float, str | None, int]:
    """Get GPU memory information.

    Args:
        device: Device string ("cuda", "mps", or "cpu")

    Returns:
        Tuple of (available_vram_gb, total_vram_gb, gpu_name, gpu_count)
    """
    # Import device utilities
    from utils.device import _get_torch

    torch = _get_torch()

    if torch is None or device == "cpu":
        return 0.0, 0.0, None, 0

    if device == "cuda" and torch.cuda.is_available():
        try:
            # Get info from GPU 0 (primary GPU)
            free, total = torch.cuda.mem_get_info(0)
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()

            return free / (1024**3), total / (1024**3), gpu_name, gpu_count
        except Exception as e:
            logger.warning(f"Failed to get CUDA memory info: {e}")
            return 0.0, 0.0, None, 0

    if device == "mps":
        try:
            # MPS doesn't expose memory stats via PyTorch
            # Apple Silicon has unified memory, so we use system RAM
            import platform

            # Get total system memory for Apple Silicon estimate
            available_ram_gb, total_ram_gb = get_available_ram_gb()

            # MPS shares system memory, estimate 70% of RAM is usable for GPU
            # (conservative to account for OS and other processes)
            usable_vram = total_ram_gb * 0.7

            gpu_name = f"Apple Silicon (MPS) - {platform.machine()}"
            return available_ram_gb * 0.7, usable_vram, gpu_name, 1
        except Exception as e:
            logger.warning(f"Failed to get MPS memory info: {e}")
            return 0.0, 0.0, None, 0

    return 0.0, 0.0, None, 0


def get_optimal_concurrency(
    device: str,
    available_ram_gb: float,
    available_vram_gb: float,
    cpu_count: int,
) -> tuple[int, int]:
    """Calculate optimal and maximum concurrency for model loading.

    This uses conservative estimates to avoid OOM errors:
    - Each model load can use 2-4GB RAM during initialization
    - CPU concurrency is limited to prevent thread thrashing
    - GPU concurrency is limited by VRAM availability

    Args:
        device: Target device ("cuda", "mps", or "cpu")
        available_ram_gb: Available system RAM in GB
        available_vram_gb: Available GPU VRAM in GB
        cpu_count: Number of CPU cores

    Returns:
        Tuple of (optimal_concurrency, max_concurrency)
    """
    # Conservative estimate: each model needs ~3GB RAM during load
    # (includes model weights, optimizer states, temp buffers)
    GB_PER_MODEL_LOAD = 3.0

    # Calculate memory-based limits
    ram_limited_concurrency = max(1, int(available_ram_gb / GB_PER_MODEL_LOAD))

    if device == "cuda" and available_vram_gb > 0:
        # On CUDA, VRAM is the primary constraint
        # Models typically need 1.5x their size in VRAM during loading
        vram_limited_concurrency = max(
            1, int(available_vram_gb / (GB_PER_MODEL_LOAD * 1.5))
        )
        memory_limit = min(ram_limited_concurrency, vram_limited_concurrency)
    else:
        # On CPU/MPS, RAM is the constraint
        memory_limit = ram_limited_concurrency

    # CPU-based limit (avoid thread thrashing)
    # Use at most half the CPU cores for model loading
    cpu_limit = max(1, cpu_count // 2)

    # Optimal: conservative to ensure stability
    optimal = min(4, memory_limit, cpu_limit)

    # Maximum: aggressive but still safe
    # Allow up to 8 concurrent loads if memory permits
    maximum = min(8, memory_limit, cpu_count)

    logger.info(
        f"Concurrency calculation: RAM={available_ram_gb:.1f}GB, "
        f"VRAM={available_vram_gb:.1f}GB, CPUs={cpu_count}, "
        f"optimal={optimal}, max={maximum}"
    )

    return optimal, maximum


def get_resource_info(device: str | None = None) -> ResourceInfo:
    """Get comprehensive system resource information.

    Args:
        device: Optional device override. If None, auto-detects optimal device.

    Returns:
        ResourceInfo with system capabilities and recommended concurrency
    """
    # Auto-detect device if not provided
    if device is None:
        from utils.device import get_optimal_device

        device = get_optimal_device()

    # Get CPU info
    cpu_count = multiprocessing.cpu_count()

    # Get memory info
    available_ram_gb, total_ram_gb = get_available_ram_gb()
    available_vram_gb, total_vram_gb, gpu_name, gpu_count = get_gpu_memory_info(device)

    # Calculate concurrency
    optimal, maximum = get_optimal_concurrency(
        device, available_ram_gb, available_vram_gb, cpu_count
    )

    resource_info = ResourceInfo(
        cpu_count=cpu_count,
        available_ram_gb=available_ram_gb,
        total_ram_gb=total_ram_gb,
        device=device,
        gpu_count=gpu_count,
        available_vram_gb=available_vram_gb,
        total_vram_gb=total_vram_gb,
        gpu_name=gpu_name,
        optimal_concurrency=optimal,
        max_concurrency=maximum,
    )

    logger.info(
        f"Resource detection: {device} device, {cpu_count} CPUs, "
        f"{available_ram_gb:.1f}/{total_ram_gb:.1f}GB RAM"
    )
    if gpu_name:
        logger.info(
            f"  GPU: {gpu_name} with {available_vram_gb:.1f}/{total_vram_gb:.1f}GB VRAM"
        )

    return resource_info


def check_memory_available(required_gb: float, device: str | None = None) -> bool:
    """Check if sufficient memory is available for a task.

    Args:
        required_gb: Required memory in GB
        device: Optional device to check (None = auto-detect)

    Returns:
        True if sufficient memory is available, False otherwise
    """
    if device is None:
        from utils.device import get_optimal_device

        device = get_optimal_device()

    available_ram_gb, _ = get_available_ram_gb()

    if device == "cuda":
        available_vram_gb, *_ = get_gpu_memory_info(device)
        # Check both RAM and VRAM
        return available_ram_gb >= required_gb and available_vram_gb >= required_gb
    else:
        # Check RAM only
        return available_ram_gb >= required_gb


def log_resource_summary(resource_info: ResourceInfo) -> None:
    """Log a human-readable summary of system resources.

    Args:
        resource_info: Resource information to summarize
    """
    logger.info("=" * 60)
    logger.info("System Resource Summary")
    logger.info("=" * 60)
    logger.info(f"Device: {resource_info.device}")
    logger.info(f"CPU Cores: {resource_info.cpu_count}")
    logger.info(
        f"RAM: {resource_info.available_ram_gb:.1f}GB available "
        f"/ {resource_info.total_ram_gb:.1f}GB total"
    )

    if resource_info.gpu_name:
        logger.info(f"GPU: {resource_info.gpu_name}")
        if resource_info.gpu_count > 1:
            logger.info(f"  Count: {resource_info.gpu_count} GPUs")
        logger.info(
            f"  VRAM: {resource_info.available_vram_gb:.1f}GB available "
            f"/ {resource_info.total_vram_gb:.1f}GB total"
        )

    logger.info(
        f"Recommended Concurrency: {resource_info.optimal_concurrency} "
        f"(max: {resource_info.max_concurrency})"
    )
    logger.info("=" * 60)


# Environment variable override for concurrency
# Useful for testing or manual tuning
def get_concurrency_override() -> int | None:
    """Get user-specified concurrency override from environment.

    Returns:
        Concurrency value if set, None otherwise
    """
    override = os.environ.get("LF_PRELOAD_CONCURRENCY", "").strip()
    if override:
        try:
            value = int(override)
            if value > 0:
                logger.info(f"Using concurrency override from environment: {value}")
                return value
            else:
                logger.warning(
                    f"Invalid LF_PRELOAD_CONCURRENCY value: {override} "
                    "(must be positive integer)"
                )
        except ValueError:
            logger.warning(
                f"Invalid LF_PRELOAD_CONCURRENCY value: {override} (must be integer)"
            )
    return None
