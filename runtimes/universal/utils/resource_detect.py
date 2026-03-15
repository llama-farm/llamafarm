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


def _get_process_cgroup_path() -> tuple[str | None, str | None]:
    """Return (cgroup_v2_dir, cgroup_v1_memory_dir) for the current process.

    Reads /proc/self/cgroup to find the cgroup slice for this process, then
    resolves the mount point from /proc/self/mountinfo so we read the correct
    nested path rather than assuming the hierarchy root.
    """
    cgroup_v2_dir: str | None = None
    cgroup_v1_memory_dir: str | None = None

    # --- Resolve mount points from mountinfo ---
    v2_mount: str | None = None
    v1_memory_mount: str | None = None
    try:
        with open("/proc/self/mountinfo") as f:
            for line in f:
                parts = line.split()
                # fields: mount-id parent-id major:minor root mount-point mount-options [optional...] - fstype source super-options
                # We need at minimum 10 fields; the separator "-" is somewhere after field 6.
                if len(parts) < 10:
                    continue
                mount_point = parts[4]
                # Find the separator "-" and get fstype after it
                try:
                    sep = parts.index("-", 6)
                    fstype = parts[sep + 1]
                except (ValueError, IndexError):
                    continue
                if fstype == "cgroup2" and v2_mount is None:
                    v2_mount = mount_point
                elif fstype == "cgroup" and v1_memory_mount is None:
                    # Check super-options for "memory"
                    super_options = parts[sep + 3] if sep + 3 < len(parts) else ""
                    if "memory" in super_options.split(","):
                        v1_memory_mount = mount_point
    except (FileNotFoundError, PermissionError, OSError):
        pass

    # --- Resolve process-specific cgroup slice ---
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                # Format: hierarchy-id:controllers:cgroup-path
                parts = line.strip().split(":", 2)
                if len(parts) != 3:
                    continue
                hier_id, controllers, cgroup_path = parts
                if hier_id == "0":
                    # cgroup v2 unified hierarchy
                    if v2_mount is not None:
                        cgroup_v2_dir = v2_mount + cgroup_path
                    else:
                        cgroup_v2_dir = "/sys/fs/cgroup" + cgroup_path
                elif "memory" in controllers.split(","):
                    # cgroup v1 memory controller
                    if v1_memory_mount is not None:
                        cgroup_v1_memory_dir = v1_memory_mount + cgroup_path
                    else:
                        cgroup_v1_memory_dir = "/sys/fs/cgroup/memory" + cgroup_path
    except (FileNotFoundError, PermissionError, OSError):
        pass

    return cgroup_v2_dir, cgroup_v1_memory_dir


def _get_cgroup_memory_info() -> tuple[int | None, int | None]:
    """Evaluate cgroup memory hierarchy and return (effective_limit, available_bytes).

    Walks up the hierarchy to find the tightest bottleneck (limit - usage).
    Returns (limit, available) or (None, None) if no limits are found.
    """
    cgroup_v2_dir, cgroup_v1_memory_dir = _get_process_cgroup_path()

    effective_limit = None
    min_available = None

    # cgroup v2
    if cgroup_v2_dir:
        path = str(cgroup_v2_dir)
        while path and path.startswith(
            "/sys/fs/cgroup"
        ):  # Safety check for hierarchy walk
            try:
                # Read limit at this level
                limit = None
                limit_file = os.path.join(path, "memory.max")
                if os.path.exists(limit_file):
                    with open(limit_file) as f:
                        val = f.read().strip()
                        if val != "max":
                            limit = int(val)

                # If we found a limit, calculate available at this level
                if limit is not None:
                    if effective_limit is None or limit < effective_limit:
                        effective_limit = limit

                    usage_file = os.path.join(path, "memory.current")
                    if os.path.exists(usage_file):
                        try:
                            with open(usage_file) as f:
                                usage = int(f.read().strip())
                                available = max(0, limit - usage)
                                if min_available is None or available < min_available:
                                    min_available = available
                        except (ValueError, OSError):
                            # usage unreadable: conservatively assume 0 available if limit exists
                            if min_available is None or min_available > 0:
                                min_available = 0
            except (ValueError, OSError):
                pass

            # Walk up. If we are at the root (/sys/fs/cgroup), dirname is the same
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent

    # cgroup v1 (typically we only care about the specific controller mount for the process)
    if cgroup_v1_memory_dir and effective_limit is None:
        try:
            limit = None
            limit_file = os.path.join(cgroup_v1_memory_dir, "memory.limit_in_bytes")
            if os.path.exists(limit_file):
                with open(limit_file) as f:
                    val = int(f.read().strip())
                    # cgroup v1 'max' is 0x7FFFFFFFFFFFF000 or 9223372036854771712
                    if val < 9223372036854771712:
                        limit = val

            if limit is not None:
                effective_limit = limit
                usage_file = os.path.join(cgroup_v1_memory_dir, "memory.usage_in_bytes")
                if os.path.exists(usage_file):
                    try:
                        with open(usage_file) as f:
                            usage = int(f.read().strip())
                            available = max(0, limit - usage)
                            min_available = available
                    except (ValueError, OSError):
                        min_available = 0
        except (ValueError, OSError):
            pass

    return effective_limit, min_available


def get_available_ram_gb() -> tuple[float, float]:
    """Get available and total system RAM in GB.

    Returns:
        Tuple of (available_gb, total_gb)
    """
    available_gb = -1.0
    total_gb = -1.0

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

            available_gb = float(available_kb / (1024 * 1024))
            total_gb = float(total_kb / (1024 * 1024))
    except (FileNotFoundError, PermissionError, OSError):
        pass

    if available_gb < 0 or total_gb < 0:
        # Fallback: try psutil if available
        try:
            import psutil

            mem = psutil.virtual_memory()
            available_gb = float(mem.available / (1024**3))
            total_gb = float(mem.total / (1024**3))
        except ImportError:
            pass

    if available_gb < 0 or total_gb < 0:
        # Last resort: return conservative estimates
        logger.warning("Could not detect system memory, using conservative estimates")
        available_gb, total_gb = 4.0, 8.0

    # Check for container/cgroup limits (Linux only)
    cgroup_limit_bytes, cgroup_available_bytes = _get_cgroup_memory_info()
    if cgroup_limit_bytes is not None:
        cgroup_limit_gb = cgroup_limit_bytes / (1024**3)
        if cgroup_limit_gb < total_gb:
            # Always cap total by the cgroup limit, even if usage is unreadable.
            # Not doing this would overestimate safe concurrency in containers.
            total_gb = min(total_gb, cgroup_limit_gb)

            if cgroup_available_bytes is not None:
                cgroup_available_gb = cgroup_available_bytes / (1024**3)
                # Use the more restrictive of host available vs cgroup available
                available_gb = min(available_gb, cgroup_available_gb)
            else:
                # Available unreadable (but limit exists): conservativesly cap to limit
                available_gb = min(available_gb, cgroup_limit_gb)

    return available_gb, total_gb


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
