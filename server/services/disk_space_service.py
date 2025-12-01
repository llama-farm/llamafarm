"""Disk space checking service for model downloads and system health."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import psutil

from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger(__name__)

# Thresholds (not configurable in Phase 1)
WARNING_THRESHOLD_PERCENT = 10.0  # Warn when < 10% free
CRITICAL_THRESHOLD_BYTES = 100 * 1024 * 1024  # Block when < 100MB free


@dataclass
class DiskSpaceInfo:
    """Disk space information for a given path."""

    total_bytes: int
    used_bytes: int
    free_bytes: int
    path: str
    percent_free: float


@dataclass
class ValidationResult:
    """Result of disk space validation for a model download."""

    can_download: bool
    warning: bool  # True if < 10% free
    available_bytes: int
    required_bytes: int
    message: str
    cache_info: DiskSpaceInfo
    system_info: DiskSpaceInfo


class DiskSpaceService:
    """Service for checking disk space and validating downloads."""

    @staticmethod
    def check_disk_space(path: str | Path) -> DiskSpaceInfo:
        """Check disk space at the given path.

        Args:
            path: Path to check (can be file or directory)

        Returns:
            DiskSpaceInfo with space information

        Raises:
            OSError: If path cannot be accessed
        """
        path_obj = Path(path)
        # Resolve to actual path (handles symlinks)
        try:
            resolved_path = path_obj.resolve()
        except (OSError, RuntimeError):
            # If resolve fails, use original path
            resolved_path = path_obj

        try:
            usage = psutil.disk_usage(str(resolved_path))
            percent_free = (usage.free / usage.total) * 100.0 if usage.total > 0 else 0.0

            return DiskSpaceInfo(
                total_bytes=usage.total,
                used_bytes=usage.used,
                free_bytes=usage.free,
                path=str(resolved_path),
                percent_free=percent_free,
            )
        except OSError as e:
            logger.warning(f"Failed to check disk space at {resolved_path}: {e}")
            raise

    @staticmethod
    def get_cache_directory() -> Path:
        """Get HuggingFace cache directory location.

        Returns:
            Path to HuggingFace cache directory
        """
        try:
            from huggingface_hub.constants import HF_HOME

            cache_dir = Path(HF_HOME) / "hub"
            return cache_dir
        except ImportError:
            # Fallback if huggingface_hub not available
            pass

        # Fallback to default location
        home = Path.home()
        if os.name == "nt":  # Windows
            cache_dir = home / ".cache" / "huggingface" / "hub"
        else:  # Unix-like
            cache_dir = home / ".cache" / "huggingface" / "hub"

        return cache_dir

    @staticmethod
    def get_system_disk() -> Path:
        """Get system disk root path.

        Returns:
            Path to system disk root
        """
        if os.name == "nt":  # Windows
            return Path("C:\\")
        else:  # Unix-like
            return Path("/")

    @staticmethod
    def check_both_disks() -> tuple[DiskSpaceInfo, DiskSpaceInfo]:
        """Check disk space for both cache and system disk.

        Returns:
            Tuple of (cache_info, system_info)

        Raises:
            OSError: If either check fails
        """
        cache_dir = DiskSpaceService.get_cache_directory()
        system_disk = DiskSpaceService.get_system_disk()

        cache_info = DiskSpaceService.check_disk_space(cache_dir)
        system_info = DiskSpaceService.check_disk_space(system_disk)

        return cache_info, system_info

    @staticmethod
    def get_model_size(model_id: str) -> int | None:
        """Get estimated size for a model from HuggingFace API.

        Args:
            model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")

        Returns:
            Estimated size in bytes, or None if unavailable
        """
        try:
            from huggingface_hub import HfApi

            # Parse model ID to extract quantization if present
            from llamafarm_common import parse_model_with_quantization

            base_model_id, _ = parse_model_with_quantization(model_id)
            api = HfApi()

            # Try to get model info for size estimation
            try:
                model_info = api.model_info(base_model_id)
                if model_info:
                    # Check for safetensors files with sizes
                    if hasattr(model_info, "safetensors") and model_info.safetensors:
                        total_size = 0
                        for st_file in model_info.safetensors:
                            if hasattr(st_file, "size") and st_file.size:
                                total_size += st_file.size
                        if total_size > 0:
                            return total_size

                    # Check for files attribute
                    if hasattr(model_info, "files") and model_info.files:
                        total_size = 0
                        for file_info in model_info.files:
                            if hasattr(file_info, "size") and file_info.size:
                                total_size += file_info.size
                        if total_size > 0:
                            return total_size

            except Exception as e:
                logger.debug(f"Could not get model info for {base_model_id}: {e}")

        except ImportError:
            logger.warning("huggingface_hub not available for model size estimation")
        except Exception as e:
            logger.warning(f"Error estimating model size for {model_id}: {e}")

        return None

    @staticmethod
    def validate_space_for_download(model_id: str) -> ValidationResult:
        """Validate if there's sufficient disk space for a model download.

        Args:
            model_id: HuggingFace model identifier

        Returns:
            ValidationResult with validation status and messages
        """
        try:
            cache_info, system_info = DiskSpaceService.check_both_disks()
        except OSError as e:
            logger.warning(f"Failed to check disk space: {e}")
            # Graceful degradation: assume we can download if check fails
            return ValidationResult(
                can_download=True,
                warning=False,
                available_bytes=0,
                required_bytes=0,
                message="Disk space check unavailable, proceeding with download",
                cache_info=DiskSpaceInfo(0, 0, 0, "", 0.0),
                system_info=DiskSpaceInfo(0, 0, 0, "", 0.0),
            )

        # Get model size estimate
        model_size = DiskSpaceService.get_model_size(model_id)
        if model_size is None:
            # If we can't estimate size, use cache disk free space as available
            # and don't block, but warn if space is low
            available_bytes = min(cache_info.free_bytes, system_info.free_bytes)
            warning = (
                cache_info.percent_free < WARNING_THRESHOLD_PERCENT
                or system_info.percent_free < WARNING_THRESHOLD_PERCENT
            )

            message = "Model size unavailable, checking available space"
            if warning:
                message += f" - Low disk space detected ({cache_info.percent_free:.1f}% cache, {system_info.percent_free:.1f}% system)"

            return ValidationResult(
                can_download=True,
                warning=warning,
                available_bytes=available_bytes,
                required_bytes=0,
                message=message,
                cache_info=cache_info,
                system_info=system_info,
            )

        # Use the smaller of cache or system free space
        available_bytes = min(cache_info.free_bytes, system_info.free_bytes)

        # Check critical threshold (absolute minimum)
        if available_bytes < CRITICAL_THRESHOLD_BYTES:
            return ValidationResult(
                can_download=False,
                warning=False,
                available_bytes=available_bytes,
                required_bytes=model_size,
                message=(
                    f"Insufficient disk space. Required: {model_size / (1024**3):.2f} GB, "
                    f"Available: {available_bytes / (1024**3):.2f} GB. "
                    f"Please free up space before downloading."
                ),
                cache_info=cache_info,
                system_info=system_info,
            )

        # Check if model fits
        if available_bytes < model_size:
            return ValidationResult(
                can_download=False,
                warning=False,
                available_bytes=available_bytes,
                required_bytes=model_size,
                message=(
                    f"Insufficient disk space. Required: {model_size / (1024**3):.2f} GB, "
                    f"Available: {available_bytes / (1024**3):.2f} GB. "
                    f"Please free up space before downloading."
                ),
                cache_info=cache_info,
                system_info=system_info,
            )

        # Check warning threshold (percentage)
        warning = (
            cache_info.percent_free < WARNING_THRESHOLD_PERCENT
            or system_info.percent_free < WARNING_THRESHOLD_PERCENT
        )

        if warning:
            message = (
                f"Nearing disk space max - you have {available_bytes / (1024**3):.2f} GB available, "
                f"it could alter LF capabilities. Do you want to continue anyway?"
            )
        else:
            message = f"Sufficient space available ({available_bytes / (1024**3):.2f} GB free)"

        return ValidationResult(
            can_download=True,
            warning=warning,
            available_bytes=available_bytes,
            required_bytes=model_size,
            message=message,
            cache_info=cache_info,
            system_info=system_info,
        )

