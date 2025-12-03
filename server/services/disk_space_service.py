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

        # If the path doesn't exist, use its parent directory for disk usage
        # This handles cases where the cache directory hasn't been created yet
        check_path = resolved_path
        if not check_path.exists():
            # Use parent directory if path doesn't exist
            check_path = check_path.parent
            # Keep going up until we find an existing directory or reach root
            while not check_path.exists() and check_path != check_path.parent:
                check_path = check_path.parent

        try:
            usage = psutil.disk_usage(str(check_path))
            percent_free = (
                (usage.free / usage.total) * 100.0 if usage.total > 0 else 0.0
            )

            return DiskSpaceInfo(
                total_bytes=usage.total,
                used_bytes=usage.used,
                free_bytes=usage.free,
                path=str(resolved_path),
                percent_free=percent_free,
            )
        except OSError as e:
            logger.warning(f"Failed to check disk space at {check_path}: {e}")
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
            from llamafarm_common import parse_model_with_quantization

            base_model_id, _ = parse_model_with_quantization(model_id)
            api = HfApi()

            # Primary method: use model_info with files_metadata to get sizes from siblings
            try:
                model_info = api.model_info(base_model_id, files_metadata=True)
                if hasattr(model_info, "siblings") and model_info.siblings:
                    total_size = sum(
                        getattr(s, "size", 0) or 0
                        for s in model_info.siblings
                        if getattr(s, "size", None)
                    )
                    if total_size > 0:
                        logger.info(
                            f"Got model size from siblings: {total_size / (1024**3):.2f} GB"
                        )
                        return total_size
            except Exception as e:
                logger.debug(f"Could not get model info for {base_model_id}: {e}")

            # Fallback: list repo files and sum their sizes
            try:
                files = api.list_repo_files(repo_id=base_model_id, repo_type="model")
                if files:
                    total_size = 0
                    for file_path in files:
                        try:
                            file_info = api.get_path_info(
                                repo_id=base_model_id, path=file_path, repo_type="model"
                            )
                            if hasattr(file_info, "size") and file_info.size:
                                total_size += file_info.size
                        except Exception:
                            continue

                    if total_size > 0:
                        logger.info(
                            f"Got model size via file listing: {total_size / (1024**3):.2f} GB"
                        )
                        return total_size
            except Exception as e:
                logger.debug(f"Could not get model size via file listing: {e}")

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
        available_bytes = min(cache_info.free_bytes, system_info.free_bytes)

        if model_size is None:
            # Size could not be determined - warn if disk space is already low
            if cache_info.percent_free < 20.0 or system_info.percent_free < 20.0:
                return ValidationResult(
                    can_download=True,
                    warning=True,
                    available_bytes=available_bytes,
                    required_bytes=0,
                    message=(
                        f"Model size could not be determined and you have low disk space "
                        f"({available_bytes / (1024**3):.2f} GB free, "
                        f"{min(cache_info.percent_free, system_info.percent_free):.1f}% free). "
                        f"Proceed with caution."
                    ),
                    cache_info=cache_info,
                    system_info=system_info,
                )

            # If size is unknown but we have plenty of space, allow without warning
            return ValidationResult(
                can_download=True,
                warning=False,
                available_bytes=available_bytes,
                required_bytes=0,
                message=f"Sufficient space available ({available_bytes / (1024**3):.2f} GB free)",
                cache_info=cache_info,
                system_info=system_info,
            )

        # Models download to the cache directory, so we only block if cache doesn't have space
        # System disk is only used for warnings (low space warnings)
        cache_available_bytes = cache_info.free_bytes

        # Check critical threshold (absolute minimum) - only check cache disk
        if cache_available_bytes < CRITICAL_THRESHOLD_BYTES:
            return ValidationResult(
                can_download=False,
                warning=False,
                available_bytes=cache_available_bytes,
                required_bytes=model_size,
                message=(
                    f"Insufficient disk space on cache disk. Required: {model_size / (1024**3):.2f} GB, "
                    f"Available: {cache_available_bytes / (1024**3):.2f} GB. "
                    f"Please free up space before downloading."
                ),
                cache_info=cache_info,
                system_info=system_info,
            )

        # Check if model fits in cache - only check cache disk for blocking
        if cache_available_bytes < model_size:
            return ValidationResult(
                can_download=False,
                warning=False,
                available_bytes=cache_available_bytes,
                required_bytes=model_size,
                message=(
                    f"Insufficient disk space on cache disk. Required: {model_size / (1024**3):.2f} GB, "
                    f"Available: {cache_available_bytes / (1024**3):.2f} GB. "
                    f"Please free up space before downloading."
                ),
                cache_info=cache_info,
                system_info=system_info,
            )

        # Check warning threshold (percentage) - PROJECTED after download
        # Calculate what the free percentage will be after downloading the model
        # Only project cache disk since that's where the model downloads
        # System disk warning is based on current free space (not projected) since
        # cache and system may be on different volumes
        remaining_cache_after_download = cache_info.free_bytes - model_size

        projected_cache_percent = (
            (remaining_cache_after_download / cache_info.total_bytes * 100)
            if cache_info.total_bytes > 0
            else 0
        )
        # System disk warning uses current free space, not projected
        # (since model downloads to cache, not system disk)
        current_system_percent = system_info.percent_free

        warning = (
            projected_cache_percent < WARNING_THRESHOLD_PERCENT
            or current_system_percent < WARNING_THRESHOLD_PERCENT
        )

        if warning:
            # For cache disk, use projected remaining space
            # For system disk, use current free space
            cache_remaining = remaining_cache_after_download
            system_remaining = system_info.free_bytes
            min_remaining = min(cache_remaining, system_remaining)
            min_percent = min(projected_cache_percent, current_system_percent)
            message = (
                f"Downloading this model ({model_size / (1024**3):.2f} GB) will leave you with "
                f"{min_remaining / (1024**3):.2f} GB free "
                f"({min_percent:.1f}% free), "
                f"which is below the 10% threshold. This could affect LlamaFarm's capabilities. "
                f"Do you want to continue anyway?"
            )
        else:
            message = f"Sufficient space available ({cache_available_bytes / (1024**3):.2f} GB free on cache disk)"

        return ValidationResult(
            can_download=True,
            warning=warning,
            available_bytes=cache_available_bytes,
            required_bytes=model_size,
            message=message,
            cache_info=cache_info,
            system_info=system_info,
        )
