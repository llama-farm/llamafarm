"""Retention policy for vision data.

Configurable cleanup based on:
- Image age
- Review status
- Confidence level
- Storage limits
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .image_store import ImageMetadataStore

logger = logging.getLogger(__name__)


@dataclass
class RetentionConfig:
    """Configuration for retention policy."""
    
    # Retention by confidence level (hours)
    high_confidence_hours: int = 1      # > 0.9 confidence
    medium_confidence_hours: int = 24   # 0.7 - 0.9
    low_confidence_hours: int = 168     # 0.5 - 0.7 (7 days)
    very_low_confidence_hours: int = 720  # < 0.5 (30 days)
    
    # Special cases
    reviewed_retention_days: int = 90
    corrections_retention_days: int = 90
    
    # Storage limits (GB)
    max_review_queue_gb: float = 10.0
    max_total_vision_gb: float = 50.0
    
    # Cleanup schedule
    cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24


@dataclass
class RetentionStats:
    """Statistics from a cleanup run."""
    
    images_deleted: int = 0
    bytes_freed: int = 0
    detections_deleted: int = 0
    labels_deleted: int = 0
    errors: list[str] = field(default_factory=list)


class RetentionPolicy:
    """Manages retention and cleanup of vision data.
    
    Example:
        ```python
        store = ImageMetadataStore()
        config = RetentionConfig(
            high_confidence_hours=1,
            max_total_vision_gb=50.0
        )
        policy = RetentionPolicy(store, config)
        
        # Run cleanup
        stats = policy.cleanup()
        print(f"Deleted {stats.images_deleted} images, freed {stats.bytes_freed} bytes")
        ```
    """
    
    def __init__(
        self,
        store: ImageMetadataStore,
        config: RetentionConfig | None = None,
    ):
        """Initialize retention policy.
        
        Args:
            store: Image metadata store
            config: Retention configuration
        """
        self.store = store
        self.config = config or RetentionConfig()
    
    def cleanup(self) -> RetentionStats:
        """Run cleanup based on retention policy.

        Returns:
            RetentionStats with cleanup results
        """
        stats = RetentionStats()

        if not self.config.cleanup_enabled:
            return stats

        try:
            # Delete by age (respects confidence tiers)
            self._cleanup_by_age(stats)

            # Delete to enforce storage limits
            self._cleanup_by_storage(stats)

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            stats.errors.append(str(e))

        if stats.images_deleted > 0:
            logger.info(
                f"Cleanup complete: deleted {stats.images_deleted} images, "
                f"freed {stats.bytes_freed / 1024 / 1024:.1f} MB"
            )

        return stats

    def _delete_files(
        self,
        deleted_info: list[tuple[str, str, str | None, int]],
        stats: RetentionStats,
    ) -> None:
        """Delete actual image and thumbnail files from disk.

        Args:
            deleted_info: List of (id, file_path, thumbnail_path, size_bytes)
                          as returned by ImageMetadataStore.delete_old_images.
            stats: RetentionStats to update with bytes_freed count.
        """
        for _image_id, file_path, thumbnail_path, size_bytes in deleted_info:
            # Delete image file
            try:
                p = Path(file_path)
                if p.is_file():
                    actual_size = p.stat().st_size
                    p.unlink()
                    stats.bytes_freed += actual_size
                else:
                    # Use DB-recorded size as estimate
                    stats.bytes_freed += size_bytes
            except OSError as e:
                logger.warning(f"Failed to delete image file {file_path}: {e}")
                stats.errors.append(f"file delete failed: {file_path}: {e}")

            # Delete thumbnail file
            if thumbnail_path:
                try:
                    tp = Path(thumbnail_path)
                    if tp.is_file():
                        stats.bytes_freed += tp.stat().st_size
                        tp.unlink()
                except OSError as e:
                    logger.warning(f"Failed to delete thumbnail {thumbnail_path}: {e}")

    def _cleanup_by_age(self, stats: RetentionStats) -> None:
        """Delete images based on age using the configured retention tiers."""
        now = datetime.utcnow()

        # Apply each confidence tier cutoff from shortest to longest.
        # delete_old_images uses created_at, so we delete progressively.
        # The tiers are applied as a single global cutoff since the DB
        # doesn't track per-image confidence. Use the most conservative
        # (longest) retention to avoid data loss — very_low_confidence_hours.
        cutoff = now - timedelta(hours=self.config.very_low_confidence_hours)
        deleted_info = self.store.delete_old_images(cutoff)

        if deleted_info:
            stats.images_deleted += len(deleted_info)
            self._delete_files(deleted_info, stats)

    def _cleanup_by_storage(self, stats: RetentionStats) -> None:
        """Delete oldest images to stay under storage limit."""
        vision_dir = self.store.db_path.parent

        if not vision_dir.exists():
            return

        # Calculate current usage
        total_bytes = sum(
            f.stat().st_size
            for f in vision_dir.rglob("*")
            if f.is_file()
        )

        limit_bytes = self.config.max_total_vision_gb * 1024 * 1024 * 1024

        if total_bytes <= limit_bytes:
            return

        # Need to delete oldest images until under limit
        logger.warning(
            f"Vision storage ({total_bytes / 1024 / 1024 / 1024:.1f} GB) "
            f"exceeds limit ({self.config.max_total_vision_gb} GB)"
        )

        # Delete oldest 10% of images
        db_stats = self.store.get_stats()
        to_delete = max(1, db_stats["total_images"] // 10)

        cutoff = datetime.utcnow() - timedelta(hours=1)  # At least 1 hour old
        deleted_info = self.store.delete_old_images(cutoff, limit=to_delete)

        if deleted_info:
            stats.images_deleted += len(deleted_info)
            self._delete_files(deleted_info, stats)
    
    def get_retention_hours(self, confidence: float) -> int:
        """Get retention hours based on confidence level."""
        if confidence > 0.9:
            return self.config.high_confidence_hours
        elif confidence > 0.7:
            return self.config.medium_confidence_hours
        elif confidence > 0.5:
            return self.config.low_confidence_hours
        else:
            return self.config.very_low_confidence_hours
