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
            # Delete by age
            stats.images_deleted += self._cleanup_by_age()
            
            # Delete to enforce storage limits
            stats.images_deleted += self._cleanup_by_storage()
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            stats.errors.append(str(e))
        
        if stats.images_deleted > 0:
            logger.info(
                f"Cleanup complete: deleted {stats.images_deleted} images, "
                f"freed {stats.bytes_freed / 1024 / 1024:.1f} MB"
            )
        
        return stats
    
    def _cleanup_by_age(self) -> int:
        """Delete images based on age and confidence."""
        now = datetime.utcnow()
        total_deleted = 0
        
        # High confidence: short retention
        cutoff = now - timedelta(hours=self.config.high_confidence_hours)
        deleted = self.store.delete_old_images(cutoff)
        total_deleted += deleted
        
        return total_deleted
    
    def _cleanup_by_storage(self) -> int:
        """Delete images to stay under storage limit."""
        vision_dir = self.store.db_path.parent
        
        if not vision_dir.exists():
            return 0
        
        # Calculate current usage
        total_bytes = sum(
            f.stat().st_size
            for f in vision_dir.rglob("*")
            if f.is_file()
        )
        
        limit_bytes = self.config.max_total_vision_gb * 1024 * 1024 * 1024
        
        if total_bytes <= limit_bytes:
            return 0
        
        # Need to delete oldest images until under limit
        logger.warning(
            f"Vision storage ({total_bytes / 1024 / 1024 / 1024:.1f} GB) "
            f"exceeds limit ({self.config.max_total_vision_gb} GB)"
        )
        
        # Delete oldest 10% of images
        stats = self.store.get_stats()
        to_delete = max(1, stats["total_images"] // 10)

        cutoff = datetime.utcnow() - timedelta(hours=1)  # At least 1 hour old
        deleted = self.store.delete_old_images(cutoff, limit=to_delete)
        
        return deleted
    
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
