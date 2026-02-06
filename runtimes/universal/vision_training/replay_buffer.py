"""Experience replay buffer for continual learning.

Stores corrected examples for use during incremental training
to prevent catastrophic forgetting.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class ReplaySample:
    """A sample in the replay buffer."""
    
    id: str
    image_path: str
    label: str  # Class name or detection annotations
    source: Literal["correction", "low_confidence", "original"]
    confidence: float = 0.0
    priority: float = 1.0  # Higher = more likely to be sampled
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    """Experience replay buffer for continual learning.
    
    Stores corrected and low-confidence samples for use during
    incremental training. Supports priority sampling to focus
    on more important examples.
    
    Example:
        ```python
        buffer = ReplayBuffer(max_size=1000)
        
        # Add a correction
        buffer.add(ReplaySample(
            id="sample_001",
            image_path="/path/to/image.jpg",
            label="person",
            source="correction",
            priority=2.0  # Corrections have higher priority
        ))
        
        # Sample for training
        batch = buffer.sample(batch_size=32)
        ```
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        storage_dir: Path | str | None = None,
    ):
        """Initialize replay buffer.
        
        Args:
            max_size: Maximum number of samples to store
            storage_dir: Directory for persisting samples
        """
        self.max_size = max_size
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self._samples: dict[str, ReplaySample] = {}
        
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def add(self, sample: ReplaySample) -> None:
        """Add a sample to the buffer.
        
        If buffer is full, removes lowest priority sample.
        """
        # If at capacity, remove lowest priority
        if len(self._samples) >= self.max_size:
            self._evict_lowest_priority()
        
        self._samples[sample.id] = sample
        logger.debug(f"Added sample {sample.id} to replay buffer")
    
    def add_correction(
        self,
        image_id: str,
        image_path: str,
        corrected_label: str,
        original_confidence: float = 0.0,
    ) -> ReplaySample:
        """Add a human-corrected sample.
        
        Corrections get higher priority than auto-flagged samples.
        """
        sample = ReplaySample(
            id=image_id,
            image_path=image_path,
            label=corrected_label,
            source="correction",
            confidence=original_confidence,
            priority=2.0,  # Higher priority for corrections
        )
        self.add(sample)
        return sample
    
    def add_low_confidence(
        self,
        image_id: str,
        image_path: str,
        predicted_label: str,
        confidence: float,
    ) -> ReplaySample:
        """Add a low-confidence sample for review."""
        sample = ReplaySample(
            id=image_id,
            image_path=image_path,
            label=predicted_label,
            source="low_confidence",
            confidence=confidence,
            priority=1.0 - confidence,  # Lower confidence = higher priority
        )
        self.add(sample)
        return sample
    
    def get(self, sample_id: str) -> ReplaySample | None:
        """Get a sample by ID."""
        return self._samples.get(sample_id)
    
    def remove(self, sample_id: str) -> bool:
        """Remove a sample from the buffer."""
        if sample_id in self._samples:
            del self._samples[sample_id]
            return True
        return False
    
    def sample(
        self,
        batch_size: int,
        source: Literal["correction", "low_confidence", "original"] | None = None,
    ) -> list[ReplaySample]:
        """Sample from the buffer with priority weighting.
        
        Args:
            batch_size: Number of samples to return
            source: Filter to specific source type
            
        Returns:
            List of samples (may be fewer than batch_size if buffer is small)
        """
        samples = list(self._samples.values())
        
        if source:
            samples = [s for s in samples if s.source == source]
        
        if not samples:
            return []
        
        # Weighted sampling by priority
        weights = [s.priority for s in samples]
        total_weight = sum(weights)
        
        if total_weight == 0:
            # Fall back to uniform sampling
            return random.sample(samples, min(batch_size, len(samples)))
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Sample with replacement if needed
        k = min(batch_size, len(samples))
        
        try:
            return random.choices(samples, weights=weights, k=k)
        except ValueError:
            return random.sample(samples, k)
    
    def sample_stratified(
        self,
        batch_size: int,
        correction_ratio: float = 0.5,
    ) -> list[ReplaySample]:
        """Sample with stratification by source.
        
        Args:
            batch_size: Total samples to return
            correction_ratio: Proportion from corrections (rest from low_confidence)
        """
        n_corrections = int(batch_size * correction_ratio)
        n_low_conf = batch_size - n_corrections
        
        corrections = self.sample(n_corrections, source="correction")
        low_conf = self.sample(n_low_conf, source="low_confidence")
        
        return corrections + low_conf
    
    def _evict_lowest_priority(self) -> None:
        """Remove the lowest priority sample."""
        if not self._samples:
            return
        
        lowest = min(self._samples.values(), key=lambda s: s.priority)
        del self._samples[lowest.id]
        logger.debug(f"Evicted low-priority sample {lowest.id}")
    
    def clear(self) -> None:
        """Clear all samples from the buffer."""
        self._samples.clear()
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        samples = list(self._samples.values())
        
        return {
            "size": len(samples),
            "max_size": self.max_size,
            "by_source": {
                "correction": len([s for s in samples if s.source == "correction"]),
                "low_confidence": len([s for s in samples if s.source == "low_confidence"]),
                "original": len([s for s in samples if s.source == "original"]),
            },
            "avg_priority": sum(s.priority for s in samples) / len(samples) if samples else 0,
        }


# Global replay buffer
_replay_buffer: ReplayBuffer | None = None


def get_replay_buffer(max_size: int = 1000) -> ReplayBuffer:
    """Get or create the global replay buffer."""
    global _replay_buffer
    if _replay_buffer is None:
        _replay_buffer = ReplayBuffer(max_size=max_size)
    return _replay_buffer
