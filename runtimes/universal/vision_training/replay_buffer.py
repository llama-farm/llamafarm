"""Experience replay buffer for continual learning.

Stores corrected examples for use during incremental training
to prevent catastrophic forgetting.

Enhanced with:
- ModelOpinion: structured record of what each model predicted
- EscalationEnvelope context: full cascade provenance
- SQLite persistence: survives restarts
"""

from __future__ import annotations

import contextlib
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Valid source types for replay samples
SourceType = Literal[
    "correction",        # Human-corrected label
    "low_confidence",    # Auto-flagged uncertain detection
    "original",          # Original training sample
    "audit",             # Audit pipeline disagreement
    "cascade_resolved",  # Resolved by a later hop in the cascade
    "escalation_resolved",  # Resolved by the immediate next model
]


@dataclass
class ModelOpinion:
    """What one model thought about a detection.

    This is the universal record: every time a model looks at an image
    and says something, it produces one of these. The cascade accumulates
    them. Training uses them to understand what went wrong and right.
    """

    model_id: str               # "yolov8n", "yolov8x", "remote:gpu-server/yolov8x"
    node_id: str = "local"      # Atmosphere node ID or "local"
    class_name: str = ""        # What the model thinks this is
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2
    mask_polygon: list[list[float]] | None = None  # Segmentation polygon points
    inference_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "node_id": self.node_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": list(self.bbox) if self.bbox else None,
            "mask_polygon": self.mask_polygon,
            "inference_time_ms": self.inference_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelOpinion:
        bbox = tuple(d["bbox"]) if d.get("bbox") else None
        ts = d.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            model_id=d["model_id"],
            node_id=d.get("node_id", "local"),
            class_name=d.get("class_name", ""),
            confidence=d.get("confidence", 0.0),
            bbox=bbox,
            mask_polygon=d.get("mask_polygon"),
            inference_time_ms=d.get("inference_time_ms", 0.0),
            timestamp=ts or datetime.utcnow(),
        )


@dataclass
class ReplaySample:
    """A sample in the replay buffer.

    Enhanced to carry full cascade context so the training pipeline
    knows exactly what happened: which models saw this image, what
    they each predicted, and what the final answer was.
    """

    id: str
    image_path: str
    label: str  # Class name or YOLO-format detection annotations
    source: SourceType
    confidence: float = 0.0
    priority: float = 1.0  # Higher = more likely to be sampled
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Structured detection context
    opinions: list[ModelOpinion] = field(default_factory=list)
    final_label: str = ""           # The resolved/corrected answer
    final_source: str = ""          # "cascade", "human", "audit"
    bbox: tuple[float, float, float, float] | None = None  # Primary bbox
    mask_rle: str | None = None     # Run-length encoded segmentation mask
    crop_path: str | None = None    # Path to cropped bbox region on disk


class ReplayBuffer:
    """Experience replay buffer for continual learning.

    Stores corrected and low-confidence samples for use during
    incremental training. Supports priority sampling to focus
    on more important examples.

    When storage_dir is provided, samples are persisted to SQLite
    and survive restarts.

    Example:
        ```python
        buffer = ReplayBuffer(max_size=1000, storage_dir="~/.llamafarm/vision/replay")

        # Add a cascade-resolved sample with full context
        buffer.add_cascade_resolved(
            image_id="frame_001",
            image_path="/path/to/image.jpg",
            opinions=[
                ModelOpinion(model_id="yolov8n", class_name="bird", confidence=0.45),
                ModelOpinion(model_id="yolov8m", class_name="bird", confidence=0.92),
            ],
            final_label="bird",
            bbox=(100, 200, 300, 400),
        )

        # Sample for training
        batch = buffer.sample(batch_size=32)
        ```
    """

    def __init__(
        self,
        max_size: int = 1000,
        storage_dir: Path | str | None = None,
    ):
        self.max_size = max_size
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self._samples: dict[str, ReplaySample] = {}
        self._persistence: ReplayBufferPersistence | None = None

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._persistence = ReplayBufferPersistence(self.storage_dir / "replay_buffer.db")
            self._load_from_persistence()

    def _load_from_persistence(self) -> None:
        """Reload all samples from SQLite on startup."""
        if self._persistence is None:
            return
        samples = self._persistence.load_all()
        for s in samples:
            self._samples[s.id] = s

        # Trim to max_size if persisted data exceeds the limit
        while len(self._samples) > self.max_size:
            self._evict_lowest_priority()

        if samples:
            logger.info(f"Restored {len(samples)} samples from persistent storage")

    def add(self, sample: ReplaySample) -> None:
        """Add a sample to the buffer.

        If buffer is full, removes lowest priority sample.
        """
        if len(self._samples) >= self.max_size:
            self._evict_lowest_priority()

        self._samples[sample.id] = sample
        if self._persistence:
            self._persistence.save(sample)
        logger.debug(f"Added sample {sample.id} to replay buffer")

    def add_correction(
        self,
        image_id: str,
        image_path: str,
        corrected_label: str,
        original_confidence: float = 0.0,
        opinions: list[ModelOpinion] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> ReplaySample:
        """Add a human-corrected sample. Priority 2.0 (highest)."""
        sample = ReplaySample(
            id=image_id,
            image_path=image_path,
            label=corrected_label,
            source="correction",
            confidence=original_confidence,
            priority=2.0,
            opinions=opinions or [],
            final_label=corrected_label,
            final_source="human",
            bbox=bbox,
        )
        self.add(sample)
        return sample

    def add_low_confidence(
        self,
        image_id: str,
        image_path: str,
        predicted_label: str,
        confidence: float,
        opinions: list[ModelOpinion] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> ReplaySample:
        """Add a low-confidence sample for review."""
        sample = ReplaySample(
            id=image_id,
            image_path=image_path,
            label=predicted_label,
            source="low_confidence",
            confidence=confidence,
            priority=1.0 - confidence,
            opinions=opinions or [],
            bbox=bbox,
        )
        self.add(sample)
        return sample

    def add_cascade_resolved(
        self,
        image_id: str,
        image_path: str,
        opinions: list[ModelOpinion],
        final_label: str,
        bbox: tuple[float, float, float, float] | None = None,
        mask_rle: str | None = None,
        crop_path: str | None = None,
        resolving_hop: int = 1,
    ) -> ReplaySample:
        """Add a sample resolved by the cascade (a later model got it right).

        This is the automatic feedback loop: when Hop 1 or Hop 2 resolves
        what Hop 0 couldn't, it becomes a training sample for Hop 0.
        """
        source: SourceType = "escalation_resolved" if resolving_hop == 1 else "cascade_resolved"
        priority = 1.5 if resolving_hop == 1 else 1.8

        sample = ReplaySample(
            id=image_id,
            image_path=image_path,
            label=final_label,
            source=source,
            confidence=opinions[-1].confidence if opinions else 0.0,
            priority=priority,
            opinions=opinions,
            final_label=final_label,
            final_source="cascade",
            bbox=bbox,
            mask_rle=mask_rle,
            crop_path=crop_path,
        )
        self.add(sample)
        return sample

    def add_audit_disagreement(
        self,
        image_id: str,
        image_path: str,
        opinions: list[ModelOpinion],
        audit_label: str,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> ReplaySample:
        """Add an audit disagreement. Same priority as human corrections."""
        sample = ReplaySample(
            id=image_id,
            image_path=image_path,
            label=audit_label,
            source="audit",
            confidence=opinions[-1].confidence if opinions else 0.0,
            priority=2.0,
            opinions=opinions,
            final_label=audit_label,
            final_source="audit",
            bbox=bbox,
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
            if self._persistence:
                self._persistence.delete(sample_id)
            return True
        return False

    def sample(
        self,
        batch_size: int,
        source: SourceType | None = None,
    ) -> list[ReplaySample]:
        """Sample from the buffer with priority weighting."""
        samples = list(self._samples.values())

        if source:
            samples = [s for s in samples if s.source == source]

        if not samples:
            return []

        weights = [s.priority for s in samples]
        total_weight = sum(weights)

        if total_weight == 0:
            return random.sample(samples, min(batch_size, len(samples)))

        weights = [w / total_weight for w in weights]
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
        """Sample with stratification by source."""
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
        if self._persistence:
            self._persistence.delete(lowest.id)
        logger.debug(f"Evicted low-priority sample {lowest.id}")

    def clear(self) -> None:
        """Clear all samples from the buffer."""
        self._samples.clear()
        if self._persistence:
            self._persistence.clear()

    def __len__(self) -> int:
        return len(self._samples)

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        samples = list(self._samples.values())

        by_source: dict[str, int] = {}
        for s in samples:
            by_source[s.source] = by_source.get(s.source, 0) + 1

        return {
            "size": len(samples),
            "max_size": self.max_size,
            "persistent": self._persistence is not None,
            "by_source": by_source,
            "avg_priority": sum(s.priority for s in samples) / len(samples) if samples else 0,
            "with_opinions": len([s for s in samples if s.opinions]),
            "with_bbox": len([s for s in samples if s.bbox]),
        }


class ReplayBufferPersistence:
    """SQLite persistence for the replay buffer.

    Mirrors in-memory writes to SQLite so training data survives restarts.
    Opinions are stored as JSON. On startup, all samples are loaded back
    into the in-memory dict for fast priority-weighted sampling.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS replay_samples (
                    id TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    label TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    priority REAL DEFAULT 1.0,
                    created_at TEXT,
                    metadata_json TEXT DEFAULT '{}',
                    opinions_json TEXT DEFAULT '[]',
                    final_label TEXT DEFAULT '',
                    final_source TEXT DEFAULT '',
                    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                    mask_rle TEXT,
                    crop_path TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_replay_source ON replay_samples(source)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_replay_priority ON replay_samples(priority)"
            )

    def save(self, sample: ReplaySample) -> None:
        import json
        import sqlite3

        opinions_json = json.dumps([op.to_dict() for op in sample.opinions])
        metadata_json = json.dumps(sample.metadata)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO replay_samples
                (id, image_path, label, source, confidence, priority, created_at,
                 metadata_json, opinions_json, final_label, final_source,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, mask_rle, crop_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sample.id,
                sample.image_path,
                sample.label,
                sample.source,
                sample.confidence,
                sample.priority,
                sample.created_at.isoformat(),
                metadata_json,
                opinions_json,
                sample.final_label,
                sample.final_source,
                sample.bbox[0] if sample.bbox else None,
                sample.bbox[1] if sample.bbox else None,
                sample.bbox[2] if sample.bbox else None,
                sample.bbox[3] if sample.bbox else None,
                sample.mask_rle,
                sample.crop_path,
            ))

    def delete(self, sample_id: str) -> None:
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM replay_samples WHERE id = ?", (sample_id,))

    def clear(self) -> None:
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM replay_samples")

    def load_all(self) -> list[ReplaySample]:
        import json
        import sqlite3

        samples = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM replay_samples").fetchall()

            for row in rows:
                bbox = None
                if row["bbox_x1"] is not None:
                    bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])

                opinions = []
                try:
                    for od in json.loads(row["opinions_json"] or "[]"):
                        opinions.append(ModelOpinion.from_dict(od))
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Failed to deserialize opinions_json for replay sample id=%r, "
                        "defaulting to empty opinions list.",
                        row.get("id") if isinstance(row, dict) else row["id"],
                        exc_info=True,
                    )

                try:
                    metadata = json.loads(row["metadata_json"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                created_at = datetime.utcnow()
                if row["created_at"]:
                    with contextlib.suppress(ValueError):
                        created_at = datetime.fromisoformat(row["created_at"])

                samples.append(ReplaySample(
                    id=row["id"],
                    image_path=row["image_path"],
                    label=row["label"],
                    source=row["source"],
                    confidence=row["confidence"],
                    priority=row["priority"],
                    created_at=created_at,
                    metadata=metadata,
                    opinions=opinions,
                    final_label=row["final_label"] or "",
                    final_source=row["final_source"] or "",
                    bbox=bbox,
                    mask_rle=row["mask_rle"],
                    crop_path=row["crop_path"],
                ))

        return samples


# Global replay buffer
_replay_buffer: ReplayBuffer | None = None


def get_replay_buffer(
    max_size: int = 1000,
    storage_dir: Path | str | None = None,
) -> ReplayBuffer:
    """Get or create the global replay buffer."""
    global _replay_buffer
    if _replay_buffer is None:
        _replay_buffer = ReplayBuffer(max_size=max_size, storage_dir=storage_dir)
    return _replay_buffer
