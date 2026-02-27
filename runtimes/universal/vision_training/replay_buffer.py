"""Experience replay buffer for continual learning with SQLite persistence."""

from __future__ import annotations

import contextlib
import json
import logging
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SourceType = Literal["correction", "low_confidence", "cascade_resolved", "escalation_resolved"]


@dataclass
class ModelOpinion:
    """What one model thought about a detection."""
    model_id: str
    node_id: str = "local"
    class_name: str = ""
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] | None = None
    inference_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "node_id": self.node_id,
            "class_name": self.class_name, "confidence": self.confidence,
            "bbox": list(self.bbox) if self.bbox else None,
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
            model_id=d["model_id"], node_id=d.get("node_id", "local"),
            class_name=d.get("class_name", ""), confidence=d.get("confidence", 0.0),
            bbox=bbox, inference_time_ms=d.get("inference_time_ms", 0.0),
            timestamp=ts or datetime.utcnow(),
        )


@dataclass
class ReplaySample:
    """A sample in the replay buffer."""
    id: str
    image_path: str
    label: str
    source: SourceType
    confidence: float = 0.0
    priority: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    opinions: list[ModelOpinion] = field(default_factory=list)
    final_label: str = ""
    final_source: str = ""
    bbox: tuple[float, float, float, float] | None = None


class ReplayBuffer:
    """Priority-weighted replay buffer with SQLite persistence."""

    def __init__(self, max_size: int = 1000, storage_dir: Path | str | None = None):
        self.max_size = max_size
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self._samples: dict[str, ReplaySample] = {}

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = self.storage_dir / "replay_buffer.db"
            self._init_db()
            self._load_from_db()
        else:
            self._db_path = None

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS replay_samples (
                    id TEXT PRIMARY KEY, image_path TEXT NOT NULL,
                    label TEXT NOT NULL, source TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0, priority REAL DEFAULT 1.0,
                    created_at TEXT, metadata_json TEXT DEFAULT '{}',
                    opinions_json TEXT DEFAULT '[]',
                    final_label TEXT DEFAULT '', final_source TEXT DEFAULT '',
                    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL
                )
            """)

    def _load_from_db(self) -> None:
        if not self._db_path:
            return
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute("SELECT * FROM replay_samples").fetchall():
                bbox = None
                if row["bbox_x1"] is not None:
                    bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
                opinions = []
                try:
                    for od in json.loads(row["opinions_json"] or "[]"):
                        opinions.append(ModelOpinion.from_dict(od))
                except (json.JSONDecodeError, TypeError):
                    pass
                created_at = datetime.utcnow()
                if row["created_at"]:
                    with contextlib.suppress(ValueError):  # graceful fallback for non-ISO timestamps
                        created_at = datetime.fromisoformat(row["created_at"])
                self._samples[row["id"]] = ReplaySample(
                    id=row["id"], image_path=row["image_path"],
                    label=row["label"], source=row["source"],
                    confidence=row["confidence"], priority=row["priority"],
                    created_at=created_at,
                    metadata=json.loads(row["metadata_json"] or "{}"),
                    opinions=opinions, final_label=row["final_label"] or "",
                    final_source=row["final_source"] or "", bbox=bbox,
                )
        # Trim
        while len(self._samples) > self.max_size:
            self._evict_lowest()
        if self._samples:
            logger.info(f"Restored {len(self._samples)} replay samples from DB")

    def _persist(self, sample: ReplaySample) -> None:
        if not self._db_path:
            return
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO replay_samples
                (id, image_path, label, source, confidence, priority, created_at,
                 metadata_json, opinions_json, final_label, final_source,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sample.id, sample.image_path, sample.label, sample.source,
                sample.confidence, sample.priority, sample.created_at.isoformat(),
                json.dumps(sample.metadata),
                json.dumps([op.to_dict() for op in sample.opinions]),
                sample.final_label, sample.final_source,
                sample.bbox[0] if sample.bbox else None,
                sample.bbox[1] if sample.bbox else None,
                sample.bbox[2] if sample.bbox else None,
                sample.bbox[3] if sample.bbox else None,
            ))

    def _delete_from_db(self, sample_id: str) -> None:
        if not self._db_path:
            return
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM replay_samples WHERE id = ?", (sample_id,))

    def add(self, sample: ReplaySample) -> None:
        if sample.id not in self._samples and len(self._samples) >= self.max_size:
            self._evict_lowest()
        self._samples[sample.id] = sample
        self._persist(sample)

    def add_correction(self, image_id: str, image_path: str, corrected_label: str,
                       original_confidence: float = 0.0,
                       opinions: list[ModelOpinion] | None = None,
                       bbox: tuple[float, float, float, float] | None = None) -> ReplaySample:
        sample = ReplaySample(
            id=image_id, image_path=image_path, label=corrected_label,
            source="correction", confidence=original_confidence, priority=2.0,
            opinions=opinions or [], final_label=corrected_label,
            final_source="human", bbox=bbox,
        )
        self.add(sample)
        return sample

    def add_low_confidence(self, image_id: str, image_path: str,
                           predicted_label: str, confidence: float,
                           opinions: list[ModelOpinion] | None = None,
                           bbox: tuple[float, float, float, float] | None = None) -> ReplaySample:
        sample = ReplaySample(
            id=image_id, image_path=image_path, label=predicted_label,
            source="low_confidence", confidence=confidence,
            priority=1.0 - confidence, opinions=opinions or [], bbox=bbox,
        )
        self.add(sample)
        return sample

    def sample(self, batch_size: int, source: SourceType | None = None) -> list[ReplaySample]:
        samples = list(self._samples.values())
        if source:
            samples = [s for s in samples if s.source == source]
        if not samples:
            return []
        weights = [s.priority for s in samples]
        total = sum(weights)
        if total == 0:
            return random.sample(samples, min(batch_size, len(samples)))
        k = min(batch_size, len(samples))
        return random.choices(samples, weights=weights, k=k)

    def _evict_lowest(self) -> None:
        if not self._samples:
            return
        lowest = min(self._samples.values(), key=lambda s: s.priority)
        del self._samples[lowest.id]
        self._delete_from_db(lowest.id)

    def clear(self) -> None:
        self._samples.clear()
        if self._db_path:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM replay_samples")

    def __len__(self) -> int:
        return len(self._samples)

    def get_stats(self) -> dict[str, Any]:
        samples = list(self._samples.values())
        by_source: dict[str, int] = {}
        for s in samples:
            by_source[s.source] = by_source.get(s.source, 0) + 1
        return {
            "size": len(samples), "max_size": self.max_size,
            "by_source": by_source,
            "avg_priority": sum(s.priority for s in samples) / len(samples) if samples else 0,
        }


_replay_buffer: ReplayBuffer | None = None


def get_replay_buffer(max_size: int = 1000,
                      storage_dir: Path | str | None = None) -> ReplayBuffer:
    global _replay_buffer
    if _replay_buffer is None:
        _replay_buffer = ReplayBuffer(max_size=max_size, storage_dir=storage_dir)
    return _replay_buffer
