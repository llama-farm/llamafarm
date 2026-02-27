"""Simple SQLite store for training image metadata."""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    id: str
    file_path: str
    source: str = "unknown"
    class_name: str = ""
    confidence: float = 0.0
    reviewed: bool = False
    created_at: datetime | None = None
    width: int = 0
    height: int = 0


class ImageStore:
    """SQLite-based image metadata storage for vision training pipeline."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    source TEXT DEFAULT 'unknown',
                    class_name TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    reviewed BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    width INTEGER DEFAULT 0,
                    height INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_images_source ON images(source);
                CREATE INDEX IF NOT EXISTS idx_images_reviewed ON images(reviewed);
                CREATE INDEX IF NOT EXISTS idx_images_class ON images(class_name);
            """)

    def add_image(self, record: ImageRecord) -> str:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO images
                (id, file_path, source, class_name, confidence, reviewed, created_at, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id, record.file_path, record.source, record.class_name,
                record.confidence, record.reviewed,
                record.created_at or datetime.utcnow(), record.width, record.height,
            ))
        return record.id

    def get_image(self, image_id: str) -> ImageRecord | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            if not row:
                return None
            return self._to_record(row)

    def get_pending_review(self, limit: int = 50, source: str | None = None) -> list[ImageRecord]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            q = "SELECT * FROM images WHERE reviewed = FALSE"
            params: list[Any] = []
            if source:
                q += " AND source = ?"
                params.append(source)
            q += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            return [self._to_record(r) for r in conn.execute(q, params).fetchall()]

    def mark_reviewed(self, image_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE images SET reviewed = TRUE WHERE id = ?", (image_id,))

    def get_stats(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            pending = conn.execute("SELECT COUNT(*) FROM images WHERE reviewed = FALSE").fetchone()[0]
            top = conn.execute("""
                SELECT class_name, COUNT(*) as cnt FROM images
                WHERE class_name != '' GROUP BY class_name ORDER BY cnt DESC LIMIT 10
            """).fetchall()
            return {
                "total_images": total, "pending_review": pending,
                "top_classes": {r[0]: r[1] for r in top},
            }

    def _to_record(self, row: sqlite3.Row) -> ImageRecord:
        created = None
        if row["created_at"]:
            with contextlib.suppress(ValueError):  # graceful fallback for non-ISO timestamps
                created = datetime.fromisoformat(str(row["created_at"]))
        return ImageRecord(
            id=row["id"], file_path=row["file_path"], source=row["source"],
            class_name=row["class_name"], confidence=row["confidence"],
            reviewed=bool(row["reviewed"]), created_at=created,
            width=row["width"], height=row["height"],
        )

    @staticmethod
    def get_images_dir(model_id: str) -> Path:
        """Get the training images directory for a model."""
        d = Path.home() / ".llamafarm" / "models" / "vision" / model_id / "training_data" / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def get_store_for_model(model_id: str) -> ImageStore:
        """Get an ImageStore scoped to a specific model."""
        db = Path.home() / ".llamafarm" / "models" / "vision" / model_id / "image_store.db"
        return ImageStore(db)
