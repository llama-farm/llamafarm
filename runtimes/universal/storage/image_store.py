"""SQLite-based image metadata storage for vision pipeline.

Stores metadata about processed images including:
- Image info (path, dimensions, format)
- Detection results
- Classification labels
- Review status
- Timestamps for retention policy
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


def _get_vision_data_dir() -> Path:
    """Get the vision data directory."""
    from utils.safe_home import get_data_dir
    vision_dir = get_data_dir() / "vision"
    vision_dir.mkdir(parents=True, exist_ok=True)
    return vision_dir


@dataclass
class ImageRecord:
    """Image metadata record."""
    
    id: str
    file_path: str
    thumbnail_path: str | None = None
    created_at: datetime | None = None
    source: str = "unknown"
    width: int = 0
    height: int = 0
    format: str = "jpeg"
    size_bytes: int = 0
    content_hash: str = ""
    ocr_text: str | None = None
    reviewed: bool = False
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None


@dataclass
class DetectionRecord:
    """Detection result for an image."""

    id: int | None = None
    image_id: str = ""
    class_name: str = ""
    confidence: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    model: str = ""
    verified: bool = False
    created_at: datetime | None = None


@dataclass
class DetectionHistoryRecord:
    """Full cascade history for a single model opinion on a detection.

    Every time a model looks at an image during the cascade, one of these
    is recorded. This is the audit trail for the entire learning loop.
    """

    id: int | None = None
    image_id: str = ""
    model_id: str = ""
    node_id: str = "local"
    class_name: str = ""
    confidence: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    mask_rle: str | None = None
    stage: str = ""             # "primary", "secondary", "audit", "remote"
    hop_number: int = 0
    inference_time_ms: float = 0.0
    created_at: datetime | None = None


@dataclass
class LabelRecord:
    """Classification label for an image."""
    
    id: int | None = None
    image_id: str = ""
    label: str = ""
    confidence: float = 0.0
    source: Literal["model", "human"] = "model"
    created_at: datetime | None = None


class ImageMetadataStore:
    """SQLite-based image metadata storage.
    
    Provides CRUD operations for images, detections, and labels
    with support for retention policies and review queue.
    
    Example:
        ```python
        store = ImageMetadataStore()
        
        # Add image
        record = ImageRecord(
            id="img_001",
            file_path="/path/to/image.jpg",
            source="stream:camera1",
            width=1920,
            height=1080,
        )
        store.add_image(record)
        
        # Add detection
        store.add_detection(
            image_id="img_001",
            class_name="person",
            confidence=0.95,
            box=(100, 200, 300, 400),
            model="yolov8n"
        )
        
        # Get pending review
        pending = store.get_pending_review(limit=50)
        ```
    """
    
    def __init__(self, db_path: Path | str | None = None):
        """Initialize the store.
        
        Args:
            db_path: Path to SQLite database. Defaults to ~/.llamafarm/vision/images.db
        """
        if db_path is None:
            db_path = _get_vision_data_dir() / "images.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    thumbnail_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    size_bytes INTEGER,
                    content_hash TEXT UNIQUE,
                    ocr_text TEXT,
                    reviewed BOOLEAN DEFAULT FALSE,
                    reviewed_at DATETIME,
                    reviewed_by TEXT
                );
                
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL,
                    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                    model TEXT,
                    verified BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images(id)
                );
                
                CREATE TABLE IF NOT EXISTS labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL,
                    source TEXT DEFAULT 'model',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images(id)
                );
                
                CREATE TABLE IF NOT EXISTS detection_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    node_id TEXT DEFAULT 'local',
                    class_name TEXT NOT NULL,
                    confidence REAL,
                    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                    mask_rle TEXT,
                    stage TEXT,
                    hop_number INTEGER DEFAULT 0,
                    inference_time_ms REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images(id)
                );

                CREATE INDEX IF NOT EXISTS idx_images_source ON images(source);
                CREATE INDEX IF NOT EXISTS idx_images_reviewed ON images(reviewed);
                CREATE INDEX IF NOT EXISTS idx_images_created ON images(created_at);
                CREATE INDEX IF NOT EXISTS idx_detections_image ON detections(image_id);
                CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name);
                CREATE INDEX IF NOT EXISTS idx_labels_image ON labels(image_id);
                CREATE INDEX IF NOT EXISTS idx_labels_label ON labels(label);
                CREATE INDEX IF NOT EXISTS idx_detection_history_image ON detection_history(image_id);
                CREATE INDEX IF NOT EXISTS idx_detection_history_model ON detection_history(model_id);
                CREATE INDEX IF NOT EXISTS idx_detection_history_stage ON detection_history(stage);
            """)
    
    def add_image(self, record: ImageRecord) -> str:
        """Add an image metadata record.
        
        Returns:
            The image ID
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO images 
                (id, file_path, thumbnail_path, created_at, source, 
                 width, height, format, size_bytes, content_hash, ocr_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id,
                record.file_path,
                record.thumbnail_path,
                record.created_at or datetime.utcnow(),
                record.source,
                record.width,
                record.height,
                record.format,
                record.size_bytes,
                record.content_hash,
                record.ocr_text,
            ))
        
        return record.id
    
    def get_image(self, image_id: str) -> ImageRecord | None:
        """Get an image by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM images WHERE id = ?",
                (image_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            return self._row_to_image_record(row)
    
    def add_detection(
        self,
        image_id: str,
        class_name: str,
        confidence: float,
        box: tuple[float, float, float, float],
        model: str,
    ) -> int:
        """Add a detection result for an image.
        
        Returns:
            Detection ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO detections 
                (image_id, class_name, confidence, x1, y1, x2, y2, model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (image_id, class_name, confidence, *box, model))
            return cursor.lastrowid
    
    def get_detections(self, image_id: str) -> list[DetectionRecord]:
        """Get all detections for an image."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM detections WHERE image_id = ? ORDER BY confidence DESC",
                (image_id,)
            ).fetchall()
            
            return [self._row_to_detection_record(row) for row in rows]
    
    def add_label(
        self,
        image_id: str,
        label: str,
        confidence: float,
        source: Literal["model", "human"] = "model",
    ) -> int:
        """Add a classification label for an image."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO labels (image_id, label, confidence, source)
                VALUES (?, ?, ?, ?)
            """, (image_id, label, confidence, source))
            return cursor.lastrowid
    
    def get_labels(self, image_id: str) -> list[LabelRecord]:
        """Get all labels for an image."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM labels WHERE image_id = ? ORDER BY confidence DESC",
                (image_id,)
            ).fetchall()
            
            return [self._row_to_label_record(row) for row in rows]
    
    def add_detection_history(
        self,
        image_id: str,
        model_id: str,
        class_name: str,
        confidence: float,
        box: tuple[float, float, float, float],
        stage: str = "primary",
        hop_number: int = 0,
        node_id: str = "local",
        mask_rle: str | None = None,
        inference_time_ms: float = 0.0,
    ) -> int:
        """Record a model opinion in the cascade history.

        Every hop in the cascade produces one of these. This is the
        audit trail AND the source for building validation sets.

        Args:
            image_id: The image this opinion is about
            model_id: Which model produced this opinion
            class_name: What the model thinks this is
            confidence: How sure the model is
            box: Bounding box (x1, y1, x2, y2)
            stage: "primary", "secondary", "audit", "remote"
            hop_number: Which hop in the cascade (0-based)
            node_id: Atmosphere node ID or "local"
            mask_rle: Run-length encoded segmentation mask
            inference_time_ms: How long inference took

        Returns:
            Detection history record ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO detection_history
                (image_id, model_id, node_id, class_name, confidence,
                 x1, y1, x2, y2, mask_rle, stage, hop_number, inference_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_id, model_id, node_id, class_name, confidence,
                *box, mask_rle, stage, hop_number, inference_time_ms,
            ))
            return cursor.lastrowid

    def get_detection_history(
        self,
        image_id: str,
    ) -> list[DetectionHistoryRecord]:
        """Get all cascade opinions for an image, ordered by hop number."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM detection_history WHERE image_id = ? ORDER BY hop_number, created_at",
                (image_id,)
            ).fetchall()

            return [self._row_to_detection_history_record(row) for row in rows]

    def get_recent_high_confidence(
        self,
        min_confidence: float = 0.7,
        stage: str = "primary",
        limit: int = 50,
    ) -> list[DetectionHistoryRecord]:
        """Get recent high-confidence primary detections for auditing.

        The audit pipeline pulls from here: images the primary model
        was confident about, to re-check with a bigger model.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM detection_history
                WHERE confidence >= ? AND stage = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (min_confidence, stage, limit)).fetchall()

            return [self._row_to_detection_history_record(row) for row in rows]

    def get_pending_review(
        self,
        limit: int = 50,
        source: str | None = None,
    ) -> list[ImageRecord]:
        """Get images pending human review."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM images WHERE reviewed = FALSE"
            params: list[Any] = []
            
            if source:
                query += " AND source = ?"
                params.append(source)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_image_record(row) for row in rows]
    
    def mark_reviewed(
        self,
        image_id: str,
        reviewed_by: str,
        decision: Literal["approved", "rejected", "corrected"] = "approved",
    ) -> None:
        """Mark an image as reviewed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE images 
                SET reviewed = TRUE, reviewed_at = CURRENT_TIMESTAMP, reviewed_by = ?
                WHERE id = ?
            """, (f"{reviewed_by}:{decision}", image_id))
    
    def verify_detection(self, detection_id: int, verified: bool = True) -> None:
        """Mark a detection as verified."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE detections SET verified = ? WHERE id = ?",
                (verified, detection_id)
            )
    
    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            stats["total_images"] = conn.execute(
                "SELECT COUNT(*) FROM images"
            ).fetchone()[0]
            
            stats["pending_review"] = conn.execute(
                "SELECT COUNT(*) FROM images WHERE reviewed = FALSE"
            ).fetchone()[0]
            
            stats["total_detections"] = conn.execute(
                "SELECT COUNT(*) FROM detections"
            ).fetchone()[0]
            
            stats["total_labels"] = conn.execute(
                "SELECT COUNT(*) FROM labels"
            ).fetchone()[0]

            stats["total_detection_history"] = conn.execute(
                "SELECT COUNT(*) FROM detection_history"
            ).fetchone()[0]

            # Top classes
            top_classes = conn.execute("""
                SELECT class_name, COUNT(*) as count
                FROM detections
                GROUP BY class_name
                ORDER BY count DESC
                LIMIT 10
            """).fetchall()
            stats["top_classes"] = {row[0]: row[1] for row in top_classes}
            
            return stats
    
    def delete_old_images(self, before: datetime, limit: int | None = None) -> int:
        """Delete images older than the given timestamp.

        Args:
            before: Delete images created before this timestamp
            limit: Max number of images to delete (oldest first)

        Returns:
            Number of images deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get IDs to delete
            query = "SELECT id FROM images WHERE created_at < ? ORDER BY created_at ASC"
            params: list[Any] = [before]
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            rows = conn.execute(query, params).fetchall()
            
            ids = [row[0] for row in rows]
            
            if not ids:
                return 0
            
            # Delete in order (labels, detection_history, detections, images)
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM labels WHERE image_id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM detection_history WHERE image_id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM detections WHERE image_id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM images WHERE id IN ({placeholders})", ids)
            
            return len(ids)
    
    @staticmethod
    def _row_to_image_record(row: sqlite3.Row) -> ImageRecord:
        """Convert database row to ImageRecord."""
        return ImageRecord(
            id=row["id"],
            file_path=row["file_path"],
            thumbnail_path=row["thumbnail_path"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            source=row["source"] or "unknown",
            width=row["width"] or 0,
            height=row["height"] or 0,
            format=row["format"] or "jpeg",
            size_bytes=row["size_bytes"] or 0,
            content_hash=row["content_hash"] or "",
            ocr_text=row["ocr_text"],
            reviewed=bool(row["reviewed"]),
            reviewed_at=datetime.fromisoformat(row["reviewed_at"]) if row["reviewed_at"] else None,
            reviewed_by=row["reviewed_by"],
        )
    
    @staticmethod
    def _row_to_detection_record(row: sqlite3.Row) -> DetectionRecord:
        """Convert database row to DetectionRecord."""
        return DetectionRecord(
            id=row["id"],
            image_id=row["image_id"],
            class_name=row["class_name"],
            confidence=row["confidence"],
            x1=row["x1"],
            y1=row["y1"],
            x2=row["x2"],
            y2=row["y2"],
            model=row["model"],
            verified=bool(row["verified"]),
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )
    
    @staticmethod
    def _row_to_label_record(row: sqlite3.Row) -> LabelRecord:
        """Convert database row to LabelRecord."""
        return LabelRecord(
            id=row["id"],
            image_id=row["image_id"],
            label=row["label"],
            confidence=row["confidence"],
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    @staticmethod
    def _row_to_detection_history_record(row: sqlite3.Row) -> DetectionHistoryRecord:
        """Convert database row to DetectionHistoryRecord."""
        return DetectionHistoryRecord(
            id=row["id"],
            image_id=row["image_id"],
            model_id=row["model_id"],
            node_id=row["node_id"] or "local",
            class_name=row["class_name"],
            confidence=row["confidence"],
            x1=row["x1"],
            y1=row["y1"],
            x2=row["x2"],
            y2=row["y2"],
            mask_rle=row["mask_rle"],
            stage=row["stage"] or "",
            hop_number=row["hop_number"] or 0,
            inference_time_ms=row["inference_time_ms"] or 0.0,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute a hash for image content (for deduplication)."""
    return hashlib.sha256(image_bytes).hexdigest()[:16]


class ImageStore:
    """Async wrapper for ImageMetadataStore with file storage.
    
    Provides async methods for storing images and their metadata,
    with support for thumbnails and review queue.
    
    Example:
        ```python
        store = ImageStore()
        
        # Save image for review
        await store.save_for_review(
            image_id="img_001",
            image_bytes=frame_bytes,
            detections=[...],
            confidence=0.45,
            model_id="yolov8n",
            source="stream:camera1",
        )
        
        # List pending reviews
        pending = await store.list_for_review(status="pending", limit=50)
        
        # Mark as reviewed
        await store.mark_reviewed(
            image_id="img_001",
            decision="corrected",
            corrected_class="truck",
        )
        ```
    """
    
    def __init__(self, data_dir: Path | str | None = None):
        """Initialize the store.
        
        Args:
            data_dir: Directory for storing images. Defaults to ~/.llamafarm/vision/
        """
        if data_dir is None:
            data_dir = _get_vision_data_dir()
        
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.thumbnails_dir = self.data_dir / "thumbnails"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        
        self._metadata_store = ImageMetadataStore(self.data_dir / "images.db")
    
    async def save_image(
        self,
        image_id: str,
        image_bytes: bytes,
        source: str = "unknown",
        create_thumbnail: bool = True,
    ) -> str:
        """Save an image and return its file path.
        
        Args:
            image_id: Unique image identifier
            image_bytes: Image data
            source: Source identifier
            create_thumbnail: Whether to create a thumbnail
            
        Returns:
            Path to saved image
        """
        import asyncio
        
        # Determine format from magic bytes
        fmt = self._detect_format(image_bytes)
        
        # Sanitize image_id to prevent path traversal
        safe_id = Path(image_id).name
        if safe_id != image_id or ".." in image_id or "/" in image_id or "\\" in image_id:
            raise ValueError(f"Invalid image_id: {image_id!r}")

        # Save image file
        image_path = self.images_dir / f"{safe_id}.{fmt}"
        await asyncio.to_thread(image_path.write_bytes, image_bytes)
        
        # Create thumbnail
        thumbnail_path = None
        if create_thumbnail:
            try:
                thumbnail_path = await self._create_thumbnail(
                    image_id, image_bytes, fmt
                )
            except Exception as e:
                logger.warning(f"Failed to create thumbnail: {e}")
        
        # Get image dimensions
        width, height = await self._get_dimensions(image_bytes)
        
        # Save metadata
        record = ImageRecord(
            id=image_id,
            file_path=str(image_path),
            thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
            created_at=datetime.utcnow(),
            source=source,
            width=width,
            height=height,
            format=fmt,
            size_bytes=len(image_bytes),
            content_hash=compute_image_hash(image_bytes),
        )
        
        await asyncio.to_thread(self._metadata_store.add_image, record)
        
        return str(image_path)
    
    async def save_for_review(
        self,
        image_id: str,
        image_bytes: bytes,
        detections: list,
        confidence: float,
        model_id: str,
        source: str = "unknown",
    ) -> str:
        """Save an image for human review.
        
        Args:
            image_id: Unique image identifier
            image_bytes: Image data
            detections: List of DetectionBox objects
            confidence: Best confidence score
            model_id: Model that produced the detections
            source: Source identifier
            
        Returns:
            Path to saved image
        """
        import asyncio
        
        # Save the image
        image_path = await self.save_image(
            image_id=image_id,
            image_bytes=image_bytes,
            source=source,
        )
        
        # Save detection metadata
        for det in detections:
            await asyncio.to_thread(
                self._metadata_store.add_detection,
                image_id=image_id,
                class_name=det.class_name,
                confidence=det.confidence,
                box=(det.x1, det.y1, det.x2, det.y2),
                model=model_id,
            )
        
        return image_path
    
    async def get_image(self, image_id: str) -> ImageRecord | None:
        """Get image metadata by ID."""
        import asyncio
        return await asyncio.to_thread(self._metadata_store.get_image, image_id)
    
    async def list_for_review(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
        source: str | None = None,
    ) -> list[ImageRecord]:
        """List images for review.
        
        Args:
            status: Filter by review status (pending, approved, rejected, corrected)
            limit: Max items to return
            offset: Pagination offset
            source: Filter by source
            
        Returns:
            List of ImageRecord objects
        """
        import asyncio
        
        # For now, use the pending review method
        # TODO: Add full status filtering to metadata store
        if status is None or status == "pending":
            return await asyncio.to_thread(
                self._metadata_store.get_pending_review,
                limit=limit,
                source=source,
            )
        
        # For other statuses, query by reviewed_by field which stores "reviewer:decision"
        def _query_by_status():
            import sqlite3 as _sqlite3
            with _sqlite3.connect(self._metadata_store.db_path) as conn:
                conn.row_factory = _sqlite3.Row
                query = "SELECT * FROM images WHERE reviewed_by LIKE ?"
                params: list[Any] = [f"%:{status}"]
                if source:
                    query += " AND source = ?"
                    params.append(source)
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                rows = conn.execute(query, params).fetchall()
                return [self._metadata_store._row_to_image_record(row) for row in rows]

        return await asyncio.to_thread(_query_by_status)
    
    async def count_reviews(self, status: str | None = None) -> int:
        """Count reviews by status."""
        import asyncio
        
        def _count():
            with sqlite3.connect(self._metadata_store.db_path) as conn:
                if status == "pending":
                    return conn.execute(
                        "SELECT COUNT(*) FROM images WHERE reviewed = FALSE"
                    ).fetchone()[0]
                elif status is None:
                    return conn.execute(
                        "SELECT COUNT(*) FROM images"
                    ).fetchone()[0]
                else:
                    return conn.execute(
                        "SELECT COUNT(*) FROM images WHERE reviewed_by LIKE ?",
                        (f"%:{status}",)
                    ).fetchone()[0]
        
        return await asyncio.to_thread(_count)
    
    async def count_reviews_today(self) -> int:
        """Count images reviewed today."""
        import asyncio
        
        def _count():
            with sqlite3.connect(self._metadata_store.db_path) as conn:
                return conn.execute("""
                    SELECT COUNT(*) FROM images 
                    WHERE reviewed = TRUE 
                    AND date(reviewed_at) = date('now')
                """).fetchone()[0]
        
        return await asyncio.to_thread(_count)
    
    async def mark_reviewed(
        self,
        image_id: str,
        decision: str,
        reviewed_by: str = "anonymous",
        corrected_class: str | None = None,
    ) -> None:
        """Mark an image as reviewed.
        
        Args:
            image_id: Image to mark
            decision: Review decision (approved, rejected, corrected)
            reviewed_by: Reviewer identifier
            corrected_class: If corrected, the correct class name
        """
        import asyncio
        
        await asyncio.to_thread(
            self._metadata_store.mark_reviewed,
            image_id=image_id,
            reviewed_by=reviewed_by,
            decision=decision,
        )
        
        # If corrected, add a human label
        if decision == "corrected" and corrected_class:
            await asyncio.to_thread(
                self._metadata_store.add_label,
                image_id=image_id,
                label=corrected_class,
                confidence=1.0,
                source="human",
            )
    
    async def _create_thumbnail(
        self,
        image_id: str,
        image_bytes: bytes,
        fmt: str,
        size: int = 256,
    ) -> Path:
        """Create a thumbnail for an image."""
        import asyncio
        from PIL import Image
        import io
        
        def _create():
            img = Image.open(io.BytesIO(image_bytes))
            img.thumbnail((size, size))
            
            thumb_path = self.thumbnails_dir / f"{image_id}_thumb.jpg"
            img.save(thumb_path, "JPEG", quality=85)
            return thumb_path
        
        return await asyncio.to_thread(_create)
    
    async def _get_dimensions(self, image_bytes: bytes) -> tuple[int, int]:
        """Get image dimensions."""
        import asyncio
        
        def _get():
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_bytes))
                return img.size
            except Exception:
                return (0, 0)
        
        return await asyncio.to_thread(_get)
    
    def _detect_format(self, image_bytes: bytes) -> str:
        """Detect image format from magic bytes."""
        if image_bytes[:3] == b'\xff\xd8\xff':
            return "jpg"
        elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return "png"
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            return "gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            return "webp"
        else:
            return "jpg"  # Default
