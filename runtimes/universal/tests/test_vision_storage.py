"""Tests for vision storage layer (image store, retention).

Tests SQLite-based storage, metadata tracking, and retention policies.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from storage.image_store import (
    DetectionHistoryRecord,
    DetectionRecord,
    ImageMetadataStore,
    ImageRecord,
    LabelRecord,
    compute_image_hash,
)

# -----------------------------------------------------------------------------
# Image Store Tests
# -----------------------------------------------------------------------------

class TestImageMetadataStore:
    """Tests for ImageMetadataStore SQLite backend."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_vision.db"
            yield db_path

    @pytest.fixture
    def store(self, temp_db):
        """Create an ImageMetadataStore instance."""
        return ImageMetadataStore(db_path=temp_db)

    def _make_image(self, image_id: str = "img_001", source: str = "test_camera", **kwargs) -> ImageRecord:
        """Helper to create an ImageRecord with defaults."""
        defaults = dict(
            id=image_id,
            file_path=f"/tmp/images/{image_id}.jpg",
            source=source,
            width=640,
            height=480,
            format="jpeg",
            size_bytes=1024,
            content_hash=f"hash_{image_id}",
        )
        defaults.update(kwargs)
        return ImageRecord(**defaults)

    def test_init_creates_tables(self, temp_db):
        """Test that __init__ creates required tables."""
        import sqlite3

        ImageMetadataStore(db_path=temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "images" in tables
        assert "detections" in tables
        assert "labels" in tables
        assert "detection_history" in tables

    def test_add_image(self, store):
        """Test adding an image record."""
        record = self._make_image()
        image_id = store.add_image(record)

        assert image_id == "img_001"

    def test_get_image(self, store):
        """Test retrieving an image by ID."""
        record = self._make_image(source="camera_1")
        store.add_image(record)

        result = store.get_image("img_001")

        assert result is not None
        assert result.id == "img_001"
        assert result.source == "camera_1"
        assert result.file_path == "/tmp/images/img_001.jpg"
        assert result.width == 640
        assert result.height == 480

    def test_get_nonexistent_image(self, store):
        """Test getting image that doesn't exist."""
        result = store.get_image("nonexistent_id")
        assert result is None

    def test_add_detection(self, store):
        """Test adding a detection record."""
        store.add_image(self._make_image())

        detection_id = store.add_detection(
            image_id="img_001",
            class_name="person",
            confidence=0.95,
            box=(10.0, 20.0, 100.0, 200.0),
            model="yolov8n",
        )

        assert detection_id is not None
        assert isinstance(detection_id, int)

    def test_get_detections_for_image(self, store):
        """Test getting all detections for an image."""
        store.add_image(self._make_image())

        store.add_detection("img_001", "person", 0.9, (10, 10, 50, 50), "yolo")
        store.add_detection("img_001", "car", 0.85, (100, 100, 200, 200), "yolo")

        detections = store.get_detections("img_001")

        assert len(detections) == 2
        assert {d.class_name for d in detections} == {"person", "car"}
        # Ordered by confidence DESC
        assert detections[0].confidence >= detections[1].confidence

    def test_detection_bbox_coords(self, store):
        """Test that detection bbox coordinates are stored correctly."""
        store.add_image(self._make_image())
        store.add_detection("img_001", "person", 0.9, (10.5, 20.5, 100.5, 200.5), "yolo")

        dets = store.get_detections("img_001")
        assert len(dets) == 1
        assert dets[0].x1 == 10.5
        assert dets[0].y1 == 20.5
        assert dets[0].x2 == 100.5
        assert dets[0].y2 == 200.5

    def test_add_and_get_label(self, store):
        """Test storing and retrieving a label."""
        store.add_image(self._make_image())

        label_id = store.add_label(
            image_id="img_001",
            label="cat",
            confidence=0.92,
            source="human",
        )

        labels = store.get_labels("img_001")

        assert len(labels) == 1
        assert labels[0].label == "cat"
        assert labels[0].source == "human"
        assert labels[0].confidence == 0.92

    def test_add_model_label(self, store):
        """Test storing a model-generated label."""
        store.add_image(self._make_image())

        store.add_label("img_001", "dog", 0.88, "model")

        labels = store.get_labels("img_001")
        assert len(labels) == 1
        assert labels[0].source == "model"

    def test_get_pending_review(self, store):
        """Test getting images pending review."""
        # Add 3 images, all unreviewed by default
        for i in range(3):
            store.add_image(self._make_image(
                image_id=f"img_{i}",
                content_hash=f"hash_{i}",
            ))

        pending = store.get_pending_review(limit=10)
        assert len(pending) == 3

    def test_get_pending_review_by_source(self, store):
        """Test filtering pending review by source."""
        store.add_image(self._make_image("img_a", source="camera_a", content_hash="ha"))
        store.add_image(self._make_image("img_b", source="camera_a", content_hash="hb"))
        store.add_image(self._make_image("img_c", source="camera_b", content_hash="hc"))

        results = store.get_pending_review(limit=10, source="camera_a")
        assert len(results) == 2
        assert all(r.source == "camera_a" for r in results)

    def test_mark_reviewed(self, store):
        """Test marking an image as reviewed."""
        store.add_image(self._make_image())

        store.mark_reviewed("img_001", reviewed_by="alice", decision="approved")

        record = store.get_image("img_001")
        assert record.reviewed is True
        assert record.reviewed_by == "alice:approved"

        # Should no longer appear in pending
        pending = store.get_pending_review()
        assert len(pending) == 0

    def test_verify_detection(self, store):
        """Test marking a detection as verified."""
        store.add_image(self._make_image())
        det_id = store.add_detection("img_001", "person", 0.9, (10, 10, 50, 50), "yolo")

        store.verify_detection(det_id, verified=True)

        dets = store.get_detections("img_001")
        assert dets[0].verified is True

    def test_get_stats(self, store):
        """Test getting storage statistics."""
        store.add_image(self._make_image("img_1", content_hash="h1"))
        store.add_image(self._make_image("img_2", source="cam_b", content_hash="h2"))
        store.add_detection("img_1", "person", 0.9, (0, 0, 10, 10), "yolo")
        store.add_label("img_1", "indoor", 0.8, "model")

        stats = store.get_stats()

        assert stats["total_images"] == 2
        assert stats["total_detections"] == 1
        assert stats["total_labels"] == 1
        assert stats["pending_review"] == 2
        assert "top_classes" in stats

    def test_delete_old_images(self, store):
        """Test deleting images older than a cutoff."""
        import sqlite3

        store.add_image(self._make_image("old_img", content_hash="hold"))

        # Backdate the image using the same format SQLite/Python adapter uses
        old_time = datetime.utcnow() - timedelta(hours=25)
        with sqlite3.connect(store.db_path) as conn:
            conn.execute(
                "UPDATE images SET created_at = ? WHERE id = ?",
                (old_time, "old_img")
            )

        cutoff = datetime.utcnow() - timedelta(hours=24)
        deleted = store.delete_old_images(cutoff)

        assert len(deleted) == 1
        assert deleted[0][0] == "old_img"  # id
        assert deleted[0][1] == "/tmp/images/old_img.jpg"  # file_path
        assert store.get_image("old_img") is None

    def test_delete_old_cascades_detections(self, store):
        """Test that deleting images also removes related detections and labels."""
        import sqlite3

        store.add_image(self._make_image("old_img", content_hash="hold"))
        store.add_detection("old_img", "person", 0.9, (0, 0, 10, 10), "yolo")
        store.add_label("old_img", "indoor", 0.8, "model")
        store.add_detection_history(
            "old_img", "yolov8n", "person", 0.9,
            (0, 0, 10, 10), stage="primary",
        )

        # Backdate using datetime object (same format as Python sqlite3 adapter)
        old_time = datetime.utcnow() - timedelta(hours=25)
        with sqlite3.connect(store.db_path) as conn:
            conn.execute(
                "UPDATE images SET created_at = ? WHERE id = ?",
                (old_time, "old_img")
            )

        cutoff = datetime.utcnow() - timedelta(hours=24)
        store.delete_old_images(cutoff)

        assert store.get_detections("old_img") == []
        assert store.get_labels("old_img") == []
        assert store.get_detection_history("old_img") == []


# -----------------------------------------------------------------------------
# Detection History Tests
# -----------------------------------------------------------------------------

class TestDetectionHistory:
    """Tests for cascade detection history tracking."""

    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_history.db"
            store = ImageMetadataStore(db_path=db_path)
            # Pre-add an image
            store.add_image(ImageRecord(
                id="img_001",
                file_path="/tmp/img.jpg",
                source="test",
                content_hash="test_hash",
            ))
            yield store

    def test_add_detection_history(self, store):
        """Test recording a model opinion."""
        record_id = store.add_detection_history(
            image_id="img_001",
            model_id="yolov8n",
            class_name="person",
            confidence=0.85,
            box=(10, 20, 100, 200),
            stage="primary",
            hop_number=0,
        )

        assert isinstance(record_id, int)

    def test_get_detection_history(self, store):
        """Test retrieving cascade history ordered by hop."""
        store.add_detection_history(
            "img_001", "yolov8n", "person", 0.55,
            (10, 20, 100, 200), stage="primary", hop_number=0,
        )
        store.add_detection_history(
            "img_001", "yolov8m", "person", 0.88,
            (12, 22, 98, 198), stage="secondary", hop_number=1,
        )

        history = store.get_detection_history("img_001")

        assert len(history) == 2
        assert history[0].model_id == "yolov8n"
        assert history[0].hop_number == 0
        assert history[1].model_id == "yolov8m"
        assert history[1].hop_number == 1

    def test_detection_history_with_mask(self, store):
        """Test recording detection with segmentation mask."""
        store.add_detection_history(
            "img_001", "yolov8n-seg", "person", 0.9,
            (10, 20, 100, 200),
            stage="primary",
            mask_rle="encoded_mask_data",
            inference_time_ms=5.2,
        )

        history = store.get_detection_history("img_001")
        assert history[0].mask_rle == "encoded_mask_data"
        assert history[0].inference_time_ms == 5.2

    def test_detection_history_remote_node(self, store):
        """Test recording detection from a remote node."""
        store.add_detection_history(
            "img_001", "yolov8x", "person", 0.95,
            (10, 20, 100, 200),
            stage="remote",
            hop_number=2,
            node_id="gpu-server-1",
        )

        history = store.get_detection_history("img_001")
        assert history[0].node_id == "gpu-server-1"
        assert history[0].stage == "remote"

    def test_get_recent_high_confidence(self, store):
        """Test querying recent high-confidence detections for audit."""
        # Add a high-confidence primary detection
        store.add_detection_history(
            "img_001", "yolov8n", "person", 0.92,
            (10, 20, 100, 200), stage="primary",
        )

        results = store.get_recent_high_confidence(
            min_confidence=0.7, stage="primary", limit=10,
        )

        assert len(results) == 1
        assert results[0].confidence >= 0.7

    def test_get_recent_high_confidence_filters_low(self, store):
        """Test that low-confidence detections are filtered out."""
        store.add_detection_history(
            "img_001", "yolov8n", "person", 0.45,
            (10, 20, 100, 200), stage="primary",
        )

        results = store.get_recent_high_confidence(min_confidence=0.7)
        assert len(results) == 0


# -----------------------------------------------------------------------------
# Retention Policy Tests
# -----------------------------------------------------------------------------

class TestRetentionPolicy:
    """Tests for retention policy management."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        return ImageMetadataStore(db_path=temp_dir / "test.db")

    @pytest.fixture
    def policy(self, store):
        from storage.retention import RetentionConfig, RetentionPolicy
        return RetentionPolicy(store, RetentionConfig(
            high_confidence_hours=1,
            cleanup_enabled=True,
        ))

    def _add_images(self, store, count, source="test"):
        """Helper to add multiple images."""
        ids = []
        for i in range(count):
            record = ImageRecord(
                id=f"img_{i}",
                file_path=f"/tmp/img_{i}.jpg",
                source=source,
                content_hash=f"hash_{i}",
            )
            store.add_image(record)
            ids.append(record.id)
        return ids

    def test_cleanup_by_age(self, store, policy):
        """Test cleanup removes old images past the very_low_confidence retention."""
        import sqlite3

        self._add_images(store, 3)

        # Backdate all images beyond very_low_confidence_hours (720h default)
        old_time = datetime.utcnow() - timedelta(hours=800)
        with sqlite3.connect(store.db_path) as conn:
            conn.execute("UPDATE images SET created_at = ?", (old_time,))

        stats = policy.cleanup()
        assert stats.images_deleted >= 1

    def test_cleanup_preserves_recent(self, store, policy):
        """Test cleanup does not remove recent images."""
        self._add_images(store, 3)

        stats = policy.cleanup()
        assert stats.images_deleted == 0

    def test_cleanup_disabled(self, store):
        """Test that cleanup is a no-op when disabled."""
        from storage.retention import RetentionConfig, RetentionPolicy

        policy = RetentionPolicy(store, RetentionConfig(cleanup_enabled=False))
        self._add_images(store, 3)

        stats = policy.cleanup()
        assert stats.images_deleted == 0

    def test_retention_hours_by_confidence(self, store, policy):
        """Test retention hours mapping by confidence level."""
        assert policy.get_retention_hours(0.95) == 1       # high
        assert policy.get_retention_hours(0.80) == 24      # medium
        assert policy.get_retention_hours(0.60) == 168     # low
        assert policy.get_retention_hours(0.30) == 720     # very low


# -----------------------------------------------------------------------------
# Data Classes Tests
# -----------------------------------------------------------------------------

class TestStorageDataclasses:
    """Tests for storage dataclasses."""

    def test_image_record_defaults(self):
        """Test ImageRecord with required fields only."""
        record = ImageRecord(
            id="test_id",
            file_path="/tmp/test.jpg",
        )

        assert record.id == "test_id"
        assert record.source == "unknown"
        assert record.width == 0
        assert record.height == 0
        assert record.reviewed is False

    def test_image_record_full(self):
        """Test ImageRecord with all fields."""
        now = datetime.utcnow()
        record = ImageRecord(
            id="test_id",
            file_path="/tmp/test.jpg",
            thumbnail_path="/tmp/test_thumb.jpg",
            created_at=now,
            source="camera_1",
            width=1920,
            height=1080,
            format="png",
            size_bytes=2048,
            content_hash="abc123",
            ocr_text="hello world",
        )

        assert record.source == "camera_1"
        assert record.width == 1920
        assert record.format == "png"
        assert record.content_hash == "abc123"

    def test_detection_record(self):
        """Test DetectionRecord dataclass."""
        record = DetectionRecord(
            id=1,
            image_id="img_1",
            class_name="person",
            confidence=0.95,
            x1=10, y1=20, x2=100, y2=200,
            model="yolov8n",
        )

        assert record.class_name == "person"
        assert record.confidence == 0.95
        assert record.x1 == 10
        assert record.y2 == 200
        assert record.verified is False

    def test_detection_history_record(self):
        """Test DetectionHistoryRecord dataclass."""
        record = DetectionHistoryRecord(
            image_id="img_1",
            model_id="yolov8n",
            class_name="person",
            confidence=0.85,
            x1=10, y1=20, x2=100, y2=200,
            stage="primary",
            hop_number=0,
            node_id="local",
            inference_time_ms=5.3,
        )

        assert record.model_id == "yolov8n"
        assert record.stage == "primary"
        assert record.node_id == "local"

    def test_label_record(self):
        """Test LabelRecord dataclass."""
        record = LabelRecord(
            id=1,
            image_id="img_1",
            label="cat",
            confidence=0.9,
            source="human",
        )

        assert record.label == "cat"
        assert record.source == "human"

    def test_compute_image_hash(self):
        """Test image hash computation."""
        hash1 = compute_image_hash(b"image_data_1")
        hash2 = compute_image_hash(b"image_data_2")
        hash_dup = compute_image_hash(b"image_data_1")

        assert hash1 != hash2
        assert hash1 == hash_dup
        assert len(hash1) == 16  # sha256 truncated to 16 chars


# -----------------------------------------------------------------------------
# Concurrent Access Tests
# -----------------------------------------------------------------------------

class TestConcurrentAccess:
    """Tests for concurrent database access."""

    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_concurrent.db"
            yield ImageMetadataStore(db_path=db_path)

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, store):
        """Test concurrent writes don't corrupt data."""

        async def add_images(batch: int, count: int):
            for i in range(count):
                record = ImageRecord(
                    id=f"img_{batch}_{i}",
                    file_path=f"/tmp/img_{batch}_{i}.jpg",
                    source=f"source_{batch}",
                    content_hash=f"hash_{batch}_{i}",
                )
                store.add_image(record)
                await asyncio.sleep(0)

        await asyncio.gather(
            add_images(0, 10),
            add_images(1, 10),
            add_images(2, 10),
        )

        stats = store.get_stats()
        assert stats["total_images"] == 30

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, store):
        """Test concurrent reads and writes."""
        # Pre-populate
        ids = []
        for i in range(10):
            record = ImageRecord(
                id=f"img_{i}",
                file_path=f"/tmp/img_{i}.jpg",
                source="test",
                content_hash=f"hash_{i}",
            )
            store.add_image(record)
            ids.append(record.id)

        async def read_images():
            for img_id in ids:
                store.get_image(img_id)
                await asyncio.sleep(0)

        async def write_images():
            for i in range(10):
                record = ImageRecord(
                    id=f"new_{i}",
                    file_path=f"/tmp/new_{i}.jpg",
                    source="test",
                    content_hash=f"new_hash_{i}",
                )
                store.add_image(record)
                await asyncio.sleep(0)

        await asyncio.gather(
            read_images(),
            write_images(),
            read_images(),
        )

        stats = store.get_stats()
        assert stats["total_images"] == 20


# -----------------------------------------------------------------------------
# Edge Cases Tests
# -----------------------------------------------------------------------------

class TestStorageEdgeCases:
    """Tests for edge cases in storage."""

    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_edge.db"
            yield ImageMetadataStore(db_path=db_path)

    def test_add_image_replace_on_duplicate_hash(self, store):
        """Test that images with same content_hash are replaced."""
        store.add_image(ImageRecord(
            id="img_1",
            file_path="/tmp/img_1.jpg",
            content_hash="same_hash",
        ))
        # Same ID replaces (INSERT OR REPLACE)
        store.add_image(ImageRecord(
            id="img_1",
            file_path="/tmp/img_1_v2.jpg",
            content_hash="same_hash",
        ))

        record = store.get_image("img_1")
        assert record.file_path == "/tmp/img_1_v2.jpg"

    def test_special_characters_in_source(self, store):
        """Test special characters in source name."""
        special_source = "camera/path:with'special\"chars"
        store.add_image(ImageRecord(
            id="img_special",
            file_path="/tmp/special.jpg",
            source=special_source,
            content_hash="hash_special",
        ))

        record = store.get_image("img_special")
        assert record.source == special_source

    def test_unicode_class_names(self, store):
        """Test Unicode characters in class names."""
        store.add_image(ImageRecord(
            id="img_uni",
            file_path="/tmp/uni.jpg",
            content_hash="hash_uni",
        ))
        store.add_detection(
            "img_uni",
            class_name="\u4eba\u7269",  # Chinese for "person"
            confidence=0.9,
            box=(0, 0, 10, 10),
            model="test",
        )

        detections = store.get_detections("img_uni")
        assert detections[0].class_name == "\u4eba\u7269"

    def test_empty_detection_history(self, store):
        """Test getting history for image with no cascade records."""
        store.add_image(ImageRecord(
            id="img_empty",
            file_path="/tmp/empty.jpg",
            content_hash="hash_empty",
        ))

        history = store.get_detection_history("img_empty")
        assert history == []

    def test_multiple_labels_same_image(self, store):
        """Test adding multiple labels to the same image."""
        store.add_image(ImageRecord(
            id="img_multi",
            file_path="/tmp/multi.jpg",
            content_hash="hash_multi",
        ))

        store.add_label("img_multi", "outdoor", 0.8, "model")
        store.add_label("img_multi", "sunny", 0.7, "model")
        store.add_label("img_multi", "park", 0.95, "human")

        labels = store.get_labels("img_multi")
        assert len(labels) == 3
        # Ordered by confidence DESC
        assert labels[0].label == "park"
        assert labels[0].confidence == 0.95

    def test_ocr_text_storage(self, store):
        """Test storing and retrieving OCR text."""
        store.add_image(ImageRecord(
            id="img_ocr",
            file_path="/tmp/ocr.jpg",
            content_hash="hash_ocr",
            ocr_text="Hello World\nLine 2",
        ))

        record = store.get_image("img_ocr")
        assert record.ocr_text == "Hello World\nLine 2"
