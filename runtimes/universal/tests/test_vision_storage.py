"""Tests for vision storage layer (image store, retention).

Tests SQLite-based storage, metadata tracking, and retention policies.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import pytest


# -----------------------------------------------------------------------------
# Image Store Tests
# -----------------------------------------------------------------------------

class TestImageStore:
    """Tests for ImageStore SQLite backend."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_vision.db"
            yield str(db_path)

    @pytest.fixture
    def image_store(self, temp_db):
        """Create an ImageStore instance."""
        from storage.image_store import ImageStore
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        return store

    def test_initialize_creates_tables(self, temp_db):
        """Test that initialize creates required tables."""
        from storage.image_store import ImageStore
        import sqlite3
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        
        # Check tables exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        assert "images" in tables
        assert "detections" in tables
        assert "labels" in tables

    def test_store_image(self, image_store):
        """Test storing an image record."""
        image_id = image_store.store_image(
            image_data=b"fake_image_bytes",
            source="test_camera",
            metadata={"width": 640, "height": 480},
        )
        
        assert image_id is not None
        assert len(image_id) > 0

    def test_get_image(self, image_store):
        """Test retrieving an image by ID."""
        image_id = image_store.store_image(
            image_data=b"test_data",
            source="camera_1",
        )
        
        record = image_store.get_image(image_id)
        
        assert record is not None
        assert record.id == image_id
        assert record.image_data == b"test_data"
        assert record.source == "camera_1"

    def test_get_nonexistent_image(self, image_store):
        """Test getting image that doesn't exist."""
        record = image_store.get_image("nonexistent_id")
        assert record is None

    def test_store_detection(self, image_store):
        """Test storing a detection record."""
        # First store an image
        image_id = image_store.store_image(b"image", "test")
        
        # Store detection
        detection_id = image_store.store_detection(
            image_id=image_id,
            class_name="person",
            confidence=0.95,
            box={"x1": 10, "y1": 20, "x2": 100, "y2": 200},
            model_id="yolov8n",
        )
        
        assert detection_id is not None

    def test_get_detections_for_image(self, image_store):
        """Test getting all detections for an image."""
        image_id = image_store.store_image(b"image", "test")
        
        # Store multiple detections
        image_store.store_detection(image_id, "person", 0.9, {"x1": 10, "y1": 10, "x2": 50, "y2": 50}, "yolo")
        image_store.store_detection(image_id, "car", 0.85, {"x1": 100, "y1": 100, "x2": 200, "y2": 200}, "yolo")
        
        detections = image_store.get_detections(image_id)
        
        assert len(detections) == 2
        assert {d.class_name for d in detections} == {"person", "car"}

    def test_store_and_get_label(self, image_store):
        """Test storing and retrieving a label."""
        image_id = image_store.store_image(b"image", "test")
        
        # Store label
        label_id = image_store.store_label(
            image_id=image_id,
            class_name="cat",
            annotator="human",
            box={"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        )
        
        labels = image_store.get_labels(image_id)
        
        assert len(labels) == 1
        assert labels[0].class_name == "cat"
        assert labels[0].annotator == "human"

    def test_list_images_pagination(self, image_store):
        """Test listing images with pagination."""
        # Store 10 images
        for i in range(10):
            image_store.store_image(f"image_{i}".encode(), f"source_{i}")
        
        # Get first page
        page1 = image_store.list_images(limit=5, offset=0)
        assert len(page1) == 5
        
        # Get second page
        page2 = image_store.list_images(limit=5, offset=5)
        assert len(page2) == 5
        
        # Verify no overlap
        ids1 = {img.id for img in page1}
        ids2 = {img.id for img in page2}
        assert ids1.isdisjoint(ids2)

    def test_list_images_by_source(self, image_store):
        """Test filtering images by source."""
        image_store.store_image(b"img1", "camera_a")
        image_store.store_image(b"img2", "camera_a")
        image_store.store_image(b"img3", "camera_b")
        
        results = image_store.list_images(source="camera_a")
        
        assert len(results) == 2
        assert all(r.source == "camera_a" for r in results)

    def test_delete_image_cascades(self, image_store):
        """Test that deleting image removes detections and labels."""
        image_id = image_store.store_image(b"image", "test")
        image_store.store_detection(image_id, "person", 0.9, {}, "yolo")
        image_store.store_label(image_id, "person", "human", {})
        
        # Delete image
        deleted = image_store.delete_image(image_id)
        
        assert deleted is True
        assert image_store.get_image(image_id) is None
        assert image_store.get_detections(image_id) == []
        assert image_store.get_labels(image_id) == []

    def test_count_images(self, image_store):
        """Test counting total images."""
        for i in range(5):
            image_store.store_image(f"img_{i}".encode(), "test")
        
        count = image_store.count_images()
        assert count == 5

    def test_get_stats(self, image_store):
        """Test getting storage statistics."""
        image_store.store_image(b"img1", "cam_a")
        image_store.store_image(b"img2", "cam_b")
        
        stats = image_store.get_stats()
        
        assert stats["total_images"] == 2
        assert "sources" in stats
        assert len(stats["sources"]) == 2


# -----------------------------------------------------------------------------
# Retention Policy Tests
# -----------------------------------------------------------------------------

class TestRetentionPolicy:
    """Tests for retention policy management."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_retention.db"
            yield str(db_path)

    @pytest.fixture
    def retention_manager(self, temp_db):
        """Create a RetentionManager instance."""
        from storage.retention import RetentionManager
        from storage.image_store import ImageStore
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        
        return RetentionManager(store)

    def test_apply_max_age_policy(self, retention_manager, temp_db):
        """Test applying max age retention policy."""
        from storage.image_store import ImageStore
        import sqlite3
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        
        # Store an old image (manually set timestamp)
        image_id = store.store_image(b"old_image", "test")
        
        # Manually backdate the image
        conn = sqlite3.connect(temp_db)
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        conn.execute(
            "UPDATE images SET created_at = ? WHERE id = ?",
            (old_time, image_id)
        )
        conn.commit()
        conn.close()
        
        # Apply 24-hour retention
        deleted = retention_manager.apply_max_age(max_hours=24)
        
        assert deleted >= 1
        assert store.get_image(image_id) is None

    def test_apply_max_count_policy(self, retention_manager, temp_db):
        """Test applying max count retention policy."""
        from storage.image_store import ImageStore
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        
        # Store 10 images
        for i in range(10):
            store.store_image(f"img_{i}".encode(), "test")
            time.sleep(0.01)  # Small delay to ensure ordering
        
        # Apply max 5 images
        deleted = retention_manager.apply_max_count(max_count=5)
        
        assert deleted == 5
        assert store.count_images() == 5

    def test_apply_max_size_policy(self, retention_manager, temp_db):
        """Test applying max size retention policy."""
        from storage.image_store import ImageStore
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        
        # Store large images (1KB each)
        for i in range(10):
            store.store_image(b"x" * 1024, "test")
            time.sleep(0.01)
        
        # Apply max 5KB limit
        deleted = retention_manager.apply_max_size(max_bytes=5 * 1024)
        
        assert deleted > 0
        # Should have ~5 images left
        assert store.count_images() <= 6

    def test_run_all_policies(self, retention_manager, temp_db):
        """Test running all retention policies."""
        from storage.image_store import ImageStore
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        
        # Store some images
        for i in range(5):
            store.store_image(f"img_{i}".encode(), "test")
        
        # Run with permissive limits
        result = retention_manager.run_all(
            max_hours=1000,
            max_count=100,
            max_bytes=1024 * 1024,
        )
        
        assert "total_deleted" in result
        assert result["total_deleted"] == 0


# -----------------------------------------------------------------------------
# Data Classes Tests
# -----------------------------------------------------------------------------

class TestStorageDataclasses:
    """Tests for storage dataclasses."""

    def test_image_record(self):
        """Test ImageRecord dataclass."""
        from storage.image_store import ImageRecord
        
        record = ImageRecord(
            id="test_id",
            image_data=b"data",
            source="camera_1",
            created_at=time.time(),
            metadata={"width": 640},
        )
        
        assert record.id == "test_id"
        assert record.source == "camera_1"
        assert record.metadata["width"] == 640

    def test_detection_record(self):
        """Test DetectionRecord dataclass."""
        from storage.image_store import DetectionRecord
        
        record = DetectionRecord(
            id="det_1",
            image_id="img_1",
            class_name="person",
            confidence=0.95,
            box={"x1": 10, "y1": 20, "x2": 100, "y2": 200},
            model_id="yolov8n",
            created_at=time.time(),
        )
        
        assert record.class_name == "person"
        assert record.confidence == 0.95

    def test_label_record(self):
        """Test LabelRecord dataclass."""
        from storage.image_store import LabelRecord
        
        record = LabelRecord(
            id="label_1",
            image_id="img_1",
            class_name="cat",
            annotator="human_reviewer",
            box={"x1": 0, "y1": 0, "x2": 50, "y2": 50},
            created_at=time.time(),
        )
        
        assert record.annotator == "human_reviewer"


# -----------------------------------------------------------------------------
# Concurrent Access Tests
# -----------------------------------------------------------------------------

class TestConcurrentAccess:
    """Tests for concurrent database access."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_concurrent.db"
            yield str(db_path)

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, temp_db):
        """Test concurrent writes don't corrupt data."""
        from storage.image_store import ImageStore
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        
        async def store_images(n):
            for i in range(n):
                store.store_image(f"data_{n}_{i}".encode(), f"source_{n}")
                await asyncio.sleep(0)
        
        # Run concurrent writes
        await asyncio.gather(
            store_images(10),
            store_images(10),
            store_images(10),
        )
        
        # Verify all images stored
        count = store.count_images()
        assert count == 30

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, temp_db):
        """Test concurrent reads and writes."""
        from storage.image_store import ImageStore
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        
        # Pre-populate
        ids = [store.store_image(f"img_{i}".encode(), "test") for i in range(10)]
        
        async def read_images():
            for img_id in ids:
                store.get_image(img_id)
                await asyncio.sleep(0)
        
        async def write_images():
            for i in range(10):
                store.store_image(f"new_{i}".encode(), "test")
                await asyncio.sleep(0)
        
        # Run concurrent reads and writes
        await asyncio.gather(
            read_images(),
            write_images(),
            read_images(),
        )
        
        # Verify data integrity
        assert store.count_images() == 20


# -----------------------------------------------------------------------------
# Edge Cases Tests
# -----------------------------------------------------------------------------

class TestStorageEdgeCases:
    """Tests for edge cases in storage."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_edge.db"
            yield str(db_path)

    def test_store_empty_image(self, temp_db):
        """Test storing empty image data."""
        from storage.image_store import ImageStore
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        
        image_id = store.store_image(b"", "test")
        record = store.get_image(image_id)
        
        assert record.image_data == b""

    def test_store_large_metadata(self, temp_db):
        """Test storing large metadata dict."""
        from storage.image_store import ImageStore
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        
        image_id = store.store_image(b"data", "test", metadata=large_metadata)
        record = store.get_image(image_id)
        
        assert len(record.metadata) == 100

    def test_special_characters_in_source(self, temp_db):
        """Test special characters in source name."""
        from storage.image_store import ImageStore
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        
        special_source = "camera/path:with'special\"chars"
        image_id = store.store_image(b"data", special_source)
        
        record = store.get_image(image_id)
        assert record.source == special_source

    def test_unicode_class_names(self, temp_db):
        """Test Unicode characters in class names."""
        from storage.image_store import ImageStore
        
        store = ImageStore(db_path=temp_db)
        store.initialize()
        
        image_id = store.store_image(b"data", "test")
        store.store_detection(
            image_id, 
            class_name="人物",  # Chinese for "person"
            confidence=0.9,
            box={},
            model_id="test"
        )
        
        detections = store.get_detections(image_id)
        assert detections[0].class_name == "人物"
