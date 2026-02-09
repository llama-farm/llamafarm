"""Tests for replay buffer SQLite persistence.

Verifies that the replay buffer:
- Persists samples to SQLite and restores on restart
- Handles ModelOpinion serialization/deserialization
- Correctly evicts low-priority samples
- Survives crash/restart cycles (data integrity)
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from vision_training.replay_buffer import (
    ModelOpinion,
    ReplayBuffer,
    ReplayBufferPersistence,
    ReplaySample,
)


@pytest.fixture
def temp_storage():
    """Create a temp directory for replay buffer persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def persistent_buffer(temp_storage):
    """Create a replay buffer with SQLite persistence."""
    return ReplayBuffer(max_size=100, storage_dir=temp_storage)


class TestReplayBufferPersistence:
    """Tests for SQLite persistence layer."""

    def test_persist_and_restore_basic(self, temp_storage):
        """Samples added to buffer survive a restart."""
        # Write some samples
        buf1 = ReplayBuffer(max_size=100, storage_dir=temp_storage)
        buf1.add_correction(
            image_id="img_001",
            image_path="/path/to/image.jpg",
            corrected_label="truck",
            original_confidence=0.4,
        )
        buf1.add_low_confidence(
            image_id="img_002",
            image_path="/path/to/image2.jpg",
            predicted_label="car",
            confidence=0.55,
        )
        assert len(buf1) == 2

        # Create a new buffer pointing at the same storage
        buf2 = ReplayBuffer(max_size=100, storage_dir=temp_storage)
        assert len(buf2) == 2

        sample = buf2.get("img_001")
        assert sample is not None
        assert sample.label == "truck"
        assert sample.source == "correction"
        assert sample.priority == 2.0

        sample2 = buf2.get("img_002")
        assert sample2 is not None
        assert sample2.source == "low_confidence"

    def test_persist_cascade_resolved(self, temp_storage):
        """Cascade-resolved samples persist with full opinion chain."""
        buf = ReplayBuffer(max_size=100, storage_dir=temp_storage)

        opinions = [
            ModelOpinion(
                model_id="yolov8n",
                node_id="local",
                class_name="bird",
                confidence=0.45,
                bbox=(100.0, 200.0, 300.0, 400.0),
                inference_time_ms=5.2,
            ),
            ModelOpinion(
                model_id="yolov8m",
                node_id="local",
                class_name="bird",
                confidence=0.92,
                bbox=(105.0, 198.0, 305.0, 402.0),
                inference_time_ms=28.1,
            ),
        ]

        buf.add_cascade_resolved(
            image_id="frame_001",
            image_path="/frames/001.jpg",
            opinions=opinions,
            final_label="bird",
            bbox=(100.0, 200.0, 300.0, 400.0),
            mask_rle="abc123",
            crop_path="/crops/001.jpg",
            resolving_hop=1,
        )

        # Restore
        buf2 = ReplayBuffer(max_size=100, storage_dir=temp_storage)
        sample = buf2.get("frame_001")

        assert sample is not None
        assert sample.source == "escalation_resolved"
        assert sample.final_label == "bird"
        assert sample.final_source == "cascade"
        assert sample.bbox == (100.0, 200.0, 300.0, 400.0)
        assert sample.mask_rle == "abc123"
        assert sample.crop_path == "/crops/001.jpg"

        # Opinions restored correctly
        assert len(sample.opinions) == 2
        assert sample.opinions[0].model_id == "yolov8n"
        assert sample.opinions[0].confidence == 0.45
        assert sample.opinions[0].bbox == (100.0, 200.0, 300.0, 400.0)
        assert sample.opinions[1].model_id == "yolov8m"
        assert sample.opinions[1].confidence == 0.92

    def test_persist_audit_disagreement(self, temp_storage):
        """Audit disagreement samples persist correctly."""
        buf = ReplayBuffer(max_size=100, storage_dir=temp_storage)

        opinions = [
            ModelOpinion(model_id="yolov8n", class_name="airplane", confidence=0.82),
            ModelOpinion(model_id="yolov8x", class_name="bird", confidence=0.91),
        ]

        buf.add_audit_disagreement(
            image_id="audit_001",
            image_path="/images/audit_001.jpg",
            opinions=opinions,
            audit_label="bird",
            bbox=(50.0, 60.0, 200.0, 300.0),
        )

        buf2 = ReplayBuffer(max_size=100, storage_dir=temp_storage)
        sample = buf2.get("audit_001")

        assert sample is not None
        assert sample.source == "audit"
        assert sample.final_source == "audit"
        assert sample.final_label == "bird"
        assert sample.priority == 2.0

    def test_eviction_persists(self, temp_storage):
        """When buffer evicts a sample, it's also removed from SQLite."""
        buf = ReplayBuffer(max_size=3, storage_dir=temp_storage)

        # Fill buffer with low-priority items
        for i in range(3):
            buf.add_low_confidence(
                image_id=f"low_{i}",
                image_path=f"/p/{i}.jpg",
                predicted_label="car",
                confidence=0.6,
            )

        assert len(buf) == 3

        # Add a high-priority correction, should evict one
        buf.add_correction(
            image_id="correction_1",
            image_path="/p/c.jpg",
            corrected_label="truck",
        )
        assert len(buf) == 3

        # Verify persistence reflects the eviction
        buf2 = ReplayBuffer(max_size=100, storage_dir=temp_storage)
        assert len(buf2) == 3
        assert buf2.get("correction_1") is not None

    def test_remove_persists(self, temp_storage):
        """Explicit removal is reflected in SQLite."""
        buf = ReplayBuffer(max_size=100, storage_dir=temp_storage)
        buf.add_correction("c1", "/p/1.jpg", "car")
        buf.add_correction("c2", "/p/2.jpg", "truck")
        assert len(buf) == 2

        buf.remove("c1")
        assert len(buf) == 1

        buf2 = ReplayBuffer(max_size=100, storage_dir=temp_storage)
        assert len(buf2) == 1
        assert buf2.get("c1") is None
        assert buf2.get("c2") is not None

    def test_clear_persists(self, temp_storage):
        """Clear removes everything from SQLite."""
        buf = ReplayBuffer(max_size=100, storage_dir=temp_storage)
        for i in range(10):
            buf.add_correction(f"c_{i}", f"/p/{i}.jpg", "car")
        assert len(buf) == 10

        buf.clear()
        assert len(buf) == 0

        buf2 = ReplayBuffer(max_size=100, storage_dir=temp_storage)
        assert len(buf2) == 0

    def test_update_replaces(self, temp_storage):
        """Adding a sample with existing ID replaces the old one."""
        buf = ReplayBuffer(max_size=100, storage_dir=temp_storage)
        buf.add_low_confidence("img_1", "/p/1.jpg", "car", 0.5)

        # Now correct the same image
        buf.add_correction(
            image_id="img_1",
            image_path="/p/1.jpg",
            corrected_label="truck",
        )

        assert len(buf) == 1
        sample = buf.get("img_1")
        assert sample.label == "truck"
        assert sample.source == "correction"

        # Verify persistence has the correction, not the original
        buf2 = ReplayBuffer(max_size=100, storage_dir=temp_storage)
        assert len(buf2) == 1
        assert buf2.get("img_1").label == "truck"


class TestModelOpinionSerialization:
    """Tests for ModelOpinion to_dict/from_dict round-trips."""

    def test_full_opinion_round_trip(self):
        """All fields survive serialization."""
        original = ModelOpinion(
            model_id="yolov8x",
            node_id="gpu-server-1",
            class_name="bird",
            confidence=0.92,
            bbox=(100.0, 200.0, 300.0, 400.0),
            mask_polygon=[[100, 200], [150, 200], [150, 250], [100, 250]],
            inference_time_ms=28.5,
            timestamp=datetime(2025, 6, 15, 12, 30, 0),
        )

        d = original.to_dict()
        restored = ModelOpinion.from_dict(d)

        assert restored.model_id == "yolov8x"
        assert restored.node_id == "gpu-server-1"
        assert restored.class_name == "bird"
        assert restored.confidence == 0.92
        assert restored.bbox == (100.0, 200.0, 300.0, 400.0)
        assert restored.mask_polygon == [[100, 200], [150, 200], [150, 250], [100, 250]]
        assert restored.inference_time_ms == 28.5
        assert restored.timestamp == datetime(2025, 6, 15, 12, 30, 0)

    def test_minimal_opinion_round_trip(self):
        """Minimal opinion with defaults survives serialization."""
        original = ModelOpinion(model_id="yolov8n")

        d = original.to_dict()
        restored = ModelOpinion.from_dict(d)

        assert restored.model_id == "yolov8n"
        assert restored.node_id == "local"
        assert restored.class_name == ""
        assert restored.confidence == 0.0
        assert restored.bbox is None
        assert restored.mask_polygon is None

    def test_opinion_without_bbox(self):
        """Opinion with no bbox serializes None correctly."""
        op = ModelOpinion(model_id="clip", class_name="sunset", confidence=0.8)
        d = op.to_dict()
        assert d["bbox"] is None

        restored = ModelOpinion.from_dict(d)
        assert restored.bbox is None


class TestReplayBufferPersistenceRaw:
    """Tests for the ReplayBufferPersistence class directly."""

    @pytest.fixture
    def db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_replay.db"

    def test_save_and_load_all(self, db_path):
        """Basic save/load cycle."""
        persistence = ReplayBufferPersistence(db_path)

        sample = ReplaySample(
            id="test_1",
            image_path="/path/to/img.jpg",
            label="truck",
            source="correction",
            confidence=0.4,
            priority=2.0,
            opinions=[
                ModelOpinion(model_id="yolov8n", class_name="car", confidence=0.4),
            ],
            final_label="truck",
            final_source="human",
            bbox=(10.0, 20.0, 100.0, 200.0),
            mask_rle="rle_data",
            crop_path="/crops/test.jpg",
            metadata={"session_id": "sess_123"},
        )

        persistence.save(sample)
        loaded = persistence.load_all()

        assert len(loaded) == 1
        s = loaded[0]
        assert s.id == "test_1"
        assert s.label == "truck"
        assert s.source == "correction"
        assert s.priority == 2.0
        assert s.final_label == "truck"
        assert s.final_source == "human"
        assert s.bbox == (10.0, 20.0, 100.0, 200.0)
        assert s.mask_rle == "rle_data"
        assert s.crop_path == "/crops/test.jpg"
        assert s.metadata == {"session_id": "sess_123"}
        assert len(s.opinions) == 1
        assert s.opinions[0].model_id == "yolov8n"

    def test_delete(self, db_path):
        """Delete removes from database."""
        persistence = ReplayBufferPersistence(db_path)

        persistence.save(ReplaySample(id="a", image_path="/a", label="x", source="correction"))
        persistence.save(ReplaySample(id="b", image_path="/b", label="y", source="correction"))

        persistence.delete("a")
        loaded = persistence.load_all()

        assert len(loaded) == 1
        assert loaded[0].id == "b"

    def test_clear(self, db_path):
        """Clear removes all samples."""
        persistence = ReplayBufferPersistence(db_path)

        for i in range(5):
            persistence.save(ReplaySample(
                id=f"s_{i}", image_path=f"/p/{i}", label="x", source="correction"
            ))

        persistence.clear()
        assert len(persistence.load_all()) == 0

    def test_upsert(self, db_path):
        """Saving with same ID updates existing record."""
        persistence = ReplayBufferPersistence(db_path)

        persistence.save(ReplaySample(
            id="s1", image_path="/p/1", label="car", source="low_confidence", priority=0.5
        ))
        persistence.save(ReplaySample(
            id="s1", image_path="/p/1", label="truck", source="correction", priority=2.0
        ))

        loaded = persistence.load_all()
        assert len(loaded) == 1
        assert loaded[0].label == "truck"
        assert loaded[0].priority == 2.0

    def test_load_handles_null_bbox(self, db_path):
        """Sample with no bbox loads correctly."""
        persistence = ReplayBufferPersistence(db_path)

        persistence.save(ReplaySample(
            id="no_bbox", image_path="/p/x", label="cat", source="correction"
        ))

        loaded = persistence.load_all()
        assert loaded[0].bbox is None

    def test_load_handles_empty_opinions(self, db_path):
        """Sample with no opinions loads correctly."""
        persistence = ReplayBufferPersistence(db_path)

        persistence.save(ReplaySample(
            id="no_ops", image_path="/p/x", label="cat", source="correction"
        ))

        loaded = persistence.load_all()
        assert loaded[0].opinions == []


class TestReplayBufferStats:
    """Tests for buffer stats with new source types."""

    def test_stats_include_new_sources(self, persistent_buffer):
        """Stats track all source types including cascade and audit."""
        buf = persistent_buffer

        buf.add_correction("c1", "/p/1.jpg", "car")
        buf.add_low_confidence("l1", "/p/2.jpg", "car", 0.5)
        buf.add_cascade_resolved(
            "cr1", "/p/3.jpg",
            opinions=[ModelOpinion(model_id="yolov8n", class_name="bird", confidence=0.45)],
            final_label="bird",
        )
        buf.add_audit_disagreement(
            "a1", "/p/4.jpg",
            opinions=[ModelOpinion(model_id="yolov8n", class_name="airplane", confidence=0.82)],
            audit_label="bird",
        )

        stats = buf.get_stats()

        assert stats["size"] == 4
        assert stats["by_source"]["correction"] == 1
        assert stats["by_source"]["low_confidence"] == 1
        assert stats["by_source"]["escalation_resolved"] == 1
        assert stats["by_source"]["audit"] == 1
        assert stats["with_opinions"] == 2  # cascade and audit have opinions
        assert stats["persistent"] is True

    def test_stats_with_bbox(self, persistent_buffer):
        """Stats count samples with bbox."""
        buf = persistent_buffer

        buf.add_correction("c1", "/p/1.jpg", "car", bbox=(1, 2, 3, 4))
        buf.add_correction("c2", "/p/2.jpg", "car")  # no bbox

        stats = buf.get_stats()
        assert stats["with_bbox"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
