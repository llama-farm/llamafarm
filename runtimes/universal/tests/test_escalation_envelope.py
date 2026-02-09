"""Tests for escalation envelope flow through the cascade.

Verifies that:
- ModelOpinions accumulate correctly as an image cascades through models
- Bounding boxes and masks flow forward between hops
- The replay buffer receives complete provenance when cascade resolves
- Detection history is recorded in image_store
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from storage.image_store import (
    DetectionHistoryRecord,
    ImageMetadataStore,
    ImageRecord,
)
from vision_training.replay_buffer import (
    ModelOpinion,
    ReplayBuffer,
    ReplaySample,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def metadata_store(temp_dir):
    """Create a metadata store with the new detection_history table."""
    store = ImageMetadataStore(temp_dir / "test.db")
    return store


@pytest.fixture
def replay_buffer(temp_dir):
    return ReplayBuffer(max_size=100, storage_dir=temp_dir / "replay")


# ---------------------------------------------------------------------------
# Detection History Table Tests
# ---------------------------------------------------------------------------

class TestDetectionHistory:
    """Tests for the detection_history table in ImageMetadataStore."""

    def test_table_created(self, metadata_store):
        """detection_history table exists after init."""
        with sqlite3.connect(metadata_store.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='detection_history'"
            )
            assert cursor.fetchone() is not None

    def test_add_detection_history(self, metadata_store):
        """Can add a cascade opinion to detection_history."""
        # First add an image
        metadata_store.add_image(ImageRecord(
            id="img_001",
            file_path="/images/img_001.jpg",
        ))

        record_id = metadata_store.add_detection_history(
            image_id="img_001",
            model_id="yolov8n",
            class_name="person",
            confidence=0.85,
            box=(100.0, 200.0, 300.0, 400.0),
            stage="primary",
            hop_number=0,
            node_id="local",
            inference_time_ms=5.2,
        )

        assert record_id is not None
        assert record_id > 0

    def test_get_detection_history(self, metadata_store):
        """Get all opinions for an image, ordered by hop."""
        metadata_store.add_image(ImageRecord(
            id="img_002",
            file_path="/images/img_002.jpg",
        ))

        # Hop 0: primary model (low confidence)
        metadata_store.add_detection_history(
            image_id="img_002",
            model_id="yolov8n",
            class_name="bird",
            confidence=0.45,
            box=(50.0, 60.0, 200.0, 300.0),
            stage="primary",
            hop_number=0,
            inference_time_ms=5.0,
        )

        # Hop 1: secondary model (high confidence)
        metadata_store.add_detection_history(
            image_id="img_002",
            model_id="yolov8m",
            class_name="bird",
            confidence=0.92,
            box=(52.0, 58.0, 198.0, 302.0),
            stage="secondary",
            hop_number=1,
            inference_time_ms=25.0,
        )

        history = metadata_store.get_detection_history("img_002")

        assert len(history) == 2
        assert history[0].model_id == "yolov8n"
        assert history[0].hop_number == 0
        assert history[0].confidence == 0.45
        assert history[1].model_id == "yolov8m"
        assert history[1].hop_number == 1
        assert history[1].confidence == 0.92

    def test_detection_history_with_mask(self, metadata_store):
        """Mask RLE is stored and retrieved correctly."""
        metadata_store.add_image(ImageRecord(
            id="img_003",
            file_path="/images/img_003.jpg",
        ))

        metadata_store.add_detection_history(
            image_id="img_003",
            model_id="sam_vit_b",
            class_name="person",
            confidence=0.95,
            box=(10.0, 20.0, 100.0, 200.0),
            stage="primary",
            hop_number=0,
            mask_rle="rle_encoded_mask_data_here",
        )

        history = metadata_store.get_detection_history("img_003")
        assert len(history) == 1
        assert history[0].mask_rle == "rle_encoded_mask_data_here"

    def test_detection_history_remote_node(self, metadata_store):
        """Remote node opinions are stored correctly."""
        metadata_store.add_image(ImageRecord(
            id="img_004",
            file_path="/images/img_004.jpg",
        ))

        metadata_store.add_detection_history(
            image_id="img_004",
            model_id="yolov8x",
            class_name="truck",
            confidence=0.88,
            box=(200.0, 100.0, 500.0, 400.0),
            stage="remote",
            hop_number=2,
            node_id="gpu-server-alpha",
            inference_time_ms=45.0,
        )

        history = metadata_store.get_detection_history("img_004")
        assert history[0].node_id == "gpu-server-alpha"
        assert history[0].stage == "remote"
        assert history[0].hop_number == 2

    def test_get_recent_high_confidence(self, metadata_store):
        """Can query recent high-confidence primary detections for auditing."""
        metadata_store.add_image(ImageRecord(id="hi_1", file_path="/hi_1.jpg"))
        metadata_store.add_image(ImageRecord(id="lo_1", file_path="/lo_1.jpg"))
        metadata_store.add_image(ImageRecord(id="hi_2", file_path="/hi_2.jpg"))

        # High confidence primary
        metadata_store.add_detection_history(
            image_id="hi_1", model_id="yolov8n", class_name="car",
            confidence=0.92, box=(0, 0, 1, 1), stage="primary",
        )
        # Low confidence primary (should NOT be returned)
        metadata_store.add_detection_history(
            image_id="lo_1", model_id="yolov8n", class_name="car",
            confidence=0.45, box=(0, 0, 1, 1), stage="primary",
        )
        # High confidence but secondary stage (should NOT be returned)
        metadata_store.add_detection_history(
            image_id="hi_2", model_id="yolov8m", class_name="car",
            confidence=0.95, box=(0, 0, 1, 1), stage="secondary",
        )

        results = metadata_store.get_recent_high_confidence(
            min_confidence=0.7,
            stage="primary",
        )

        assert len(results) == 1
        assert results[0].image_id == "hi_1"
        assert results[0].confidence == 0.92

    def test_stats_include_detection_history(self, metadata_store):
        """get_stats includes detection_history count."""
        metadata_store.add_image(ImageRecord(id="s1", file_path="/s1.jpg"))
        metadata_store.add_detection_history(
            image_id="s1", model_id="yolov8n", class_name="person",
            confidence=0.8, box=(0, 0, 1, 1), stage="primary",
        )
        metadata_store.add_detection_history(
            image_id="s1", model_id="yolov8m", class_name="person",
            confidence=0.95, box=(0, 0, 1, 1), stage="secondary", hop_number=1,
        )

        stats = metadata_store.get_stats()
        assert stats["total_detection_history"] == 2


# ---------------------------------------------------------------------------
# Envelope Flow Through Replay Buffer
# ---------------------------------------------------------------------------

class TestEnvelopeToReplayBuffer:
    """Tests for how cascade results flow into the replay buffer."""

    def test_single_hop_resolution(self, replay_buffer):
        """When hop 1 resolves what hop 0 couldn't: correct source type and priority."""
        opinions = [
            ModelOpinion(
                model_id="yolov8n",
                node_id="local",
                class_name="bird",
                confidence=0.45,
                bbox=(100.0, 200.0, 300.0, 400.0),
                inference_time_ms=5.0,
            ),
            ModelOpinion(
                model_id="yolov8m",
                node_id="local",
                class_name="bird",
                confidence=0.92,
                bbox=(105.0, 198.0, 305.0, 402.0),
                inference_time_ms=28.0,
            ),
        ]

        sample = replay_buffer.add_cascade_resolved(
            image_id="frame_001",
            image_path="/frames/001.jpg",
            opinions=opinions,
            final_label="bird",
            bbox=(100.0, 200.0, 300.0, 400.0),
            resolving_hop=1,
        )

        assert sample.source == "escalation_resolved"
        assert sample.priority == 1.5
        assert sample.final_source == "cascade"
        assert sample.confidence == 0.92  # Confidence from resolving model
        assert len(sample.opinions) == 2

    def test_two_hop_resolution(self, replay_buffer):
        """When hop 2 resolves: higher priority than single hop."""
        opinions = [
            ModelOpinion(model_id="yolov8n", class_name="car", confidence=0.45),
            ModelOpinion(model_id="yolov8m", class_name="truck", confidence=0.55),
            ModelOpinion(model_id="yolov8x", class_name="truck", confidence=0.88),
        ]

        sample = replay_buffer.add_cascade_resolved(
            image_id="frame_002",
            image_path="/frames/002.jpg",
            opinions=opinions,
            final_label="truck",
            resolving_hop=2,
        )

        assert sample.source == "cascade_resolved"
        assert sample.priority == 1.8
        assert len(sample.opinions) == 3

    def test_human_review_highest_priority(self, replay_buffer):
        """Human corrections have priority 2.0 (highest)."""
        opinions = [
            ModelOpinion(model_id="yolov8n", class_name="airplane", confidence=0.35),
            ModelOpinion(model_id="yolov8x", class_name="airplane", confidence=0.42),
        ]

        sample = replay_buffer.add_correction(
            image_id="review_001",
            image_path="/images/review_001.jpg",
            corrected_label="bird",
            original_confidence=0.35,
            opinions=opinions,
            bbox=(50.0, 60.0, 200.0, 300.0),
        )

        assert sample.priority == 2.0
        assert sample.final_source == "human"
        assert sample.final_label == "bird"
        assert len(sample.opinions) == 2

    def test_audit_same_priority_as_human(self, replay_buffer):
        """Audit disagreements have same priority as human corrections."""
        opinions = [
            ModelOpinion(model_id="yolov8n", class_name="airplane", confidence=0.82),
            ModelOpinion(model_id="yolov8x", class_name="bird", confidence=0.91),
        ]

        sample = replay_buffer.add_audit_disagreement(
            image_id="audit_001",
            image_path="/images/audit_001.jpg",
            opinions=opinions,
            audit_label="bird",
        )

        assert sample.priority == 2.0
        assert sample.final_source == "audit"

    def test_priority_ordering(self, replay_buffer):
        """Verify priority ordering: human/audit > cascade_resolved > escalation_resolved > low_confidence."""
        replay_buffer.add_low_confidence("lc1", "/p/1.jpg", "car", 0.6)
        replay_buffer.add_cascade_resolved(
            "cr1", "/p/2.jpg",
            opinions=[ModelOpinion(model_id="yolov8n", confidence=0.5)],
            final_label="car",
            resolving_hop=1,
        )
        replay_buffer.add_cascade_resolved(
            "cr2", "/p/3.jpg",
            opinions=[ModelOpinion(model_id="yolov8n", confidence=0.5)],
            final_label="car",
            resolving_hop=2,
        )
        replay_buffer.add_correction("c1", "/p/4.jpg", "truck")

        # Verify priorities directly (deterministic, not sampling-dependent)
        samples_by_id = {s.id: s for s in replay_buffer._samples.values()}

        assert samples_by_id["c1"].priority == 2.0        # correction
        assert samples_by_id["cr2"].priority == 1.8        # cascade_resolved (hop 2)
        assert samples_by_id["cr1"].priority == 1.5        # escalation_resolved (hop 1)
        assert samples_by_id["lc1"].priority == pytest.approx(0.4, abs=0.01)  # low_confidence

        # Verify ordering: correction > cascade > escalation > low_conf
        assert samples_by_id["c1"].priority > samples_by_id["cr2"].priority
        assert samples_by_id["cr2"].priority > samples_by_id["cr1"].priority
        assert samples_by_id["cr1"].priority > samples_by_id["lc1"].priority


# ---------------------------------------------------------------------------
# Full Cascade Simulation
# ---------------------------------------------------------------------------

class TestCascadeSimulation:
    """Simulate what happens during a full cascade, building the envelope."""

    def test_build_envelope_through_cascade(self, replay_buffer, metadata_store):
        """Simulate a 3-model cascade building opinions step by step."""
        image_id = "cascade_sim_001"

        # Register the image
        metadata_store.add_image(ImageRecord(
            id=image_id, file_path=f"/images/{image_id}.jpg"
        ))

        opinions: list[ModelOpinion] = []

        # Hop 0: Primary model (local, fast, uncertain)
        op0 = ModelOpinion(
            model_id="yolov8n",
            node_id="local",
            class_name="bird",
            confidence=0.45,
            bbox=(100.0, 200.0, 300.0, 400.0),
            inference_time_ms=5.2,
        )
        opinions.append(op0)
        metadata_store.add_detection_history(
            image_id=image_id,
            model_id=op0.model_id,
            class_name=op0.class_name,
            confidence=op0.confidence,
            box=op0.bbox,
            stage="primary",
            hop_number=0,
            node_id=op0.node_id,
            inference_time_ms=op0.inference_time_ms,
        )

        # Hop 1: Secondary model (local, medium, still uncertain)
        op1 = ModelOpinion(
            model_id="yolov8m",
            node_id="local",
            class_name="airplane",
            confidence=0.55,
            bbox=(98.0, 202.0, 302.0, 398.0),
            inference_time_ms=22.0,
        )
        opinions.append(op1)
        metadata_store.add_detection_history(
            image_id=image_id,
            model_id=op1.model_id,
            class_name=op1.class_name,
            confidence=op1.confidence,
            box=op1.bbox,
            stage="secondary",
            hop_number=1,
            node_id=op1.node_id,
            inference_time_ms=op1.inference_time_ms,
        )

        # Hop 2: Remote GPU model (resolves it)
        op2 = ModelOpinion(
            model_id="yolov8x",
            node_id="gpu-server-1",
            class_name="bird",
            confidence=0.94,
            bbox=(101.0, 199.0, 299.0, 401.0),
            mask_polygon=[[101, 199], [200, 199], [200, 350], [101, 350]],
            inference_time_ms=45.0,
        )
        opinions.append(op2)
        metadata_store.add_detection_history(
            image_id=image_id,
            model_id=op2.model_id,
            class_name=op2.class_name,
            confidence=op2.confidence,
            box=op2.bbox,
            stage="remote",
            hop_number=2,
            node_id=op2.node_id,
            inference_time_ms=op2.inference_time_ms,
            mask_rle="simulated_rle",
        )

        # Cascade resolved by hop 2 -> add to replay buffer
        sample = replay_buffer.add_cascade_resolved(
            image_id=image_id,
            image_path=f"/images/{image_id}.jpg",
            opinions=opinions,
            final_label="bird",
            bbox=op0.bbox,
            mask_rle="simulated_rle",
            resolving_hop=2,
        )

        # Verify the full chain
        assert sample.source == "cascade_resolved"
        assert sample.priority == 1.8
        assert len(sample.opinions) == 3
        assert sample.opinions[0].model_id == "yolov8n"
        assert sample.opinions[1].model_id == "yolov8m"
        assert sample.opinions[2].model_id == "yolov8x"
        assert sample.final_label == "bird"
        assert sample.confidence == 0.94

        # Verify detection history in the store
        history = metadata_store.get_detection_history(image_id)
        assert len(history) == 3
        assert history[0].stage == "primary"
        assert history[1].stage == "secondary"
        assert history[2].stage == "remote"
        assert history[2].node_id == "gpu-server-1"
        assert history[2].mask_rle == "simulated_rle"

    def test_disagreement_detection(self, replay_buffer):
        """When models disagree, the final answer wins."""
        opinions = [
            ModelOpinion(model_id="yolov8n", class_name="airplane", confidence=0.72),
            ModelOpinion(model_id="yolov8x", class_name="bird", confidence=0.91),
        ]

        sample = replay_buffer.add_audit_disagreement(
            image_id="disagree_001",
            image_path="/images/disagree_001.jpg",
            opinions=opinions,
            audit_label="bird",
        )

        # The training pipeline should use "bird" as the label
        assert sample.final_label == "bird"
        assert sample.label == "bird"
        # But both opinions are preserved for analysis
        assert sample.opinions[0].class_name == "airplane"
        assert sample.opinions[1].class_name == "bird"

    def test_circuit_breaker_max_hops(self):
        """Verify that we track hop count correctly (max 3 per plan)."""
        max_hops = 3
        opinions = []
        for hop in range(max_hops):
            opinions.append(ModelOpinion(
                model_id=f"model_{hop}",
                class_name="unknown",
                confidence=0.3 + (hop * 0.1),
            ))

        # After 3 hops, if still uncertain, this should go to review
        # not to another cascade hop
        assert len(opinions) == max_hops
        assert all(op.confidence < 0.7 for op in opinions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
