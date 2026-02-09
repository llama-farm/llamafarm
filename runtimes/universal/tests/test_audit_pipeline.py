"""Tests for audit pipeline (model-checking-model).

Verifies:
- Audit detects disagreements between primary and audit model
- Disagreements are added to replay buffer as training samples
- Agreements are marked as verified
- IoU-based bbox comparison
- Auto-run scheduling
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.vision_base import DetectionBox, DetectionResult
from storage.image_store import (
    ImageMetadataStore,
    ImageRecord,
)
from vision_training.audit_pipeline import (
    AuditConfig,
    AuditPipeline,
    AuditReport,
    _compute_iou,
)
from vision_training.replay_buffer import ReplayBuffer


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def metadata_store(temp_dir):
    return ImageMetadataStore(temp_dir / "test.db")


@pytest.fixture
def replay_buffer(temp_dir):
    return ReplayBuffer(max_size=100, storage_dir=temp_dir / "replay")


@pytest.fixture
def mock_image_store(temp_dir):
    """Mock ImageStore that returns file paths."""
    store = AsyncMock()

    # Write a test image
    img_path = temp_dir / "test_img.jpg"
    img_path.write_bytes(b"fake_jpeg_data")

    mock_record = MagicMock()
    mock_record.file_path = str(img_path)
    store.get_image.return_value = mock_record

    return store


def make_audit_model(audit_class: str, audit_confidence: float, audit_box=None):
    """Create a mock audit model that returns specific results."""
    model = MagicMock()

    box = audit_box or DetectionBox(
        x1=0.1, y1=0.1, x2=0.3, y2=0.3,
        class_name=audit_class,
        class_id=0,
        confidence=audit_confidence,
    )

    async def detect(image, confidence_threshold=0.5):
        return DetectionResult(
            confidence=audit_confidence,
            inference_time_ms=45.0,
            model_name="audit_model",
            boxes=[box],
        )

    model.detect = detect
    return model


class TestIoUComputation:
    """Tests for bbox IoU computation."""

    def test_perfect_overlap(self):
        assert _compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_no_overlap(self):
        assert _compute_iou((0, 0, 5, 5), (10, 10, 15, 15)) == 0.0

    def test_partial_overlap(self):
        iou = _compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.1 < iou < 0.2  # ~14%

    def test_contained_box(self):
        iou = _compute_iou((0, 0, 10, 10), (2, 2, 8, 8))
        assert 0.3 < iou < 0.4  # ~36%


class TestAuditPipeline:
    """Tests for the audit pipeline."""

    @pytest.mark.asyncio
    async def test_agreement_detected(
        self, metadata_store, replay_buffer, mock_image_store
    ):
        """When audit model agrees with primary, sample is marked verified."""
        # Setup: primary said "person" at 0.85
        metadata_store.add_image(ImageRecord(id="img_001", file_path="/test.jpg"))
        metadata_store.add_detection_history(
            image_id="img_001",
            model_id="yolov8n",
            class_name="person",
            confidence=0.85,
            box=(0.1, 0.1, 0.3, 0.3),
            stage="primary",
        )

        # Audit model also says "person"
        audit_model = make_audit_model("person", 0.92)

        async def loader(model_id):
            return audit_model

        pipeline = AuditPipeline(
            config=AuditConfig(audit_model_id="yolov8x", enabled=True),
            model_loader=loader,
            image_store=mock_image_store,
            replay_buffer=replay_buffer,
            metadata_store=metadata_store,
        )

        report = await pipeline.run_audit(sample_size=10)

        assert report.samples_audited == 1
        assert report.agreements == 1
        assert report.disagreements == 0
        assert report.accuracy_estimate == 1.0
        assert report.verified_samples_marked == 1

        # Replay buffer should NOT have this (it's an agreement)
        assert len(replay_buffer) == 0

    @pytest.mark.asyncio
    async def test_disagreement_creates_training_sample(
        self, metadata_store, replay_buffer, mock_image_store
    ):
        """When audit model disagrees, sample is added to replay buffer."""
        # Primary said "airplane" at 0.82
        metadata_store.add_image(ImageRecord(id="img_002", file_path="/test.jpg"))
        metadata_store.add_detection_history(
            image_id="img_002",
            model_id="yolov8n",
            class_name="airplane",
            confidence=0.82,
            box=(0.1, 0.1, 0.3, 0.3),
            stage="primary",
        )

        # Audit model says "bird" (disagrees)
        audit_model = make_audit_model("bird", 0.91)

        async def loader(model_id):
            return audit_model

        pipeline = AuditPipeline(
            config=AuditConfig(audit_model_id="yolov8x", enabled=True),
            model_loader=loader,
            image_store=mock_image_store,
            replay_buffer=replay_buffer,
            metadata_store=metadata_store,
        )

        report = await pipeline.run_audit(sample_size=10)

        assert report.samples_audited == 1
        assert report.agreements == 0
        assert report.disagreements == 1
        assert report.disagreement_samples_added == 1

        # Replay buffer should have the disagreement
        assert len(replay_buffer) == 1
        sample = replay_buffer.sample(1)[0]
        assert sample.source == "audit"
        assert sample.final_label == "bird"
        assert sample.priority == 2.0
        assert len(sample.opinions) == 2
        assert sample.opinions[0].class_name == "airplane"
        assert sample.opinions[1].class_name == "bird"

    @pytest.mark.asyncio
    async def test_no_samples_to_audit(self, metadata_store, replay_buffer, mock_image_store):
        """Empty metadata store returns empty report."""
        async def loader(model_id):
            return make_audit_model("person", 0.9)

        pipeline = AuditPipeline(
            config=AuditConfig(audit_model_id="yolov8x", enabled=True),
            model_loader=loader,
            image_store=mock_image_store,
            replay_buffer=replay_buffer,
            metadata_store=metadata_store,
        )

        report = await pipeline.run_audit()
        assert report.samples_audited == 0

    @pytest.mark.asyncio
    async def test_no_audit_model_configured(self, metadata_store):
        """Returns empty report if no audit model configured."""
        pipeline = AuditPipeline(
            config=AuditConfig(audit_model_id="", enabled=True),
            metadata_store=metadata_store,
        )

        report = await pipeline.run_audit()
        assert report.samples_audited == 0


class TestAuditScheduling:
    """Tests for audit auto-scheduling."""

    def test_should_run_first_time(self):
        """should_run() returns True on first check."""
        pipeline = AuditPipeline(
            config=AuditConfig(enabled=True, auto_interval_minutes=30),
        )
        assert pipeline.should_run() is True

    def test_should_not_run_when_disabled(self):
        """should_run() returns False when disabled."""
        pipeline = AuditPipeline(
            config=AuditConfig(enabled=False),
        )
        assert pipeline.should_run() is False

    def test_should_not_run_too_soon(self):
        """should_run() returns False if run recently."""
        pipeline = AuditPipeline(
            config=AuditConfig(enabled=True, auto_interval_minutes=30),
        )
        pipeline._last_run = datetime.utcnow()
        assert pipeline.should_run() is False

    def test_should_run_after_interval(self):
        """should_run() returns True after interval elapses."""
        pipeline = AuditPipeline(
            config=AuditConfig(enabled=True, auto_interval_minutes=30),
        )
        pipeline._last_run = datetime.utcnow() - timedelta(minutes=31)
        assert pipeline.should_run() is True


class TestAuditResults:
    """Tests for AuditResult and AuditReport."""

    def test_report_accuracy_calculation(self):
        """Accuracy is correctly calculated from results."""
        from vision_training.audit_pipeline import AuditResult

        report = AuditReport()
        report.results = [
            AuditResult("img1", "car", 0.9, "car", 0.95, agreement=True),
            AuditResult("img2", "car", 0.8, "truck", 0.85, agreement=False),
            AuditResult("img3", "person", 0.9, "person", 0.92, agreement=True),
        ]
        report.samples_audited = 3
        report.agreements = 2
        report.disagreements = 1
        report.accuracy_estimate = report.agreements / report.samples_audited

        assert report.accuracy_estimate == pytest.approx(0.667, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
