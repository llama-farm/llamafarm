"""Tests for validation gate (blue/green model swap).

Verifies:
- Validation set building from verified images
- Candidate vs current model comparison
- Promotion when candidate is better
- Rejection when candidate is worse
- Rollback to previous version
- Auto-trainer integration with validation gate
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.vision_base import DetectionBox, DetectionResult
from storage.image_store import ImageMetadataStore, ImageRecord
from vision_training.validation_gate import (
    ModelVersion,
    ValidationGate,
    ValidationMetrics,
    ValidationResult,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def metadata_store(temp_dir):
    return ImageMetadataStore(temp_dir / "test.db")


@pytest.fixture
def mock_image_store(temp_dir):
    """Mock ImageStore that returns file paths."""
    store = AsyncMock()
    img_path = temp_dir / "test_img.jpg"
    img_path.write_bytes(b"fake_jpeg_data")

    mock_record = MagicMock()
    mock_record.file_path = str(img_path)
    store.get_image.return_value = mock_record
    return store


def make_model(class_name: str, confidence: float, box=None):
    """Create a mock detection model."""
    model = MagicMock()

    box = box or DetectionBox(
        x1=0.1, y1=0.1, x2=0.3, y2=0.3,
        class_name=class_name,
        class_id=0,
        confidence=confidence,
    )

    async def detect(image, confidence_threshold=0.5, classes=None):
        return DetectionResult(
            confidence=confidence,
            inference_time_ms=10.0,
            model_name="test_model",
            boxes=[box],
        )

    model.detect = detect
    return model


class TestBuildValidationSet:
    """Tests for building validation sets from verified images."""

    def test_empty_metadata(self, metadata_store, temp_dir):
        """Empty metadata store returns empty validation set."""
        import asyncio

        gate = ValidationGate(
            metadata_store=metadata_store,
            models_dir=temp_dir / "models",
        )

        val_set = asyncio.get_event_loop().run_until_complete(
            gate.build_validation_set()
        )
        assert len(val_set) == 0

    def test_builds_from_audit_verified(self, metadata_store, temp_dir):
        """Validation set includes audit-verified samples."""
        import asyncio

        # Add audit-verified detections
        metadata_store.add_image(ImageRecord(id="img_001", file_path="/test.jpg"))
        metadata_store.add_detection_history(
            image_id="img_001",
            model_id="yolov8x",
            class_name="person",
            confidence=0.95,
            box=(0.1, 0.1, 0.3, 0.3),
            stage="audit",
        )

        gate = ValidationGate(
            metadata_store=metadata_store,
            models_dir=temp_dir / "models",
        )

        val_set = asyncio.get_event_loop().run_until_complete(
            gate.build_validation_set()
        )
        assert len(val_set) == 1
        assert val_set[0]["expected_class"] == "person"
        assert val_set[0]["source"] == "audit_verified"

    def test_supplements_with_primary(self, metadata_store, temp_dir):
        """When not enough audit samples, supplements with high-confidence primary."""
        import asyncio

        metadata_store.add_image(ImageRecord(id="img_001", file_path="/test.jpg"))
        metadata_store.add_detection_history(
            image_id="img_001",
            model_id="yolov8n",
            class_name="car",
            confidence=0.95,
            box=(0.2, 0.2, 0.5, 0.5),
            stage="primary",
        )

        gate = ValidationGate(
            metadata_store=metadata_store,
            models_dir=temp_dir / "models",
        )

        val_set = asyncio.get_event_loop().run_until_complete(
            gate.build_validation_set()
        )
        assert len(val_set) == 1
        assert val_set[0]["source"] == "primary_high_confidence"


class TestValidateCandidate:
    """Tests for candidate vs current model comparison."""

    @pytest.mark.asyncio
    async def test_not_enough_samples(self, metadata_store, temp_dir):
        """Validation fails if not enough samples."""
        gate = ValidationGate(
            metadata_store=metadata_store,
            models_dir=temp_dir / "models",
            min_validation_samples=10,
        )

        result = await gate.validate_candidate(
            candidate_path="/fake/model.pt",
            current_model_id="yolov8n",
        )

        assert result.passed is False
        assert "Not enough validation samples" in result.reason

    @pytest.mark.asyncio
    async def test_candidate_better_passes(
        self, metadata_store, mock_image_store, temp_dir
    ):
        """Candidate that beats current model passes validation."""
        # Setup validation data
        for i in range(15):
            img_id = f"img_{i:03d}"
            metadata_store.add_image(ImageRecord(id=img_id, file_path="/test.jpg"))
            metadata_store.add_detection_history(
                image_id=img_id,
                model_id="yolov8x",
                class_name="person",
                confidence=0.95,
                box=(0.1, 0.1, 0.3, 0.3),
                stage="audit",
            )

        # Current model: sometimes wrong (returns wrong class for every 3rd call)
        # Candidate model: always right
        # Use a single model instance per loader call, with internal counter
        def make_flaky_model():
            """Model that gets every 3rd detection wrong."""
            model = MagicMock()
            counter = {"n": 0}

            async def detect(image, confidence_threshold=0.5, classes=None):
                counter["n"] += 1
                if counter["n"] % 3 == 0:
                    cls = "car"  # Wrong
                else:
                    cls = "person"  # Right
                return DetectionResult(
                    confidence=0.75,
                    inference_time_ms=10.0,
                    model_name="current",
                    boxes=[DetectionBox(
                        x1=0.1, y1=0.1, x2=0.3, y2=0.3,
                        class_name=cls, class_id=0, confidence=0.75,
                    )],
                )

            model.detect = detect
            return model

        async def loader(model_id):
            if model_id == "yolov8n":
                return make_flaky_model()
            else:
                return make_model("person", 0.92)

        gate = ValidationGate(
            model_loader=loader,
            metadata_store=metadata_store,
            image_store=mock_image_store,
            models_dir=temp_dir / "models",
            min_validation_samples=10,
        )

        result = await gate.validate_candidate(
            candidate_path="/fake/candidate.pt",
            current_model_id="yolov8n",
        )

        assert result.passed is True
        assert result.improvement > 0
        assert result.candidate_metrics.accuracy > result.current_metrics.accuracy

    @pytest.mark.asyncio
    async def test_candidate_worse_fails(
        self, metadata_store, mock_image_store, temp_dir
    ):
        """Candidate worse than current model fails validation."""
        for i in range(15):
            img_id = f"img_{i:03d}"
            metadata_store.add_image(ImageRecord(id=img_id, file_path="/test.jpg"))
            metadata_store.add_detection_history(
                image_id=img_id,
                model_id="yolov8x",
                class_name="person",
                confidence=0.95,
                box=(0.1, 0.1, 0.3, 0.3),
                stage="audit",
            )

        call_count = {"candidate": 0}

        async def loader(model_id):
            if model_id == "yolov8n":
                return make_model("person", 0.9)
            else:
                call_count["candidate"] += 1
                # Candidate is bad: always says "cat"
                return make_model("cat", 0.8)

        gate = ValidationGate(
            model_loader=loader,
            metadata_store=metadata_store,
            image_store=mock_image_store,
            models_dir=temp_dir / "models",
            min_validation_samples=10,
            min_improvement=-0.02,
        )

        result = await gate.validate_candidate(
            candidate_path="/fake/bad_candidate.pt",
            current_model_id="yolov8n",
        )

        assert result.passed is False
        assert result.improvement < 0


class TestPromoteAndRollback:
    """Tests for model promotion and rollback."""

    @pytest.mark.asyncio
    async def test_promote_creates_version(self, temp_dir):
        """Promote creates a new model version."""
        models_dir = temp_dir / "models"

        # Create a candidate file
        candidate = temp_dir / "candidate.pt"
        candidate.write_bytes(b"model_weights_v2")

        gate = ValidationGate(models_dir=models_dir)

        version = await gate.promote(
            candidate_path=str(candidate),
            model_id="yolov8n",
        )

        assert version.version == 1
        assert version.is_active is True
        assert version.model_id == "yolov8n"

        # Current file should exist
        current = models_dir / "yolov8n" / "current.pt"
        assert current.exists()
        assert current.read_bytes() == b"model_weights_v2"

    @pytest.mark.asyncio
    async def test_promote_backs_up_previous(self, temp_dir):
        """Promote backs up the previous version."""
        models_dir = temp_dir / "models"
        model_dir = models_dir / "yolov8n"
        model_dir.mkdir(parents=True)

        # Put an existing "current" model
        current = model_dir / "current.pt"
        current.write_bytes(b"model_weights_v1")

        # Create candidate
        candidate = temp_dir / "candidate.pt"
        candidate.write_bytes(b"model_weights_v2")

        gate = ValidationGate(models_dir=models_dir)

        version = await gate.promote(
            candidate_path=str(candidate),
            model_id="yolov8n",
        )

        assert version.version == 1
        # Current should be the new model
        assert current.read_bytes() == b"model_weights_v2"
        # Backup should exist
        backup = model_dir / "v0.pt"
        assert backup.exists()
        assert backup.read_bytes() == b"model_weights_v1"

    @pytest.mark.asyncio
    async def test_sequential_promotions(self, temp_dir):
        """Multiple promotions increment version correctly."""
        models_dir = temp_dir / "models"
        gate = ValidationGate(models_dir=models_dir)

        for i in range(3):
            candidate = temp_dir / f"candidate_{i}.pt"
            candidate.write_bytes(f"weights_v{i+1}".encode())

            version = await gate.promote(
                candidate_path=str(candidate),
                model_id="yolov8n",
            )
            assert version.version == i + 1

        versions = gate.get_versions("yolov8n")
        assert len(versions) == 3
        assert versions[-1].is_active is True
        assert versions[0].is_active is False

    @pytest.mark.asyncio
    async def test_rollback(self, temp_dir):
        """Rollback restores previous version."""
        models_dir = temp_dir / "models"
        gate = ValidationGate(models_dir=models_dir)

        # Promote v1
        v1 = temp_dir / "v1.pt"
        v1.write_bytes(b"weights_v1")
        await gate.promote(str(v1), "yolov8n")

        # Promote v2
        v2 = temp_dir / "v2.pt"
        v2.write_bytes(b"weights_v2")
        await gate.promote(str(v2), "yolov8n")

        # Current should be v2
        current = models_dir / "yolov8n" / "current.pt"
        assert current.read_bytes() == b"weights_v2"

        # Rollback
        restored = await gate.rollback("yolov8n")
        assert restored is not None
        assert restored.is_active is True
        assert current.read_bytes() == b"weights_v1"

    @pytest.mark.asyncio
    async def test_rollback_no_previous(self, temp_dir):
        """Rollback returns None when no previous version exists."""
        gate = ValidationGate(models_dir=temp_dir / "models")
        result = await gate.rollback("yolov8n")
        assert result is None

    @pytest.mark.asyncio
    async def test_promote_callback(self, temp_dir):
        """on_model_promoted callback fires on promotion."""
        callback_called = []

        def on_promoted(version):
            callback_called.append(version)

        gate = ValidationGate(models_dir=temp_dir / "models")
        gate.set_on_model_promoted(on_promoted)

        candidate = temp_dir / "candidate.pt"
        candidate.write_bytes(b"weights")
        await gate.promote(str(candidate), "yolov8n")

        assert len(callback_called) == 1
        assert callback_called[0].model_id == "yolov8n"


class TestAutoTrainerValidation:
    """Tests for auto-trainer integration with validation gate."""

    @pytest.mark.asyncio
    async def test_auto_trainer_validates_before_promote(self, temp_dir):
        """Auto-trainer uses validation gate after training completes."""
        from vision_training.auto_trainer import AutoTrainConfig, AutoTrainer
        from vision_training.replay_buffer import ReplayBuffer
        from vision_training.trainer import IncrementalTrainer, TrainingJob, TrainingStatus

        # Mock trainer that "completes" immediately
        trainer = MagicMock(spec=IncrementalTrainer)
        completed_job = TrainingJob(
            job_id="test-job",
            model_id="yolov8n",
            dataset_path="/tmp/dataset",
            task="detection",
            config=MagicMock(),
            status=TrainingStatus.COMPLETED,
            metrics={"model_path": str(temp_dir / "trained.pt")},
        )

        async def wait_for(*args, **kwargs):
            return completed_job

        trainer.wait_for_job = wait_for

        # Mock validation gate
        validation_gate = MagicMock(spec=ValidationGate)
        validation_gate.validate_candidate = AsyncMock(
            return_value=ValidationResult(
                passed=True,
                current_metrics=ValidationMetrics(model_id="yolov8n", accuracy=0.7),
                candidate_metrics=ValidationMetrics(model_id="candidate", accuracy=0.85),
                improvement=0.15,
                reason="Improved",
            )
        )
        validation_gate.promote = AsyncMock(
            return_value=ModelVersion(
                model_id="yolov8n",
                version=1,
                model_path=str(temp_dir / "promoted.pt"),
                is_active=True,
            )
        )

        auto_trainer = AutoTrainer(
            replay_buffer=ReplayBuffer(max_size=100),
            trainer=trainer,
            config=AutoTrainConfig(enabled=True),
            model_id="yolov8n",
            validation_gate=validation_gate,
        )

        # Simulate _wait_for_completion
        await auto_trainer._wait_for_completion("test-job", 50)

        # Validation gate should have been called
        validation_gate.validate_candidate.assert_called_once()
        validation_gate.promote.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_trainer_rejects_bad_candidate(self, temp_dir):
        """Auto-trainer does NOT promote when validation fails."""
        from vision_training.auto_trainer import AutoTrainConfig, AutoTrainer
        from vision_training.replay_buffer import ReplayBuffer
        from vision_training.trainer import TrainingJob, TrainingStatus

        trainer = MagicMock()
        completed_job = TrainingJob(
            job_id="test-job",
            model_id="yolov8n",
            dataset_path="/tmp/dataset",
            task="detection",
            config=MagicMock(),
            status=TrainingStatus.COMPLETED,
            metrics={"model_path": str(temp_dir / "trained.pt")},
        )

        async def wait_for(*args, **kwargs):
            return completed_job

        trainer.wait_for_job = wait_for

        # Validation gate rejects
        validation_gate = MagicMock(spec=ValidationGate)
        validation_gate.validate_candidate = AsyncMock(
            return_value=ValidationResult(
                passed=False,
                current_metrics=ValidationMetrics(model_id="yolov8n", accuracy=0.85),
                candidate_metrics=ValidationMetrics(model_id="candidate", accuracy=0.6),
                improvement=-0.25,
                reason="Candidate is worse",
            )
        )

        auto_trainer = AutoTrainer(
            replay_buffer=ReplayBuffer(max_size=100),
            trainer=trainer,
            config=AutoTrainConfig(enabled=True),
            model_id="yolov8n",
            validation_gate=validation_gate,
        )

        # Track callback
        callback_results = []
        auto_trainer.set_on_training_complete(lambda r: callback_results.append(r))

        await auto_trainer._wait_for_completion("test-job", 50)

        # Validation was called but promote was NOT
        validation_gate.validate_candidate.assert_called_once()
        validation_gate.promote.assert_not_called()

        # Callback should report not promoted
        assert len(callback_results) == 1
        assert callback_results[0]["promoted"] is False


class TestCreateDataset:
    """Tests for auto-trainer dataset creation."""

    @pytest.mark.asyncio
    async def test_creates_class_map_from_samples(self, temp_dir):
        """Dataset creation builds class map from actual sample labels."""
        from vision_training.auto_trainer import AutoTrainConfig, AutoTrainer
        from vision_training.replay_buffer import ReplayBuffer

        buffer = ReplayBuffer(max_size=100, storage_dir=temp_dir / "replay")

        # Create test images
        for i in range(5):
            img_path = temp_dir / f"img_{i}.jpg"
            img_path.write_bytes(b"fake_image")

        # Add samples with different classes
        buffer.add_cascade_resolved(
            image_id="sample_0",
            image_path=str(temp_dir / "img_0.jpg"),
            opinions=[],
            final_label="person",
            bbox=(0.1, 0.1, 0.3, 0.3),
        )
        buffer.add_cascade_resolved(
            image_id="sample_1",
            image_path=str(temp_dir / "img_1.jpg"),
            opinions=[],
            final_label="car",
            bbox=(0.2, 0.2, 0.5, 0.5),
        )
        buffer.add_cascade_resolved(
            image_id="sample_2",
            image_path=str(temp_dir / "img_2.jpg"),
            opinions=[],
            final_label="person",
            bbox=(0.1, 0.1, 0.4, 0.4),
        )

        auto_trainer = AutoTrainer(
            replay_buffer=buffer,
            trainer=MagicMock(),
            config=AutoTrainConfig(enabled=True),
        )

        # Get all samples (not random.choices which may return duplicates)
        samples = list(buffer._samples.values())
        dataset_path = await auto_trainer._create_dataset(samples)

        dataset_dir = Path(dataset_path)
        assert (dataset_dir / "data.yaml").exists()
        assert (dataset_dir / "class_map.json").exists()

        import json
        import yaml

        # Check class map
        class_map = json.loads((dataset_dir / "class_map.json").read_text())
        assert "person" in class_map
        assert "car" in class_map
        assert len(class_map) == 2

        # Check data.yaml has correct nc
        data = yaml.safe_load((dataset_dir / "data.yaml").read_text())
        assert data["nc"] == 2
        assert set(data["names"]) == {"person", "car"}


class TestValidationMetrics:
    """Tests for ValidationMetrics and ValidationResult."""

    def test_empty_metrics(self):
        metrics = ValidationMetrics(model_id="test")
        assert metrics.is_empty is True
        assert metrics.accuracy == 0.0

    def test_validation_result_improvement(self):
        result = ValidationResult(
            passed=True,
            current_metrics=ValidationMetrics(model_id="current", accuracy=0.7),
            candidate_metrics=ValidationMetrics(model_id="candidate", accuracy=0.85),
            improvement=0.15,
        )
        assert result.improvement == 0.15
        assert result.passed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
