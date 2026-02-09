"""Audit pipeline: automatic model-checking-model.

A bigger model periodically reviews what the small model has been saying.
100% automatic. No human needed.

Disagreements become training samples.
Agreements become validation samples (marked as verified).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuditConfig:
    """Configuration for the audit pipeline."""

    audit_model_id: str = ""            # Model to use for auditing (local or "remote:...")
    sample_size: int = 20               # How many recent images to audit per run
    min_confidence_to_audit: float = 0.7  # Only audit predictions above this
    iou_threshold: float = 0.5          # IoU threshold for bbox agreement
    agreement_threshold: float = 0.7     # Min confidence from audit model to count
    auto_interval_minutes: float = 30.0  # How often to run automatically
    enabled: bool = False


@dataclass
class AuditResult:
    """Result of a single audit comparison."""

    image_id: str
    primary_class: str
    primary_confidence: float
    audit_class: str
    audit_confidence: float
    agreement: bool
    iou: float = 0.0  # Bbox overlap


@dataclass
class AuditReport:
    """Report from an audit run."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    samples_audited: int = 0
    agreements: int = 0
    disagreements: int = 0
    accuracy_estimate: float = 0.0
    results: list[AuditResult] = field(default_factory=list)
    disagreement_samples_added: int = 0
    verified_samples_marked: int = 0


def _compute_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union between two boxes (x1,y1,x2,y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


class AuditPipeline:
    """Periodically re-checks primary model predictions with a bigger model.

    Runs automatically. No human needed.

    Disagreements become training samples (priority 2.0).
    Agreements become validation samples (marked verified in image_store).

    The audit model can be:
    - Standalone: a larger local model (yolov8x)
    - Mesh: a RemoteModelProxy pointing at a GPU node

    Same code either way. The audit_model_id can be "yolov8x"
    or "remote:gpu-server/yolov8x".

    Example:
        ```python
        audit = AuditPipeline(
            config=AuditConfig(audit_model_id="yolov8x", enabled=True),
            model_loader=load_model,
            image_store=store,
            replay_buffer=buffer,
            metadata_store=metadata,
        )

        report = await audit.run_audit()
        print(f"Accuracy: {report.accuracy_estimate:.1%}")
        print(f"Disagreements: {report.disagreements}")
        ```
    """

    def __init__(
        self,
        config: AuditConfig,
        model_loader: Any = None,
        image_store: Any = None,
        replay_buffer: Any = None,
        metadata_store: Any = None,
    ):
        self.config = config
        self._model_loader = model_loader
        self._image_store = image_store
        self._replay_buffer = replay_buffer
        self._metadata_store = metadata_store
        self._audit_model = None
        self._last_run: datetime | None = None

    async def _ensure_audit_model(self) -> bool:
        """Load the audit model if not already loaded."""
        if self._audit_model is not None:
            return True

        if not self.config.audit_model_id:
            logger.warning("No audit model configured")
            return False

        if self._model_loader is None:
            logger.warning("No model loader configured")
            return False

        try:
            self._audit_model = await self._model_loader(self.config.audit_model_id)
            return True
        except Exception as e:
            logger.error(f"Failed to load audit model: {e}")
            return False

    async def run_audit(self, sample_size: int | None = None) -> AuditReport:
        """Run an audit pass.

        1. Pull N recent high-confidence images from metadata_store
        2. Re-run through audit model
        3. Compare: class name match + IoU
        4. Disagreements -> replay buffer (source="audit", priority=2.0)
        5. Agreements -> mark verified in metadata_store
        6. Return report with accuracy estimate

        Args:
            sample_size: Override default sample size

        Returns:
            AuditReport with results and metrics
        """
        report = AuditReport()
        size = sample_size or self.config.sample_size

        if not await self._ensure_audit_model():
            logger.error("Cannot run audit: no audit model available")
            return report

        if self._metadata_store is None:
            logger.error("Cannot run audit: no metadata store")
            return report

        # 1. Get recent high-confidence primary detections
        recent = self._metadata_store.get_recent_high_confidence(
            min_confidence=self.config.min_confidence_to_audit,
            stage="primary",
            limit=size,
        )

        if not recent:
            logger.info("No samples to audit")
            return report

        # 2. For each, re-run through audit model
        for record in recent:
            try:
                result = await self._audit_single(record, report)
                if result:
                    report.results.append(result)
            except Exception as e:
                logger.error(f"Audit failed for {record.image_id}: {e}")

        # Calculate summary
        report.samples_audited = len(report.results)
        report.agreements = sum(1 for r in report.results if r.agreement)
        report.disagreements = sum(1 for r in report.results if not r.agreement)
        report.accuracy_estimate = (
            report.agreements / report.samples_audited
            if report.samples_audited > 0 else 0.0
        )

        self._last_run = datetime.utcnow()

        logger.info(
            f"Audit complete: {report.samples_audited} samples, "
            f"{report.accuracy_estimate:.1%} agreement, "
            f"{report.disagreements} disagreements"
        )

        return report

    async def _audit_single(self, record, report: AuditReport) -> AuditResult | None:
        """Audit a single detection record."""
        from vision_training.replay_buffer import ModelOpinion

        # Load image from store
        image_bytes = None
        if self._image_store:
            img_record = await self._image_store.get_image(record.image_id)
            if img_record and img_record.file_path:
                try:
                    from pathlib import Path
                    image_bytes = Path(img_record.file_path).read_bytes()
                except FileNotFoundError:
                    return None

        if image_bytes is None:
            return None

        # Run audit model
        audit_result = await self._audit_model.detect(
            image=image_bytes,
            confidence_threshold=self.config.agreement_threshold,
        )

        if not audit_result.boxes:
            return None

        # Find best matching detection
        primary_box = (record.x1, record.y1, record.x2, record.y2)
        best_audit_box = None
        best_iou = 0.0

        for box in audit_result.boxes:
            audit_box = (box.x1, box.y1, box.x2, box.y2)
            iou = _compute_iou(primary_box, audit_box)
            if iou > best_iou:
                best_iou = iou
                best_audit_box = box

        if best_audit_box is None:
            return None

        # Check agreement: same class and reasonable IoU
        agreement = (
            best_audit_box.class_name == record.class_name
            and best_iou >= self.config.iou_threshold
        )

        result = AuditResult(
            image_id=record.image_id,
            primary_class=record.class_name,
            primary_confidence=record.confidence,
            audit_class=best_audit_box.class_name,
            audit_confidence=best_audit_box.confidence,
            agreement=agreement,
            iou=best_iou,
        )

        if agreement:
            # Mark as verified in metadata store
            if self._metadata_store:
                try:
                    # Record audit agreement as detection_history
                    self._metadata_store.add_detection_history(
                        image_id=record.image_id,
                        model_id=self.config.audit_model_id,
                        class_name=best_audit_box.class_name,
                        confidence=best_audit_box.confidence,
                        box=(best_audit_box.x1, best_audit_box.y1,
                             best_audit_box.x2, best_audit_box.y2),
                        stage="audit",
                        node_id="local",
                    )
                    report.verified_samples_marked += 1
                except Exception as e:
                    logger.warning(f"Failed to mark verified: {e}")
        else:
            # Disagreement -> replay buffer
            if self._replay_buffer is not None:
                try:
                    opinions = [
                        ModelOpinion(
                            model_id=record.model_id,
                            class_name=record.class_name,
                            confidence=record.confidence,
                            bbox=primary_box,
                        ),
                        ModelOpinion(
                            model_id=self.config.audit_model_id,
                            class_name=best_audit_box.class_name,
                            confidence=best_audit_box.confidence,
                            bbox=(best_audit_box.x1, best_audit_box.y1,
                                  best_audit_box.x2, best_audit_box.y2),
                        ),
                    ]

                    img_record = await self._image_store.get_image(record.image_id) if self._image_store else None
                    image_path = img_record.file_path if img_record else ""

                    self._replay_buffer.add_audit_disagreement(
                        image_id=f"audit_{record.image_id}",
                        image_path=image_path,
                        opinions=opinions,
                        audit_label=best_audit_box.class_name,
                        bbox=primary_box,
                    )
                    report.disagreement_samples_added += 1
                except Exception as e:
                    logger.warning(f"Failed to add audit disagreement: {e}")

        return result

    @property
    def last_run(self) -> datetime | None:
        return self._last_run

    def should_run(self) -> bool:
        """Check if enough time has passed since last audit."""
        if not self.config.enabled:
            return False
        if self._last_run is None:
            return True

        elapsed = (datetime.utcnow() - self._last_run).total_seconds()
        return elapsed >= self.config.auto_interval_minutes * 60
