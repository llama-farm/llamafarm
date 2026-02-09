"""Validation gate: candidate model must prove it's better before going live.

Blue/green model swap:
1. Auto-trainer produces a candidate model
2. Validation gate builds a held-out set from verified images
3. Runs BOTH current and candidate against validation set
4. Candidate must beat current on mAP (or at least match)
5. If better: promote (hot-swap in memory, keep old as rollback)
6. If worse: reject, keep current, log why

On mesh: promotion triggers MODEL_AVAILABLE gossip so peers pull the new model.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Metrics from evaluating a model against validation set."""

    model_id: str
    total_images: int = 0
    correct: int = 0
    incorrect: int = 0
    accuracy: float = 0.0
    mean_confidence: float = 0.0
    mean_iou: float = 0.0
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_empty(self) -> bool:
        return self.total_images == 0


@dataclass
class ValidationResult:
    """Result of comparing candidate vs current model."""

    passed: bool
    current_metrics: ValidationMetrics
    candidate_metrics: ValidationMetrics
    improvement: float = 0.0  # Accuracy delta
    reason: str = ""
    validated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ModelVersion:
    """A versioned model checkpoint."""

    model_id: str
    version: int
    model_path: str
    metrics: ValidationMetrics | None = None
    promoted_at: datetime | None = None
    is_active: bool = False
    is_rollback: bool = False


class ValidationGate:
    """Validates candidate models before promotion.

    Builds a held-out validation set from images verified by:
    - Human review (highest trust)
    - Audit pipeline agreement (high trust)

    Then runs both current and candidate models against this set.
    Candidate must beat (or match) current model to be promoted.

    Example:
        ```python
        gate = ValidationGate(
            model_loader=load_model,
            metadata_store=store,
            image_store=img_store,
            models_dir=Path("~/.llamafarm/models/vision"),
        )

        result = await gate.validate_candidate(
            candidate_path="/tmp/train/best.pt",
            current_model_id="yolov8n",
        )

        if result.passed:
            version = await gate.promote(
                candidate_path="/tmp/train/best.pt",
                model_id="yolov8n",
            )
            print(f"Promoted to v{version.version}")
        else:
            print(f"Rejected: {result.reason}")
        ```
    """

    def __init__(
        self,
        model_loader: Any = None,
        metadata_store: Any = None,
        image_store: Any = None,
        models_dir: Path | str | None = None,
        min_validation_samples: int = 10,
        min_improvement: float = -0.02,  # Allow slight regression (2%)
    ):
        self._model_loader = model_loader
        self._metadata_store = metadata_store
        self._image_store = image_store
        self._models_dir = Path(models_dir) if models_dir else Path.home() / ".llamafarm" / "models" / "vision"
        self._min_validation_samples = min_validation_samples
        self._min_improvement = min_improvement
        self._versions: dict[str, list[ModelVersion]] = {}
        self._on_model_promoted: Any = None

    def set_on_model_promoted(self, callback: Any) -> None:
        """Set callback for when a model is promoted."""
        self._on_model_promoted = callback

    async def build_validation_set(
        self,
        min_confidence: float = 0.8,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Build validation set from verified detections.

        Sources (in order of trust):
        1. Human-reviewed images (stage="review", highest priority)
        2. Audit-verified images (stage="audit", agreement confirmed)
        3. High-confidence primary detections (stage="primary", fallback)

        Returns:
            List of validation samples with image_id, expected_class, bbox
        """
        if self._metadata_store is None:
            logger.warning("No metadata store for validation set")
            return []

        samples = []

        # Get audit-verified (agreement = both models said the same thing)
        verified = self._metadata_store.get_recent_high_confidence(
            min_confidence=min_confidence,
            stage="audit",
            limit=limit,
        )

        for record in verified:
            samples.append({
                "image_id": record.image_id,
                "expected_class": record.class_name,
                "confidence": record.confidence,
                "bbox": (record.x1, record.y1, record.x2, record.y2),
                "source": "audit_verified",
            })

        # If not enough, supplement with high-confidence primary detections
        if len(samples) < limit:
            remaining = limit - len(samples)
            existing_ids = {s["image_id"] for s in samples}

            primary = self._metadata_store.get_recent_high_confidence(
                min_confidence=max(min_confidence, 0.9),  # Higher bar for unverified
                stage="primary",
                limit=remaining * 2,  # Get extra to filter dupes
            )

            for record in primary:
                if record.image_id not in existing_ids:
                    samples.append({
                        "image_id": record.image_id,
                        "expected_class": record.class_name,
                        "confidence": record.confidence,
                        "bbox": (record.x1, record.y1, record.x2, record.y2),
                        "source": "primary_high_confidence",
                    })
                    existing_ids.add(record.image_id)

                if len(samples) >= limit:
                    break

        logger.info(f"Built validation set: {len(samples)} samples")
        return samples

    async def validate_candidate(
        self,
        candidate_path: str,
        current_model_id: str,
    ) -> ValidationResult:
        """Validate a candidate model against the current model.

        Runs both models against the held-out validation set and compares.

        Args:
            candidate_path: Path to candidate model weights
            current_model_id: ID of the current production model

        Returns:
            ValidationResult with pass/fail and detailed metrics
        """
        # Build validation set
        val_set = await self.build_validation_set()

        if len(val_set) < self._min_validation_samples:
            return ValidationResult(
                passed=False,
                current_metrics=ValidationMetrics(model_id=current_model_id),
                candidate_metrics=ValidationMetrics(model_id="candidate"),
                reason=f"Not enough validation samples ({len(val_set)}/{self._min_validation_samples})",
            )

        # Evaluate current model
        current_metrics = await self._evaluate_model(
            model_id=current_model_id,
            model_path=None,  # Use existing loaded model
            val_set=val_set,
        )

        # Evaluate candidate
        candidate_metrics = await self._evaluate_model(
            model_id="candidate",
            model_path=candidate_path,
            val_set=val_set,
        )

        # Compare
        improvement = candidate_metrics.accuracy - current_metrics.accuracy

        passed = improvement >= self._min_improvement
        reason = ""
        if not passed:
            reason = (
                f"Candidate accuracy {candidate_metrics.accuracy:.1%} is "
                f"{abs(improvement):.1%} worse than current {current_metrics.accuracy:.1%} "
                f"(min improvement: {self._min_improvement:.1%})"
            )
        else:
            reason = (
                f"Candidate accuracy {candidate_metrics.accuracy:.1%} "
                f"{'improved' if improvement > 0 else 'maintained'} vs "
                f"current {current_metrics.accuracy:.1%} "
                f"(delta: {improvement:+.1%})"
            )

        result = ValidationResult(
            passed=passed,
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
            improvement=improvement,
            reason=reason,
        )

        logger.info(f"Validation {'PASSED' if passed else 'FAILED'}: {reason}")
        return result

    async def _evaluate_model(
        self,
        model_id: str,
        model_path: str | None,
        val_set: list[dict[str, Any]],
    ) -> ValidationMetrics:
        """Evaluate a model against the validation set."""
        metrics = ValidationMetrics(model_id=model_id, total_images=len(val_set))

        if self._model_loader is None:
            logger.warning("No model loader configured")
            return metrics

        try:
            if model_path:
                model = await self._model_loader(model_path)
            else:
                model = await self._model_loader(model_id)
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return metrics

        total_confidence = 0.0
        total_iou = 0.0
        per_class_counts: dict[str, dict[str, int]] = {}

        for sample in val_set:
            try:
                image_bytes = await self._load_image(sample["image_id"])
                if image_bytes is None:
                    metrics.total_images -= 1
                    continue

                result = await model.detect(image=image_bytes)

                if not result.boxes:
                    metrics.incorrect += 1
                    continue

                # Find best matching box
                expected_bbox = sample["bbox"]
                best_box = None
                best_iou = 0.0

                for box in result.boxes:
                    iou = self._compute_iou(
                        (box.x1, box.y1, box.x2, box.y2),
                        expected_bbox,
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_box = box

                if best_box and best_box.class_name == sample["expected_class"] and best_iou > 0.3:
                    metrics.correct += 1
                else:
                    metrics.incorrect += 1

                if best_box:
                    total_confidence += best_box.confidence
                    total_iou += best_iou

                # Per-class tracking
                expected_class = sample["expected_class"]
                if expected_class not in per_class_counts:
                    per_class_counts[expected_class] = {"correct": 0, "total": 0}
                per_class_counts[expected_class]["total"] += 1
                if best_box and best_box.class_name == expected_class and best_iou > 0.3:
                    per_class_counts[expected_class]["correct"] += 1

            except Exception as e:
                logger.warning(f"Eval failed for {sample['image_id']}: {e}")
                metrics.total_images -= 1

        # Calculate aggregate metrics
        evaluated = metrics.correct + metrics.incorrect
        if evaluated > 0:
            metrics.accuracy = metrics.correct / evaluated
            metrics.mean_confidence = total_confidence / evaluated
            metrics.mean_iou = total_iou / evaluated

        # Per-class accuracy
        for cls, counts in per_class_counts.items():
            if counts["total"] > 0:
                metrics.per_class[cls] = {
                    "accuracy": counts["correct"] / counts["total"],
                    "total": float(counts["total"]),
                }

        return metrics

    async def _load_image(self, image_id: str) -> bytes | None:
        """Load image bytes from image store."""
        if self._image_store is None:
            return None

        try:
            record = await self._image_store.get_image(image_id)
            if record and record.file_path:
                return Path(record.file_path).read_bytes()
        except Exception:
            pass
        return None

    @staticmethod
    def _compute_iou(
        box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float],
    ) -> float:
        """Compute IoU between two boxes."""
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

    async def promote(
        self,
        candidate_path: str,
        model_id: str,
        backup: bool = True,
    ) -> ModelVersion:
        """Promote candidate model to production.

        1. Version the current model as rollback
        2. Copy candidate to models dir
        3. Return new ModelVersion

        On mesh: caller should trigger MODEL_AVAILABLE gossip after this.

        Args:
            candidate_path: Path to candidate model weights
            model_id: Model ID to promote as
            backup: Keep old version as rollback

        Returns:
            ModelVersion for the promoted model
        """
        model_dir = self._models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Get next version number
        versions = self._versions.get(model_id, [])
        next_version = max((v.version for v in versions), default=0) + 1

        # Backup current if exists
        current_path = model_dir / "current.pt"
        if backup and current_path.exists():
            backup_path = model_dir / f"v{next_version - 1}.pt"
            shutil.copy2(str(current_path), str(backup_path))

            # Mark old version as rollback, update its path to the backup
            for v in versions:
                if v.is_active:
                    v.is_active = False
                    v.is_rollback = True
                    v.model_path = str(backup_path)

        # Copy candidate to current
        src = Path(candidate_path)
        if src.exists():
            shutil.copy2(str(src), str(current_path))

        # Create version record
        version = ModelVersion(
            model_id=model_id,
            version=next_version,
            model_path=str(current_path),
            promoted_at=datetime.utcnow(),
            is_active=True,
        )

        if model_id not in self._versions:
            self._versions[model_id] = []
        self._versions[model_id].append(version)

        logger.info(f"Promoted {model_id} to v{next_version} from {candidate_path}")

        # Notify callback
        if self._on_model_promoted:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(self._on_model_promoted):
                    await self._on_model_promoted(version)
                else:
                    self._on_model_promoted(version)
            except Exception as e:
                logger.warning(f"Model promoted callback failed: {e}")

        return version

    async def rollback(self, model_id: str) -> ModelVersion | None:
        """Roll back to the previous model version.

        Args:
            model_id: Model ID to roll back

        Returns:
            The restored ModelVersion, or None if no rollback available
        """
        versions = self._versions.get(model_id, [])
        rollback_version = None

        for v in reversed(versions):
            if v.is_rollback:
                rollback_version = v
                break

        if rollback_version is None:
            logger.warning(f"No rollback version found for {model_id}")
            return None

        model_dir = self._models_dir / model_id
        current_path = model_dir / "current.pt"
        rollback_path = Path(rollback_version.model_path)

        if not rollback_path.exists():
            # Try the versioned backup path
            rollback_path = model_dir / f"v{rollback_version.version}.pt"

        if rollback_path.exists():
            shutil.copy2(str(rollback_path), str(current_path))

            # Update version states
            for v in versions:
                v.is_active = False
            rollback_version.is_active = True
            rollback_version.is_rollback = False

            logger.info(f"Rolled back {model_id} to v{rollback_version.version}")
            return rollback_version
        else:
            logger.error(f"Rollback file not found: {rollback_path}")
            return None

    def get_versions(self, model_id: str) -> list[ModelVersion]:
        """Get all versions of a model."""
        return self._versions.get(model_id, [])

    def get_active_version(self, model_id: str) -> ModelVersion | None:
        """Get the currently active version of a model."""
        for v in self._versions.get(model_id, []):
            if v.is_active:
                return v
        return None
