"""Auto-training trigger for continuous learning.

Monitors the replay buffer and triggers training when:
1. Buffer size exceeds threshold
2. Minimum interval since last training has passed
3. System is not currently training

After training completes:
4. Validation gate checks candidate vs current model
5. If candidate is better: blue/green swap (promote)
6. If worse: reject, keep current

Integrates with:
- ReplayBuffer for training samples
- IncrementalTrainer for actual training
- ValidationGate for blue/green promotion
- Streaming detector for buffer updates
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class AutoTrainConfig:
    """Configuration for auto-training."""
    
    enabled: bool = True
    threshold: int = 50  # Samples before auto-training
    min_interval_hours: float = 6.0  # Minimum hours between training
    epochs: int = 5
    batch_size: int = 16
    use_ewc: bool = True
    use_replay: bool = True
    replay_ratio: float = 0.3


@dataclass
class AutoTrainState:
    """State of the auto-training system."""
    
    last_training_at: datetime | None = None
    last_training_job_id: str | None = None
    training_in_progress: bool = False
    current_job_id: str | None = None
    total_trainings: int = 0
    total_samples_trained: int = 0


class AutoTrainer:
    """Automatic training trigger based on replay buffer size.
    
    Monitors the replay buffer and triggers incremental training
    when the buffer exceeds the configured threshold and the minimum
    interval since the last training has passed.
    
    Example:
        ```python
        from vision_training.replay_buffer import get_replay_buffer
        from vision_training.trainer import get_trainer
        
        auto_trainer = AutoTrainer(
            replay_buffer=get_replay_buffer(),
            trainer=get_trainer(),
            config=AutoTrainConfig(threshold=50, min_interval_hours=6),
        )
        
        # Start monitoring
        await auto_trainer.start()
        
        # Or manually check
        triggered = await auto_trainer.check_and_train()
        
        # Get status
        status = auto_trainer.get_status()
        ```
    """
    
    def __init__(
        self,
        replay_buffer: Any,
        trainer: Any,
        config: AutoTrainConfig | None = None,
        model_id: str = "yolov8n",
        validation_gate: Any = None,
    ):
        """Initialize auto-trainer.

        Args:
            replay_buffer: ReplayBuffer instance
            trainer: IncrementalTrainer instance
            config: Auto-training configuration
            model_id: Model to train
            validation_gate: ValidationGate for blue/green promotion
        """
        self._replay_buffer = replay_buffer
        self._trainer = trainer
        self._config = config or AutoTrainConfig()
        self._model_id = model_id
        self._validation_gate = validation_gate
        self._state = AutoTrainState()
        self._monitor_task: asyncio.Task | None = None
        self._on_training_complete: Callable[[dict], Any] | None = None
    
    def set_on_training_complete(self, callback: Callable[[dict], Any]) -> None:
        """Set callback for when training completes."""
        self._on_training_complete = callback
    
    def get_status(self) -> dict[str, Any]:
        """Get current auto-training status."""
        next_eligible_at = None
        if self._state.last_training_at:
            next_eligible_at = (
                self._state.last_training_at +
                timedelta(hours=self._config.min_interval_hours)
            )
        
        buffer_size = len(self._replay_buffer) if self._replay_buffer else 0
        
        return {
            "enabled": self._config.enabled,
            "threshold": self._config.threshold,
            "buffer_size": buffer_size,
            "training_eligible": self._is_training_eligible(),
            "last_training_at": (
                self._state.last_training_at.isoformat()
                if self._state.last_training_at else None
            ),
            "next_eligible_at": (
                next_eligible_at.isoformat()
                if next_eligible_at else None
            ),
            "training_in_progress": self._state.training_in_progress,
            "current_job_id": self._state.current_job_id,
            "total_trainings": self._state.total_trainings,
            "total_samples_trained": self._state.total_samples_trained,
        }
    
    def _is_training_eligible(self) -> bool:
        """Check if training can be triggered."""
        if not self._config.enabled:
            return False
        
        if self._state.training_in_progress:
            return False
        
        # Check buffer size
        buffer_size = len(self._replay_buffer) if self._replay_buffer else 0
        if buffer_size < self._config.threshold:
            return False
        
        # Check minimum interval
        if self._state.last_training_at:
            min_delta = timedelta(hours=self._config.min_interval_hours)
            if datetime.utcnow() - self._state.last_training_at < min_delta:
                return False
        
        return True
    
    async def check_and_train(self) -> bool:
        """Check if training should be triggered and start if so.
        
        Returns:
            True if training was triggered
        """
        if not self._is_training_eligible():
            return False
        
        logger.info("Auto-training triggered")
        return await self._start_training()
    
    async def _start_training(self) -> bool:
        """Start a training job."""
        if not self._trainer:
            logger.warning("No trainer configured")
            return False
        
        try:
            self._state.training_in_progress = True
            
            # Get samples from replay buffer
            buffer_size = len(self._replay_buffer)
            samples = self._replay_buffer.sample(buffer_size)
            
            logger.info(f"Starting training with {len(samples)} samples")
            
            # Create dataset from samples
            dataset_path = await self._create_dataset(samples)
            
            # Start training job
            from vision_training.trainer import TrainingConfig
            
            job = await self._trainer.start_training(
                model_id=self._model_id,
                dataset_path=dataset_path,
                task="detection",
                config=TrainingConfig(
                    epochs=self._config.epochs,
                    batch_size=self._config.batch_size,
                    use_ewc=self._config.use_ewc,
                    use_replay=self._config.use_replay,
                    replay_ratio=self._config.replay_ratio,
                ),
            )
            
            self._state.current_job_id = job.job_id
            
            # Wait for completion in background
            asyncio.create_task(self._wait_for_completion(job.job_id, len(samples)))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            self._state.training_in_progress = False
            return False
    
    async def _wait_for_completion(self, job_id: str, sample_count: int) -> None:
        """Wait for training to complete, validate, and promote if better."""
        try:
            job = await self._trainer.wait_for_job(job_id, timeout=3600)  # 1 hour max

            self._state.training_in_progress = False
            self._state.current_job_id = None
            self._state.last_training_at = datetime.utcnow()
            self._state.last_training_job_id = job_id
            self._state.total_trainings += 1
            self._state.total_samples_trained += sample_count

            promoted = False
            validation_result = None
            model_path = job.metrics.get("model_path") if job and job.metrics else None

            # Validation gate: candidate must beat current model
            if (
                self._validation_gate is not None
                and job is not None
                and job.status.value == "completed"
                and model_path
            ):
                try:
                    validation_result = await self._validation_gate.validate_candidate(
                        candidate_path=model_path,
                        current_model_id=self._model_id,
                    )

                    if validation_result.passed:
                        version = await self._validation_gate.promote(
                            candidate_path=model_path,
                            model_id=self._model_id,
                        )
                        promoted = True
                        logger.info(
                            f"Candidate promoted to v{version.version}: "
                            f"{validation_result.reason}"
                        )
                    else:
                        logger.info(
                            f"Candidate rejected: {validation_result.reason}"
                        )
                except Exception as e:
                    logger.error(f"Validation gate failed: {e}")

            result = {
                "job_id": job_id,
                "status": job.status.value if job else "unknown",
                "samples_trained": sample_count,
                "metrics": job.metrics if job else {},
                "promoted": promoted,
                "validation": {
                    "passed": validation_result.passed if validation_result else None,
                    "improvement": validation_result.improvement if validation_result else None,
                    "reason": validation_result.reason if validation_result else None,
                } if validation_result else None,
            }

            logger.info(f"Training completed: {result}")

            if self._on_training_complete:
                if asyncio.iscoroutinefunction(self._on_training_complete):
                    await self._on_training_complete(result)
                else:
                    self._on_training_complete(result)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._state.training_in_progress = False
            self._state.current_job_id = None
    
    async def _create_dataset(self, samples: list) -> str:
        """Create a YOLO-format dataset from replay buffer samples.

        Builds a proper class map from actual sample labels instead of
        hardcoding nc=80. Uses final_label and bbox from enhanced samples.

        Returns:
            Path to dataset directory
        """
        import shutil
        import tempfile
        from pathlib import Path

        import yaml

        dataset_dir = Path(tempfile.mkdtemp(prefix="vision_train_"))
        images_dir = dataset_dir / "images" / "train"
        labels_dir = dataset_dir / "labels" / "train"
        val_images_dir = dataset_dir / "images" / "val"
        val_labels_dir = dataset_dir / "labels" / "val"

        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        val_images_dir.mkdir(parents=True)
        val_labels_dir.mkdir(parents=True)

        # First pass: collect all class names from samples
        class_names: list[str] = []
        class_map: dict[str, int] = {}

        for sample in samples:
            label_name = getattr(sample, "final_label", None) or getattr(sample, "label", "")
            if label_name and label_name not in class_map:
                class_map[label_name] = len(class_names)
                class_names.append(label_name)

        # Second pass: copy images and write labels
        valid_count = 0
        for i, sample in enumerate(samples):
            if not sample.image_path:
                continue

            src_path = Path(sample.image_path)
            if not src_path.exists():
                continue

            # Determine label name and class ID
            label_name = getattr(sample, "final_label", None) or getattr(sample, "label", "")
            if not label_name or label_name not in class_map:
                continue

            class_id = class_map[label_name]

            # Use validation split: ~20% to val
            is_val = (i % 5 == 0)
            dst_img_dir = val_images_dir if is_val else images_dir
            dst_lbl_dir = val_labels_dir if is_val else labels_dir

            # Copy image
            suffix = src_path.suffix or ".jpg"
            dst_image = dst_img_dir / f"{i:06d}{suffix}"
            shutil.copy(src_path, dst_image)

            # Write YOLO-format label
            dst_label = dst_lbl_dir / f"{i:06d}.txt"
            bbox = getattr(sample, "bbox", None)
            if bbox:
                # Convert to YOLO format: class_id cx cy w h (normalized)
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                dst_label.write_text(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            elif hasattr(sample, "label") and sample.label:
                # Fallback to raw label string (backward compat)
                dst_label.write_text(sample.label)
            else:
                dst_label.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")

            valid_count += 1

        # Create data.yaml with actual class map
        data_config = {
            "train": str(images_dir),
            "val": str(val_images_dir),
            "nc": len(class_names),
            "names": class_names,
        }
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(yaml.dump(data_config, default_flow_style=False))

        # Also write class_map.json for reference
        import json
        class_map_file = dataset_dir / "class_map.json"
        class_map_file.write_text(json.dumps(class_map, indent=2))

        logger.info(
            f"Created dataset: {valid_count} samples, "
            f"{len(class_names)} classes at {dataset_dir}"
        )

        return str(dataset_dir)
    
    async def start(self, check_interval_minutes: float = 30) -> None:
        """Start background monitoring.
        
        Args:
            check_interval_minutes: How often to check if training is needed
        """
        if self._monitor_task and not self._monitor_task.done():
            logger.warning("Monitor already running")
            return
        
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(check_interval_minutes)
        )
        logger.info(f"Auto-training monitor started (interval: {check_interval_minutes}m)")
    
    async def stop(self) -> None:
        """Stop background monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Auto-training monitor stopped")
    
    async def _monitor_loop(self, interval_minutes: float) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await self.check_and_train()
            except Exception as e:
                logger.error(f"Error in auto-train monitor: {e}")
            
            await asyncio.sleep(interval_minutes * 60)
    
    async def on_buffer_update(self, buffer_size: int) -> None:
        """Callback when replay buffer is updated.
        
        Can be used as the training trigger callback for StreamingVisionDetector.
        
        Args:
            buffer_size: Current buffer size
        """
        if buffer_size >= self._config.threshold:
            await self.check_and_train()


# Global auto-trainer instance
_auto_trainer: AutoTrainer | None = None


def get_auto_trainer() -> AutoTrainer | None:
    """Get the global auto-trainer instance."""
    return _auto_trainer


def init_auto_trainer(
    replay_buffer: Any,
    trainer: Any,
    config: AutoTrainConfig | None = None,
    model_id: str = "yolov8n",
    validation_gate: Any = None,
) -> AutoTrainer:
    """Initialize the global auto-trainer.

    Args:
        replay_buffer: ReplayBuffer instance
        trainer: IncrementalTrainer instance
        config: Auto-training configuration
        model_id: Model to train
        validation_gate: ValidationGate for blue/green promotion

    Returns:
        AutoTrainer instance
    """
    global _auto_trainer
    _auto_trainer = AutoTrainer(
        replay_buffer=replay_buffer,
        trainer=trainer,
        config=config,
        model_id=model_id,
        validation_gate=validation_gate,
    )
    return _auto_trainer
