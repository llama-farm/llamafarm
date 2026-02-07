"""Auto-training trigger for continuous learning.

Monitors the replay buffer and triggers training when:
1. Buffer size exceeds threshold
2. Minimum interval since last training has passed
3. System is not currently training

Integrates with:
- ReplayBuffer for training samples
- IncrementalTrainer for actual training
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
    ):
        """Initialize auto-trainer.
        
        Args:
            replay_buffer: ReplayBuffer instance
            trainer: IncrementalTrainer instance
            config: Auto-training configuration
            model_id: Model to train
        """
        self._replay_buffer = replay_buffer
        self._trainer = trainer
        self._config = config or AutoTrainConfig()
        self._model_id = model_id
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
        """Wait for training to complete and update state."""
        try:
            job = await self._trainer.wait_for_job(job_id, timeout=3600)  # 1 hour max
            
            self._state.training_in_progress = False
            self._state.current_job_id = None
            self._state.last_training_at = datetime.utcnow()
            self._state.last_training_job_id = job_id
            self._state.total_trainings += 1
            self._state.total_samples_trained += sample_count
            
            result = {
                "job_id": job_id,
                "status": job.status.value if job else "unknown",
                "samples_trained": sample_count,
                "metrics": job.metrics if job else {},
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
        """Create a dataset from replay buffer samples.
        
        Returns:
            Path to dataset directory
        """
        import tempfile
        from pathlib import Path
        
        # Create temp directory for dataset
        dataset_dir = Path(tempfile.mkdtemp(prefix="vision_train_"))
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        
        images_dir.mkdir()
        labels_dir.mkdir()
        
        # Copy samples
        for i, sample in enumerate(samples):
            if not sample.image_path:
                continue
            
            src_path = Path(sample.image_path)
            if not src_path.exists():
                continue
            
            # Copy image
            dst_image = images_dir / f"{i:06d}{src_path.suffix}"
            import shutil
            shutil.copy(src_path, dst_image)
            
            # Write label
            dst_label = labels_dir / f"{i:06d}.txt"
            dst_label.write_text(sample.label)
        
        # Create data.yaml
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(f"""
train: {images_dir}
val: {images_dir}
nc: 80
names: []  # Will be auto-detected from labels
""")
        
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
) -> AutoTrainer:
    """Initialize the global auto-trainer.
    
    Args:
        replay_buffer: ReplayBuffer instance
        trainer: IncrementalTrainer instance
        config: Auto-training configuration
        model_id: Model to train
        
    Returns:
        AutoTrainer instance
    """
    global _auto_trainer
    _auto_trainer = AutoTrainer(
        replay_buffer=replay_buffer,
        trainer=trainer,
        config=config,
        model_id=model_id,
    )
    return _auto_trainer
