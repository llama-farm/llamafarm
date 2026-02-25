"""Incremental trainer for vision models with continual learning.

Supports:
- Fine-tuning YOLO/CLIP models on custom datasets
- Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
- Experience replay from correction buffer
- Async training job management
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from models.yolo_model import YOLOModel

logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    """Training job status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.001
    
    # Continual learning
    use_ewc: bool = True
    ewc_lambda: float = 0.4
    use_replay: bool = True
    replay_ratio: float = 0.3
    
    # Validation
    validation_split: float = 0.2
    early_stopping_patience: int = 5


@dataclass
class TrainingJob:
    """Training job state."""
    
    job_id: str
    model_id: str
    dataset_path: str
    task: Literal["detection", "classification", "segmentation"]
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.QUEUED
    progress: float = 0.0
    current_epoch: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class IncrementalTrainer:
    """Manages incremental training of vision models.
    
    Example:
        ```python
        trainer = IncrementalTrainer()
        
        # Start training job
        job = await trainer.start_training(
            model_id="yolov8n",
            dataset_path="/path/to/dataset",
            task="detection",
            config=TrainingConfig(epochs=10)
        )
        
        # Check progress
        status = trainer.get_job(job.job_id)
        print(f"Progress: {status.progress:.1%}")
        
        # Wait for completion
        final = await trainer.wait_for_job(job.job_id)
        print(f"Metrics: {final.metrics}")
        ```
    """
    
    def __init__(
        self,
        model_loader: Callable[[str], Any] | None = None,
        output_dir: Path | str | None = None,
    ):
        """Initialize trainer.

        Args:
            model_loader: Async function to load models
            output_dir: Directory for trained models
        """
        self._model_loader = model_loader
        self._output_dir = Path(output_dir) if output_dir else Path.home() / ".llamafarm" / "models" / "vision" / "training"
        self._jobs: dict[str, TrainingJob] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._version_counter: dict[str, int] = {}
    
    def set_model_loader(self, loader: Callable[[str], Any]) -> None:
        """Set the model loader function."""
        self._model_loader = loader
    
    async def start_training(
        self,
        model_id: str,
        dataset_path: str,
        task: Literal["detection", "classification", "segmentation"],
        config: TrainingConfig | None = None,
        base_model: str | None = None,
    ) -> TrainingJob:
        """Start a new training job.
        
        Args:
            model_id: Model to train (or name for new model)
            dataset_path: Path to training dataset
            task: Training task type
            config: Training configuration
            base_model: Base model for transfer learning
            
        Returns:
            TrainingJob with job_id
        """
        config = config or TrainingConfig()
        job_id = str(uuid.uuid4())[:8]
        
        job = TrainingJob(
            job_id=job_id,
            model_id=model_id,
            dataset_path=dataset_path,
            task=task,
            config=config,
        )
        
        self._jobs[job_id] = job
        
        # Start training in background
        task_obj = asyncio.create_task(
            self._run_training(job, base_model)
        )
        self._running_tasks[job_id] = task_obj
        
        logger.info(f"Started training job {job_id} for {model_id}")
        
        return job
    
    async def _run_training(
        self,
        job: TrainingJob,
        base_model: str | None,
    ) -> None:
        """Run the training job."""
        try:
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.utcnow()
            
            # Load base model
            if self._model_loader is None:
                raise RuntimeError("Model loader not configured")
            
            model = await self._model_loader(base_model or job.model_id)

            # Run training (delegates to model's train method)
            if job.task == "detection":
                metrics = await self._train_detection(model, job)
            elif job.task == "classification":
                metrics = await self._train_classification(model, job)
            else:
                raise ValueError(f"Unsupported task: {job.task}")
            
            job.metrics = metrics
            job.status = TrainingStatus.COMPLETED
            job.progress = 1.0
            job.completed_at = datetime.utcnow()
            
            logger.info(f"Training job {job.job_id} completed: {metrics}")
            
        except Exception as e:
            logger.error(f"Training job {job.job_id} failed: {e}")
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow()
    
    async def _train_detection(
        self,
        model: YOLOModel,
        job: TrainingJob,
    ) -> dict[str, Any]:
        """Train a detection model."""
        config = job.config

        # Build training kwargs
        train_kwargs: dict[str, Any] = {
            "dataset_path": job.dataset_path,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
        }

        # Pass EWC config if enabled
        if config.use_ewc:
            train_kwargs["use_ewc"] = True
            train_kwargs["ewc_lambda"] = config.ewc_lambda

        # Use YOLO's built-in training (synchronous — run in thread to avoid blocking)
        if asyncio.iscoroutinefunction(getattr(model, "train", None)):
            metrics = await model.train(**train_kwargs)
        else:
            metrics = await asyncio.to_thread(model.train, **train_kwargs)

        # Update progress
        job.progress = 1.0
        job.current_epoch = config.epochs

        # Save versioned model checkpoint
        model_path = self._save_versioned_model(job)
        if model_path:
            metrics["model_path"] = model_path

        return metrics

    def _save_versioned_model(self, job: TrainingJob) -> str | None:
        """Save the trained model into the vision models directory.

        Creates a subdirectory per model with metadata.json and current.pt,
        matching the structure expected by the /v1/vision/models list endpoint.

        Returns path to saved model, or None if source not found.
        """
        import json
        import shutil
        from datetime import UTC, datetime

        # Look for the trained model output
        # YOLO saves to runs/detect/train/weights/best.pt (or classify/train/...)
        model_id = job.model_id
        # Validate model_id to prevent path traversal
        if not model_id or '/' in model_id or '\\' in model_id or model_id in ('.', '..') or '..' in model_id:
            raise ValueError(f"Invalid model_id: {model_id!r}")
        dataset_path = job.dataset_path
        dataset_dir = Path(dataset_path)
        candidates = [
            dataset_dir / "runs" / "detect" / "train" / "weights" / "best.pt",
            dataset_dir / "runs" / "classify" / "train" / "weights" / "best.pt",
            dataset_dir / "best.pt",
            dataset_dir / "weights" / "best.pt",
        ]

        src_path = None
        for candidate in candidates:
            if candidate.exists():
                src_path = candidate
                break

        if src_path is None:
            logger.warning(f"No trained model found for {model_id}")
            return None

        # Save into models dir as {model_id}/current.pt + metadata.json
        model_dir = self._output_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        dst_path = model_dir / "current.pt"
        shutil.copy2(str(src_path), str(dst_path))

        # Write metadata
        meta = {
            "name": model_id,
            "task": job.task,
            "base_model": getattr(job, "base_model", None),
            "dataset": dataset_path,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        logger.info(f"Saved model {model_id} to {dst_path}")
        return str(dst_path)
    
    async def _train_classification(
        self,
        model: Any,
        job: TrainingJob,
    ) -> dict[str, Any]:
        """Train a classification model."""
        # For CLIP, we'd use few-shot training
        # This is a placeholder for the full implementation
        return {
            "status": "classification training not yet implemented",
            "epochs": job.config.epochs,
        }
    
    def get_job(self, job_id: str) -> TrainingJob | None:
        """Get a training job by ID."""
        return self._jobs.get(job_id)
    
    def list_jobs(
        self,
        status: TrainingStatus | None = None,
    ) -> list[TrainingJob]:
        """List training jobs."""
        jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        
        task = self._running_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            return True
        
        return False
    
    async def wait_for_job(
        self,
        job_id: str,
        timeout: float | None = None,
    ) -> TrainingJob | None:
        """Wait for a training job to complete."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        
        task = self._running_tasks.get(job_id)
        if task:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=timeout)
        
        return self._jobs.get(job_id)


# Global trainer instance
_trainer: IncrementalTrainer | None = None


def get_trainer() -> IncrementalTrainer:
    """Get or create the global trainer."""
    global _trainer
    if _trainer is None:
        _trainer = IncrementalTrainer()
    return _trainer


def set_trainer_output_dir(d: Path | str) -> None:
    """Set the output directory for trained models."""
    get_trainer()._output_dir = Path(d)


def set_trainer_model_loader(loader: Callable[[str], Any]) -> None:
    """Set the model loader for the global trainer."""
    get_trainer().set_model_loader(loader)
