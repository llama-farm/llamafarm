"""Incremental trainer for vision models. Simple MVP — no EWC."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

VISION_MODELS_DIR = Path.home() / ".llamafarm" / "models" / "vision"


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.001
    validation_split: float = 0.2


@dataclass
class TrainingJob:
    job_id: str
    model_id: str
    dataset_path: str
    task: Literal["detection", "classification"]
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
    """Manages async training jobs for vision models."""

    def __init__(self, model_loader: Callable | None = None):
        self._model_loader = model_loader
        self._jobs: dict[str, TrainingJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def set_model_loader(self, loader: Callable) -> None:
        self._model_loader = loader

    async def start_training(self, model_id: str, dataset_path: str,
                             task: Literal["detection", "classification"],
                             config: TrainingConfig | None = None,
                             base_model: str | None = None) -> TrainingJob:
        config = config or TrainingConfig()
        job_id = str(uuid.uuid4())[:8]
        job = TrainingJob(job_id=job_id, model_id=model_id,
                          dataset_path=dataset_path, task=task, config=config)
        self._jobs[job_id] = job
        self._tasks[job_id] = asyncio.create_task(self._run(job, base_model))
        logger.info(f"Started training job {job_id} for {model_id}")
        return job

    async def _run(self, job: TrainingJob, base_model: str | None) -> None:
        try:
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.utcnow()

            if self._model_loader is None:
                raise RuntimeError("Model loader not configured")

            # Load a FRESH model for training — don't corrupt inference cache
            from ultralytics import YOLO
            model_id = base_model or job.model_id
            model_path = model_id  # Could be a path or a variant name
            device = 'cpu'
            
            # Try to get model path from cached model's info
            try:
                cached = await self._model_loader(model_id)
                if hasattr(cached, '_model_path') and cached._model_path:
                    model_path = cached._model_path
                device = cached.device if hasattr(cached, 'device') else 'cpu'
            except Exception:
                pass

            training_yolo = YOLO(model_path)
            
            # Train using the fresh YOLO instance
            logger.info(f"Starting YOLO training: {job.config.epochs} epochs, batch {job.config.batch_size}")
            train_args = {
                "data": job.dataset_path,
                "epochs": job.config.epochs,
                "batch": job.config.batch_size,
                "device": device if device != "auto" else None,
                "imgsz": 640,
                "patience": 50,
                "save": True,
                "verbose": True,
            }
            results = await asyncio.to_thread(training_yolo.train, **train_args)
            
            metrics = {}
            if hasattr(results, "results_dict"):
                metrics = results.results_dict
            metrics["model_path"] = str(results.save_dir) if hasattr(results, "save_dir") else None
            metrics["epochs"] = job.config.epochs

            job.metrics = metrics
            job.progress = 1.0
            job.current_epoch = job.config.epochs
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow()

            # Save versioned model and auto-export ONNX
            await self._save_versioned(job, training_yolo)
            logger.info(f"Training job {job.job_id} completed")

        except Exception as e:
            logger.error(f"Training job {job.job_id} failed: {e}")
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow()

    async def _save_versioned(self, job: TrainingJob, model: Any) -> None:
        """Save model as versioned checkpoint and auto-export ONNX."""
        model_dir = VISION_MODELS_DIR / job.model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Find next version
        existing = list(model_dir.glob("v*.pt"))
        versions = []
        for p in existing:
            m = re.match(r'v(\d+)\.pt$', p.name)
            if m:
                versions.append(int(m.group(1)))
        version = max(versions, default=0) + 1
        dst = model_dir / f"v{version}.pt"

        # Find trained weights
        train_result_path = job.metrics.get("model_path")
        if not train_result_path:
            logger.warning(f"No model_path in training metrics for {job.model_id}")
            return

        best_pt = Path(train_result_path) / "weights" / "best.pt"
        if not best_pt.exists():
            logger.warning(f"Best weights not found at {best_pt}")
            return

        shutil.copy2(str(best_pt), str(dst))
        shutil.copy2(str(best_pt), str(model_dir / "current.pt"))
        logger.info(f"Saved {job.model_id} v{version} to {dst}")

        # Auto-export ONNX from trained weights (best effort, awaited)
        try:
            await self._export_onnx(str(dst), str(model_dir))
        except Exception as e:
            logger.warning(f"ONNX auto-export failed for {job.model_id}: {e}")

    async def _export_onnx(self, model_path: str, output_dir: str) -> None:
        """Export trained weights to ONNX."""
        try:
            from ultralytics import YOLO
            trained = YOLO(model_path)
            trained.export(format="onnx", simplify=True)
            logger.info(f"ONNX exported for {model_path}")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

    def get_job(self, job_id: str) -> TrainingJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self, status: TrainingStatus | None = None) -> list[TrainingJob]:
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    async def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            return True
        return False

    async def wait_for_job(self, job_id: str, timeout: float | None = None) -> TrainingJob | None:
        task = self._tasks.get(job_id)
        if task:
            with contextlib.suppress(asyncio.TimeoutError):  # best-effort wait; training may outlast timeout
                await asyncio.wait_for(task, timeout=timeout)
        return self._jobs.get(job_id)


_trainer: IncrementalTrainer | None = None


def get_trainer() -> IncrementalTrainer:
    global _trainer
    if _trainer is None:
        _trainer = IncrementalTrainer()
    return _trainer


def set_trainer_model_loader(loader: Callable) -> None:
    get_trainer().set_model_loader(loader)
