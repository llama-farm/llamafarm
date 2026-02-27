"""Incremental trainer for vision models. Simple MVP — no EWC."""

from __future__ import annotations

import asyncio
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
    imgsz: int = 640
    patience: int = 50


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
    """Manages async training jobs for vision models.

    Jobs run sequentially (one at a time) to prevent OOM from concurrent
    model training. New jobs are queued and start when the current job finishes.
    """

    def __init__(self, model_loader: Callable | None = None):
        self._model_loader = model_loader
        self._jobs: dict[str, TrainingJob] = {}
        self._queue: asyncio.Queue[tuple[TrainingJob, str | None, Callable | None]] = (
            asyncio.Queue()
        )
        self._worker_task: asyncio.Task | None = None

    def set_model_loader(self, loader: Callable) -> None:
        self._model_loader = loader

    def _ensure_worker(self) -> None:
        """Start the sequential job worker if not running."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._job_worker())

    async def _job_worker(self) -> None:
        """Process training jobs one at a time from the queue."""
        while True:
            job, base_model, on_complete = await self._queue.get()
            try:
                logger.info(
                    f"Job worker: starting {job.job_id} ({job.model_id}), "
                    f"{self._queue.qsize()} remaining in queue"
                )
                await self._run(job, base_model, on_complete)
            except Exception as e:
                logger.error(f"Job worker: unhandled error for {job.job_id}: {e}")
            finally:
                self._queue.task_done()

    async def start_training(
        self,
        model_id: str,
        dataset_path: str,
        task: Literal["detection", "classification"],
        config: TrainingConfig | None = None,
        base_model: str | None = None,
        on_complete: Callable | None = None,
    ) -> TrainingJob:
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
        await self._queue.put((job, base_model, on_complete))
        self._ensure_worker()
        queue_pos = self._queue.qsize()
        logger.info(
            f"Queued training job {job_id} for {model_id} (position {queue_pos})"
        )
        return job

    async def _run(
        self,
        job: TrainingJob,
        base_model: str | None,
        on_complete: Callable | None = None,
    ) -> None:
        # Skip if cancelled while queued
        if job.status == TrainingStatus.CANCELLED:
            logger.info(f"Skipping cancelled job {job.job_id}")
            return
        try:
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.utcnow()

            if self._model_loader is None:
                raise RuntimeError("Model loader not configured")

            # Load a FRESH model for training — don't corrupt inference cache
            from ultralytics import YOLO

            model_id = base_model or job.model_id
            model_path = model_id  # Could be a path or a variant name
            # Auto-detect best device: MPS (Apple GPU) > CUDA > CPU
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            # Try to get model path from cached model's info
            try:
                cached = await self._model_loader(model_id)
                if hasattr(cached, "_model_path") and cached._model_path:
                    model_path = cached._model_path
                # Use cached model's device if available
                if hasattr(cached, "device") and cached.device:
                    device = cached.device
            except Exception:
                pass

            training_yolo = YOLO(model_path)

            # Auto-convert COCO JSON to YOLO format if needed
            dataset_path = job.dataset_path
            from vision_training.coco_converter import is_coco_format

            if is_coco_format(dataset_path):
                from pathlib import Path as _Path

                from vision_training.coco_converter import convert_coco_to_yolo

                coco_out = (
                    _Path(dataset_path).parent / f".yolo_{_Path(dataset_path).stem}"
                )
                logger.info(
                    f"Auto-converting COCO JSON → YOLO: {dataset_path} → {coco_out}"
                )
                data_yaml = await asyncio.to_thread(
                    convert_coco_to_yolo,
                    dataset_path,
                    coco_out,
                )
                dataset_path = str(data_yaml)
                logger.info(f"COCO conversion complete: {dataset_path}")

            # Train using the fresh YOLO instance
            logger.info(
                f"Starting YOLO training: {job.config.epochs} epochs, batch {job.config.batch_size}"
            )
            train_args = {
                "data": dataset_path,
                "epochs": job.config.epochs,
                "batch": job.config.batch_size,
                "device": device if device != "auto" else None,
                "imgsz": job.config.imgsz,
                "patience": job.config.patience,
                "save": True,
                "verbose": True,
            }

            # Register cancellation callback — checked between epochs
            def _check_cancelled(trainer_obj):
                if job.status == TrainingStatus.CANCELLED:
                    logger.info(f"Job {job.job_id} cancelled — stopping training")
                    raise KeyboardInterrupt("Training cancelled")

            training_yolo.add_callback("on_train_epoch_end", _check_cancelled)

            results = await asyncio.to_thread(training_yolo.train, **train_args)

            metrics = {}
            if hasattr(results, "results_dict"):
                metrics = results.results_dict
            metrics["model_path"] = (
                str(results.save_dir) if hasattr(results, "save_dir") else None
            )
            metrics["epochs"] = job.config.epochs

            job.metrics = metrics
            job.progress = 1.0
            job.current_epoch = job.config.epochs
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow()

            # Save versioned model and auto-export ONNX
            await self._save_versioned(job, training_yolo)
            logger.info(f"Training job {job.job_id} completed")

            # Run on_complete callback (e.g., auto-eval)
            if on_complete:
                # Validate model_id: basename only, no traversal
                safe_id = Path(job.model_id).name
                if (
                    safe_id != job.model_id
                    or ".." in job.model_id
                    or ":" in job.model_id
                    or "\\" in job.model_id
                ):
                    logger.error(f"Invalid model_id for on_complete: {job.model_id}")
                else:
                    model_dir = VISION_MODELS_DIR / safe_id
                    current_pt = model_dir / "current.pt"
                    if current_pt.exists():
                        try:
                            await on_complete(job.model_id, str(current_pt))
                        except Exception as cb_err:
                            logger.error(
                                f"on_complete callback failed for {job.job_id}: {cb_err}"
                            )

        except KeyboardInterrupt:
            # Raised by cancellation callback
            if job.status != TrainingStatus.CANCELLED:
                job.status = TrainingStatus.CANCELLED
                job.completed_at = datetime.utcnow()
            logger.info(f"Training job {job.job_id} stopped by cancellation")
        except Exception as e:
            if job.status == TrainingStatus.CANCELLED:
                logger.info(f"Training job {job.job_id} stopped by cancellation")
                return
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
            m = re.match(r"v(\d+)\.pt$", p.name)
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
        """Cancel a training job.

        QUEUED jobs are cancelled immediately. RUNNING jobs are marked
        for cancellation — the training loop checks this flag between
        epochs and will stop at the next checkpoint.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status == TrainingStatus.QUEUED:
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            logger.info(f"Cancelled queued job {job_id}")
            return True
        if job.status == TrainingStatus.RUNNING:
            # Mark for cancellation — the on_train_epoch_end callback in _run()
            # checks this flag and raises KeyboardInterrupt to stop training.
            # completed_at is set when the job actually stops (in _run's except block).
            job.status = TrainingStatus.CANCELLED
            logger.info(
                f"Marked running job {job_id} for cancellation "
                f"(will stop at next epoch boundary)"
            )
            return True
        return False

    async def wait_for_job(
        self, job_id: str, timeout: float | None = None
    ) -> TrainingJob | None:
        """Poll job status until complete or timeout."""
        import time

        deadline = time.time() + timeout if timeout else None
        while True:
            job = self._jobs.get(job_id)
            if not job or job.status in (
                TrainingStatus.COMPLETED,
                TrainingStatus.FAILED,
                TrainingStatus.CANCELLED,
            ):
                return job
            if deadline and time.time() > deadline:
                return job
            await asyncio.sleep(1)


_trainer: IncrementalTrainer | None = None


def get_trainer() -> IncrementalTrainer:
    global _trainer
    if _trainer is None:
        _trainer = IncrementalTrainer()
    return _trainer


def set_trainer_model_loader(loader: Callable) -> None:
    get_trainer().set_model_loader(loader)
