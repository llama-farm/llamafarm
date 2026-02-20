"""Cross-platform fine-tuning trainer with Unsloth."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Output directory for fine-tuned models
FINETUNE_DIR = Path.home() / ".llamafarm" / "models" / "llm"


class JobStatus(str, Enum):
    """Training job status."""
    
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SFTJobConfig:
    """Configuration for SFT training job."""
    
    job_id: str
    model: str
    dataset: list[dict]
    dataset_format: str | None = None
    chat_template: str | None = None
    train_on_responses_only: bool = True
    output_dir: str | None = None
    output_gguf: bool = True
    quantization: str = "q8_0"
    lora_rank: int = 16
    lora_alpha: int = 16
    target_modules: list[str] | None = None
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    max_steps: int | None = None
    warmup_steps: int = 10
    gradient_accumulation_steps: int = 2


@dataclass
class CPTJobConfig:
    """Configuration for CPT training job."""
    
    job_id: str
    model: str
    dataset: list[dict]
    output_dir: str | None = None
    output_gguf: bool = True
    quantization: str = "q8_0"
    lora_rank: int = 16
    lora_alpha: int = 16
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 5e-5
    embedding_learning_rate: float = 5e-6
    max_seq_length: int = 2048
    max_steps: int | None = None


@dataclass
class TrainingResult:
    """Result of a training job."""
    
    job_id: str
    status: str
    output_dir: str | None = None
    metrics: dict = field(default_factory=dict)
    error: str | None = None
    duration_seconds: float | None = None


@dataclass
class TrainingJob:
    """Training job tracker."""
    
    job_id: str
    type: str  # "sft" or "cpt"
    model: str
    config: SFTJobConfig | CPTJobConfig
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    metrics: dict = field(default_factory=dict)
    output_dir: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    process: asyncio.subprocess.Process | None = None


def detect_backend() -> str:
    """Detect which backend to use (MLX or CUDA).
    
    Returns:
        "mlx" on macOS, "cuda" otherwise
    """
    if platform.system() == "Darwin":
        return "mlx"
    else:
        return "cuda"


def import_training_libs(backend: str | None = None):
    """Import training libraries based on backend.
    
    Args:
        backend: "mlx", "cuda", or None (auto-detect)
    
    Returns:
        Tuple of (FastLanguageModel, SFTTrainer, actual_backend)
    
    Raises:
        ImportError: If required libraries are not available
    """
    if backend is None:
        backend = detect_backend()
    
    try:
        if backend == "mlx":
            from unsloth_mlx import FastLanguageModel
            # Try to import SFTTrainer and SFTConfig from unsloth_mlx
            try:
                from unsloth_mlx import SFTTrainer
            except ImportError:
                # Fallback to trl if unsloth_mlx doesn't have it
                from trl import SFTTrainer
            
            return FastLanguageModel, SFTTrainer, "mlx"
        
        else:  # cuda
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            
            return FastLanguageModel, SFTTrainer, "cuda"
    
    except ImportError as e:
        logger.error(f"Failed to import training libraries for {backend}: {e}")
        raise ImportError(
            f"Fine-tuning libraries not available for {backend}. "
            f"Install unsloth_mlx (macOS) or unsloth (Linux/CUDA)."
        ) from e


class FineTuneTrainer:
    """Manages fine-tuning jobs with sequential queue."""
    
    def __init__(self):
        self._jobs: dict[str, TrainingJob] = {}
        self._queue: asyncio.Queue[TrainingJob] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._running = False
    
    async def start(self):
        """Start the job worker."""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Fine-tuning worker started")
    
    async def stop(self):
        """Stop the job worker."""
        self._running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        # Cancel any running jobs
        for job in self._jobs.values():
            if job.status == JobStatus.RUNNING and job.process:
                try:
                    job.process.terminate()
                    await asyncio.wait_for(job.process.wait(), timeout=5.0)
                except Exception as e:
                    logger.warning(f"Error terminating job {job.job_id}: {e}")
        
        logger.info("Fine-tuning worker stopped")
    
    async def queue_sft_job(self, config: SFTJobConfig) -> TrainingJob:
        """Queue a supervised fine-tuning job.
        
        Args:
            config: SFT job configuration
        
        Returns:
            TrainingJob object
        """
        job = TrainingJob(
            job_id=config.job_id,
            type="sft",
            model=config.model,
            config=config,
            status=JobStatus.QUEUED
        )
        
        self._jobs[job.job_id] = job
        await self._queue.put(job)
        logger.info(f"Queued SFT job {job.job_id} for model {config.model}")
        
        return job
    
    async def queue_cpt_job(self, config: CPTJobConfig) -> TrainingJob:
        """Queue a continued pre-training job.
        
        Args:
            config: CPT job configuration
        
        Returns:
            TrainingJob object
        """
        job = TrainingJob(
            job_id=config.job_id,
            type="cpt",
            model=config.model,
            config=config,
            status=JobStatus.QUEUED
        )
        
        self._jobs[job.job_id] = job
        await self._queue.put(job)
        logger.info(f"Queued CPT job {job.job_id} for model {config.model}")
        
        return job
    
    def get_job(self, job_id: str) -> TrainingJob | None:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    def list_jobs(self) -> list[TrainingJob]:
        """List all jobs."""
        return list(self._jobs.values())
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job.
        
        Args:
            job_id: Job ID to cancel
        
        Returns:
            True if cancelled, False if not found or already completed
        """
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        if job.status == JobStatus.RUNNING and job.process:
            try:
                job.process.terminate()
                await asyncio.wait_for(job.process.wait(), timeout=5.0)
            except Exception as e:
                logger.warning(f"Error terminating job {job_id}: {e}")
        
        job.status = JobStatus.CANCELLED
        job.completed_at = time.time()
        logger.info(f"Cancelled job {job_id}")
        
        return True
    
    async def _worker(self):
        """Background worker that processes jobs sequentially."""
        logger.info("Fine-tuning worker loop started")
        
        while self._running:
            try:
                # Wait for next job (with timeout to allow shutdown)
                try:
                    job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process the job
                await self._run_job(job)
            
            except asyncio.CancelledError:
                logger.info("Worker cancelled")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
    
    async def _run_job(self, job: TrainingJob):
        """Run a training job in a subprocess.
        
        Args:
            job: Job to run
        """
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        
        try:
            # Prepare output directory
            if job.config.output_dir:
                output_dir = Path(job.config.output_dir)
            else:
                output_dir = FINETUNE_DIR / job.job_id
            
            output_dir.mkdir(parents=True, exist_ok=True)
            job.output_dir = str(output_dir)
            
            # Save config
            config_path = output_dir / "config.json"
            with open(config_path, "w") as f:
                config_dict = {
                    "job_id": job.job_id,
                    "type": job.type,
                    "model": job.model,
                    "created_at": job.created_at,
                }
                if isinstance(job.config, SFTJobConfig):
                    config_dict.update({
                        "dataset_size": len(job.config.dataset),
                        "epochs": job.config.epochs,
                        "lora_rank": job.config.lora_rank,
                        "learning_rate": job.config.learning_rate,
                    })
                elif isinstance(job.config, CPTJobConfig):
                    config_dict.update({
                        "dataset_size": len(job.config.dataset),
                        "epochs": job.config.epochs,
                        "lora_rank": job.config.lora_rank,
                        "learning_rate": job.config.learning_rate,
                        "embedding_learning_rate": job.config.embedding_learning_rate,
                    })
                json.dump(config_dict, f, indent=2)
            
            # Run training in subprocess
            result = await self._run_training_subprocess(job, output_dir)
            
            # Update job with result
            job.metrics = result.metrics
            job.status = JobStatus(result.status)
            job.error = result.error
            job.completed_at = time.time()
            job.progress = 1.0 if job.status == JobStatus.COMPLETED else job.progress
            
            logger.info(f"Job {job.job_id} finished with status {job.status}")
        
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
    
    async def _run_training_subprocess(self, job: TrainingJob, output_dir: Path) -> TrainingResult:
        """Run training in a subprocess for OOM isolation.
        
        Args:
            job: Training job
            output_dir: Output directory
        
        Returns:
            TrainingResult
        """
        # Create a worker script path
        worker_script = Path(__file__).parent / "training_worker.py"
        
        # Prepare job config file
        job_config_path = output_dir / "job_config.json"
        with open(job_config_path, "w") as f:
            config_data = {
                "job_id": job.job_id,
                "type": job.type,
                "model": job.model,
                "output_dir": str(output_dir),
                "dataset": job.config.dataset,
            }
            
            if isinstance(job.config, SFTJobConfig):
                config_data.update({
                    "dataset_format": job.config.dataset_format,
                    "chat_template": job.config.chat_template,
                    "train_on_responses_only": job.config.train_on_responses_only,
                    "output_gguf": job.config.output_gguf,
                    "quantization": job.config.quantization,
                    "lora_rank": job.config.lora_rank,
                    "lora_alpha": job.config.lora_alpha,
                    "target_modules": job.config.target_modules,
                    "epochs": job.config.epochs,
                    "batch_size": job.config.batch_size,
                    "learning_rate": job.config.learning_rate,
                    "max_seq_length": job.config.max_seq_length,
                    "max_steps": job.config.max_steps,
                    "warmup_steps": job.config.warmup_steps,
                    "gradient_accumulation_steps": job.config.gradient_accumulation_steps,
                })
            elif isinstance(job.config, CPTJobConfig):
                config_data.update({
                    "output_gguf": job.config.output_gguf,
                    "quantization": job.config.quantization,
                    "lora_rank": job.config.lora_rank,
                    "lora_alpha": job.config.lora_alpha,
                    "epochs": job.config.epochs,
                    "batch_size": job.config.batch_size,
                    "learning_rate": job.config.learning_rate,
                    "embedding_learning_rate": job.config.embedding_learning_rate,
                    "max_seq_length": job.config.max_seq_length,
                    "max_steps": job.config.max_steps,
                })
            
            json.dump(config_data, f, indent=2)
        
        # Run subprocess
        log_path = output_dir / "training.log"
        
        try:
            with open(log_path, "w") as log_file:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(worker_script),
                    str(job_config_path),
                    stdout=log_file,
                    stderr=asyncio.subprocess.STDOUT,
                )
                
                job.process = process
                returncode = await process.wait()
            
            # Read result
            result_path = output_dir / "result.json"
            if result_path.exists():
                with open(result_path) as f:
                    result_data = json.load(f)
                
                return TrainingResult(
                    job_id=job.job_id,
                    status=result_data.get("status", "failed"),
                    output_dir=str(output_dir),
                    metrics=result_data.get("metrics", {}),
                    error=result_data.get("error"),
                    duration_seconds=result_data.get("duration_seconds")
                )
            else:
                # Worker failed before writing result
                return TrainingResult(
                    job_id=job.job_id,
                    status="failed",
                    error=f"Worker process exited with code {returncode}, no result file"
                )
        
        except Exception as e:
            logger.error(f"Subprocess execution failed: {e}")
            return TrainingResult(
                job_id=job.job_id,
                status="failed",
                error=str(e)
            )
