"""
Agent Scheduler for OpenClaw Lite.

Provides cron-like scheduling for agent tasks:
- One-shot reminders
- Recurring jobs
- Job persistence
- Missed job handling
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Awaitable
from functools import cached_property

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


class JobType(Enum):
    ONCE = "once"  # Run once at scheduled time
    CRON = "cron"  # Recurring cron schedule
    INTERVAL = "interval"  # Fixed interval


@dataclass
class JobRun:
    """Record of a job execution."""
    run_id: str
    job_id: str
    started_at: float
    completed_at: Optional[float] = None
    status: JobStatus = JobStatus.RUNNING
    result: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status.value,
            "result": self.result,
            "error": self.error
        }


@dataclass
class CronJob:
    """
    A scheduled job.
    
    Supports:
    - One-shot: runs once at next_run
    - Cron: recurring schedule (cron_expr)
    - Interval: runs every interval_sec
    """
    
    job_id: str
    name: str
    task: str  # The task/prompt to execute
    job_type: JobType = JobType.ONCE
    
    # Scheduling
    next_run: Optional[float] = None  # Unix timestamp
    cron_expr: Optional[str] = None  # e.g., "0 9 * * 1" (Mon 9am)
    interval_sec: Optional[int] = None  # For interval type
    
    # Configuration
    agent_id: Optional[str] = None  # Target agent
    session_key: Optional[str] = None  # Target session
    channel: Optional[str] = None  # Announce to channel
    timeout_sec: int = 300
    
    # State
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    last_run: Optional[float] = None
    run_count: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "task": self.task,
            "job_type": self.job_type.value,
            "next_run": self.next_run,
            "cron_expr": self.cron_expr,
            "interval_sec": self.interval_sec,
            "agent_id": self.agent_id,
            "session_key": self.session_key,
            "channel": self.channel,
            "timeout_sec": self.timeout_sec,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "run_count": self.run_count,
            "enabled": self.enabled,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CronJob":
        return cls(
            job_id=data["job_id"],
            name=data["name"],
            task=data["task"],
            job_type=JobType(data.get("job_type", "once")),
            next_run=data.get("next_run"),
            cron_expr=data.get("cron_expr"),
            interval_sec=data.get("interval_sec"),
            agent_id=data.get("agent_id"),
            session_key=data.get("session_key"),
            channel=data.get("channel"),
            timeout_sec=data.get("timeout_sec", 300),
            status=JobStatus(data.get("status", "pending")),
            created_at=data.get("created_at", time.time()),
            last_run=data.get("last_run"),
            run_count=data.get("run_count", 0),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {})
        )
    
    def calculate_next_run(self) -> Optional[float]:
        """Calculate the next run time."""
        now = time.time()
        
        if self.job_type == JobType.ONCE:
            # One-shot: no recurrence
            return None
        
        elif self.job_type == JobType.INTERVAL:
            if self.interval_sec:
                return now + self.interval_sec
            return None
        
        elif self.job_type == JobType.CRON:
            if self.cron_expr:
                return self._parse_cron_next(self.cron_expr)
            return None
        
        return None
    
    def _parse_cron_next(self, expr: str) -> Optional[float]:
        """
        Parse a cron expression and return next run time.
        
        Simplified cron: minute hour day month weekday
        Supports: *, specific values, ranges
        """
        try:
            parts = expr.split()
            if len(parts) != 5:
                logger.warning(f"Invalid cron expression: {expr}")
                return None
            
            minute, hour, day, month, weekday = parts
            now = datetime.now(timezone.utc)
            
            # Start from next minute
            candidate = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            
            # Simple implementation: check next 7 days
            for _ in range(7 * 24 * 60):  # 7 days of minutes
                if self._matches_cron(candidate, minute, hour, day, month, weekday):
                    return candidate.timestamp()
                candidate += timedelta(minutes=1)
            
            return None
            
        except Exception as e:
            logger.error(f"Cron parse error: {e}")
            return None
    
    def _matches_cron(
        self,
        dt: datetime,
        minute: str,
        hour: str,
        day: str,
        month: str,
        weekday: str
    ) -> bool:
        """Check if datetime matches cron pattern."""
        
        def matches_field(value: int, pattern: str) -> bool:
            if pattern == "*":
                return True
            if pattern.isdigit():
                return value == int(pattern)
            if "-" in pattern:
                start, end = pattern.split("-")
                return int(start) <= value <= int(end)
            if "," in pattern:
                return value in [int(v) for v in pattern.split(",")]
            return False
        
        return (
            matches_field(dt.minute, minute) and
            matches_field(dt.hour, hour) and
            matches_field(dt.day, day) and
            matches_field(dt.month, month) and
            matches_field(dt.isoweekday() % 7, weekday)  # Sunday = 0
        )


class AgentScheduler:
    """
    Scheduler for agent tasks.
    
    Features:
    - Add/remove/update scheduled jobs
    - Persistent job storage
    - Background scheduler loop
    - Job execution with timeout
    - Run history tracking
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_runs_history: int = 100
    ):
        self.storage_path = storage_path
        self.max_runs_history = max_runs_history
        
        self._jobs: Dict[str, CronJob] = {}
        self._runs: List[JobRun] = []
        self._executor: Optional[Callable] = None
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        
        if storage_path:
            self._load()
    
    def set_executor(
        self,
        executor: Callable[[CronJob], Awaitable[str]]
    ) -> None:
        """
        Set the job executor function.
        
        The executor receives a job and should return the result string.
        """
        self._executor = executor
    
    def add_job(
        self,
        name: str,
        task: str,
        job_type: JobType = JobType.ONCE,
        run_at: Optional[datetime] = None,
        delay_sec: Optional[int] = None,
        cron_expr: Optional[str] = None,
        interval_sec: Optional[int] = None,
        agent_id: Optional[str] = None,
        session_key: Optional[str] = None,
        channel: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> CronJob:
        """Add a scheduled job."""
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        
        # Calculate initial next_run
        if run_at:
            next_run = run_at.timestamp()
        elif delay_sec:
            next_run = time.time() + delay_sec
        elif cron_expr:
            job = CronJob(
                job_id=job_id,
                name=name,
                task=task,
                job_type=JobType.CRON,
                cron_expr=cron_expr
            )
            next_run = job.calculate_next_run()
        elif interval_sec:
            next_run = time.time() + interval_sec
        else:
            next_run = time.time()  # Run immediately
        
        job = CronJob(
            job_id=job_id,
            name=name,
            task=task,
            job_type=job_type,
            next_run=next_run,
            cron_expr=cron_expr,
            interval_sec=interval_sec,
            agent_id=agent_id,
            session_key=session_key,
            channel=channel,
            metadata=metadata or {}
        )
        
        self._jobs[job_id] = job
        self._save()
        
        logger.info(f"Added job: {job_id} ({name}) - next run: {datetime.fromtimestamp(next_run, tz=timezone.utc).isoformat() if next_run else 'N/A'}")
        
        return job
    
    def get_job(self, job_id: str) -> Optional[CronJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    def list_jobs(
        self,
        include_disabled: bool = False,
        agent_id: Optional[str] = None
    ) -> List[CronJob]:
        """List all jobs."""
        jobs = list(self._jobs.values())
        
        if not include_disabled:
            jobs = [j for j in jobs if j.enabled]
        if agent_id:
            jobs = [j for j in jobs if j.agent_id == agent_id]
        
        # Sort by next_run
        jobs.sort(key=lambda j: j.next_run or float('inf'))
        
        return jobs
    
    def update_job(
        self,
        job_id: str,
        **updates
    ) -> Optional[CronJob]:
        """Update a job."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        
        for key, value in updates.items():
            if hasattr(job, key):
                setattr(job, key, value)
        
        self._save()
        logger.info(f"Updated job: {job_id}")
        
        return job
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a job."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save()
            logger.info(f"Removed job: {job_id}")
            return True
        return False
    
    def disable_job(self, job_id: str) -> bool:
        """Disable a job (keep but don't run)."""
        job = self._jobs.get(job_id)
        if job:
            job.enabled = False
            job.status = JobStatus.DISABLED
            self._save()
            return True
        return False
    
    def enable_job(self, job_id: str) -> bool:
        """Enable a disabled job."""
        job = self._jobs.get(job_id)
        if job:
            job.enabled = True
            job.status = JobStatus.PENDING
            if job.next_run and job.next_run < time.time():
                # Recalculate if past due
                job.next_run = job.calculate_next_run() or time.time()
            self._save()
            return True
        return False
    
    async def run_job(self, job_id: str) -> Optional[JobRun]:
        """Manually run a job immediately."""
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Job not found: {job_id}")
            return None
        
        return await self._execute_job(job)
    
    def get_runs(
        self,
        job_id: Optional[str] = None,
        limit: int = 20
    ) -> List[JobRun]:
        """Get job run history."""
        runs = self._runs
        if job_id:
            runs = [r for r in runs if r.job_id == job_id]
        return runs[-limit:]
    
    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Scheduler started")
    
    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_jobs()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
            
            await asyncio.sleep(1)  # Check every second
    
    async def _check_jobs(self) -> None:
        """Check for jobs that need to run."""
        now = time.time()
        
        for job in list(self._jobs.values()):
            if not job.enabled:
                continue
            if job.status == JobStatus.RUNNING:
                continue
            if not job.next_run or job.next_run > now:
                continue
            
            # Job is due
            logger.debug(f"Executing due job: {job.job_id}")
            await self._execute_job(job)
    
    async def _execute_job(self, job: CronJob) -> JobRun:
        """Execute a job."""
        run = JobRun(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            job_id=job.job_id,
            started_at=time.time()
        )
        
        job.status = JobStatus.RUNNING
        
        try:
            if self._executor:
                result = await asyncio.wait_for(
                    self._executor(job),
                    timeout=job.timeout_sec
                )
                run.result = result
                run.status = JobStatus.COMPLETED
                job.status = JobStatus.COMPLETED
            else:
                run.result = "No executor configured"
                run.status = JobStatus.COMPLETED
                job.status = JobStatus.COMPLETED
                
        except asyncio.TimeoutError:
            run.error = "Timeout"
            run.status = JobStatus.FAILED
            job.status = JobStatus.FAILED
            
        except Exception as e:
            run.error = str(e)
            run.status = JobStatus.FAILED
            job.status = JobStatus.FAILED
            logger.error(f"Job execution error: {e}")
        
        run.completed_at = time.time()
        job.last_run = run.started_at
        job.run_count += 1
        
        # Calculate next run for recurring jobs
        if job.job_type != JobType.ONCE:
            job.next_run = job.calculate_next_run()
            if job.next_run:
                job.status = JobStatus.PENDING
        else:
            job.enabled = False  # Disable one-shot jobs after running
        
        # Store run history
        self._runs.append(run)
        if len(self._runs) > self.max_runs_history:
            self._runs = self._runs[-self.max_runs_history:]
        
        self._save()
        
        logger.info(
            f"Job {job.job_id} completed - "
            f"status: {run.status.value}, "
            f"duration: {run.completed_at - run.started_at:.2f}s"
        )
        
        return run
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "jobs": {k: v.to_dict() for k, v in self._jobs.items()},
            "runs": [r.to_dict() for r in self._runs[-self.max_runs_history:]]
        }
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            data = json.loads(self.storage_path.read_text())
            for key, job_data in data.get("jobs", {}).items():
                self._jobs[key] = CronJob.from_dict(job_data)
            for run_data in data.get("runs", []):
                self._runs.append(JobRun(
                    run_id=run_data["run_id"],
                    job_id=run_data["job_id"],
                    started_at=run_data["started_at"],
                    completed_at=run_data.get("completed_at"),
                    status=JobStatus(run_data.get("status", "completed")),
                    result=run_data.get("result"),
                    error=run_data.get("error")
                ))
            logger.info(f"Loaded {len(self._jobs)} jobs from storage")
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}")


# Global scheduler instance
_scheduler: Optional[AgentScheduler] = None


def get_scheduler(storage_path: Optional[Path] = None) -> AgentScheduler:
    """Get or create the global scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AgentScheduler(storage_path=storage_path)
    return _scheduler
