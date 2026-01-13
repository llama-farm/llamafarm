"""
Training executor for non-blocking ML training.

Provides a ThreadPoolExecutor for offloading CPU-bound training code
(sklearn fit, SetFit training, PyTorch loops) from the async event loop.
This prevents training from blocking health checks and other requests.

Usage:
    from utils.training_executor import get_training_executor

    async def fit(self, data):
        loop = asyncio.get_running_loop()
        executor = get_training_executor()
        result = await loop.run_in_executor(executor, self._fit_sync, data)
        return result

Environment Variables:
    ML_TRAINING_WORKERS: Max concurrent training jobs (default: 2)
"""

import atexit
import logging
import os
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Singleton executor instance
_executor: ThreadPoolExecutor | None = None


def get_training_executor() -> ThreadPoolExecutor:
    """Get or create the training thread pool executor.

    Returns a singleton ThreadPoolExecutor for running CPU-bound training
    operations without blocking the async event loop.

    The executor uses threads (not processes) to avoid pickling issues
    with complex model objects like sklearn estimators and PyTorch models.

    Returns:
        ThreadPoolExecutor configured for ML training workloads.
    """
    global _executor

    if _executor is None:
        max_workers = int(os.environ.get("ML_TRAINING_WORKERS", "2"))
        logger.info(f"Creating training executor with max_workers={max_workers}")
        _executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ml-training",
        )
        # Register cleanup on exit
        atexit.register(_shutdown_executor)

    return _executor


def _shutdown_executor() -> None:
    """Shutdown the executor gracefully."""
    global _executor
    if _executor is not None:
        logger.info("Shutting down training executor")
        _executor.shutdown(wait=True, cancel_futures=False)
        _executor = None


def shutdown_training_executor() -> None:
    """Explicitly shutdown the training executor.

    Call this during application shutdown for clean cleanup.
    Safe to call multiple times.
    """
    _shutdown_executor()


def get_training_executor_stats() -> dict:
    """Get statistics about the training executor.

    Returns:
        Dict with executor stats or empty dict if not initialized.
    """
    if _executor is None:
        return {"initialized": False}

    # ThreadPoolExecutor doesn't expose much info, but we can get basics
    return {
        "initialized": True,
        "max_workers": _executor._max_workers,
        "thread_name_prefix": _executor._thread_name_prefix,
    }
