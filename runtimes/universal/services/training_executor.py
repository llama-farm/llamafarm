"""Training executor service for non-blocking ML model training.

This service provides a ThreadPoolExecutor to offload CPU-bound training
operations from the async event loop, preventing the server from blocking
during model training.

Issue addressed: In server.py, the /v1/anomaly/fit and /v1/classifier/fit
endpoints are async, but the underlying training calls execute synchronous
CPU-bound code (scikit-learn or SetFit) directly in the main thread.
Impact: While a model is training, the entire API freezes.

Solution: Offload heavy training tasks to a thread pool.
"""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic return type
T = TypeVar("T")

# Global executor for training tasks
# Using a bounded pool to prevent resource exhaustion
# Default: 4 workers (reasonable for most systems)
_training_executor: ThreadPoolExecutor | None = None
_MAX_TRAINING_WORKERS = 4


def get_training_executor() -> ThreadPoolExecutor:
    """Get or create the global training executor.

    Returns:
        ThreadPoolExecutor for training tasks
    """
    global _training_executor
    if _training_executor is None:
        _training_executor = ThreadPoolExecutor(
            max_workers=_MAX_TRAINING_WORKERS,
            thread_name_prefix="training-",
        )
        logger.info(f"Created training executor with {_MAX_TRAINING_WORKERS} workers")
    return _training_executor


def shutdown_training_executor() -> None:
    """Shutdown the global training executor.

    Call this during application shutdown to clean up resources.
    """
    global _training_executor
    if _training_executor is not None:
        logger.info("Shutting down training executor...")
        _training_executor.shutdown(wait=True)
        _training_executor = None
        logger.info("Training executor shut down")


async def run_in_executor(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run a synchronous function in the training thread pool.

    This is the primary method for offloading CPU-bound training operations
    from the async event loop.

    Args:
        func: Synchronous function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function

    Example:
        # In an async endpoint:
        result = await run_in_executor(
            model._fit_sync,
            X_scaled,
            epochs=100,
            batch_size=32
        )
    """
    loop = asyncio.get_running_loop()
    executor = get_training_executor()

    # Wrap the function call to handle kwargs
    def _wrapped() -> T:
        return func(*args, **kwargs)

    return await loop.run_in_executor(executor, _wrapped)


async def run_training_task(
    task_name: str,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run a training task with logging and error handling.

    This is a convenience wrapper around run_in_executor that adds
    logging and standardized error handling.

    Args:
        task_name: Name of the task for logging
        func: Synchronous function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function

    Raises:
        Exception: Re-raises any exception from the training function
    """
    logger.info(f"Starting training task: {task_name}")

    try:
        result = await run_in_executor(func, *args, **kwargs)
        logger.info(f"Completed training task: {task_name}")
        return result
    except Exception as e:
        logger.error(f"Training task failed: {task_name} - {e}", exc_info=True)
        raise


class TrainingContext:
    """Context manager for training tasks with progress tracking.

    Provides a structured way to run training with proper cleanup
    and progress reporting.

    Example:
        async with TrainingContext("classifier-fit") as ctx:
            result = await ctx.run(model._fit_sync, X, y)
    """

    def __init__(self, task_name: str):
        """Initialize training context.

        Args:
            task_name: Name of the training task
        """
        self.task_name = task_name
        self._start_time: float | None = None

    async def __aenter__(self) -> "TrainingContext":
        """Enter the context."""
        import time

        self._start_time = time.time()
        logger.info(f"Training context started: {self.task_name}")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context."""
        import time

        duration = time.time() - (self._start_time or time.time())
        if exc_type is not None:
            logger.error(
                f"Training context failed: {self.task_name} "
                f"(duration: {duration:.2f}s) - {exc_val}"
            )
        else:
            logger.info(
                f"Training context completed: {self.task_name} "
                f"(duration: {duration:.2f}s)"
            )

    async def run(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run a function in the training executor.

        Args:
            func: Function to run
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function
        """
        return await run_in_executor(func, *args, **kwargs)
