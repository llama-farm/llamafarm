"""
Concurrent model loader for preloading multiple models in parallel.

Provides controlled parallel loading with:
- Semaphore-based concurrency control
- Progress tracking per model
- Error isolation (one failure doesn't block others)
- Detailed timing and status reporting
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LoadStatus(str, Enum):
    """Status of a model load operation."""

    PENDING = "pending"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    SKIPPED = "skipped"
    ALREADY_LOADED = "already_loaded"


@dataclass
class ModelLoadResult:
    """Result of loading a single model."""

    model_name: str
    """Name/identifier of the model"""

    status: LoadStatus
    """Final status of the load operation"""

    pinned: bool
    """Whether the model was pinned after loading"""

    load_time_seconds: float | None = None
    """Time taken to load (None if not loaded)"""

    error_message: str | None = None
    """Error message if status is FAILED"""

    model_path: str | None = None
    """Full model path/ID that was loaded"""


@dataclass
class BatchLoadSummary:
    """Summary of a batch model loading operation."""

    total_models: int
    """Total number of models in the batch"""

    loaded_count: int
    """Number successfully loaded"""

    failed_count: int
    """Number that failed to load"""

    skipped_count: int
    """Number skipped (e.g., already loaded)"""

    already_loaded_count: int
    """Number that were already in cache"""

    total_time_seconds: float
    """Total wall-clock time for batch"""

    concurrency_used: int
    """Actual concurrency level used"""

    results: dict[str, ModelLoadResult]
    """Per-model results (keyed by model_name)"""


class ConcurrentModelLoader:
    """Parallel model loader with semaphore control and error isolation.

    This class orchestrates concurrent loading of multiple models while:
    - Limiting parallelism to prevent OOM
    - Tracking progress for each model
    - Isolating errors (one failure doesn't stop others)
    - Providing detailed timing and status info
    """

    def __init__(self, concurrency: int = 3):
        """Initialize concurrent loader.

        Args:
            concurrency: Maximum number of parallel model loads (default: 3)
        """
        self.concurrency = max(1, concurrency)
        self._semaphore = asyncio.Semaphore(self.concurrency)
        logger.info(
            f"ConcurrentModelLoader initialized with concurrency={self.concurrency}"
        )

    async def load_one(
        self,
        model_name: str,
        model_path: str,
        pin: bool,
        load_fn: Callable[[str, bool], Awaitable[Any]],
        is_loaded_fn: Callable[[str], bool] | None = None,
    ) -> ModelLoadResult:
        """Load a single model with semaphore control.

        This method:
        1. Checks if model is already loaded (if is_loaded_fn provided)
        2. Acquires semaphore slot
        3. Calls load_fn to load the model
        4. Tracks timing and errors
        5. Returns detailed result

        Args:
            model_name: Name/identifier for the model (for logging/tracking)
            model_path: Full model path/ID to load
            pin: Whether to pin the model after loading
            load_fn: Async function to load the model. Should accept (model_path, pin)
                     and return the loaded model object.
            is_loaded_fn: Optional function to check if model is already loaded.
                          Should accept model_path and return bool.

        Returns:
            ModelLoadResult with status, timing, and error info
        """
        # Check if already loaded
        if is_loaded_fn and is_loaded_fn(model_path):
            logger.info(f"Model '{model_name}' already loaded, skipping")
            return ModelLoadResult(
                model_name=model_name,
                status=LoadStatus.ALREADY_LOADED,
                pinned=pin,
                model_path=model_path,
            )

        # Acquire semaphore and load
        async with self._semaphore:
            start_time = time.perf_counter()
            logger.info(f"Loading model '{model_name}' (path: {model_path}, pin={pin})")

            try:
                # Call the load function
                await load_fn(model_path, pin)

                elapsed = time.perf_counter() - start_time
                logger.info(
                    f"✓ Model '{model_name}' loaded successfully in {elapsed:.2f}s "
                    f"(pinned={pin})"
                )

                return ModelLoadResult(
                    model_name=model_name,
                    status=LoadStatus.LOADED,
                    pinned=pin,
                    load_time_seconds=round(elapsed, 2),
                    model_path=model_path,
                )

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                error_msg = str(e)

                logger.error(
                    f"✗ Model '{model_name}' failed to load after {elapsed:.2f}s: {error_msg}",
                    exc_info=True,
                )

                # Provide specific error hints
                if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                    logger.error(
                        f"  → OOM error loading '{model_name}'. Try:\n"
                        f"     1. Reduce concurrency (currently {self.concurrency})\n"
                        f"     2. Use smaller/quantized models\n"
                        f"     3. Unpin less critical models to free memory"
                    )
                elif (
                    "connection" in error_msg.lower() or "download" in error_msg.lower()
                ):
                    logger.error(
                        f"  → Network error loading '{model_name}'. Check:\n"
                        f"     1. Internet connection\n"
                        f"     2. HuggingFace token (for gated models)\n"
                        f"     3. Model ID is correct"
                    )

                return ModelLoadResult(
                    model_name=model_name,
                    status=LoadStatus.FAILED,
                    pinned=False,
                    load_time_seconds=round(elapsed, 2),
                    error_message=error_msg,
                    model_path=model_path,
                )

    async def load_many(
        self,
        models: list[tuple[str, str, bool]],
        load_fn: Callable[[str, bool], Awaitable[Any]],
        is_loaded_fn: Callable[[str], bool] | None = None,
    ) -> BatchLoadSummary:
        """Load multiple models concurrently.

        This method:
        1. Creates load tasks for all models
        2. Executes them with controlled concurrency
        3. Gathers results (with error isolation)
        4. Computes summary statistics

        Args:
            models: List of (model_name, model_path, pin) tuples
            load_fn: Async function to load each model
            is_loaded_fn: Optional function to check if model is loaded

        Returns:
            BatchLoadSummary with per-model results and aggregate stats
        """
        if not models:
            logger.warning("load_many called with empty model list")
            return BatchLoadSummary(
                total_models=0,
                loaded_count=0,
                failed_count=0,
                skipped_count=0,
                already_loaded_count=0,
                total_time_seconds=0.0,
                concurrency_used=self.concurrency,
                results={},
            )

        logger.info(
            f"Starting batch load of {len(models)} models with concurrency={self.concurrency}"
        )
        start_time = time.perf_counter()

        # Create load tasks
        tasks = [
            self.load_one(
                model_name=name,
                model_path=path,
                pin=pin,
                load_fn=load_fn,
                is_loaded_fn=is_loaded_fn,
            )
            for name, path, pin in models
        ]

        # Execute with error isolation (return_exceptions=True)
        # This ensures one failure doesn't cancel other loads
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.perf_counter() - start_time

        # Process results
        results: dict[str, ModelLoadResult] = {}
        loaded_count = 0
        failed_count = 0
        skipped_count = 0
        already_loaded_count = 0

        for i, result in enumerate(results_list):
            model_name = models[i][0]

            # Append a unique suffix if model_name is already in results
            base_model_name = model_name
            duplicate_counter = 1
            while model_name in results:
                model_name = f"{base_model_name}_{duplicate_counter}"
                duplicate_counter += 1

            if isinstance(result, BaseException):
                # Exception during load (shouldn't happen with proper error handling in load_one)
                logger.error(
                    f"Unexpected exception loading '{base_model_name}': {result}",
                    exc_info=result,
                )
                results[model_name] = ModelLoadResult(
                    model_name=base_model_name,
                    status=LoadStatus.FAILED,
                    pinned=False,
                    error_message=f"Unexpected error: {result}",
                    model_path=models[i][1],
                )
                failed_count += 1
            else:
                # Normal result.
                # Store using the (possibly synthetic) unique dict key, but do NOT mutate
                # result.model_name — keeping the original name preserves stable identity
                # semantics and avoids confusion when correlating results with inputs.
                results[model_name] = result

                if result.status == LoadStatus.LOADED:
                    loaded_count += 1
                elif result.status == LoadStatus.FAILED:
                    failed_count += 1
                elif result.status == LoadStatus.SKIPPED:
                    skipped_count += 1
                elif result.status == LoadStatus.ALREADY_LOADED:
                    already_loaded_count += 1

        logger.info(
            f"Batch load complete in {total_time:.2f}s: "
            f"{loaded_count} loaded, {failed_count} failed, "
            f"{already_loaded_count} already loaded, {skipped_count} skipped"
        )

        return BatchLoadSummary(
            total_models=len(models),
            loaded_count=loaded_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            already_loaded_count=already_loaded_count,
            total_time_seconds=round(total_time, 2),
            concurrency_used=self.concurrency,
            results=results,
        )


def format_load_summary(summary: BatchLoadSummary) -> str:
    """Format batch load summary as human-readable string.

    Args:
        summary: Batch load summary to format

    Returns:
        Multi-line formatted summary string
    """
    lines = [
        "=" * 60,
        "Model Preload Summary",
        "=" * 60,
        f"Total Models:      {summary.total_models}",
        f"Loaded:            {summary.loaded_count}",
        f"Failed:            {summary.failed_count}",
        f"Already Loaded:    {summary.already_loaded_count}",
        f"Skipped:           {summary.skipped_count}",
        f"Total Time:        {summary.total_time_seconds:.2f}s",
        f"Concurrency Used:  {summary.concurrency_used}",
        "=" * 60,
    ]

    # Add per-model details if any failed
    if summary.failed_count > 0:
        lines.append("")
        lines.append("Failed Models:")
        lines.append("-" * 60)
        for name, result in summary.results.items():
            if result.status == LoadStatus.FAILED:
                lines.append(f"  {name}: {result.error_message}")
        lines.append("=" * 60)

    return "\n".join(lines)


def get_load_status_emoji(status: LoadStatus) -> str:
    """Get emoji representation of load status.

    Args:
        status: Load status

    Returns:
        Emoji string
    """
    return {
        LoadStatus.LOADED: "✓",
        LoadStatus.FAILED: "✗",
        LoadStatus.ALREADY_LOADED: "↻",
        LoadStatus.SKIPPED: "⊘",
        LoadStatus.PENDING: "○",
        LoadStatus.LOADING: "⋯",
    }.get(status, "?")
