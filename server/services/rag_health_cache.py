"""
RAG Health Cache Service

This service manages cached health status for the RAG service to avoid blocking
health checks while still providing up-to-date information.
"""

import logging
import time
from datetime import UTC, datetime, timedelta
from threading import Lock, Thread
from typing import Any

from celery import signature

logger = logging.getLogger(__name__)


class RAGHealthCache:
    """
    Manages cached RAG health status with periodic background updates.

    This class provides non-blocking access to RAG health information by:
    1. Maintaining a cache of the last known health status
    2. Periodically updating the cache in the background
    3. Falling back to fast ping checks when needed
    """

    def __init__(self, update_interval: int = 30, timeout: float = 5.0):
        """
        Initialize the RAG health cache.

        Args:
            update_interval: Seconds between background health checks
            timeout: Timeout for health check tasks in seconds
        """
        self.update_interval = update_interval
        self.timeout = timeout
        self.cache: dict[str, Any] | None = None
        self.cache_timestamp: float = 0
        self.lock = Lock()
        self.background_thread: Thread | None = None
        self.running = False

    def start_background_updates(self):
        """Start the background thread that periodically updates health status."""
        if self.background_thread and self.background_thread.is_alive():
            return

        self.running = True
        self.background_thread = Thread(
            target=self._background_update_loop, daemon=True
        )
        self.background_thread.start()
        logger.info("RAG health cache background updates started")

    def stop_background_updates(self):
        """Stop the background update thread."""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=1.0)
        logger.info("RAG health cache background updates stopped")

    def _background_update_loop(self):
        """Background thread loop that periodically updates the cache."""
        while self.running:
            try:
                # Perform health check
                health_data = self._perform_health_check()

                # Update cache
                with self.lock:
                    self.cache = health_data
                    self.cache_timestamp = time.time()

                logger.debug(
                    "RAG health cache updated",
                    extra={
                        "status": health_data.get("status", "unknown")
                        if health_data
                        else "failed",
                        "cache_age": 0,
                    },
                )

            except Exception as e:
                logger.warning(f"Background RAG health check failed: {e}")

            # Wait for next update
            for _ in range(self.update_interval):
                if not self.running:
                    break
                time.sleep(1)

    def _perform_health_check(self) -> dict[str, Any] | None:
        """
        Perform the actual health check by calling the RAG health task.

        Returns:
            Health data dict or None if check failed
        """
        try:
            from celery import current_task

            from core.celery.celery import app as celery_app  # type: ignore

            # Check if we're already inside a Celery task context
            # Use multiple methods to detect task context for better reliability
            in_task_context = False
            try:
                if current_task is not None:
                    task_id = (
                        getattr(current_task.request, "id", None)
                        if hasattr(current_task, "request")
                        else None
                    )
                    in_task_context = bool(task_id)
                    logger.debug(
                        f"Health check task context detection: current_task={current_task}, task_id={task_id}, in_context={in_task_context}"
                    )
            except Exception as e:
                logger.debug(f"Health check task context detection failed: {e}")
                # Default to assuming we're in a task context to be safe
                in_task_context = True

            # Try comprehensive health check first
            health_task = signature("rag.health_check", app=celery_app)
            result = health_task.apply_async(
                expires=datetime.now(UTC) + timedelta(seconds=10)
            )

            try:
                health_data = self._safe_get_result(
                    result, self.timeout, in_task_context
                )
                if isinstance(health_data, dict):
                    return health_data
            except Exception:
                # Comprehensive check failed, try simple ping
                ping_task = signature("rag.ping", app=celery_app)
                ping_result = ping_task.apply_async(
                    expires=datetime.now(UTC) + timedelta(seconds=10)
                )

                try:
                    ping_data = self._safe_get_result(
                        ping_result, 2.0, in_task_context
                    )  # Shorter timeout for ping
                    if isinstance(ping_data, dict):
                        # Convert ping response to health format
                        return {
                            "status": ping_data.get("status", "degraded"),
                            "timestamp": ping_data.get("timestamp", int(time.time())),
                            "message": "RAG worker responding (ping only)",
                            "worker_id": ping_data.get("worker_id", "unknown"),
                            "checks": {
                                "connectivity": {
                                    "status": "healthy",
                                    "message": "RAG worker reachable",
                                }
                            },
                            "metrics": {"latency_ms": ping_data.get("latency_ms", 0)},
                            "errors": [],
                            "ping_only": True,
                        }
                except Exception:
                    logger.warning(
                        "RAG health check failed: Ping check failed", exc_info=True
                    )
                    pass

            return None

        except Exception as e:
            logger.warning(f"RAG health check failed: {e}")
            return None

    def _safe_get_result(self, result, timeout: float, in_task_context: bool):
        """
        Safely get result from a Celery task, avoiding result.get() within task context.

        Args:
            result: Celery AsyncResult object
            timeout: Timeout in seconds
            in_task_context: Whether we're currently inside a Celery task

        Returns:
            Task result data

        Raises:
            Exception: If task fails or times out
        """
        # Always use polling approach to be extra safe
        # This avoids any potential issues with task context detection
        logger.debug(
            f"Getting task result, in_task_context={in_task_context}, timeout={timeout}"
        )

        poll_interval = 0.1  # 100ms polling
        waited = 0.0

        while waited < timeout:
            if result.status not in ("PENDING", "STARTED"):
                break
            time.sleep(poll_interval)
            waited += poll_interval

        logger.debug(f"Task completed with status: {result.status}")

        if result.status == "SUCCESS":
            return result.result
        elif result.status == "FAILURE":
            # Get the exception info and raise it
            if hasattr(result, "traceback") and result.traceback:
                raise Exception(f"Task failed: {result.traceback}")
            else:
                raise Exception(f"Task failed with status: {result.status}")
        else:
            # Timeout or other status
            raise Exception(f"Task timed out or failed: {result.status}")

    def get_cached_health(self) -> dict[str, Any]:
        """
        Get the current cached health status.

        Returns:
            Health status dict with cache metadata
        """
        with self.lock:
            now = time.time()
            cache_age = (
                int(now - self.cache_timestamp) if self.cache_timestamp > 0 else -1
            )

            if self.cache is None:
                # No cache yet, try immediate check with short timeout
                immediate_health = self._perform_quick_check()
                return {
                    "status": "degraded" if immediate_health else "unhealthy",
                    "message": "RAG worker responding"
                    if immediate_health
                    else "RAG worker not responding",
                    "timestamp": int(now),
                    "cache_age_seconds": 0,
                    "source": "immediate_check",
                }

            # Return cached data with metadata
            cached_health = self.cache.copy()
            cached_health["cache_age_seconds"] = cache_age
            cached_health["source"] = "cache"

            # Mark as stale if cache is too old
            if cache_age > self.update_interval * 2:
                cached_health["status"] = "degraded"
                cached_health["message"] = f"Cached status (stale: {cache_age}s old)"

            return cached_health

    def _perform_quick_check(self) -> bool:
        """
        Perform a quick connectivity check.

        Returns:
            True if RAG worker is reachable, False otherwise
        """
        try:
            from celery import current_task

            from core.celery.celery import app as celery_app  # type: ignore

            # Check if we're already inside a Celery task context
            # Use multiple methods to detect task context for better reliability
            in_task_context = False
            try:
                if current_task is not None:
                    task_id = (
                        getattr(current_task.request, "id", None)
                        if hasattr(current_task, "request")
                        else None
                    )
                    in_task_context = bool(task_id)
                    logger.debug(
                        f"Quick check task context detection: current_task={current_task}, task_id={task_id}, in_context={in_task_context}"
                    )
            except Exception as e:
                logger.debug(f"Quick check task context detection failed: {e}")
                # Default to assuming we're in a task context to be safe
                in_task_context = True

            ping_task = signature("rag.ping", app=celery_app)
            result = ping_task.apply_async()
            ping_data = self._safe_get_result(
                result, 1.0, in_task_context
            )  # Very short timeout

            return isinstance(ping_data, dict) and ping_data.get("status") == "healthy"

        except Exception:
            return False

    def force_update(self) -> dict[str, Any]:
        """
        Force an immediate health check update.

        Returns:
            Updated health status
        """
        health_data = self._perform_health_check()

        with self.lock:
            if health_data:
                self.cache = health_data
                self.cache_timestamp = time.time()

        return self.get_cached_health()


# Global cache instance
_rag_health_cache: RAGHealthCache | None = None


def get_rag_health_cache() -> RAGHealthCache:
    """Get the global RAG health cache instance."""
    global _rag_health_cache

    if _rag_health_cache is None:
        _rag_health_cache = RAGHealthCache()
        _rag_health_cache.start_background_updates()

    return _rag_health_cache


def shutdown_rag_health_cache():
    """Shutdown the global RAG health cache."""
    global _rag_health_cache

    if _rag_health_cache:
        _rag_health_cache.stop_background_updates()
        _rag_health_cache = None
