"""
RAG Health Cache Service

This service manages cached health status for the RAG service to avoid blocking
health checks while still providing up-to-date information.
"""

import logging
import os
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Any

logger = logging.getLogger(__name__)


def _get_pid_dir() -> Path:
    """Get the PID directory (mirrors llamafarm_common.pidfile logic)."""
    try:
        _home = Path.home()
    except RuntimeError:
        _fb = (
            os.environ.get("USERPROFILE")
            or os.environ.get("APPDATA")
            or os.environ.get("LOCALAPPDATA")
        )
        _home = Path(_fb) if _fb else Path.cwd()
    lf_data_dir = os.getenv("LF_DATA_DIR", str(_home / ".llamafarm"))
    return Path(lf_data_dir) / "pids"


def _is_process_alive(pid: int) -> bool:
    """Check whether a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


class RAGHealthCache:
    """
    Manages cached RAG health status with periodic background updates.

    This class provides non-blocking access to RAG health information by:
    1. Maintaining a cache of the last known health status
    2. Periodically updating the cache in the background
    3. Using PID file checks for reliable cross-deploy-mode health detection
    """

    def __init__(self, update_interval: int = 10, timeout: float = 5.0):
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
        Check RAG worker health via its PID file.

        This approach works reliably in both source and binary (PyApp)
        deploy modes because it avoids Celery imports and subprocess
        invocations that behave differently across environments.

        Returns:
            Health data dict or None if the worker is not running
        """
        pid_file = _get_pid_dir() / "rag.pid"

        if not pid_file.exists():
            return None

        try:
            pid_text = pid_file.read_text().strip()
            if not pid_text:
                return None
            pid = int(pid_text)
        except (ValueError, OSError):
            return None

        if not _is_process_alive(pid):
            return None

        return {
            "status": "healthy",
            "timestamp": int(time.time()),
            "message": "RAG worker responding",
            "worker_id": f"pid:{pid}",
            "checks": {
                "connectivity": {
                    "status": "healthy",
                    "message": "RAG worker process alive",
                }
            },
            "metrics": {"latency_ms": 0},
            "errors": [],
        }

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
                # No cache yet — return unhealthy immediately.
                # Background thread will populate the cache shortly.
                # Do NOT perform sync checks here as they block the event loop.
                return {
                    "status": "unhealthy",
                    "message": "RAG worker not responding (waiting for background check)",
                    "timestamp": int(now),
                    "cache_age_seconds": 0,
                    "source": "initial",
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
