"""
RAG Health Cache Service

This service manages cached health status for the RAG service to avoid blocking
health checks while still providing up-to-date information.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional
from threading import Lock, Thread

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
        self.cache: Optional[Dict[str, Any]] = None
        self.cache_timestamp: float = 0
        self.lock = Lock()
        self.background_thread: Optional[Thread] = None
        self.running = False
        
    def start_background_updates(self):
        """Start the background thread that periodically updates health status."""
        if self.background_thread and self.background_thread.is_alive():
            return
            
        self.running = True
        self.background_thread = Thread(target=self._background_update_loop, daemon=True)
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
                        "status": health_data.get("status", "unknown") if health_data else "failed",
                        "cache_age": 0
                    }
                )
                
            except Exception as e:
                logger.warning(f"Background RAG health check failed: {e}")
                
            # Wait for next update
            for _ in range(self.update_interval):
                if not self.running:
                    break
                time.sleep(1)
                
    def _perform_health_check(self) -> Optional[Dict[str, Any]]:
        """
        Perform the actual health check by calling the RAG health task.
        
        Returns:
            Health data dict or None if check failed
        """
        try:
            from core.celery.celery import app as celery_app  # type: ignore
            
            # Try comprehensive health check first
            health_task = signature("rag.health_check", app=celery_app)
            result = health_task.apply_async()
            
            try:
                health_data = result.get(timeout=self.timeout)
                if isinstance(health_data, dict):
                    return health_data
            except Exception:
                # Comprehensive check failed, try simple ping
                ping_task = signature("rag.ping", app=celery_app)
                ping_result = ping_task.apply_async()
                
                try:
                    ping_data = ping_result.get(timeout=2.0)  # Shorter timeout for ping
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
                                    "message": "RAG worker reachable"
                                }
                            },
                            "metrics": {
                                "latency_ms": ping_data.get("latency_ms", 0)
                            },
                            "errors": [],
                            "ping_only": True
                        }
                except Exception:
                    pass
                    
            return None
            
        except Exception as e:
            logger.warning(f"RAG health check failed: {e}")
            return None
            
    def get_cached_health(self) -> Dict[str, Any]:
        """
        Get the current cached health status.
        
        Returns:
            Health status dict with cache metadata
        """
        with self.lock:
            now = time.time()
            cache_age = int(now - self.cache_timestamp) if self.cache_timestamp > 0 else -1
            
            if self.cache is None:
                # No cache yet, try immediate check with short timeout
                immediate_health = self._perform_quick_check()
                return {
                    "status": "degraded" if immediate_health else "unhealthy",
                    "message": "RAG worker responding" if immediate_health else "RAG worker not responding",
                    "timestamp": int(now),
                    "cache_age_seconds": 0,
                    "source": "immediate_check"
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
            from core.celery.celery import app as celery_app  # type: ignore
            
            ping_task = signature("rag.ping", app=celery_app)
            result = ping_task.apply_async()
            ping_data = result.get(timeout=1.0)  # Very short timeout
            
            return isinstance(ping_data, dict) and ping_data.get("status") == "healthy"
            
        except Exception:
            return False
            
    def force_update(self) -> Dict[str, Any]:
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
_rag_health_cache: Optional[RAGHealthCache] = None


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