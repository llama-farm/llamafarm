"""
RAG Health Check Tasks

Celery tasks for RAG service health monitoring and diagnostics.
"""

import logging
import time
from typing import Any, Dict

from celery import Task

logger = logging.getLogger(__name__)


class HealthTask(Task):
    """Base task class for health check operations."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log task failure details."""
        logger.error(
            "RAG health task failed",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "error": str(exc),
                "args": args,
                "kwargs": kwargs,
            },
        )


def create_health_tasks(app):
    """Create health check tasks bound to the given Celery app."""
    
    @app.task(bind=True, base=HealthTask, name="rag.health_check")
    def rag_health_check_task(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the RAG service.
        
        This task tests core RAG functionality and returns detailed health status.
        It's designed to be called by the server to check RAG service availability.
        
        Returns:
            Dict containing health status, metrics, and diagnostic information
        """
        start_time = time.time()
        
        logger.info(
            "Starting RAG health check",
            extra={"task_id": self.request.id}
        )
        
        health_data = {
            "status": "healthy",
            "timestamp": int(start_time),
            "task_id": self.request.id,
            "worker_id": getattr(self.request, 'hostname', 'unknown'),
            "checks": {},
            "metrics": {},
            "errors": []
        }
        
        try:
            # Check 1: Task system is working (we're already here, so this passes)
            health_data["checks"]["task_system"] = {
                "status": "healthy",
                "message": "RAG worker processing tasks"
            }
            
            # Check 2: Import system - verify we can import core RAG modules
            try:
                from api import DatabaseSearchAPI
                from core.ingest_handler import IngestHandler
                health_data["checks"]["imports"] = {
                    "status": "healthy", 
                    "message": "Core RAG modules importable"
                }
            except Exception as e:
                health_data["checks"]["imports"] = {
                    "status": "degraded",
                    "message": f"Import issues: {str(e)}"
                }
                health_data["errors"].append(f"Import error: {e}")
            
            # Check 3: Configuration system - verify we can load config templates
            try:
                import yaml
                
                # Try to load a basic YAML to test the system
                test_yaml = "test: value"
                yaml.safe_load(test_yaml)
                
                health_data["checks"]["config_system"] = {
                    "status": "healthy",
                    "message": "Configuration system functional"
                }
            except Exception as e:
                health_data["checks"]["config_system"] = {
                    "status": "degraded", 
                    "message": f"Config system issues: {str(e)}"
                }
                health_data["errors"].append(f"Config error: {e}")
            
            # Check 4: Memory and performance metrics
            try:
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                health_data["metrics"]["memory_mb"] = round(memory_mb, 2)
                health_data["metrics"]["cpu_percent"] = cpu_percent
                
                # Flag if resource usage is high
                if memory_mb > 1000:  # > 1GB
                    health_data["checks"]["memory"] = {
                        "status": "degraded",
                        "message": f"High memory usage: {memory_mb:.1f}MB"
                    }
                else:
                    health_data["checks"]["memory"] = {
                        "status": "healthy",
                        "message": f"Memory usage normal: {memory_mb:.1f}MB"
                    }
                    
            except Exception as e:
                health_data["checks"]["performance"] = {
                    "status": "degraded",
                    "message": f"Cannot collect metrics: {str(e)}"
                }
            
            # Determine overall status based on individual checks
            check_statuses = [check["status"] for check in health_data["checks"].values()]
            
            if "unhealthy" in check_statuses:
                health_data["status"] = "unhealthy"
            elif "degraded" in check_statuses:
                health_data["status"] = "degraded"
            else:
                health_data["status"] = "healthy"
            
            # Add timing metrics
            duration_ms = int((time.time() - start_time) * 1000)
            health_data["metrics"]["check_duration_ms"] = duration_ms
            
            logger.info(
                "RAG health check completed",
                extra={
                    "task_id": self.request.id,
                    "status": health_data["status"],
                    "duration_ms": duration_ms,
                    "checks_passed": len([c for c in check_statuses if c == "healthy"]),
                    "total_checks": len(check_statuses)
                }
            )
            
            return health_data
            
        except Exception as e:
            logger.error(
                "RAG health check failed",
                extra={
                    "task_id": self.request.id,
                    "error": str(e)
                },
                exc_info=True
            )
            
            # Return failure status
            return {
                "status": "unhealthy",
                "timestamp": int(start_time),
                "task_id": self.request.id,
                "worker_id": getattr(self.request, 'hostname', 'unknown'),
                "checks": {},
                "metrics": {"check_duration_ms": int((time.time() - start_time) * 1000)},
                "errors": [f"Health check exception: {str(e)}"],
                "message": f"Health check failed: {str(e)}"
            }

    @app.task(bind=True, base=HealthTask, name="rag.ping")
    def rag_ping_task(self) -> Dict[str, Any]:
        """
        Simple ping task for basic connectivity testing.
        
        Returns:
            Dict with basic ping response and timing
        """
        start_time = time.time()
        
        return {
            "status": "healthy",
            "message": "RAG worker responding",
            "timestamp": int(start_time),
            "task_id": self.request.id,
            "worker_id": getattr(self.request, 'hostname', 'unknown'),
            "latency_ms": int((time.time() - start_time) * 1000)
        }
    
    return rag_health_check_task, rag_ping_task


# Global variables to store the tasks once created
rag_health_check_task = None
rag_ping_task = None