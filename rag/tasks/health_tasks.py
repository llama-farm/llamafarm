"""
RAG Health Check Tasks

Celery tasks for RAG service health monitoring and diagnostics.
"""

import logging
import time
from typing import Any

from celery import Task

from celery_app import app

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
                "task_args": args,
                "task_kwargs": kwargs,
            },
        )


@app.task(bind=True, base=HealthTask, name="rag.health_check")
def rag_health_check_task(self) -> dict[str, Any]:
    """
    Perform a comprehensive health check of the RAG service.

    This task tests core RAG functionality and returns detailed health status.
    It's designed to be called by the server to check RAG service availability.

    Returns:
        Dict containing health status, metrics, and diagnostic information
    """
    start_time = time.time()

    logger.info("Starting RAG health check", extra={"task_id": self.request.id})

    health_data = {
        "status": "healthy",
        "message": "RAG service is healthy",
        "timestamp": int(start_time),
        "task_id": self.request.id,
        "worker_id": getattr(self.request, "hostname", "unknown"),
        "checks": {},
        "metrics": {},
        "errors": [],
    }

    try:
        # Check 1: Task system is working (we're already here, so this passes)
        health_data["checks"]["task_system"] = {
            "status": "healthy",
            "message": "RAG worker processing tasks",
        }

        # Check 2: Import system - verify we can import core RAG modules
        try:
            from api import DatabaseSearchAPI  # noqa: F401
            from core.ingest_handler import IngestHandler  # noqa: F401

            health_data["checks"]["imports"] = {
                "status": "healthy",
                "message": "Core RAG modules importable",
            }
        except Exception as e:
            health_data["checks"]["imports"] = {
                "status": "degraded",
                "message": f"Import issues: {str(e)}",
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
                "message": "Configuration system functional",
            }
        except Exception as e:
            health_data["checks"]["config_system"] = {
                "status": "degraded",
                "message": f"Config system issues: {str(e)}",
            }
            health_data["errors"].append(f"Config error: {e}")

        # Check 4: Memory and performance metrics
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()

            health_data["metrics"]["memory_mb"] = round(memory_mb, 2)
            health_data["metrics"]["cpu_percent"] = cpu_percent

            # Flag if resource usage is high
            if memory_mb > 1000:  # > 1GB
                health_data["checks"]["memory"] = {
                    "status": "degraded",
                    "message": f"High memory usage: {memory_mb:.1f}MB",
                }
            else:
                health_data["checks"]["memory"] = {
                    "status": "healthy",
                    "message": f"Memory usage normal: {memory_mb:.1f}MB",
                }

        except Exception as e:
            health_data["checks"]["performance"] = {
                "status": "degraded",
                "message": f"Cannot collect metrics: {str(e)}",
            }

        # Determine overall status based on individual checks
        check_statuses = [check["status"] for check in health_data["checks"].values()]

        # Check if all checks are healthy
        all_healthy = all(status == "healthy" for status in check_statuses)

        if all_healthy:
            health_data["status"] = "healthy"
            health_data["message"] = "RAG service is healthy"
        else:
            health_data["status"] = "degraded"
            health_data["message"] = "RAG service may be degraded"

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
                "total_checks": len(check_statuses),
            },
        )

        return health_data

    except Exception as e:
        logger.error(
            "RAG health check failed",
            extra={"task_id": self.request.id, "error": str(e)},
            exc_info=True,
        )

        # Return failure status
        return {
            "status": "unhealthy",
            "timestamp": int(start_time),
            "task_id": self.request.id,
            "worker_id": getattr(self.request, "hostname", "unknown"),
            "checks": {},
            "metrics": {"check_duration_ms": int((time.time() - start_time) * 1000)},
            "errors": [f"Health check exception: {str(e)}"],
            "message": f"Health check failed: {str(e)}",
        }


@app.task(bind=True, base=HealthTask, name="rag.health_check_database")
def rag_health_check_database_task(
    self, project_dir: str, database: str
) -> dict[str, Any]:
    """
    Perform a health check of a specific database.

    This task focuses on database-specific health: configuration, connectivity,
    queryability, and database metrics.

    Args:
        project_dir: Directory containing llamafarm.yaml config
        database: Database name to check health for

    Returns:
        Dict containing database health status, metrics, and diagnostic information
    """
    start_time = time.time()

    logger.info(
        "Starting database health check",
        extra={
            "task_id": self.request.id,
            "project_dir": project_dir,
            "database": database,
        },
    )

    health_data = {
        "status": "healthy",
        "message": "Database is healthy",
        "database": database,
        "timestamp": int(start_time),
        "task_id": self.request.id,
        "worker_id": getattr(self.request, "hostname", "unknown"),
        "checks": {},
        "metrics": {},
        "errors": [],
    }

    try:
        # Check 1: Database configuration - verify database exists in project config
        database_config = None
        try:
            from pathlib import Path

            import yaml

            cfg_path = Path(project_dir) / "llamafarm.yaml"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    config = yaml.safe_load(f)
                    # Find the specific database configuration
                    if config and "rag" in config and "databases" in config["rag"]:
                        for db_cfg in config["rag"]["databases"]:
                            if db_cfg.get("name") == database:
                                database_config = db_cfg
                                break

                        if database_config:
                            health_data["checks"]["database_config"] = {
                                "status": "healthy",
                                "message": f"Database '{database}' found in configuration",
                            }
                            # Store database type and other config info
                            health_data["metrics"]["database_type"] = (
                                database_config.get("type", "unknown")
                            )
                            health_data["metrics"]["database_config"] = {
                                k: v
                                for k, v in database_config.items()
                                if k
                                not in [
                                    "credentials",
                                    "password",
                                    "api_key",
                                ]  # Exclude sensitive info
                            }
                        else:
                            health_data["checks"]["database_config"] = {
                                "status": "unhealthy",
                                "message": f"Database '{database}' not found in configuration",
                            }
                            health_data["errors"].append(
                                f"Database '{database}' not configured"
                            )
                    else:
                        health_data["checks"]["database_config"] = {
                            "status": "unhealthy",
                            "message": "RAG not configured in project",
                        }
                        health_data["errors"].append("RAG configuration missing")
            else:
                health_data["checks"]["database_config"] = {
                    "status": "unhealthy",
                    "message": "Project configuration file not found",
                }
                health_data["errors"].append("llamafarm.yaml not found")

        except Exception as e:
            health_data["checks"]["database_config"] = {
                "status": "degraded",
                "message": f"Configuration check failed: {str(e)}",
            }
            health_data["errors"].append(f"Config error: {e}")

        # Check 2: Database connectivity and basic operations
        if database_config:
            try:
                from pathlib import Path
                from api import DatabaseSearchAPI

                cfg_path = Path(project_dir) / "llamafarm.yaml"
                # Initialize the search API to test database connectivity
                search_api = DatabaseSearchAPI(
                    config_path=str(cfg_path), database=database
                )

                # Try to get database stats/info
                # This will test if we can actually connect to and query the database
                db_stats = search_api.vector_store.get_collection_info()

                if db_stats:
                    health_data["checks"]["database_connectivity"] = {
                        "status": "healthy",
                        "message": f"Database '{database}' is accessible and queryable",
                    }

                    # Add database metrics if available
                    health_data["metrics"]["document_count"] = db_stats.get("count", 0)
                    health_data["metrics"]["collection_name"] = db_stats.get(
                        "name", database
                    )
                    if "error" not in db_stats:
                        health_data["metrics"]["collection_status"] = "active"

                else:
                    health_data["checks"]["database_connectivity"] = {
                        "status": "degraded",
                        "message": f"Database '{database}' exists but returned no stats",
                    }

            except Exception as e:
                health_data["checks"]["database_connectivity"] = {
                    "status": "degraded",
                    "message": f"Database connectivity issues: {str(e)}",
                }
                health_data["errors"].append(f"Database connection error: {e}")

        # Check 3: Database query performance test
        if (
            database_config
            and health_data["checks"].get("database_connectivity", {}).get("status")
            == "healthy"
        ):
            try:
                # Perform a simple test query to measure response time
                query_start = time.time()

                # Use a simple test query
                search_api.search(database=database, query="test", top_k=1)

                query_time_ms = (time.time() - query_start) * 1000
                health_data["metrics"]["query_latency_ms"] = round(query_time_ms, 2)

                if query_time_ms < 1000:  # Less than 1 second
                    health_data["checks"]["database_performance"] = {
                        "status": "healthy",
                        "message": f"Query performance good: {query_time_ms:.1f}ms",
                    }
                elif query_time_ms < 5000:  # Less than 5 seconds
                    health_data["checks"]["database_performance"] = {
                        "status": "degraded",
                        "message": f"Query performance slow: {query_time_ms:.1f}ms",
                    }
                else:
                    health_data["checks"]["database_performance"] = {
                        "status": "degraded",
                        "message": f"Query performance very slow: {query_time_ms:.1f}ms",
                    }

            except Exception as e:
                health_data["checks"]["database_performance"] = {
                    "status": "degraded",
                    "message": f"Performance test failed: {str(e)}",
                }
                health_data["errors"].append(f"Performance test error: {e}")

        # Determine overall status based on individual checks
        check_statuses = [check["status"] for check in health_data["checks"].values()]

        if not check_statuses:
            health_data["status"] = "unknown"
            health_data["message"] = "No checks performed"
        else:
            has_unhealthy = any(status == "unhealthy" for status in check_statuses)
            has_degraded = any(status == "degraded" for status in check_statuses)
            all_healthy = all(status == "healthy" for status in check_statuses)

            if has_unhealthy:
                health_data["status"] = "unhealthy"
                health_data["message"] = "Database has critical issues"
            elif has_degraded:
                health_data["status"] = "degraded"
                health_data["message"] = "Database may be degraded"
            elif all_healthy:
                health_data["status"] = "healthy"
                health_data["message"] = "Database is healthy"
            else:
                health_data["status"] = "unknown"
                health_data["message"] = "Database status unknown"

        # Add timing metrics
        duration_ms = int((time.time() - start_time) * 1000)
        health_data["metrics"]["check_duration_ms"] = duration_ms

        logger.info(
            "Database health check completed",
            extra={
                "task_id": self.request.id,
                "database": database,
                "status": health_data["status"],
                "duration_ms": duration_ms,
                "checks_passed": len([c for c in check_statuses if c == "healthy"]),
                "total_checks": len(check_statuses),
            },
        )

        return health_data

    except Exception as e:
        logger.error(
            "Database health check failed",
            extra={
                "task_id": self.request.id,
                "database": database,
                "error": str(e),
            },
            exc_info=True,
        )

        # Return failure status
        return {
            "status": "unhealthy",
            "database": database,
            "timestamp": int(start_time),
            "task_id": self.request.id,
            "worker_id": getattr(self.request, "hostname", "unknown"),
            "checks": {},
            "metrics": {"check_duration_ms": int((time.time() - start_time) * 1000)},
            "errors": [f"Health check exception: {str(e)}"],
            "message": f"Database health check failed: {str(e)}",
        }


@app.task(bind=True, base=HealthTask, name="rag.ping")
def rag_ping_task(self) -> dict[str, Any]:
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
        "worker_id": getattr(self.request, "hostname", "unknown"),
        "latency_ms": int((time.time() - start_time) * 1000),
    }
