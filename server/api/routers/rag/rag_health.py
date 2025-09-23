"""RAG Health endpoint for system health checks."""

from typing import Optional, Dict, Any
import time
from datetime import datetime, timezone
import structlog
from fastapi import HTTPException
from pydantic import BaseModel, Field

from config.datamodel import LlamaFarmConfig
from core.celery.tasks.rag_tasks import get_rag_health

logger = structlog.get_logger()


class ComponentHealth(BaseModel):
    """Health status of a single RAG component."""

    name: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: float = Field(..., alias="latency")
    message: Optional[str] = None


class RAGHealthResponse(BaseModel):
    """RAG system health response matching Go CLI RAGHealth struct."""

    status: str  # healthy, degraded, unhealthy
    database: str
    components: Dict[str, ComponentHealth]
    last_check: datetime
    issues: Optional[list[str]] = None

    class Config:
        populate_by_name = True


async def handle_rag_health(
    project_config: LlamaFarmConfig, project_dir: str, database: Optional[str] = None
) -> RAGHealthResponse:
    """Handle RAG health check request using Celery service."""
    start_time = time.time()

    # Determine which database to check
    database_name = database
    if not database_name and project_config.rag and project_config.rag.databases:
        # Use first database as default
        database_name = project_config.rag.databases[0].name
        logger.info(f"Using default database for health check: {database_name}")

    if not database_name:
        raise HTTPException(
            status_code=400, detail="No database specified and no default available"
        )

    # Validate database exists
    database_exists = False
    if project_config.rag:
        for db in project_config.rag.databases:
            if db.name == database_name:
                database_exists = True
                break

    if not database_exists:
        raise HTTPException(
            status_code=404, detail=f"Database '{database_name}' not found"
        )

    try:
        # Use Celery service to perform health check
        logger.info(
            f"Performing RAG health check via Celery service for database: '{database_name}'"
        )

        health_data = get_rag_health(project_dir=project_dir, database=database_name)

        # Convert the health data to match the Go CLI structure
        components = {}
        issues = []
        overall_status = "healthy"

        # Process individual checks from the RAG health task
        if "checks" in health_data:
            for check_name, check_data in health_data["checks"].items():
                # Map latency from metrics if available
                latency_ms = 0.0
                if (
                    "metrics" in health_data
                    and f"{check_name}_latency_ms" in health_data["metrics"]
                ):
                    latency_ms = health_data["metrics"][f"{check_name}_latency_ms"]
                elif (
                    "metrics" in health_data
                    and "check_duration_ms" in health_data["metrics"]
                ):
                    # Use overall check duration as fallback
                    latency_ms = health_data["metrics"]["check_duration_ms"] / len(
                        health_data["checks"]
                    )

                components[check_name] = ComponentHealth(
                    name=check_name,
                    status=check_data.get("status", "unknown"),
                    latency=latency_ms,
                    message=check_data.get("message"),
                )

                # Collect degraded/unhealthy statuses for overall status
                if (
                    check_data.get("status") == "degraded"
                    and overall_status == "healthy"
                ):
                    overall_status = "degraded"
                elif check_data.get("status") == "unhealthy":
                    overall_status = "unhealthy"

        # Collect errors/issues
        if "errors" in health_data and health_data["errors"]:
            issues.extend(health_data["errors"])

        # Use the status from the health check if available
        if "status" in health_data:
            overall_status = health_data["status"]

        processing_time = (time.time() - start_time) * 1000
        logger.info(
            f"Health check completed in {processing_time:.2f}ms with status: {overall_status}"
        )

        return RAGHealthResponse(
            status=overall_status,
            database=database_name,
            components=components,
            last_check=datetime.now(timezone.utc),
            issues=issues if issues else None,
        )

    except Exception as e:
        logger.error(f"Error during RAG health check: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
