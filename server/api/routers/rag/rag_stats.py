"""RAG Stats endpoint for database statistics."""

from datetime import datetime
from typing import Any

import structlog
from config.datamodel import LlamaFarmConfig
from fastapi import HTTPException
from pydantic import BaseModel, Field

from core.celery.rag_client import get_rag_stats

logger = structlog.get_logger()


class RAGStatsResponse(BaseModel):
    """RAG database statistics response matching Go CLI RAGStats struct."""

    database: str
    vector_count: int
    document_count: int
    chunk_count: int
    collection_size_bytes: int = Field(alias="collection_size_bytes")
    index_size_bytes: int = Field(alias="index_size_bytes")
    embedding_dimension: int = Field(alias="embedding_dimension")
    distance_metric: str = Field(alias="distance_metric")
    last_updated: datetime = Field(alias="last_updated")
    metadata: dict[str, Any] | None = None

    class Config:
        populate_by_name = True


async def handle_rag_stats(
    project_config: LlamaFarmConfig, project_dir: str, database: str | None = None
) -> RAGStatsResponse:
    """Handle RAG stats request using Celery service."""

    # Determine which database to get stats for
    database_name = database
    if not database_name and project_config.rag and project_config.rag.databases:
        # Use first database as default
        database_name = project_config.rag.databases[0].name
        logger.info(f"Using default database for stats: {database_name}")

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
        # Use Celery service to get stats
        logger.info(
            f"Fetching RAG stats via Celery service for database: '{database_name}'"
        )

        stats_data = get_rag_stats(project_dir=project_dir, database=database_name)

        # Parse last_updated if it's a string
        last_updated = stats_data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
        elif not isinstance(last_updated, datetime):
            from datetime import UTC

            last_updated = datetime.now(UTC)

        return RAGStatsResponse(
            database=stats_data.get("database", database_name),
            vector_count=stats_data.get("vector_count", 0),
            document_count=stats_data.get("document_count", 0),
            chunk_count=stats_data.get("chunk_count", 0),
            collection_size_bytes=stats_data.get("collection_size_bytes", 0),
            index_size_bytes=stats_data.get("index_size_bytes", 0),
            embedding_dimension=stats_data.get("embedding_dimension", 0),
            distance_metric=stats_data.get("distance_metric", "unknown"),
            last_updated=last_updated,
            metadata=stats_data.get("metadata"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve RAG stats: {str(e)}"
        ) from e
