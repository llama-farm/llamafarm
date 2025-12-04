"""
RAG Stats Tasks

Celery tasks for retrieving RAG database statistics.
"""

import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from celery import Task

from api import DatabaseSearchAPI
from celery_app import app
from core.logging import RAGStructLogger

# Add the repo root to the path to find the config module
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from config import load_config  # noqa: E402

logger = RAGStructLogger("rag.tasks.stats")


class StatsTask(Task):
    """Base task class for stats operations."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log task failure details."""
        logger.error(
            "RAG stats task failed",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "error": str(exc),
                "task_args": args,
                "task_kwargs": kwargs,
            },
        )


@app.task(bind=True, base=StatsTask, name="rag.get_database_stats")
def rag_get_database_stats_task(
    self, project_dir: str, database: str
) -> dict[str, Any]:
    """
    Get comprehensive statistics for a specific RAG database.

    This task retrieves vector counts, document counts, storage info,
    and configuration details for the specified database.

    Args:
        project_dir: Directory containing llamafarm.yaml config
        database: Database name to get stats for

    Returns:
        Dict containing database statistics matching the CLI's RAGStats struct
    """
    start_time = time.time()

    logger.info(
        "Starting database stats retrieval",
        extra={
            "task_id": self.request.id,
            "project_dir": project_dir,
            "database": database,
        },
    )

    # Initialize response with defaults
    stats_data = {
        "database": database,
        "vector_count": 0,
        "document_count": 0,
        "chunk_count": 0,
        "collection_size_bytes": 0,
        "index_size_bytes": 0,
        "embedding_dimension": 0,
        "distance_metric": "cosine",
        "last_updated": datetime.now(UTC).isoformat(),
        "metadata": {},
    }

    try:
        # Load project configuration to get database settings
        config = load_config(config_path=project_dir, validate=True)

        if not config or not config.rag or not config.rag.databases:
            return {
                **stats_data,
                "metadata": {"error": "RAG not configured in project"},
            }

        # Find the specific database configuration
        database_config = None
        for db_cfg in config.rag.databases:
            if db_cfg.name == database:
                database_config = db_cfg
                break

        if not database_config:
            return {
                **stats_data,
                "metadata": {
                    "error": f"Database '{database}' not found in configuration"
                },
            }

        # Extract configuration details
        # Distance metric from database config
        if database_config.config and isinstance(database_config.config, dict):
            stats_data["distance_metric"] = database_config.config.get(
                "distance_function", "cosine"
            )

        # Embedding dimension from default embedding strategy
        stats_data["embedding_dimension"] = _get_embedding_dimension(database_config)

        # Initialize search API to access vector store
        try:
            search_api = DatabaseSearchAPI(project_dir=project_dir, database=database)

            # Get collection info from vector store
            collection_info = search_api.vector_store.get_collection_info()

            if collection_info and "error" not in collection_info:
                # Vector count is the total number of entries in the collection
                # This includes all chunks
                stats_data["vector_count"] = collection_info.get("count", 0)
                stats_data["chunk_count"] = collection_info.get("count", 0)

                # Try to get unique document count by querying metadata
                # For now, use vector count as an approximation
                # In the future, we could query unique source_file metadata
                stats_data["document_count"] = _estimate_document_count(
                    search_api, stats_data["chunk_count"]
                )

                # Add collection metadata
                stats_data["metadata"]["collection_name"] = collection_info.get(
                    "name", database
                )
                stats_data["metadata"]["persist_directory"] = collection_info.get(
                    "persist_directory", ""
                )

                # Try to get storage size from persist directory
                persist_dir = collection_info.get("persist_directory")
                if persist_dir:
                    collection_size, index_size = _get_storage_sizes(persist_dir)
                    stats_data["collection_size_bytes"] = collection_size
                    stats_data["index_size_bytes"] = index_size

            else:
                stats_data["metadata"]["collection_error"] = collection_info.get(
                    "error", "Unknown error"
                )

        except Exception as e:
            logger.warning(
                f"Could not access vector store for stats: {e}",
                extra={"database": database},
            )
            stats_data["metadata"]["access_error"] = str(e)

        # Record timing
        duration_ms = int((time.time() - start_time) * 1000)
        stats_data["metadata"]["retrieval_duration_ms"] = duration_ms

        logger.info(
            "Database stats retrieval completed",
            extra={
                "task_id": self.request.id,
                "database": database,
                "vector_count": stats_data["vector_count"],
                "duration_ms": duration_ms,
            },
        )

        return stats_data

    except Exception as e:
        logger.error(
            "Database stats retrieval failed",
            extra={
                "task_id": self.request.id,
                "database": database,
                "error": str(e),
            },
            exc_info=True,
        )

        return {
            **stats_data,
            "metadata": {
                "error": f"Stats retrieval failed: {str(e)}",
                "retrieval_duration_ms": int((time.time() - start_time) * 1000),
            },
        }


def _get_embedding_dimension(database_config) -> int:
    """
    Extract embedding dimension from database config.

    Uses the default embedding strategy if specified, otherwise the first strategy.

    Args:
        database_config: Database configuration object

    Returns:
        Embedding dimension (defaults to 768 if not found)
    """
    strategies = database_config.embedding_strategies
    if not strategies:
        return 768

    # Find target strategy: default if specified, otherwise first
    default_name = database_config.default_embedding_strategy
    target = (
        next(
            (s for s in strategies if s.name == default_name),
            strategies[0],
        )
        if default_name
        else strategies[0]
    )

    # Extract dimension from config dict
    config = target.config
    if isinstance(config, dict):
        return config.get("dimension", 768)

    return 768


def _estimate_document_count(search_api: DatabaseSearchAPI, chunk_count: int) -> int:
    """
    Estimate the number of unique documents in the database.

    This tries to count unique source files. Falls back to chunk_count / 10
    as a rough estimate if metadata query fails.

    Args:
        search_api: Initialized DatabaseSearchAPI
        chunk_count: Total number of chunks in the database

    Returns:
        Estimated document count
    """
    try:
        # Try to get unique document IDs from the collection
        # ChromaDB allows getting all IDs but not metadata directly
        if hasattr(search_api.vector_store, "collection"):
            collection = search_api.vector_store.collection
            # Get a sample of metadata to count unique source files
            result = collection.get(
                limit=min(chunk_count, 10000), include=["metadatas"]
            )
            if result and result.get("metadatas"):
                unique_sources = set()
                for meta in result["metadatas"]:
                    if meta:
                        # Try different metadata keys that might identify source
                        source = (
                            meta.get("source_file")
                            or meta.get("source")
                            or meta.get("filename")
                        )
                        if source:
                            unique_sources.add(source)
                if unique_sources:
                    return len(unique_sources)
    except Exception:
        pass

    # Fallback: estimate based on typical chunk ratio
    # Assume ~10 chunks per document on average
    if chunk_count > 0:
        return max(1, chunk_count // 10)
    return 0


def _get_storage_sizes(persist_dir: str) -> tuple[int, int]:
    """
    Calculate storage sizes for the database.

    Args:
        persist_dir: Path to the persist directory

    Returns:
        Tuple of (collection_size_bytes, index_size_bytes)
    """
    collection_size = 0
    index_size = 0

    try:
        persist_path = Path(persist_dir)
        if persist_path.exists():
            for file_path in persist_path.rglob("*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    # Index files typically have specific extensions
                    if file_path.suffix in (".idx", ".index", ".bin"):
                        index_size += file_size
                    else:
                        collection_size += file_size
    except Exception:
        pass

    return collection_size, index_size
