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


@app.task(bind=True, base=StatsTask, name="rag.list_database_documents")
def rag_list_database_documents_task(
    self,
    project_dir: str,
    database: str,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """
    List documents in a RAG database with metadata.

    This task retrieves chunks from the vector store and aggregates them
    by source file to provide document-level information.

    Args:
        project_dir: Directory containing llamafarm.yaml config
        database: Database name to list documents from
        limit: Maximum number of documents to return
        offset: Number of documents to skip (for pagination)

    Returns:
        Dict containing 'documents' list and 'total_count'
    """
    start_time = time.time()

    logger.info(
        "Starting document list retrieval",
        extra={
            "task_id": self.request.id,
            "project_dir": project_dir,
            "database": database,
            "limit": limit,
            "offset": offset,
        },
    )

    result = {
        "documents": [],
        "total_count": 0,
        "database": database,
    }

    try:
        # Load project configuration
        config = load_config(config_path=project_dir, validate=True)

        if not config or not config.rag or not config.rag.databases:
            logger.warning("RAG not configured in project")
            return result

        # Find the specific database configuration
        database_config = None
        for db_cfg in config.rag.databases:
            if db_cfg.name == database:
                database_config = db_cfg
                break

        if not database_config:
            logger.warning(f"Database '{database}' not found in configuration")
            return result

        # Initialize search API to access vector store
        search_api = DatabaseSearchAPI(project_dir=project_dir, database=database)

        # Fetch all chunks by paginating through the entire collection.
        # We need all chunks because multiple chunks can belong to the same
        # document, and we must aggregate them before applying document-level
        # pagination.
        chunks: list[Any] = []
        chunk_page_size = 10000
        chunk_offset = 0

        while True:
            page_chunks, total_chunks = search_api.vector_store.list_documents(
                limit=chunk_page_size,
                offset=chunk_offset,
                include_content=False,
            )
            chunks.extend(page_chunks)

            # Stop if we've fetched all chunks or got an empty page
            if len(page_chunks) < chunk_page_size or len(chunks) >= total_chunks:
                break
            chunk_offset += chunk_page_size

        # Aggregate chunks by source file
        documents_map: dict[str, dict[str, Any]] = {}

        for chunk in chunks:
            source = chunk.source or chunk.metadata.get("source") or "unknown"

            if source not in documents_map:
                # Get date_ingested from various metadata fields, with fallback
                date_ingested = (
                    chunk.metadata.get("processing_timestamp")
                    or chunk.metadata.get("processing_date")
                    or chunk.metadata.get("ingested_at")
                    or datetime.now(UTC).isoformat()
                )
                # Get file size - this is stored per chunk but represents the whole file
                file_size = (
                    chunk.metadata.get("size") or chunk.metadata.get("file_size") or 0
                )
                if isinstance(file_size, str):
                    try:
                        file_size = int(file_size)
                    except ValueError:
                        file_size = 0
                documents_map[source] = {
                    "id": chunk.metadata.get("document_hash")
                    or chunk.metadata.get("file_hash")
                    or chunk.id,
                    "filename": _extract_filename(source),
                    "source": source,
                    "chunk_count": 0,
                    "size_bytes": int(file_size)
                    if isinstance(file_size, (int, float))
                    else 0,
                    "parser_used": chunk.metadata.get("parser_type")
                    or chunk.metadata.get("parser")
                    or "unknown",
                    "date_ingested": date_ingested,
                    "metadata": {},
                }

            doc_info = documents_map[source]
            doc_info["chunk_count"] += 1

            # Collect interesting metadata from any chunk
            for key in ["category", "language", "source_type", "title"]:
                if key in chunk.metadata and key not in doc_info["metadata"]:
                    doc_info["metadata"][key] = chunk.metadata[key]

        # Convert to list and apply pagination
        documents_list = list(documents_map.values())
        result["total_count"] = len(documents_list)

        # Sort by filename for consistent ordering
        documents_list.sort(key=lambda d: d["filename"].lower())

        # Apply pagination at document level
        paginated_docs = documents_list[offset : offset + limit]
        result["documents"] = paginated_docs

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "Document list retrieval completed",
            extra={
                "task_id": self.request.id,
                "database": database,
                "documents_returned": len(paginated_docs),
                "total_documents": result["total_count"],
                "duration_ms": duration_ms,
            },
        )

        return result

    except Exception as e:
        logger.error(
            "Document list retrieval failed",
            extra={
                "task_id": self.request.id,
                "database": database,
                "error": str(e),
            },
            exc_info=True,
        )
        return result


def _extract_filename(source: str) -> str:
    """Extract filename from a source path."""
    if not source:
        return "unknown"
    # Handle both forward and back slashes
    parts = source.replace("\\", "/").split("/")
    return parts[-1] if parts else source


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
