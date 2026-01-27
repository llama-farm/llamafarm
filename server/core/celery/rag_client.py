"""
Client utilities for interacting with RAG Celery tasks.

These helpers centralize the logic for building task signatures, dispatching
them, and polling for completion so that both FastAPI handlers and Celery
tasks can reuse the same behavior.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from celery import signature

from core.celery import app


def build_ingest_signature(
    project_dir: str,
    data_processing_strategy_name: str,
    database_name: str,
    source_path: str,
    filename: str | None = None,
    dataset_name: str | None = None,
    parser_overrides: dict[str, Any] | None = None,
):
    """
    Build a Celery signature for the rag.ingest_file task.

    This helper ensures all signatures are constructed consistently.
    """

    return signature(
        "rag.ingest_file",
        args=[
            project_dir,
            data_processing_strategy_name,
            database_name,
            source_path,
            filename,
            dataset_name,
            parser_overrides or {},
        ],
        app=app,
    )


def build_delete_signature(
    project_dir: str,
    database_name: str,
    file_hash: str,
):
    """
    Build a Celery signature for the rag.delete_file task.

    This helper ensures all signatures are constructed consistently.
    """

    return signature(
        "rag.delete_file",
        args=[project_dir, database_name, file_hash],
        app=app,
    )


async def delete_file_from_rag(
    project_dir: str,
    database_name: str,
    file_hash: str,
    timeout: int = 60,
    poll_interval: float = 0.5,
) -> dict[str, Any]:
    """
    Delete all chunks for a file from the RAG database.

    Uses asyncio.sleep() for non-blocking polling, suitable for async contexts.

    Args:
        project_dir: Path to the project directory
        database_name: Name of the RAG database
        file_hash: SHA-256 hash of the file content
        timeout: Maximum time to wait for task completion
        poll_interval: Time between status checks

    Returns:
        Dictionary with deletion results including deleted_count
    """
    task = build_delete_signature(
        project_dir=project_dir,
        database_name=database_name,
        file_hash=file_hash,
    )
    result = task.apply_async()

    waited = 0.0
    while waited < timeout:
        try:
            status = result.status
            if status not in ("PENDING", "STARTED"):
                break
        except Exception:
            await asyncio.sleep(poll_interval)
            waited += poll_interval
            continue

        await asyncio.sleep(poll_interval)
        waited += poll_interval

    try:
        final_status = result.status
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Failed to get task status: {exc}",
            "file_hash": file_hash,
            "deleted_count": 0,
        }

    if final_status == "SUCCESS":
        try:
            return result.result
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Failed to get task result: {exc}",
                "file_hash": file_hash,
                "deleted_count": 0,
            }

    if final_status == "FAILURE":
        error_msg = "Task failed"
        if hasattr(result, "traceback") and result.traceback:
            error_msg = f"Task failed: {result.traceback}"
        return {
            "status": "error",
            "error": error_msg,
            "file_hash": file_hash,
            "deleted_count": 0,
        }

    return {
        "status": "error",
        "error": f"Task timed out or failed with status: {final_status}",
        "file_hash": file_hash,
        "deleted_count": 0,
    }


async def ingest_file_with_rag(
    project_dir: str,
    data_processing_strategy_name: str,
    database_name: str,
    source_path: str,
    filename: str | None = None,
    dataset_name: str | None = None,
    parser_overrides: dict[str, Any] | None = None,
    timeout: int = 300,
    poll_interval: float = 2.0,
) -> tuple[bool, dict[str, Any]]:
    """
    Dispatch rag.ingest_file and poll for completion from async contexts.
    """

    sig = build_ingest_signature(
        project_dir=project_dir,
        data_processing_strategy_name=data_processing_strategy_name,
        database_name=database_name,
        source_path=source_path,
        filename=filename,
        dataset_name=dataset_name,
        parser_overrides=parser_overrides,
    )
    result = sig.apply_async()

    waited = 0.0
    while waited < timeout:
        try:
            status = result.status
            if status not in ("PENDING", "STARTED"):
                break
        except Exception:
            await asyncio.sleep(poll_interval)
            waited += poll_interval
            continue

        await asyncio.sleep(poll_interval)
        waited += poll_interval

    try:
        final_status = result.status
    except Exception as exc:  # pragma: no cover - defensive
        return False, {"error": f"Failed to get task status: {exc}"}

    if final_status == "SUCCESS":
        try:
            return result.result
        except Exception as exc:  # pragma: no cover - defensive
            return False, {"error": f"Failed to get task result: {exc}"}

    if final_status == "FAILURE":
        # Propagate the remote traceback if available
        if hasattr(result, "traceback") and result.traceback:
            raise Exception(f"Task failed: {result.traceback}")  # noqa: BLE001
        raise Exception("Task failed")  # noqa: BLE001

    return (
        False,
        {"error": f"Task timed out or failed with status: {final_status}"},
    )


def search_with_rag_database(
    project_dir: str,
    database: str,
    query: str,
    top_k: int = 5,
    retrieval_strategy: str | None = None,
    score_threshold: float | None = None,
    metadata_filters: dict[str, Any] | None = None,
    distance_metric: str | None = None,
    hybrid_alpha: float | None = None,
    rerank_model: str | None = None,
    query_expansion: bool | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run a search query via rag.search_with_database.
    """

    task = signature(
        "rag.search_with_database",
        args=[
            project_dir,
            database,
            query,
            top_k,
            retrieval_strategy,
            score_threshold,
            metadata_filters,
            distance_metric,
            hybrid_alpha,
            rerank_model,
            query_expansion,
            max_tokens,
        ],
        app=app,
    )
    return _run_sync_task_with_polling(task, timeout=30, poll_interval=0.5) or []


def handle_rag_query(
    project_dir: str,
    database: str,
    query: str,
    context: dict[str, Any] | None = None,
    top_k: int = 5,
    retrieval_strategy: str | None = None,
) -> dict[str, Any]:
    """
    Execute rag.handle_rag_query and return the response.
    """

    task = signature(
        "rag.handle_rag_query",
        args=[project_dir, database, query, context, top_k, retrieval_strategy],
        app=app,
    )
    return _run_sync_task_with_polling(task, timeout=60, poll_interval=1) or {
        "query": query,
        "database": database,
        "results": [],
        "total_results": 0,
        "retrieval_strategy": retrieval_strategy,
        "context": context,
        "error": "Task timed out or failed",
    }


def get_rag_health(project_dir: str, database: str) -> dict[str, Any]:
    """
    Fetch RAG health diagnostics via rag.health_check_database.
    """

    task = signature(
        "rag.health_check_database",
        args=[project_dir, database],
        app=app,
    )
    return _run_sync_task_with_polling(task, timeout=30, poll_interval=0.5) or {
        "status": "degraded",
        "message": "Health check task timed out or failed",
        "database": database,
        "checks": {},
        "metrics": {},
        "errors": ["Unable to contact RAG worker"],
    }


def get_rag_stats(project_dir: str, database: str) -> dict[str, Any]:
    """
    Fetch RAG database statistics via rag.get_database_stats.
    """
    from datetime import UTC, datetime

    task = signature(
        "rag.get_database_stats",
        args=[project_dir, database],
        app=app,
    )
    return _run_sync_task_with_polling(task, timeout=30, poll_interval=0.5) or {
        "database": database,
        "vector_count": 0,
        "document_count": 0,
        "chunk_count": 0,
        "collection_size_bytes": 0,
        "index_size_bytes": 0,
        "embedding_dimension": 0,
        "distance_metric": "unknown",
        "last_updated": datetime.now(UTC).isoformat(),
        "metadata": {"error": "Stats retrieval task timed out or failed"},
    }


def list_rag_documents(
    project_dir: str,
    database: str,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """
    List documents in a RAG database via rag.list_database_documents.

    Args:
        project_dir: Path to the project directory
        database: Name of the database to list documents from
        limit: Maximum number of documents to return
        offset: Number of documents to skip for pagination

    Returns:
        Dict with 'documents' list, 'total_count', and 'database' name
    """
    task = signature(
        "rag.list_database_documents",
        args=[project_dir, database, limit, offset],
        app=app,
    )
    return _run_sync_task_with_polling(task, timeout=60, poll_interval=0.5) or {
        "documents": [],
        "total_count": 0,
        "database": database,
    }


def batch_search(
    project_dir: str,
    database: str,
    queries: list[str],
    top_k: int = 5,
    retrieval_strategy: str | None = None,
) -> list[dict[str, Any]]:
    """
    Execute rag.batch_search for a list of queries.
    """

    task = signature(
        "rag.batch_search",
        args=[project_dir, database, queries, top_k, retrieval_strategy],
        app=app,
    )
    timeout = max(30, len(queries) * 10)
    return _run_sync_task_with_polling(task, timeout=timeout, poll_interval=1) or []


def preview_document(
    project_dir: str,
    file_path: str,
    database: str | None = None,
    data_processing_strategy_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    chunk_strategy: str | None = None,
    original_filename: str | None = None,
) -> dict[str, Any]:
    """
    Dispatch rag.preview_document task and wait for result.

    Args:
        project_dir: Path to the project directory
        file_path: Path to the file to preview
        database: Name of the database (for config lookup)
        data_processing_strategy_name: Data processing strategy to use for preview.
            If not provided, falls back to the first available strategy.
        chunk_size: Override chunk size
        chunk_overlap: Override chunk overlap
        chunk_strategy: Override chunk strategy
        original_filename: Original filename with extension (for parser detection)

    Returns:
        Dict with preview results including chunks and statistics
    """
    task = signature(
        "rag.preview_document",
        args=[project_dir, file_path],
        kwargs={
            "database": database,
            "data_processing_strategy_name": data_processing_strategy_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_strategy": chunk_strategy,
            "original_filename": original_filename,
        },
        app=app,
    )
    return _run_sync_task_with_polling(task, timeout=60.0, poll_interval=0.5) or {
        "original_text": "",
        "chunks": [],
        "file_info": {"filename": "unknown", "size": 0},
        "parser_used": "unknown",
        "chunk_strategy": "unknown",
        "chunk_size": 0,
        "chunk_overlap": 0,
        "total_chunks": 0,
        "avg_chunk_size": 0.0,
        "total_size_with_overlaps": 0,
        "warnings": ["Preview task timed out or failed"],
    }


def _run_sync_task_with_polling(task_signature, timeout: float, poll_interval: float):
    """
    Helper used by synchronous contexts to poll a Celery AsyncResult safely.
    """

    result = task_signature.apply_async()
    waited = 0.0

    while waited < timeout:
        try:
            status = result.status
            if status not in ("PENDING", "STARTED"):
                break
        except Exception:
            time.sleep(poll_interval)
            waited += poll_interval
            continue
        time.sleep(poll_interval)
        waited += poll_interval

    try:
        final_status = result.status
    except Exception as exc:  # pragma: no cover - defensive
        raise Exception(f"Failed to get task status: {exc}") from exc  # noqa: BLE001

    if final_status == "SUCCESS":
        try:
            return result.result
        except Exception as exc:  # pragma: no cover - defensive
            raise Exception(f"Failed to get task result: {exc}") from exc  # noqa: BLE001

    if final_status == "FAILURE":
        if hasattr(result, "traceback") and result.traceback:
            raise Exception(f"Task failed: {result.traceback}")  # noqa: BLE001
        raise Exception("Task failed")  # noqa: BLE001

    return None
