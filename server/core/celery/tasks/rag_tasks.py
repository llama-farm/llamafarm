"""
Server-side RAG Task Definitions

This module defines the Celery tasks that the server can call to interact
with the RAG service. These are task signatures - the actual implementations
are in the RAG container.
"""

import time
from typing import Any

from celery import signature

from core.celery import app


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
    Search directly against a RAG database via Celery task.
    This version is safe to call from both regular code and within Celery tasks.

    Args:
        project_dir: Directory containing llamafarm.yaml config
        database: Database name to search
        query: Search query string
        top_k: Maximum number of results to return
        retrieval_strategy: Optional retrieval strategy name
        score_threshold: Minimum similarity score filter
        metadata_filters: Filter results by metadata fields
        distance_metric: Distance metric to use for similarity calculation
        hybrid_alpha: Hybrid alpha for hybrid retrieval
        rerank_model: Rerank model to use for reranking
        query_expansion: Enable query expansion
        max_tokens: Maximum tokens to generate for each result

    Returns:
        List of search results as dictionaries
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
    result = task.apply_async()

    # Always use polling approach to avoid any result.get() issues
    timeout = 30
    poll_interval = 0.5
    waited = 0.0

    while waited < timeout:
        if result.status not in ("PENDING", "STARTED"):
            break
        time.sleep(poll_interval)
        waited += poll_interval

    if result.status == "SUCCESS":
        return result.result
    elif result.status == "FAILURE":
        # Get the exception info and raise it without using result.get()
        if hasattr(result, "traceback") and result.traceback:
            raise Exception(f"Task failed: {result.traceback}")
        else:
            raise Exception(f"Task failed with status: {result.status}")
    else:
        return []  # Return empty list on timeout or other status


async def ingest_file_with_rag(
    project_dir: str,
    data_processing_strategy_name: str,
    database_name: str,
    source_path: str,
    filename: str | None = None,
    dataset_name: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Ingest a single file using the RAG system via Celery task.
    This version is safe to call from both regular code and within Celery tasks.

    Args:
        project_dir: The directory of the project
        data_processing_strategy_name: Name of the data processing strategy to use
        database_name: Name of the database to use
        source_path: Path to the file to ingest
        filename: Optional original filename (for display purposes)
        dataset_name: Optional dataset name for logging

    Returns:
        Tuple of (success: bool, details: dict) with processing information
    """
    task = signature(
        "rag.ingest_file",
        args=[
            project_dir,
            data_processing_strategy_name,
            database_name,
            source_path,
            filename,
            dataset_name,
        ],
        app=app,
    )
    result = task.apply_async()

    poll_interval = 2  # seconds
    max_wait = 120  # seconds
    waited = 0

    # Always use polling approach to avoid any result.get() issues
    while True:
        status = result.status
        if status not in ("PENDING", "STARTED"):
            break
        if waited >= max_wait:
            break
        time.sleep(poll_interval)
        waited += poll_interval

    # Get the result without using result.get() to avoid the error
    if result.status == "SUCCESS":
        return result.result
    elif result.status == "FAILURE":
        # Get the exception info and raise it without using result.get()
        if hasattr(result, "traceback") and result.traceback:
            raise Exception(f"Task failed: {result.traceback}")
        else:
            raise Exception(f"Task failed with status: {result.status}")
    else:
        # Timeout or other status
        return False, {
            "error": f"Task timed out or failed with status: {result.status}"
        }


def handle_rag_query(
    project_dir: str,
    database: str,
    query: str,
    context: dict[str, Any] | None = None,
    top_k: int = 5,
    retrieval_strategy: str | None = None,
) -> dict[str, Any]:
    """
    Handle complex RAG query operations via Celery task.
    This version is safe to call from both regular code and within Celery tasks.

    Args:
        project_dir: Directory containing llamafarm.yaml config
        database: Database name to query
        query: Query string
        context: Optional context for the query
        top_k: Maximum number of results to return
        retrieval_strategy: Optional retrieval strategy name

    Returns:
        Dictionary containing query results and metadata
    """
    task = signature(
        "rag.handle_rag_query",
        args=[project_dir, database, query, context, top_k, retrieval_strategy],
        app=app,
    )
    result = task.apply_async()

    # Always use polling approach to avoid any result.get() issues
    timeout = 60
    poll_interval = 1
    waited = 0

    while waited < timeout:
        if result.status not in ("PENDING", "STARTED"):
            break
        time.sleep(poll_interval)
        waited += poll_interval

    if result.status == "SUCCESS":
        return result.result
    elif result.status == "FAILURE":
        # Get the exception info and raise it without using result.get()
        if hasattr(result, "traceback") and result.traceback:
            raise Exception(f"Task failed: {result.traceback}")
        else:
            raise Exception(f"Task failed with status: {result.status}")
    else:
        # Return empty result on timeout or other status
        return {
            "query": query,
            "database": database,
            "results": [],
            "total_results": 0,
            "retrieval_strategy": retrieval_strategy,
            "context": context,
            "error": f"Task timed out or failed: {result.status}",
        }


def get_rag_health(
    project_dir: str,
    database: str,
) -> dict[str, Any]:
    """
    Get RAG system health status via Celery task.
    This version is safe to call from both regular code and within Celery tasks.

    Args:
        project_dir: Directory containing llamafarm.yaml config
        database: Database name to check health for

    Returns:
        Dictionary containing health status and diagnostic information
    """
    task = signature(
        "rag.health_check_database",
        args=[project_dir, database],
        app=app,
    )
    result = task.apply_async()

    # Always use polling approach to avoid any result.get() issues
    timeout = 30
    poll_interval = 0.5
    waited = 0.0

    while waited < timeout:
        if result.status not in ("PENDING", "STARTED"):
            break
        time.sleep(poll_interval)
        waited += poll_interval

    if result.status == "SUCCESS":
        return result.result
    elif result.status == "FAILURE":
        # Get the exception info and raise it without using result.get()
        if hasattr(result, "traceback") and result.traceback:
            raise Exception(f"Health check task failed: {result.traceback}")
        else:
            raise Exception(f"Health check task failed with status: {result.status}")
    else:
        # Return degraded status on timeout or other status
        return {
            "status": "degraded",
            "message": f"Health check timed out or failed: {result.status}",
            "database": database,
            "checks": {},
            "metrics": {},
            "errors": [f"Health check task status: {result.status}"],
        }


def batch_search(
    project_dir: str,
    database: str,
    queries: list[str],
    top_k: int = 5,
    retrieval_strategy: str | None = None,
) -> list[dict[str, Any]]:
    """
    Handle batch search operations via Celery task.
    This version is safe to call from both regular code and within Celery tasks.

    Args:
        project_dir: Directory containing llamafarm.yaml config
        database: Database name to query
        queries: List of query strings
        top_k: Maximum number of results per query
        retrieval_strategy: Optional retrieval strategy name

    Returns:
        List of search results for each query
    """
    task = signature(
        "rag.batch_search",
        args=[project_dir, database, queries, top_k, retrieval_strategy],
        app=app,
    )
    result = task.apply_async()

    timeout = len(queries) * 10  # 10 seconds per query timeout

    # Always use polling approach to avoid any result.get() issues
    poll_interval = 1
    waited = 0

    while waited < timeout:
        if result.status not in ("PENDING", "STARTED"):
            break
        time.sleep(poll_interval)
        waited += poll_interval

    if result.status == "SUCCESS":
        return result.result
    elif result.status == "FAILURE":
        # Get the exception info and raise it without using result.get()
        if hasattr(result, "traceback") and result.traceback:
            raise Exception(f"Task failed: {result.traceback}")
        else:
            raise Exception(f"Task failed with status: {result.status}")
    else:
        return []  # Return empty list on timeout or other status
