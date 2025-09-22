"""
Server-side RAG Task Definitions

This module defines the Celery tasks that the server can call to interact
with the RAG service. These are task signatures - the actual implementations
are in the RAG container.
"""

from typing import Any, Dict, List, Optional, Tuple

from celery import signature

from core.celery import app


def search_with_rag_database(
    project_dir: str,
    database: str,
    query: str,
    top_k: int = 5,
    retrieval_strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search directly against a RAG database via Celery task.

    Args:
        project_dir: Directory containing llamafarm.yaml config
        database: Database name to search
        query: Search query string
        top_k: Maximum number of results to return
        retrieval_strategy: Optional retrieval strategy name

    Returns:
        List of search results as dictionaries
    """
    task = signature(
        "rag.search_with_database",
        args=[project_dir, database, query, top_k, retrieval_strategy],
        app=app,
    )
    result = task.apply_async()
    return result.get(timeout=30)  # 30 second timeout


def ingest_file_with_rag(
    project_dir: str,
    data_processing_strategy_name: str,
    database_name: str,
    source_path: str,
    filename: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Ingest a single file using the RAG system via Celery task.

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
    return result.get(timeout=120)  # 2 minute timeout for file processing


def handle_rag_query(
    project_dir: str,
    database: str,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    retrieval_strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Handle complex RAG query operations via Celery task.

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
    return result.get(timeout=60)  # 1 minute timeout


def batch_search(
    project_dir: str,
    database: str,
    queries: List[str],
    top_k: int = 5,
    retrieval_strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Handle batch search operations via Celery task.

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
    return result.get(timeout=len(queries) * 10)  # 10 seconds per query timeout
