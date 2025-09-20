"""
RAG Service

This service handles RAG operations by delegating to the RAG container
via Celery tasks instead of subprocess calls.
"""

from typing import Any, Dict, List, Optional, Tuple
from core.logging import FastAPIStructLogger
from core.celery.tasks.rag_tasks import (
    search_with_rag_database as search_task,
    ingest_file_with_rag as ingest_task,
    handle_rag_query as query_task,
    batch_search as batch_search_task,
)
from config.datamodel import LlamaFarmConfig

logger = FastAPIStructLogger()


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
    logger.info(
        "Starting RAG database search via Celery",
        project_dir=project_dir,
        database=database,
        query=query[:100] + "..." if len(query) > 100 else query,
        top_k=top_k,
        retrieval_strategy=retrieval_strategy,
    )

    try:
        results = search_task(
            project_dir=project_dir,
            database=database,
            query=query,
            top_k=top_k,
            retrieval_strategy=retrieval_strategy,
        )

        logger.info(
            "RAG database search completed",
            results_count=len(results),
        )

        return results

    except Exception as e:
        logger.error(
            "RAG database search failed",
            error=str(e),
            project_dir=project_dir,
            database=database,
        )
        return []


def ingest_file_with_rag(
    project_dir: str,
    project_config: LlamaFarmConfig,
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
        project_config: The full project configuration (not used in Celery version)
        data_processing_strategy_name: Name of the data processing strategy to use
        database_name: Name of the database to use
        source_path: Path to the file to ingest
        filename: Optional original filename
        dataset_name: Optional dataset name

    Returns:
        Tuple of (success: bool, details: dict) with processing information
    """
    logger.info(
        "Starting RAG file ingestion via Celery",
        project_dir=project_dir,
        strategy=data_processing_strategy_name,
        database=database_name,
        source_path=source_path,
        filename=filename,
        dataset_name=dataset_name,
    )

    try:
        success, details = ingest_task(
            project_dir=project_dir,
            data_processing_strategy_name=data_processing_strategy_name,
            database_name=database_name,
            source_path=source_path,
            filename=filename,
            dataset_name=dataset_name,
        )

        logger.info(
            "RAG file ingestion completed",
            success=success,
            details=details,
        )

        return success, details

    except Exception as e:
        logger.error(
            "RAG file ingestion failed",
            error=str(e),
            project_dir=project_dir,
            database=database_name,
            strategy=data_processing_strategy_name,
        )
        return False, {
            "filename": filename or source_path.split("/")[-1],
            "error": str(e),
            "parser": None,
            "extractors": [],
            "chunks": None,
            "chunk_size": None,
            "embedder": None,
            "reason": None,
            "result": None,
        }


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
    logger.info(
        "Starting RAG query processing via Celery",
        project_dir=project_dir,
        database=database,
        query=query[:100] + "..." if len(query) > 100 else query,
        top_k=top_k,
        retrieval_strategy=retrieval_strategy,
        has_context=context is not None,
    )

    try:
        result = query_task(
            project_dir=project_dir,
            database=database,
            query=query,
            context=context,
            top_k=top_k,
            retrieval_strategy=retrieval_strategy,
        )

        logger.info(
            "RAG query processing completed",
            results_count=result.get("total_results", 0),
        )

        return result

    except Exception as e:
        logger.error(
            "RAG query processing failed",
            error=str(e),
            project_dir=project_dir,
            database=database,
        )
        return {
            "query": query,
            "database": database,
            "results": [],
            "total_results": 0,
            "retrieval_strategy": retrieval_strategy,
            "context": context,
            "error": str(e),
        }


def search_with_rag(
    project_dir: str,
    dataset: str,
    query: str,
    top_k: int = 5,
    retrieval_strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run a search via RAG API (dataset-based search).

    This is a compatibility wrapper - for new code, use search_with_rag_database
    with the specific database name.

    Args:
        project_dir: Directory containing llamafarm.yaml config
        dataset: Dataset name (mapped to database)
        query: Search query string
        top_k: Maximum number of results to return
        retrieval_strategy: Optional retrieval strategy name

    Returns:
        List of search results as dictionaries
    """
    logger.info(
        "Legacy dataset search - mapping to database search",
        dataset=dataset,
        query=query[:100] + "..." if len(query) > 100 else query,
    )

    # For compatibility, use the dataset name as the database name
    # In a real implementation, you might want to load the project config
    # to map datasets to databases properly
    return search_with_rag_database(
        project_dir=project_dir,
        database=dataset,
        query=query,
        top_k=top_k,
        retrieval_strategy=retrieval_strategy,
    )
