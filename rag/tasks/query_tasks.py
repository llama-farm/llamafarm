"""
RAG Query Tasks

Celery tasks for complex RAG query operations and processing.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import Task

from celery_app import app

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import DatabaseSearchAPI

logger = logging.getLogger(__name__)


class QueryTask(Task):
    """Base task class for query operations with error handling."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log task failure details."""
        logger.error(
            "RAG query task failed",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "error": str(exc),
                "task_args": args,
                "task_kwargs": kwargs,
            },
        )


@app.task(bind=True, base=QueryTask, name="rag.handle_rag_query")
def handle_rag_query_task(
    self,
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
        "Starting RAG query processing",
        extra={
            "task_id": self.request.id,
            "project_dir": project_dir,
            "database": database,
            "query": query[:100] + "..." if len(query) > 100 else query,
            "top_k": top_k,
            "retrieval_strategy": retrieval_strategy,
            "has_context": context is not None,
        },
    )

    try:
        # Build config path
        cfg_path = Path(project_dir) / "llamafarm.yaml"

        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        # Initialize search API
        api = DatabaseSearchAPI(config_path=str(cfg_path), database=database)

        # Perform search
        results = api.search(
            query=query, top_k=top_k, retrieval_strategy=retrieval_strategy
        )

        # Convert results to dictionaries
        result_dicts = [r.to_dict() for r in results]

        # Build response
        response = {
            "query": query,
            "database": database,
            "results": result_dicts,
            "total_results": len(result_dicts),
            "retrieval_strategy": retrieval_strategy,
            "context": context,
        }

        logger.info(
            "RAG query processing completed",
            extra={
                "task_id": self.request.id,
                "results_count": len(result_dicts),
            },
        )

        return response

    except Exception as e:
        logger.error(
            "RAG query processing failed",
            extra={
                "task_id": self.request.id,
                "error": str(e),
                "project_dir": project_dir,
                "database": database,
            },
        )
        # Re-raise to mark task as failed
        raise


@app.task(bind=True, base=QueryTask, name="rag.batch_search")
def batch_search_task(
    self,
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
    logger.info(
        "Starting batch RAG search",
        extra={
            "task_id": self.request.id,
            "project_dir": project_dir,
            "database": database,
            "query_count": len(queries),
            "top_k": top_k,
            "retrieval_strategy": retrieval_strategy,
        },
    )

    try:
        # Build config path
        cfg_path = Path(project_dir) / "llamafarm.yaml"

        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        # Initialize search API
        api = DatabaseSearchAPI(config_path=str(cfg_path), database=database)

        # Process each query
        all_results = []
        for i, query in enumerate(queries):
            logger.info(
                "Processing batch query",
                extra={
                    "task_id": self.request.id,
                    "query_index": i + 1,
                    "query": query[:50] + "..." if len(query) > 50 else query,
                },
            )

            # Perform search
            results = api.search(
                query=query, top_k=top_k, retrieval_strategy=retrieval_strategy
            )

            # Convert results to dictionaries
            result_dicts = [r.to_dict() for r in results]

            all_results.append(
                {
                    "query": query,
                    "results": result_dicts,
                    "total_results": len(result_dicts),
                }
            )

        logger.info(
            "Batch RAG search completed",
            extra={
                "task_id": self.request.id,
                "total_queries": len(queries),
                "total_results": sum(r["total_results"] for r in all_results),
            },
        )

        return all_results

    except Exception as e:
        logger.error(
            "Batch RAG search failed",
            extra={
                "task_id": self.request.id,
                "error": str(e),
                "project_dir": project_dir,
                "database": database,
            },
        )
        # Re-raise to mark task as failed
        raise
