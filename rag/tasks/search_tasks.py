"""
RAG Search Tasks

Celery tasks for RAG search operations including database searches
and retrieval operations.
"""

import sys
from pathlib import Path
from typing import Any, Optional

from celery import Task

from celery_app import app

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import DatabaseSearchAPI
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.tasks.search")


class SearchTask(Task):
    """Base task class for search operations with error handling."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log task failure details."""
        logger.error(
            "RAG search task failed",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "error": str(exc),
                "task_args": args,
                "task_kwargs": kwargs,
            },
        )


@app.task(bind=True, base=SearchTask, name="rag.search_with_database")
def search_with_rag_database_task(
    self,
    project_dir: str,
    database: str,
    query: str,
    top_k: int = 5,
    retrieval_strategy: Optional[str] = None,
    score_threshold: Optional[float] = None,
    metadata_filters: Optional[dict[str, Any]] = None,
    distance_metric: Optional[str] = None,
    hybrid_alpha: Optional[float] = None,
    rerank_model: Optional[str] = None,
    query_expansion: Optional[bool] = None,
    max_tokens: Optional[int] = None,
) -> list[dict[str, Any]]:
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
        "Starting RAG database search",
        extra={
            "task_id": self.request.id,
            "project_dir": project_dir,
            "database": database,
            "query": query[:100] + "..." if len(query) > 100 else query,
            "top_k": top_k,
            "retrieval_strategy": retrieval_strategy,
        },
    )

    try:
        # Initialize search API
        api = DatabaseSearchAPI(project_dir=project_dir, database=database)

        # Perform search
        # Map to API parameters
        api_kwargs: dict[str, Any] = {
            "retrieval_strategy": retrieval_strategy,
            "top_k": top_k,
        }
        if score_threshold is not None:
            api_kwargs["min_score"] = score_threshold
        if metadata_filters is not None:
            api_kwargs["metadata_filter"] = metadata_filters
        # Strategy-specific hints; passed through to strategy.retrieve(**kwargs)
        if distance_metric is not None:
            api_kwargs["distance_metric"] = distance_metric
        if hybrid_alpha is not None:
            api_kwargs["hybrid_alpha"] = hybrid_alpha
        if rerank_model is not None:
            api_kwargs["rerank_model"] = rerank_model
        if query_expansion is not None:
            api_kwargs["query_expansion"] = query_expansion
        if max_tokens is not None:
            api_kwargs["max_tokens"] = max_tokens

        results = api.search(query=query, **api_kwargs)

        # Convert results to dictionaries
        result_dicts = [r.to_dict() for r in results]

        logger.info(
            "RAG database search completed",
            extra={
                "task_id": self.request.id,
                "results_count": len(result_dicts),
            },
        )

        return result_dicts

    except Exception as e:
        logger.error(
            "RAG database search failed",
            extra={
                "task_id": self.request.id,
                "error": str(e),
                "project_dir": project_dir,
                "database": database,
            },
        )
        # Re-raise to mark task as failed
        raise
