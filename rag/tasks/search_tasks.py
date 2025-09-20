"""
RAG Search Tasks

Celery tasks for RAG search operations including database searches
and retrieval operations.
"""

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import Task
from celery_app import app

# Add parent directory to path for imports - ensure we can import from rag modules
rag_root = Path(__file__).parent.parent
if str(rag_root) not in sys.path:
    sys.path.insert(0, str(rag_root))

try:
    from api import DatabaseSearchAPI
except ImportError as e:
    # Fallback: try importing from absolute path
    api_path = rag_root / "api.py"
    if api_path.exists():
        spec = importlib.util.spec_from_file_location("api", api_path)
        if spec and spec.loader:
            api_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api_module)
            DatabaseSearchAPI = api_module.DatabaseSearchAPI
    else:
        raise ImportError(f"Could not import DatabaseSearchAPI: {e}")

logger = logging.getLogger(__name__)


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
                "args": args,
                "kwargs": kwargs,
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
