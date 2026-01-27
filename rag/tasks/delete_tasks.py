"""
RAG Document Deletion Tasks

Celery tasks for deleting documents from vector stores by file_hash.
"""

import sys
from pathlib import Path
from typing import Literal, TypedDict

from celery import Task

from celery_app import app

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.document_manager import DocumentManager
from core.ingest_handler import IngestHandler
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.tasks.delete")


class DeleteTask(Task):
    """Base task class for deletion operations with error handling."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log task failure details."""
        logger.error(
            "RAG delete task failed",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            task_args=args,
            task_kwargs=kwargs,
            exc_info=True,
        )


class DeleteFileTaskResult(TypedDict):
    status: Literal["success", "error"]
    file_hash: str
    deleted_count: int
    error: str | None


@app.task(bind=True, base=DeleteTask, name="rag.delete_file")
def delete_file_task(
    self,
    project_dir: str,
    database_name: str,
    file_hash: str,
) -> DeleteFileTaskResult:
    """
    Delete all chunks belonging to a file from the vector store.

    Args:
        project_dir: The directory of the project
        database_name: Name of the database to delete from
        file_hash: SHA-256 hash of the file content to delete

    Returns:
        Dictionary with deletion results including deleted_count
    """
    logger.info(
        "Starting RAG file deletion",
        extra={
            "task_id": self.request.id,
            "project_dir": project_dir,
            "database": database_name,
            "file_hash": file_hash[:16] + "...",
        },
    )

    try:
        # Configuration path
        config_path = Path(project_dir) / "llamafarm.yaml"

        if not config_path.exists():
            error_msg = f"Config file not found: {config_path}"
            logger.error(error_msg, extra={"task_id": self.request.id})
            return {
                "status": "error",
                "error": error_msg,
                "file_hash": file_hash,
                "deleted_count": 0,
            }

        # Initialize handler to get vector store
        # We use a dummy strategy since we only need the database connection
        # TODO: Refactor rag code so that we don't need to use an IngestHandler just
        #       to get a vector store instance
        handler = IngestHandler(
            config_path=str(config_path),
            data_processing_strategy=_get_first_strategy(config_path),
            database=database_name,
        )

        # Use DocumentManager to handle the deletion logic
        doc_manager = DocumentManager(handler.vector_store)
        result = doc_manager.delete_by_file_hash(file_hash)

        logger.info(
            "RAG file deletion completed",
            extra={
                "task_id": self.request.id,
                "file_hash": file_hash[:16] + "...",
                "deleted_count": result.get("deleted_count", 0),
            },
        )

        return DeleteFileTaskResult(
            status="success" if not result.get("error") else "error",
            file_hash=file_hash,
            deleted_count=result.get("deleted_count", 0),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(
            "RAG file deletion failed",
            extra={
                "task_id": self.request.id,
                "error": str(e),
                "file_hash": file_hash[:16] + "...",
                "database": database_name,
            },
            exc_info=True,
        )
        return DeleteFileTaskResult(
            status="error",
            error=str(e),
            file_hash=file_hash,
            deleted_count=0,
        )


def _get_first_strategy(config_path: Path) -> str:
    """Get the first available data processing strategy from config.

    Returns 'universal_rag' if no strategies are defined, which triggers
    the built-in default strategy.
    """
    try:
        from config import load_config

        config = load_config(str(config_path))
        if config.rag and config.rag.data_processing_strategies:
            return config.rag.data_processing_strategies[0].name
    except Exception:
        pass
    return "universal_rag"
