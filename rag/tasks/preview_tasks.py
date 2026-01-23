"""RAG Preview Tasks - Celery tasks for document preview."""

import sys
from pathlib import Path
from typing import Any

from celery import Task

from celery_app import app

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logging import RAGStructLogger
from core.preview_handler import PreviewHandler

logger = RAGStructLogger("rag.tasks.preview")


class PreviewTask(Task):
    """Base task class for preview operations."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(
            "RAG preview task failed",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            exc_info=True,
        )


def _get_blob_processor(
    project_dir: str,
    database: str | None = None,
    data_processing_strategy_name: str | None = None,
):
    """Get a configured BlobProcessor for the project.

    Args:
        project_dir: Path to the project directory
        database: Database name (currently unused, reserved for future config lookup)
        data_processing_strategy_name: Specific strategy name to use. If not provided,
            falls back to the first available strategy.
    """
    from core.blob_processor import BlobProcessor
    from core.strategies.handler import SchemaHandler

    config_path = Path(project_dir) / "llamafarm.yaml"
    schema_handler = SchemaHandler(str(config_path))

    # Get available strategy names
    strategy_names = schema_handler.get_data_processing_strategy_names()
    if not strategy_names:
        raise ValueError("No data processing strategies found in config")

    # Use provided strategy name if valid, otherwise fall back to first strategy
    if data_processing_strategy_name and data_processing_strategy_name in strategy_names:
        strategy_to_use = data_processing_strategy_name
    else:
        strategy_to_use = strategy_names[0]

    processing_strategy = schema_handler.create_processing_config(strategy_to_use)

    return BlobProcessor(processing_strategy)


@app.task(bind=True, base=PreviewTask, name="rag.preview_document")
def preview_document_task(
    self,
    project_dir: str,
    file_path: str,
    database: str | None = None,
    data_processing_strategy_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    chunk_strategy: str | None = None,
    original_filename: str | None = None,
) -> dict[str, Any]:
    """Generate preview using same processing as ingestion.

    Args:
        project_dir: Path to the project directory
        file_path: Path to the file to preview
        database: Database name (for config lookup)
        data_processing_strategy_name: Specific strategy name to use for processing.
            If not provided, falls back to the first available strategy.
        chunk_size: Override chunk size
        chunk_overlap: Override chunk overlap
        chunk_strategy: Override chunk strategy
        original_filename: Original filename with extension (for parser detection)
    """
    logger.info(
        "Starting preview generation",
        extra={
            "task_id": self.request.id,
            "project_dir": project_dir,
            "file_path": file_path,
            "database": database,
            "data_processing_strategy_name": data_processing_strategy_name,
            "original_filename": original_filename,
        },
    )

    try:
        # Load blob_processor with config
        blob_processor = _get_blob_processor(
            project_dir, database, data_processing_strategy_name
        )
        preview_handler = PreviewHandler(blob_processor)

        # Read file
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_data = file_path_obj.read_bytes()
        # Use original_filename if provided (for proper parser detection),
        # otherwise fall back to the file path name
        filename = original_filename if original_filename else file_path_obj.name
        metadata = {
            "filename": filename,
            "filepath": str(file_path_obj),
            "size": len(file_data),
        }

        # Generate preview
        result = preview_handler.generate_preview(
            file_data,
            metadata,
            chunk_size_override=chunk_size,
            chunk_overlap_override=chunk_overlap,
            chunk_strategy_override=chunk_strategy,
        )

        logger.info(
            "Preview generation complete",
            extra={
                "task_id": self.request.id,
                "total_chunks": result.total_chunks,
                "parser_used": result.parser_used,
            },
        )

        return result.to_dict()

    except Exception as e:
        logger.error(
            "Preview generation failed",
            extra={
                "task_id": self.request.id,
                "error": str(e),
                "file_path": file_path,
            },
            exc_info=True,
        )
        raise
