"""
RAG Ingestion Tasks

Celery tasks for RAG file ingestion and processing operations.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from celery import Task

from celery_app import app

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ingest_handler import IngestHandler
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.tasks.ingest")


class IngestTask(Task):
    """Base task class for ingestion operations with error handling."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log task failure details."""
        logger.error(
            "RAG ingest task failed",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            task_args=args,
            task_kwargs=kwargs,
            exc_info=True,
        )


@app.task(bind=True, base=IngestTask, name="rag.ingest_file")
def ingest_file_with_rag_task(
    self,
    project_dir: str,
    data_processing_strategy_name: str,
    database_name: str,
    source_path: str,
    filename: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> tuple[bool, dict[str, Any]]:
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
    logger.info(
        "Starting RAG file ingestion",
        extra={
            "task_id": self.request.id,
            "project_dir": project_dir,
            "strategy": data_processing_strategy_name,
            "database": database_name,
            "source_path": source_path,
            "file_name": filename,
            "dataset_name": dataset_name,
        },
    )

    # Initialize details dict
    details = {
        "filename": filename or Path(source_path).name,
        "parser": None,
        "extractors": [],
        "chunks": None,
        "chunk_size": None,
        "embedder": None,
        "error": None,
        "reason": None,
        "result": None,
    }

    try:
        # Configuration path
        config_path = Path(project_dir) / "llamafarm.yaml"

        if not config_path.exists():
            error_msg = f"Config file not found: {config_path}"
            logger.error(error_msg, extra={"task_id": self.request.id})
            details["error"] = error_msg
            return False, details

        # Initialize the ingest handler
        handler = IngestHandler(
            config_path=str(config_path),
            data_processing_strategy=data_processing_strategy_name,
            database=database_name,
            dataset_name=dataset_name,
        )

        # Read the file
        with open(source_path, "rb") as f:
            file_data = f.read()

        # Create metadata
        file_path = Path(source_path)

        # Check if this is a hash-based file in lf_data/raw
        if "lf_data/raw" in str(source_path):
            # Extract file hash from the path
            file_hash = file_path.name
            # Try to load metadata file
            meta_dir = file_path.parent.parent / "meta"
            meta_file = meta_dir / f"{file_hash}.json"

            if meta_file.exists():
                with open(meta_file, "r") as mf:
                    meta_content = json.load(mf)
                    original_filename = meta_content.get(
                        "original_file_name", file_hash
                    )
                    mime_type = meta_content.get(
                        "mime_type", "application/octet-stream"
                    )
            else:
                original_filename = file_hash
                mime_type = "application/octet-stream"
        else:
            # Regular file path
            original_filename = file_path.name
            mime_type = "application/octet-stream"

        metadata = {
            "filename": original_filename,
            "filepath": str(file_path),
            "size": len(file_data),
            "content_type": mime_type,
        }

        # Ingest the file
        result = handler.ingest_file(file_data=file_data, metadata=metadata)

        # Process result
        details["result"] = result
        status = result.get("status")

        # Extract details from result
        if result.get("parsers_used"):
            details["parser"] = (
                result["parsers_used"][0]
                if len(result["parsers_used"]) == 1
                else ", ".join(result["parsers_used"])
            )

        if result.get("extractors_applied"):
            details["extractors"] = result["extractors_applied"]

        if result.get("embedder"):
            details["embedder"] = result["embedder"]

        if result.get("document_count"):
            details["chunks"] = result["document_count"]

        if result.get("chunk_size"):
            details["chunk_size"] = result["chunk_size"]

        # Pass through the status and counts
        if result.get("status"):
            details["status"] = result["status"]
        if "stored_count" in result:
            details["stored_count"] = result["stored_count"]
        if "skipped_count" in result:
            details["skipped_count"] = result["skipped_count"]

        # Set reason if it's a duplicate
        if status == "skipped" or result.get("reason") == "duplicate":
            details["reason"] = "duplicate"
            details["status"] = "skipped"
        elif result.get("stored_count", 0) == 0 and result.get("skipped_count", 0) > 0:
            details["reason"] = "duplicate"
            details["status"] = "skipped"

        # Determine success
        success = status in ["success", "skipped"]

        if not success:
            details["error"] = result.get("message", "Unknown error")

        logger.info(
            "RAG file ingestion completed",
            extra={
                "task_id": self.request.id,
                "success": success,
                "status": status,
                "stored_count": result.get("stored_count", 0),
                "skipped_count": result.get("skipped_count", 0),
            },
        )

        return success, details

    except Exception as e:
        logger.error(
            "RAG file ingestion failed",
            extra={
                "task_id": self.request.id,
                "error": str(e),
                "source_path": source_path,
                "database": database_name,
                "strategy": data_processing_strategy_name,
            },
            exc_info=True,
        )
        details["error"] = str(e)
        return False, details
