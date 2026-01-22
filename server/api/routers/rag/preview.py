"""RAG Preview endpoint for document chunking preview."""

import asyncio
import base64
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from core.celery.rag_client import preview_document
from core.logging import FastAPIStructLogger
from services.data_service import DataService
from services.project_service import ProjectService

logger = FastAPIStructLogger()

router = APIRouter()


class DocumentPreviewRequest(BaseModel):
    """Request model for document preview."""

    # For existing files in dataset
    dataset_id: str | None = Field(None, description="Dataset containing the file")
    file_hash: str | None = Field(None, description="Hash of the file to preview")

    # For uploaded content (base64)
    file_content: str | None = Field(None, description="Base64-encoded file content")
    filename: str | None = Field(None, description="Filename for uploaded content")

    # Data processing strategy selection
    data_processing_strategy: str | None = Field(
        None,
        description="Data processing strategy to use. If not provided, uses the "
        "dataset's configured strategy or falls back to the first available strategy.",
    )

    # Override settings
    chunk_size: int | None = Field(
        None, ge=50, le=10000, description="Override chunk size"
    )
    chunk_overlap: int | None = Field(None, ge=0, description="Override chunk overlap")
    chunk_strategy: str | None = Field(
        None,
        description="Override chunk strategy",
        pattern="^(characters|sentences|paragraphs)$",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v, info):
        if v is not None and info.data.get("chunk_size") is not None:
            if v >= info.data["chunk_size"]:
                raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class ChunkPreviewInfo(BaseModel):
    """Information about a single chunk in preview."""

    chunk_index: int
    content: str
    start_position: int
    end_position: int
    char_count: int
    word_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentPreviewResponse(BaseModel):
    """Response model for document preview."""

    original_text: str
    chunks: list[ChunkPreviewInfo]

    # File info
    filename: str
    size_bytes: int
    content_type: str | None = None

    # Processing info
    parser_used: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    total_chunks: int

    # Statistics
    avg_chunk_size: float
    total_size_with_overlaps: int
    avg_overlap_size: float = 0.0

    warnings: list[str] = Field(default_factory=list)


async def handle_preview(
    project_config: Any,
    project_dir: str,
    database_name: str,
    request: DocumentPreviewRequest,
    namespace: str,
    project: str,
) -> dict[str, Any]:
    """Handle preview request by dispatching to RAG worker."""
    # Determine file path and original filename
    original_filename: str | None = None
    dataset_id: str | None = request.dataset_id

    if request.file_hash and request.dataset_id:
        # File from dataset
        file_path = (
            Path(project_dir)
            / "lf_data"
            / "datasets"
            / request.dataset_id
            / "raw"
            / request.file_hash
        )
        if not file_path.exists():
            raise ValueError(f"File not found: {request.file_hash}")
    elif request.file_content:
        # Uploaded content - save to temp file
        content = base64.b64decode(request.file_content)
        original_filename = request.filename
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_{request.filename or 'upload'}"
        ) as tmp:
            tmp.write(content)
            file_path = Path(tmp.name)
    elif request.file_hash:
        # Just file_hash, search in all datasets
        datasets_dir = Path(project_dir) / "lf_data" / "datasets"
        file_path = None
        if datasets_dir.exists():
            for dataset_dir in datasets_dir.iterdir():
                potential_path = dataset_dir / "raw" / request.file_hash
                if potential_path.exists():
                    file_path = potential_path
                    dataset_id = dataset_dir.name
                    break
        if not file_path:
            raise ValueError(f"File not found: {request.file_hash}")
    else:
        raise ValueError("Must provide either file_hash or file_content")

    # Look up original filename from dataset metadata if we have a file_hash
    if request.file_hash and dataset_id and not original_filename:
        metadata = DataService.get_data_file_metadata_by_hash(
            namespace, project, dataset_id, request.file_hash
        )
        if metadata:
            original_filename = metadata.original_file_name

    # Determine which data processing strategy to use
    data_processing_strategy_name: str | None = request.data_processing_strategy

    # If no strategy specified and we have a dataset, use the dataset's configured strategy
    if not data_processing_strategy_name and dataset_id:
        try:
            from services.dataset_service import DatasetService

            dataset_config = DatasetService.get_dataset_config(
                namespace, project, dataset_id
            )
            data_processing_strategy_name = dataset_config.data_processing_strategy
        except Exception:
            # If we can't look up the dataset config, fall back to default behavior
            pass

    # Call preview task
    result = await asyncio.to_thread(
        preview_document,
        project_dir=project_dir,
        file_path=str(file_path),
        database=database_name,
        data_processing_strategy_name=data_processing_strategy_name,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        chunk_strategy=request.chunk_strategy,
        original_filename=original_filename,
    )

    # Clean up temp file if created
    if request.file_content:
        try:
            file_path.unlink()
        except Exception:
            pass

    return result


@router.post(
    "/databases/{database_name}/preview",
    response_model=DocumentPreviewResponse,
    operation_id="rag_preview_document",
    summary="Preview document chunking",
)
async def preview_document_endpoint(
    namespace: str,
    project: str,
    database_name: str,
    request: DocumentPreviewRequest,
):
    """
    Preview how a document will be parsed and chunked.

    Returns the document text with chunk boundaries and statistics.
    Use this to test different chunk sizes and overlap settings
    before ingesting documents.
    """
    logger.bind(namespace=namespace, project=project, database=database_name)

    # Get project
    try:
        project_obj = ProjectService.get_project(namespace, project)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    project_dir = ProjectService.get_project_dir(namespace, project)

    # Validate RAG is configured
    if not project_obj.config.rag:
        raise HTTPException(
            status_code=400, detail="RAG not configured for this project"
        )

    # Validate database exists
    database_exists = False
    for db in project_obj.config.rag.databases or []:
        if db.name == database_name:
            database_exists = True
            break

    if not database_exists:
        raise HTTPException(
            status_code=404, detail=f"Database '{database_name}' not found"
        )

    # Validate request
    if not request.file_hash and not request.file_content:
        raise HTTPException(
            status_code=400, detail="Must provide either file_hash or file_content"
        )

    try:
        result = await handle_preview(
            project_obj.config,
            str(project_dir),
            database_name,
            request,
            namespace,
            project,
        )

        # Transform to response model
        chunks = [
            ChunkPreviewInfo(
                chunk_index=c["chunk_index"],
                content=c["content"],
                start_position=c["start_position"],
                end_position=c["end_position"],
                char_count=c["char_count"],
                word_count=c["word_count"],
                metadata=c.get("metadata", {}),
            )
            for c in result.get("chunks", [])
        ]

        return DocumentPreviewResponse(
            original_text=result["original_text"],
            chunks=chunks,
            filename=result["file_info"]["filename"],
            size_bytes=result["file_info"]["size"],
            content_type=result["file_info"].get("content_type"),
            parser_used=result["parser_used"],
            chunk_strategy=result["chunk_strategy"],
            chunk_size=result["chunk_size"],
            chunk_overlap=result["chunk_overlap"],
            total_chunks=result["total_chunks"],
            avg_chunk_size=result["avg_chunk_size"],
            total_size_with_overlaps=result["total_size_with_overlaps"],
            warnings=result.get("warnings", []),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Preview failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
