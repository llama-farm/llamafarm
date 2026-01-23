"""RAG Preview endpoint for document chunking preview."""

import asyncio
import base64
import contextlib
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from api.errors import DatabaseNotFoundError
from core.celery.rag_client import preview_document
from core.logging import FastAPIStructLogger
from services.data_service import DataService
from services.database_service import DatabaseService
from services.project_service import ProjectService

logger = FastAPIStructLogger()

router = APIRouter()


def _sanitize_path_component(value: str, field_name: str) -> str:
    """Ensure path components don't contain traversal characters.

    Args:
        value: The path component to sanitize
        field_name: Name of the field for error messages

    Returns:
        The sanitized value (unchanged if valid)

    Raises:
        ValueError: If the value contains path separators or traversal patterns
    """
    if not value:
        return value
    # Path.name strips any directory components, so if it differs, traversal was attempted
    sanitized = Path(value).name
    if sanitized != value:
        raise ValueError(f"Invalid {field_name}: contains path separators")
    return sanitized


def _resolve_file_path(
    project_dir: str,
    file_hash: str | None,
    dataset_id: str | None,
    file_content: str | None,
    filename: str | None,
) -> tuple[Path, str | None, str | None, bool]:
    """Resolve file path from request parameters.

    Args:
        project_dir: Project directory path
        file_hash: Hash of file in dataset
        dataset_id: Dataset ID containing the file
        file_content: Base64-encoded file content
        filename: Filename for uploaded content

    Returns:
        Tuple of (file_path, original_filename, dataset_id, is_temp_file)

    Raises:
        ValueError: If file cannot be found or parameters are invalid
    """
    if file_hash and dataset_id:
        # File from dataset
        file_path = (
            Path(project_dir)
            / "lf_data"
            / "datasets"
            / dataset_id
            / "raw"
            / file_hash
        )
        if not file_path.exists():
            raise ValueError(f"File not found: {file_hash}")
        return file_path, None, dataset_id, False

    if file_content:
        # Uploaded content - save to temp file
        content = base64.b64decode(file_content)
        # Sanitize filename suffix to prevent path manipulation
        safe_suffix = f"_{Path(filename or 'upload').name}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=safe_suffix) as tmp:
            tmp.write(content)
            return Path(tmp.name), filename, dataset_id, True

    if file_hash:
        # Just file_hash, search in all datasets
        datasets_dir = Path(project_dir) / "lf_data" / "datasets"
        if datasets_dir.exists():
            for dataset_dir in datasets_dir.iterdir():
                potential_path = dataset_dir / "raw" / file_hash
                if potential_path.exists():
                    return potential_path, None, dataset_dir.name, False
        raise ValueError(f"File not found: {file_hash}")

    raise ValueError("Must provide either file_hash or file_content")


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
        if (
            v is not None
            and info.data.get("chunk_size") is not None
            and v >= info.data["chunk_size"]
        ):
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
    # Validate path components to prevent directory traversal attacks
    if request.file_hash:
        _sanitize_path_component(request.file_hash, "file_hash")
    if request.dataset_id:
        _sanitize_path_component(request.dataset_id, "dataset_id")

    # Resolve file path
    file_path, original_filename, dataset_id, is_temp_file = _resolve_file_path(
        project_dir,
        request.file_hash,
        request.dataset_id,
        request.file_content,
        request.filename,
    )

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

    try:
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
        return result
    finally:
        # Clean up temp file if created (even on exception)
        if is_temp_file:
            with contextlib.suppress(Exception):
                file_path.unlink()


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
    try:
        DatabaseService.get_database(namespace, project, database_name)
    except DatabaseNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Database '{database_name}' not found"
        ) from None

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
            ChunkPreviewInfo.model_validate(c) for c in result.get("chunks", [])
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
        raise HTTPException(
            status_code=500,
            detail="Preview generation failed. Check server logs for details.",
        ) from e
