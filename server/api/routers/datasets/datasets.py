import json
from enum import StrEnum
from typing import Any

from config.datamodel import Dataset
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from api.routers.datasets._models import ListDatasetsResponse
from core.logging import FastAPIStructLogger
from services.dataset_service import DatasetIngestLaunchResult, DatasetService
from services.project_service import ProjectService

logger = FastAPIStructLogger()

router = APIRouter(
    prefix="/projects/{namespace}/{project}/datasets",
    tags=["datasets"],
)

# Reusable FastAPI defaults to satisfy lint rule B008 (no call in defaults)
FILES_REQUIRED = File(...)


# Support both with and without trailing slash to avoid proxy redirect issues
@router.get(
    "/",
    operation_id="dataset_list",
    tags=["mcp"],
    responses={200: {"model": ListDatasetsResponse}},
)
@router.get("", include_in_schema=False)
async def list_datasets(
    namespace: str,
    project: str,
    include_extra_details: bool = Query(
        True, description="Include detailed file information with original filenames"
    ),
):
    logger.bind(namespace=namespace, project=project)
    if include_extra_details:
        datasets = DatasetService.list_datasets_with_file_details(namespace, project)
    else:
        datasets = DatasetService.list_datasets(namespace, project)

    return ListDatasetsResponse(
        total=len(datasets),
        datasets=datasets,
    )


class AvailableStrategiesResponse(BaseModel):
    data_processing_strategies: list[str]
    databases: list[str]


@router.get(
    "/strategies",
    operation_id="dataset_strategies_list",
    tags=["mcp"],
    summary="List available data processing strategies and databases for the project",
    description="List available data processing strategies and databases for the project",
    responses={200: {"model": AvailableStrategiesResponse}},
)
async def get_available_strategies(namespace: str, project: str):
    """Get available data processing strategies and databases for the project"""
    logger.bind(namespace=namespace, project=project)
    data_processing_strategies = (
        DatasetService.get_supported_data_processing_strategies(namespace, project)
    )
    databases = DatasetService.get_supported_databases(namespace, project)
    return AvailableStrategiesResponse(
        data_processing_strategies=data_processing_strategies,
        databases=databases,
    )


class CreateDatasetRequest(BaseModel):
    name: str
    data_processing_strategy: str
    database: str


class CreateDatasetResponse(BaseModel):
    dataset: Dataset


@router.post(
    "/",
    operation_id="dataset_create",
    tags=["mcp"],
    responses={200: {"model": CreateDatasetResponse}},
)
async def create_dataset(namespace: str, project: str, request: CreateDatasetRequest):
    logger.bind(namespace=namespace, project=project)
    try:
        dataset = DatasetService.create_dataset(
            namespace=namespace,
            project=project,
            name=request.name,
            data_processing_strategy=request.data_processing_strategy,
            database=request.database,
        )
        return CreateDatasetResponse(dataset=dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class DeleteDatasetResponse(BaseModel):
    dataset: Dataset


@router.delete(
    "/{dataset}",
    operation_id="dataset_delete",
    tags=["mcp"],
    responses={200: {"model": DeleteDatasetResponse}},
)
async def delete_dataset(namespace: str, project: str, dataset: str):
    logger.bind(namespace=namespace, project=project)
    try:
        deleted_dataset = DatasetService.delete_dataset(
            namespace=namespace, project=project, name=dataset
        )
        return DeleteDatasetResponse(dataset=deleted_dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class DatasetActionType(StrEnum):
    PROCESS = "process"
    DELETE_FILE_CHUNKS = "delete_file_chunks"
    DELETE_DATASET_CHUNKS = "delete_dataset_chunks"


class DatasetActionRequest(BaseModel):
    action_type: DatasetActionType = Field(
        ..., description="The type of action to execute"
    )
    file_hash: str | None = Field(
        None, description="File hash for delete_file_chunks action"
    )
    parser_overrides: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Optional parser config overrides for PROCESS actions",
    )


class DatasetActionResponse(BaseModel):
    message: str = Field(..., description="The status message")
    task_uri: str = Field(..., description="The URI for tracking the task")
    task_id: str = Field(..., description="The Celery task ID")


class DeleteChunksResponse(BaseModel):
    message: str = Field(..., description="The status message")
    file_hash: str = Field(..., description="The file hash")
    deleted_chunks: int = Field(..., description="Number of chunks deleted")


class DeleteAllChunksResponse(BaseModel):
    message: str = Field(..., description="The status message")
    total_deleted_chunks: int = Field(..., description="Total number of chunks deleted")
    total_files_cleared: int = Field(
        ..., description="Number of files whose chunks were deleted"
    )
    total_files_failed: int = Field(
        ..., description="Number of files that failed to have chunks deleted"
    )


@router.post(
    "/{dataset}/actions",
    operation_id="dataset_actions",
    summary="Execute an action on a dataset",
    description="""Execute an action on a dataset
    - PROCESS: Process all files in the dataset using the configured data processing strategy
    - DELETE_FILE_CHUNKS: Delete chunks for a specific file from the vector store (requires file_hash)
    - DELETE_DATASET_CHUNKS: Delete chunks for ALL files from the vector store (for reprocessing entire dataset)
    """,
    tags=["mcp"],
    responses={200: {"model": DatasetActionResponse}},
)
async def actions(
    namespace: str, project: str, dataset: str, request: DatasetActionRequest, req: Request
):
    logger.bind(namespace=namespace, project=project, dataset=dataset)

    action_type = request.action_type

    def task_uri(task_id: str):
        return f"{req.base_url}v1/projects/{namespace}/{project}/tasks/{task_id}"

    if action_type == DatasetActionType.PROCESS:
        launch = DatasetService.start_dataset_ingestion(
            namespace, project, dataset, parser_overrides=request.parser_overrides
        )
        return {
            "message": launch.message,
            "task_uri": task_uri(launch.task_id),
            "task_id": launch.task_id,
        }
    elif action_type == DatasetActionType.DELETE_FILE_CHUNKS:
        if not request.file_hash:
            raise HTTPException(
                status_code=400,
                detail="file_hash required for delete_file_chunks action",
            )
        result = await DatasetService.delete_file_chunks(
            namespace, project, dataset, request.file_hash
        )
        return DeleteChunksResponse(
            message="Chunks deleted successfully",
            file_hash=request.file_hash,
            deleted_chunks=result.get("deleted_count", 0),
        )
    elif action_type == DatasetActionType.DELETE_DATASET_CHUNKS:
        result = await DatasetService.delete_dataset_chunks(namespace, project, dataset)
        return DeleteAllChunksResponse(
            message="All chunks deleted successfully",
            total_deleted_chunks=result.get("total_deleted_chunks", 0),
            total_files_cleared=result.get("total_files_cleared", 0),
            total_files_failed=result.get("total_files_failed", 0),
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Invalid action type: {action_type}"
        )


class DatasetDataUploadResponse(BaseModel):
    filename: str = Field(..., description="The name of the uploaded file")
    hash: str = Field(..., description="The hash of the uploaded file")
    processed: bool = Field(..., description="Whether the file has been processed")
    skipped: bool = Field(
        default=False, description="Whether the file was skipped (duplicate)"
    )
    task_id: str | None = Field(
        default=None, description="Celery task ID if processing was started"
    )
    status: str | None = Field(
        default=None,
        description="Upload status (processing, uploaded, skipped, or error)",
    )


class BulkDatasetDataUploadResponse(BaseModel):
    uploaded: int = Field(..., description="Number of files uploaded")
    skipped: int = Field(default=0, description="Number of files skipped (duplicates)")
    failed: int = Field(default=0, description="Number of files that failed to upload")
    task_id: str | None = Field(
        default=None, description="Celery task ID if processing was started"
    )
    status: str = Field(
        default="uploaded",
        description="Bulk upload status (processing when auto-process triggered)",
    )


def _parse_parser_overrides(raw_overrides: str | None):
    if not raw_overrides:
        return None
    if len(raw_overrides) > 10240:
        raise HTTPException(
            status_code=400,
            detail="parser_overrides payload too large (max 10KB)",
        )
    try:
        parsed = json.loads(raw_overrides)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parser_overrides JSON: {exc}",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=400,
            detail="parser_overrides must be a JSON object mapping parser type to config",
        )

    # Basic safety validation for chunk settings
    CHUNK_SIZE_MAX = 100000

    def _validate_chunk_field(name: str, value: object, allow_zero: bool):
        if not isinstance(value, int | float):
            raise HTTPException(
                status_code=400, detail=f"{name} must be a number"
            )
        if allow_zero:
            if value < 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"{name} must be greater than or equal to 0",
                )
        else:
            if value <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"{name} must be greater than 0",
                )
        if name == "chunk_size" and value > CHUNK_SIZE_MAX:
            raise HTTPException(
                status_code=400,
                detail=f"{name} must be less than or equal to {CHUNK_SIZE_MAX}",
            )

    for parser_type, override in parsed.items():
        if not isinstance(override, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Override for {parser_type or 'parser'} must be an object",
            )
        if "chunk_size" in override:
            _validate_chunk_field("chunk_size", override["chunk_size"], allow_zero=False)
        if "chunk_overlap" in override:
            _validate_chunk_field(
                "chunk_overlap", override["chunk_overlap"], allow_zero=True
            )
        if "chunk_size" in override and "chunk_overlap" in override:
            try:
                if override["chunk_overlap"] >= override["chunk_size"]:
                    raise HTTPException(
                        status_code=400,
                        detail="chunk_overlap must be less than chunk_size",
                    )
            except TypeError:
                raise HTTPException(
                    status_code=400,
                    detail="chunk_size and chunk_overlap must be numbers",
                ) from None

    return parsed


def _validate_overrides_against_default_chunking(
    namespace: str,
    project: str,
    strategy_name: str | None,
    parser_overrides: dict | None,
):
    """
    Validate merged chunk_size/chunk_overlap using default parser configs.

    Ensures that providing only chunk_overlap does not exceed the default chunk_size.
    """
    if not parser_overrides or not strategy_name:
        return

    try:
        project_config = ProjectService.load_config(namespace, project)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "Failed to load project config for parser override validation",
            namespace=namespace,
            project=project,
            error=str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail="Unable to load project config for parser override validation",
        ) from exc

    rag_config = getattr(project_config, "rag", None)
    strategies = getattr(rag_config, "data_processing_strategies", None) or []
    strategy = next(
        (s for s in strategies if getattr(s, "name", None) == strategy_name), None
    )
    if not strategy or not getattr(strategy, "parsers", None):
        return

    wildcard_override = parser_overrides.get("*") or parser_overrides.get("__all__")

    for parser in strategy.parsers or []:
        parser_type = getattr(parser, "type", None)
        merged_config: dict[str, Any] = {}
        base_config = getattr(parser, "config", None) or {}
        merged_config.update(base_config)
        if wildcard_override:
            merged_config.update(wildcard_override)
        if parser_type:
            merged_config.update(parser_overrides.get(parser_type, {}) or {})

        chunk_size = merged_config.get("chunk_size")
        chunk_overlap = merged_config.get("chunk_overlap")
        if chunk_size is None or chunk_overlap is None:
            continue

        if not isinstance(chunk_size, int | float) or not isinstance(
            chunk_overlap, int | float
        ):
            raise HTTPException(
                status_code=400,
                detail="chunk_size and chunk_overlap must be numbers after applying overrides",
            )

        if chunk_size <= 0:
            raise HTTPException(
                status_code=400,
                detail="chunk_size must be greater than 0 after applying overrides",
            )

        if chunk_overlap < 0:
            raise HTTPException(
                status_code=400,
                detail="chunk_overlap must be greater than or equal to 0 after applying overrides",
            )

        if chunk_overlap >= chunk_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"chunk_overlap ({chunk_overlap}) must be less than chunk_size "
                    f"({chunk_size}) for parser {parser_type or 'unknown'}"
                ),
            )


@router.post(
    "/{dataset}/data",
    operation_id="dataset_data_upload",
    summary="Upload a file to the dataset",
    description=(
        "Upload a file to the dataset. Processing is triggered automatically based on dataset configuration or the auto_process flag."
    ),
    responses={200: {"model": DatasetDataUploadResponse}},
)
async def upload_data(
    namespace: str,
    project: str,
    dataset: str,
    file: UploadFile,
    auto_process: bool | None = Query(
        default=None,
        description="Automatically process the file into the vector database. Defaults to dataset config (true if unspecified in config).",
    ),
    parser_overrides: str | None = Form(
        default=None,
        description="JSON object mapping parser type to override config, e.g. {'PDFParser_LlamaIndex': {'chunk_size': 1024}}",
    ),
):
    """Upload a file to the dataset. Processing triggered based on dataset config or auto_process parameter."""
    logger.bind(namespace=namespace, project=project, dataset=dataset)

    dataset_config = DatasetService.get_dataset_config(namespace, project, dataset)
    dataset_auto_process = (
        dataset_config.auto_process if dataset_config.auto_process is not None else True
    )
    effective_auto_process = (
        auto_process if auto_process is not None else dataset_auto_process
    )
    parsed_overrides = _parse_parser_overrides(parser_overrides)
    _validate_overrides_against_default_chunking(
        namespace=namespace,
        project=project,
        strategy_name=getattr(dataset_config, "data_processing_strategy", None),
        parser_overrides=parsed_overrides,
    )

    was_added, metadata_file_content = await DatasetService.add_file_to_dataset(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file=file,
    )

    if not was_added:
        logger.info(
            "File skipped (duplicate)",
            dataset=dataset,
            filename=file.filename,
            hash=metadata_file_content.hash,
        )
        return DatasetDataUploadResponse(
            filename=file.filename,
            hash=metadata_file_content.hash,
            processed=False,
            skipped=True,
            status="skipped",
        )

    logger.info(
        "File uploaded to dataset",
        dataset=dataset,
        filename=file.filename,
        hash=metadata_file_content.hash,
    )

    launch: DatasetIngestLaunchResult | None = None
    if effective_auto_process:
        launch = DatasetService.start_ingestion_for_hashes(
            namespace=namespace,
            project=project,
            dataset=dataset,
            file_hashes=[metadata_file_content.hash],
            parser_overrides=parsed_overrides,
        )
        status = "processing"
        processed = True
    else:
        status = "uploaded"
        processed = False

    return DatasetDataUploadResponse(
        filename=file.filename,
        hash=metadata_file_content.hash,
        processed=processed,
        skipped=False,
        task_id=launch.task_id if launch else None,
        status=status,
    )


@router.post(
    "/{dataset}/data/bulk",
    operation_id="dataset_data_bulk_upload",
    summary="Upload multiple files to the dataset",
    description=(
        "Bulk upload files to the dataset. Defaults to storing files without processing. "
        "Set auto_process=true to process immediately."
    ),
    responses={200: {"model": BulkDatasetDataUploadResponse}},
)
async def upload_data_bulk(
    namespace: str,
    project: str,
    dataset: str,
    files: list[UploadFile] = FILES_REQUIRED,
    auto_process: bool | None = Query(
        default=None,
        description="Process all uploaded files immediately (default: false for bulk)",
    ),
    parser_overrides: str | None = Form(
        default=None,
        description="JSON object mapping parser type to override config",
    ),
):
    logger.bind(namespace=namespace, project=project, dataset=dataset)
    dataset_config = DatasetService.get_dataset_config(namespace, project, dataset)
    parsed_overrides = _parse_parser_overrides(parser_overrides)
    _validate_overrides_against_default_chunking(
        namespace=namespace,
        project=project,
        strategy_name=getattr(dataset_config, "data_processing_strategy", None),
        parser_overrides=parsed_overrides,
    )

    if len(files) > 100:
        raise HTTPException(
            status_code=400,
            detail="Bulk upload limited to 100 files",
        )

    # Bulk uploads default to no processing unless explicitly requested.
    # We deliberately do NOT inherit dataset.auto_process here to avoid surprising
    # large batch ingestions. Precedence: explicit param > False.
    effective_auto_process = auto_process if auto_process is not None else False

    uploaded = 0
    skipped = 0
    failed = 0
    added_hashes: list[str] = []

    for file in files:
        try:
            was_added, metadata_file_content = await DatasetService.add_file_to_dataset(
                namespace=namespace,
                project=project,
                dataset=dataset,
                file=file,
            )
        except Exception as exc:
            logger.warning(
                "Failed to upload file",
                dataset=dataset,
                filename=file.filename,
                error=str(exc),
            )
            failed += 1
            continue

        if was_added:
            uploaded += 1
            added_hashes.append(metadata_file_content.hash)
        else:
            skipped += 1

    task_id = None
    status = "uploaded"
    if effective_auto_process and added_hashes:
        launch = DatasetService.start_ingestion_for_hashes(
            namespace=namespace,
            project=project,
            dataset=dataset,
            file_hashes=added_hashes,
            parser_overrides=parsed_overrides,
        )
        task_id = launch.task_id
        status = "processing"

    return BulkDatasetDataUploadResponse(
        uploaded=uploaded,
        skipped=skipped,
        failed=failed,
        task_id=task_id,
        status=status,
    )


class DeleteDataResponse(BaseModel):
    file_hash: str
    deleted_chunks: int = Field(
        description="Number of chunks deleted from vector store"
    )


@router.delete(
    "/{dataset}/data/{file_hash}",
    operation_id="dataset_data_delete",
    summary="Delete a file from the dataset",
    description="Delete a file from the dataset and remove its chunks from the vector store.",
    responses={200: {"model": DeleteDataResponse}},
)
async def delete_data(
    namespace: str,
    project: str,
    dataset: str,
    file_hash: str,
):
    logger.bind(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file_hash=file_hash,
    )
    result = await DatasetService.remove_file_from_dataset(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file_hash=file_hash,
    )

    return DeleteDataResponse(
        file_hash=file_hash,
        deleted_chunks=result.get("deleted_count", 0),
    )
