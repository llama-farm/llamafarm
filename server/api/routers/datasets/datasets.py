from enum import Enum

from config.datamodel import Dataset
from fastapi import APIRouter, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from api.routers.datasets._models import ListDatasetsResponse
from core.logging import FastAPIStructLogger
from services.dataset_service import DatasetService

logger = FastAPIStructLogger()

router = APIRouter(
    prefix="/projects/{namespace}/{project}/datasets",
    tags=["datasets"],
)


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


class DatasetActionType(str, Enum):
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
    namespace: str, project: str, dataset: str, request: DatasetActionRequest
):
    logger.bind(namespace=namespace, project=project, dataset=dataset)

    action_type = request.action_type

    def task_uri(task_id: str):
        return (
            f"http://localhost:8000/v1/projects/{namespace}/{project}/tasks/{task_id}"
        )

    if action_type == DatasetActionType.PROCESS:
        launch = DatasetService.start_dataset_ingestion(namespace, project, dataset)
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


@router.post(
    "/{dataset}/data",
    operation_id="dataset_data_upload",
    summary="Upload a file to the dataset",
    description=(
        "Upload a file to the dataset (stores it but does NOT process into vector database. "
        "Use the dataset actions endpoint with the 'process' action_type to process the file into the vector database)"
    ),
    responses={200: {"model": DatasetDataUploadResponse}},
)
async def upload_data(
    namespace: str,
    project: str,
    dataset: str,
    file: UploadFile,
):
    """Upload a file to the dataset (stores it but does NOT process into vector database)"""
    logger.bind(namespace=namespace, project=project, dataset=dataset)

    was_added, metadata_file_content = await DatasetService.add_file_to_dataset(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file=file,
    )

    if was_added:
        logger.info(
            "File uploaded to dataset",
            dataset=dataset,
            filename=file.filename,
            hash=metadata_file_content.hash,
        )
    else:
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
        skipped=not was_added,
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


# ============================================================================
# ROUTER UTTERANCES ENDPOINTS
# ============================================================================


class RouterUtteranceInput(BaseModel):
    """Input model for a single router utterance."""

    text: str = Field(..., description="The query text")
    route: str | None = Field(None, description="Optional route label")


class AppendUtterancesRequest(BaseModel):
    """Request to append utterances to a dataset."""

    utterances: list[RouterUtteranceInput] = Field(
        ..., description="List of utterances to append", min_length=1
    )


class AppendUtterancesResponse(BaseModel):
    """Response from appending utterances."""

    count: int = Field(..., description="Number of utterances appended")


class RouterUtteranceOutput(BaseModel):
    """Output model for a router utterance."""

    text: str
    route: str | None


class GetUtterancesResponse(BaseModel):
    """Response containing utterances."""

    utterances: list[RouterUtteranceOutput]
    total: int = Field(..., description="Total count of utterances matching filter")


@router.post(
    "/{dataset}/utterances",
    operation_id="dataset_append_utterances",
    tags=["router", "mcp"],
    summary="Append utterances to a router dataset",
    description=(
        "Append utterances to a router dataset for semantic router training. "
        "Creates the utterances file if it doesn't exist. "
        "Each utterance has a 'text' (required) and 'route' (optional) field."
    ),
    responses={200: {"model": AppendUtterancesResponse}},
)
async def append_utterances(
    namespace: str,
    project: str,
    dataset: str,
    request: AppendUtterancesRequest,
):
    """Append utterances to a router dataset."""
    from services.router_dataset_service import RouterDatasetService, RouterUtterance

    logger.bind(namespace=namespace, project=project, dataset=dataset)

    # Convert input to service model
    utterances = [
        RouterUtterance(text=u.text, route=u.route) for u in request.utterances
    ]

    result = RouterDatasetService.append_utterances(
        namespace=namespace,
        project=project,
        dataset=dataset,
        utterances=utterances,
    )

    logger.info(
        "Appended utterances to dataset",
        dataset=dataset,
        count=result["count"],
    )

    return AppendUtterancesResponse(count=result["count"])


@router.get(
    "/{dataset}/utterances",
    operation_id="dataset_get_utterances",
    tags=["router", "mcp"],
    summary="Get utterances from a router dataset",
    description=(
        "Get utterances from a router dataset. "
        "Supports pagination and filtering by route."
    ),
    responses={200: {"model": GetUtterancesResponse}},
)
async def get_utterances(
    namespace: str,
    project: str,
    dataset: str,
    offset: int = Query(0, ge=0, description="Number of utterances to skip"),
    limit: int | None = Query(None, ge=1, le=1000, description="Max utterances to return"),
    route: str | None = Query(None, description="Filter by route name"),
):
    """Get utterances from a router dataset."""
    from services.router_dataset_service import RouterDatasetService

    logger.bind(namespace=namespace, project=project, dataset=dataset)

    # Get total count for response
    total = RouterDatasetService.get_utterance_count(
        namespace=namespace,
        project=project,
        dataset=dataset,
        route_filter=route,
    )

    # Get utterances with pagination
    utterances = RouterDatasetService.get_utterances(
        namespace=namespace,
        project=project,
        dataset=dataset,
        offset=offset,
        limit=limit,
        route_filter=route,
    )

    return GetUtterancesResponse(
        utterances=[
            RouterUtteranceOutput(text=u.text, route=u.route) for u in utterances
        ],
        total=total,
    )


class DeleteUtterancesResponse(BaseModel):
    """Response from deleting utterances."""

    deleted: int = Field(..., description="Number of utterances deleted")


@router.delete(
    "/{dataset}/utterances",
    operation_id="dataset_delete_utterances",
    tags=["router"],
    summary="Delete all utterances from a router dataset",
    description="Delete all utterances from a router dataset.",
    responses={200: {"model": DeleteUtterancesResponse}},
)
async def delete_utterances(
    namespace: str,
    project: str,
    dataset: str,
):
    """Delete all utterances from a router dataset."""
    from services.router_dataset_service import RouterDatasetService

    logger.bind(namespace=namespace, project=project, dataset=dataset)

    result = RouterDatasetService.delete_utterances(
        namespace=namespace,
        project=project,
        dataset=dataset,
    )

    logger.info(
        "Deleted utterances from dataset",
        dataset=dataset,
        deleted=result["deleted"],
    )

    return DeleteUtterancesResponse(deleted=result["deleted"])
