from enum import Enum

from config.datamodel import Dataset
from fastapi import APIRouter, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from api.routers.datasets._models import ListDatasetsResponse
from core.logging import FastAPIStructLogger
from services.dataset_service import DatasetService
from services.project_service import ProjectService

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


class DatasetActionRequest(BaseModel):
    action_type: DatasetActionType = Field(
        ..., description="The type of action to execute"
    )


class DatasetActionResponse(BaseModel):
    message: str = Field(..., description="The status message")
    task_uri: str = Field(..., description="The URI for tracking the task")
    task_id: str = Field(..., description="The Celery task ID")


@router.post(
    "/{dataset}/actions",
    operation_id="dataset_actions",
    summary="Execute an action on a dataset",
    description="""Execute an action on a dataset
    - INGEST: Process all files in the dataset using the configured data processing strategy
    - PROCESS: Process all files in the dataset using the configured data processing strategy
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

    if action_type in [DatasetActionType.PROCESS]:
        launch = DatasetService.start_dataset_ingestion(namespace, project, dataset)
        return {
            "message": launch.message,
            "task_uri": task_uri(launch.task_id),
            "task_id": launch.task_id,
        }
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


@router.delete("/{dataset}/data/{file_hash}")
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
    DatasetService.remove_file_from_dataset(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file_hash=file_hash,
    )

    return {"file_hash": file_hash}


@router.post(
    "/{dataset}/cleanup/{file_hash}",
    operation_id="dataset_cleanup_file",
    tags=["datasets", "mcp"],
    summary="Manually cleanup chunks for a specific file",
    description="Manually cleanup chunks for a specific file. Useful for recovery when automatic cleanup fails.",
)
async def cleanup_file_chunks(
    namespace: str,
    project: str,
    dataset: str,
    file_hash: str,
) -> dict:
    """
    Manually cleanup chunks for a specific file.

    Useful for recovery when automatic cleanup fails.

    Args:
        namespace: Project namespace
        project: Project name
        dataset: Dataset name
        file_hash: Hash of the file to cleanup

    Returns:
        Cleanup result with deleted chunk count
    """
    logger.bind(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file_hash=file_hash,
    )

    try:
        # Get project and dataset config
        project_obj = ProjectService.get_project(namespace, project)
        project_dir = ProjectService.get_project_dir(namespace, project)

        dataset_config = next(
            (ds for ds in (project_obj.config.datasets or []) if ds.name == dataset),
            None,
        )

        if not dataset_config:
            raise HTTPException(
                status_code=404, detail=f"Dataset '{dataset}' not found"
            )

        database_name = dataset_config.database
        if not database_name:
            raise HTTPException(
                status_code=400, detail=f"Dataset '{dataset}' has no database configured"
            )

        # Get database configuration
        if not project_obj.config.rag or not project_obj.config.rag.databases:
            raise HTTPException(
                status_code=400, detail="No databases configured in project"
            )

        database_config = next(
            (
                db
                for db in project_obj.config.rag.databases
                if db.name == database_name
            ),
            None,
        )

        if not database_config:
            raise HTTPException(
                status_code=404, detail=f"Database '{database_name}' not found"
            )

        # Initialize vector store
        import importlib
        from pathlib import Path

        vector_store_config = database_config.config
        vector_store_type = (
            database_config.type.value
            if hasattr(database_config.type, "value")
            else str(database_config.type)
        )

        store_name_lower = vector_store_type.replace("Store", "_store").lower()
        module_path = f"rag.components.stores.{store_name_lower}"

        try:
            module = importlib.import_module(module_path)
            store_class = getattr(module, vector_store_type)
            vector_store = store_class(
                config=vector_store_config, project_dir=Path(project_dir)
            )
        except (ImportError, AttributeError) as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize vector store: {e}"
            ) from e

        # Initialize document manager and delete chunks
        from rag.core.document_manager import DocumentManager, DeletionStrategy

        doc_manager = DocumentManager(
            vector_store=vector_store,
            config={"enable_soft_delete": False},  # Hard delete for cleanup
        )

        result = doc_manager.delete_documents(
            document_hashes=[file_hash],
            strategy=DeletionStrategy.HARD_DELETE,
        )

        deleted_count = result.get("deleted_count", 0)
        errors = result.get("errors", [])

        if errors:
            logger.warning(
                f"Errors during manual cleanup: {errors}",
                file_hash=file_hash,
            )

        return {
            "message": f"Deleted {deleted_count} chunk(s) for file {file_hash[:12]}...",
            "deleted_count": deleted_count,
            "file_hash": file_hash,
            "errors": errors if errors else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Manual cleanup failed for {file_hash[:12]}...: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Cleanup failed: {str(e)}"
        ) from e
