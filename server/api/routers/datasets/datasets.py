from enum import Enum

from config.datamodel import Dataset
from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.routers.datasets._models import ListDatasetsResponse
from core.logging import FastAPIStructLogger
from services.data_service import DataService
from services.dataset_service import (
    DatasetService,
    FileScanResult,
    ScanPageResult,
)
from services.dataset_service import (
    DocumentScanningBackend as ServiceDocumentScanningBackend,
)
from services.dataset_service import (
    DocumentScanningSettings as ServiceDocumentScanningSettings,
)

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


class DocumentScanningBackend(str, Enum):
    SURYA = "surya"
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"


class DocumentScanningSettings(BaseModel):
    """Document scanning (OCR) settings for a dataset"""

    enabled: bool = Field(
        default=False, description="Whether document scanning is enabled"
    )
    backend: DocumentScanningBackend = Field(
        default=DocumentScanningBackend.SURYA,
        description="OCR backend to use (surya recommended for best quality)",
    )
    language: str = Field(default="en", description="Language code for OCR")
    parse_by_page: bool = Field(
        default=True, description="Whether to return separate results per page"
    )


class CreateDatasetRequest(BaseModel):
    name: str
    data_processing_strategy: str
    database: str
    document_scanning: DocumentScanningSettings | None = None


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
    logger.info(f"Creating dataset with request: {request.model_dump()}")
    try:
        dataset = DatasetService.create_dataset(
            namespace=namespace,
            project=project,
            name=request.name,
            data_processing_strategy=request.data_processing_strategy,
            database=request.database,
        )
        # Save document scanning settings if provided
        logger.info(f"Document scanning settings: {request.document_scanning}")
        if request.document_scanning and request.document_scanning.enabled:
            # Convert router model to service model
            logger.info(f"Saving scan settings for dataset: {request.name}")
            service_settings = ServiceDocumentScanningSettings(
                enabled=request.document_scanning.enabled,
                backend=ServiceDocumentScanningBackend(
                    request.document_scanning.backend.value
                ),
                language=request.document_scanning.language,
                parse_by_page=request.document_scanning.parse_by_page,
            )
            logger.info(f"Service settings: {service_settings}")
            DatasetService.save_scan_settings(
                namespace, project, request.name, service_settings
            )
            logger.info("Scan settings saved successfully")
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


@router.get(
    "/{dataset}/data/{file_hash}/raw",
    operation_id="dataset_data_download",
    summary="Download raw file from dataset",
    description="Download the raw file content from the dataset by file hash.",
    responses={
        200: {"description": "Raw file content"},
        404: {"description": "File not found"},
    },
)
async def download_raw_file(
    namespace: str,
    project: str,
    dataset: str,
    file_hash: str,
):
    """Download raw file from dataset storage."""
    logger.bind(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file_hash=file_hash,
    )

    # Get dataset to find data directory
    ds = DatasetService.get_dataset(namespace, project, dataset)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")

    # Get file metadata to find the file path and mime type
    metadata = DataService.get_data_file_metadata_by_hash(
        namespace, project, dataset, file_hash
    )
    if metadata is None:
        raise HTTPException(
            status_code=404, detail=f"File '{file_hash}' not found in dataset"
        )

    # Construct path to raw file
    import os

    from core.settings import settings

    data_dir = os.path.join(settings.project_dir, "lf_data", "datasets", dataset)
    raw_file_path = os.path.join(data_dir, "raw", file_hash)

    if not os.path.exists(raw_file_path):
        raise HTTPException(status_code=404, detail="Raw file not found on disk")

    return FileResponse(
        path=raw_file_path,
        filename=metadata.original_file_name,
        media_type=metadata.mime_type,
    )


# =============================================================================
# Document Scanning Endpoints
# =============================================================================


class ScanSettingsResponse(BaseModel):
    """Response with document scanning settings for a dataset"""

    enabled: bool = Field(..., description="Whether document scanning is enabled")
    backend: str = Field(..., description="OCR backend")
    language: str = Field(..., description="Language code")
    parse_by_page: bool = Field(..., description="Whether to parse by page")


@router.get(
    "/{dataset}/scan-settings",
    operation_id="dataset_scan_settings_get",
    summary="Get document scanning settings for a dataset",
    description="Get the document scanning (OCR) settings configured for this dataset.",
    responses={
        200: {"model": ScanSettingsResponse},
        404: {"description": "Dataset not found or scanning not enabled"},
    },
)
async def get_scan_settings(namespace: str, project: str, dataset: str):
    """Get document scanning settings for a dataset."""
    logger.bind(namespace=namespace, project=project, dataset=dataset)

    settings = DatasetService.get_scan_settings(namespace, project, dataset)
    if settings is None:
        raise HTTPException(
            status_code=404, detail="Document scanning not enabled for this dataset"
        )

    return ScanSettingsResponse(
        enabled=settings.enabled,
        backend=settings.backend.value,
        language=settings.language,
        parse_by_page=settings.parse_by_page,
    )


class FileScanResultResponse(BaseModel):
    """Response with scan result for a file"""

    pages: list[dict] = Field(..., description="Array of page results")
    error: str | None = Field(None, description="Error message if scan failed")
    scanned_at: str | None = Field(None, description="Timestamp when scanned")
    backend: str | None = Field(None, description="Backend used for scanning")


@router.get(
    "/{dataset}/data/{file_hash}/scan",
    operation_id="dataset_file_scan_get",
    summary="Get scan result for a file",
    description="Get the OCR scan result for a file in the dataset.",
    responses={
        200: {"model": FileScanResultResponse},
        404: {"description": "File or scan result not found"},
    },
)
async def get_file_scan(namespace: str, project: str, dataset: str, file_hash: str):
    """Get scan result for a file in a dataset."""
    logger.bind(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file_hash=file_hash,
    )

    result = DatasetService.get_file_scan_result(namespace, project, dataset, file_hash)
    if result is None:
        raise HTTPException(status_code=404, detail="Scan result not found")

    return FileScanResultResponse(
        pages=[page.model_dump() for page in result.pages],
        error=result.error,
        scanned_at=result.scanned_at,
        backend=result.backend,
    )


class SaveScanResultRequest(BaseModel):
    """Request to save a scan result for a file"""

    pages: list[dict] = Field(
        ..., description="Array of page results with index, text, confidence"
    )
    backend: str | None = Field(None, description="Backend used for scanning")


class SaveScanResultResponse(BaseModel):
    """Response after saving a scan result"""

    file_hash: str = Field(..., description="The file hash")
    pages_saved: int = Field(..., description="Number of pages saved")


@router.post(
    "/{dataset}/data/{file_hash}/scan",
    operation_id="dataset_file_scan_save",
    summary="Save scan result for a file",
    description="Save the OCR scan result for a file in the dataset.",
    responses={200: {"model": SaveScanResultResponse}},
)
async def save_file_scan(
    namespace: str,
    project: str,
    dataset: str,
    file_hash: str,
    request: SaveScanResultRequest,
):
    """Save scan result for a file in a dataset."""
    logger.bind(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file_hash=file_hash,
    )

    from datetime import datetime

    # Convert request to FileScanResult
    pages = [
        ScanPageResult(
            index=page.get("index", i),
            text=page.get("text", ""),
            confidence=page.get("confidence", 0.0),
        )
        for i, page in enumerate(request.pages)
    ]

    result = FileScanResult(
        pages=pages,
        scanned_at=datetime.utcnow().isoformat(),
        backend=request.backend,
    )

    DatasetService.save_file_scan_result(namespace, project, dataset, file_hash, result)

    return SaveScanResultResponse(
        file_hash=file_hash,
        pages_saved=len(pages),
    )


@router.delete(
    "/{dataset}/data/{file_hash}/scan",
    operation_id="dataset_file_scan_delete",
    summary="Delete scan result for a file",
    description="Delete the OCR scan result for a file in the dataset.",
    responses={200: {"description": "Scan result deleted"}},
)
async def delete_file_scan(namespace: str, project: str, dataset: str, file_hash: str):
    """Delete scan result for a file in a dataset."""
    logger.bind(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file_hash=file_hash,
    )

    deleted = DatasetService.delete_file_scan_result(
        namespace, project, dataset, file_hash
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Scan result not found")

    return {"message": "Scan result deleted", "file_hash": file_hash}
