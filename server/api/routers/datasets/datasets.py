from config.datamodel import Dataset
from fastapi import APIRouter, HTTPException, Query, UploadFile
from pydantic import BaseModel

from core.celery.tasks import process_dataset_task
from core.logging import FastAPIStructLogger
from services.data_service import DataService, FileExistsInAnotherDatasetError
from services.dataset_service import DatasetService, DatasetWithFileDetails
from services.project_service import ProjectService
from services.rag_subprocess import ingest_file_with_rag

logger = FastAPIStructLogger()

router = APIRouter(
    prefix="/projects/{namespace}/{project}/datasets",
    tags=["datasets"],
)


class ListDatasetsResponse(BaseModel):
    total: int
    datasets: list[Dataset | DatasetWithFileDetails]


@router.get("/", response_model=ListDatasetsResponse)
async def list_datasets(
    namespace: str,
    project: str,
    include_extra_details: bool = Query(
        True, description="Include detailed file information with original filenames"
    ),
):
    logger.bind(namespace=namespace, project=project)
    if include_extra_details:
        detailed_datasets = DatasetService.list_datasets_with_file_details(
            namespace, project
        )
        datasets = [
            DatasetWithFileDetails(
                name=ds.name,
                data_processing_strategy=ds.data_processing_strategy,
                files=ds.files,
                database=ds.database,
                details=ds.details,
            )
            for ds in detailed_datasets
        ]
    else:
        # Backward compatibility: return old format for CLI
        basic_datasets = DatasetService.list_datasets(namespace, project)
        datasets = [
            Dataset(
                name=ds.name,
                database=ds.database,
                data_processing_strategy=ds.data_processing_strategy,
                files=ds.files,
            )
            for ds in basic_datasets
        ]

    return ListDatasetsResponse(
        total=len(datasets),
        datasets=datasets,
    )


class AvailableStrategiesResponse(BaseModel):
    data_processing_strategies: list[str]
    databases: list[str]


@router.get("/strategies", response_model=AvailableStrategiesResponse)
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


@router.post("/", response_model=CreateDatasetResponse)
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


@router.delete("/{dataset}", response_model=DeleteDatasetResponse)
async def delete_dataset(namespace: str, project: str, dataset: str):
    logger.bind(namespace=namespace, project=project)
    try:
        deleted_dataset = DatasetService.delete_dataset(
            namespace=namespace, project=project, name=dataset
        )
        return DeleteDatasetResponse(dataset=deleted_dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class DatasetActionRequest(BaseModel):
    action_type: str


@router.post("/{dataset}/actions")
async def actions(
    namespace: str, project: str, dataset: str, request: DatasetActionRequest
):
    logger.bind(namespace=namespace, project=project, dataset=dataset)

    action_type = request.action_type

    def task_uri(task_id: str):
        return (
            f"http://localhost:8000/v1/projects/{namespace}/{project}/tasks/{task_id}"
        )

    if action_type == "ingest":
        task = process_dataset_task.delay(namespace, project, dataset)
        return {
            "message": "Accepted",
            "task_uri": task_uri(task.id),
        }
    else:
        raise HTTPException(
            status_code=400, detail=f"Invalid action type: {action_type}"
        )


@router.post("/{dataset}/data")
async def upload_data(
    namespace: str,
    project: str,
    dataset: str,
    file: UploadFile,
):
    """Upload a file to the dataset (stores it but does NOT process into vector database)"""
    logger.bind(namespace=namespace, project=project, dataset=dataset)
    metadata_file_content = await DataService.add_data_file(
        namespace=namespace,
        project_id=project,
        file=file,
    )

    DatasetService.add_file_to_dataset(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file=metadata_file_content,
    )

    logger.info(
        "File uploaded to dataset",
        dataset=dataset,
        filename=file.filename,
        hash=metadata_file_content.hash,
    )

    return {
        "filename": file.filename,
        "hash": metadata_file_content.hash,
        "processed": False,
    }


class FileProcessingDetail(BaseModel):
    hash: str
    filename: str | None = None
    status: str  # processed, skipped, failed
    parser: str | None = None
    extractors: list[str] | None = None
    chunks: int | None = None
    chunk_size: int | None = None
    embedder: str | None = None
    error: str | None = None
    reason: str | None = None  # For skipped files (e.g., "duplicate")


class ProcessDatasetResponse(BaseModel):
    processed_files: int
    skipped_files: int
    failed_files: int
    strategy: str | None = None
    database: str | None = None
    details: list[FileProcessingDetail]


@router.post("/{dataset}/process", response_model=ProcessDatasetResponse)
async def process_dataset(
    namespace: str,
    project: str,
    dataset: str,
):
    """Process all unprocessed files in the dataset into the vector database"""
    logger.bind(namespace=namespace, project=project, dataset=dataset)

    # Get project and dataset configuration
    project_obj = ProjectService.get_project(namespace, project)
    project_dir = ProjectService.get_project_dir(namespace, project)

    dataset_config = next(
        (ds for ds in (project_obj.config.datasets or []) if ds.name == dataset),
        None,
    )

    if dataset_config is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")

    data_processing_strategy_name = dataset_config.data_processing_strategy
    database_name = dataset_config.database

    if not data_processing_strategy_name or not database_name:
        raise HTTPException(
            status_code=400,
            detail="Dataset missing data_processing_strategy or database configuration",
        )

    # Process each file in the dataset
    processed = 0
    skipped = 0
    failed = 0
    details = []

    import os

    # Safely construct the raw data directory path and validate containment
    raw_data_dir = os.path.normpath(os.path.join(project_dir, "lf_data", "raw"))
    abs_raw_data_dir = os.path.abspath(raw_data_dir)

    # Validate that raw_data_dir is inside project_dir
    abs_project_dir = os.path.abspath(project_dir)
    if not abs_raw_data_dir.startswith(abs_project_dir + os.sep):
        logger.error(
            "Raw data directory path traversal attempt", raw_data_dir=raw_data_dir
        )
        raise HTTPException(
            status_code=400, detail="Invalid raw data directory (security violation)"
        )

    for file_hash in dataset_config.files or []:
        # Safely construct and validate data path to prevent path traversal
        data_path = os.path.normpath(os.path.join(raw_data_dir, file_hash))
        abs_data_path = os.path.abspath(data_path)

        # Validate that the data path is within the raw_data_dir
        if not abs_data_path.startswith(abs_raw_data_dir + os.sep):
            logger.warning(
                "Path traversal attempt detected", hash=file_hash, path=data_path
            )
            failed += 1
            details.append(
                FileProcessingDetail(
                    hash=file_hash,
                    filename=None,
                    status="failed",
                    error="Invalid file path (security violation)",
                )
            )
            continue

        # Use the validated absolute path for all operations
        data_path = abs_data_path

        # Check if file exists
        if not os.path.exists(data_path):
            logger.warning("File not found", hash=file_hash, path=data_path)
            failed += 1
            details.append(
                FileProcessingDetail(
                    hash=file_hash,
                    filename=None,
                    status="failed",
                    error="File not found",
                )
            )
            continue

        # Check if already processed (by checking if hash exists as document ID in vector store)
        # This will be handled inside ingest_file_with_rag with duplicate detection

        logger.info(
            "Processing file into vector database",
            hash=file_hash,
            dataset=dataset,
            data_processing_strategy=data_processing_strategy_name,
            database=database_name,
        )

        # Get metadata for the file to get filename
        filename = None
        file_size = 0
        try:
            from server.services.data_service import DataService

            metadata = DataService.get_data_file_metadata_by_hash(
                namespace=namespace,
                project_id=project,
                file_content_hash=file_hash,
            )
            filename = metadata.filename
            # Get file size (data_path already validated above)
            file_size = os.path.getsize(data_path)
        except:
            filename = os.path.basename(data_path)
            # Get file size (data_path already validated above)
            file_size = os.path.getsize(data_path)

        logger.info(
            f"Processing file: {filename} ({file_hash[:8]}...) - {file_size} bytes"
        )

        # Process the file
        ok, file_details = ingest_file_with_rag(
            project_dir=project_dir,
            project_config=project_obj.config,
            data_processing_strategy_name=data_processing_strategy_name,
            database_name=database_name,
            source_path=data_path,
            filename=filename,
            dataset_name=dataset,  # Pass dataset name for logging
        )

        # Determine actual status based on file_details
        # Check multiple indicators for duplicates
        is_duplicate = (
            file_details.get("reason") == "duplicate"
            or file_details.get("status") == "skipped"
            or (
                file_details.get("stored_count", 0) == 0
                and file_details.get("skipped_count", 0) > 0
            )
        )

        if is_duplicate:
            status = "skipped"
            skipped += 1
            logger.info(f"File {filename} marked as SKIPPED (duplicate)")
        elif ok:
            status = "processed"
            processed += 1
        else:
            status = "failed"
            failed += 1

        # Create detailed response
        detail = FileProcessingDetail(
            hash=file_hash,
            filename=filename or file_details.get("filename"),
            status=status,
            parser=file_details.get("parser"),
            extractors=file_details.get("extractors"),
            chunks=file_details.get("chunks"),
            chunk_size=file_details.get("chunk_size"),
            embedder=file_details.get("embedder"),
            error=file_details.get("error") if status == "failed" else None,
            reason=file_details.get("reason"),
        )

        details.append(detail)

    logger.info(
        "Dataset processing complete",
        dataset=dataset,
        processed=processed,
        skipped=skipped,
        failed=failed,
    )

    # Add log file location info
    log_info = None
    try:
        import sys
        import os

        # Add rag module to path if needed
        rag_path = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
            )
        )
        if rag_path not in sys.path:
            sys.path.insert(0, rag_path)

        from rag.core.processing_logger import ProcessingLogger

        log_files = ProcessingLogger.get_latest_logs(project_dir, dataset)
        if log_files:
            log_info = f"Processing logs saved to: {log_files[0]}"
            logger.info(log_info)
    except Exception as e:
        logger.debug(f"Could not get log info: {e}")

    response = ProcessDatasetResponse(
        processed_files=processed,
        skipped_files=skipped,
        failed_files=failed,
        strategy=data_processing_strategy_name,
        database=database_name,
        details=details,
    )

    # Add log location to response summary if available
    if log_info:
        print(f"\n📝 {log_info}")

    return response


@router.delete("/{dataset}/data/{file_hash}")
async def delete_data(
    namespace: str,
    project: str,
    dataset: str,
    file_hash: str,
    remove_from_disk: bool = False,
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
    if remove_from_disk:
        try:
            metadata_file_content = DataService.get_data_file_metadata_by_hash(
                namespace=namespace,
                project_id=project,
                file_content_hash=file_hash,
            )

            DataService.delete_data_file(
                namespace=namespace,
                project_id=project,
                dataset=dataset,
                file=metadata_file_content,
            )
        except FileNotFoundError:
            pass
        except FileExistsInAnotherDatasetError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return {"file_hash": file_hash}
