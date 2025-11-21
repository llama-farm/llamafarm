import asyncio
from enum import Enum

from config.datamodel import Dataset
from fastapi import APIRouter, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from api.routers.datasets._models import ListDatasetsResponse
from core.celery.tasks import process_dataset_task
from core.logging import FastAPIStructLogger
from services.data_service import DataService, FileExistsInAnotherDatasetError
from services.dataset_service import DatasetService, DatasetWithFileDetails
from services.project_service import ProjectService

logger = FastAPIStructLogger()

router = APIRouter(
    prefix="/projects/{namespace}/{project}/datasets",
    tags=["datasets"],
)


@router.get(
    "/",
    operation_id="dataset_list",
    tags=["mcp"],
    responses={200: {"model": ListDatasetsResponse}},
)
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
    INGEST = "ingest"  # alias for "process"
    PROCESS = Field(
        "process",
        description="Process all files in the dataset using the configured data processing strategy",
    )


class DatasetActionRequest(BaseModel):
    action_type: DatasetActionType = Field(
        ..., description="The type of action to execute"
    )


class DatasetActionResponse(BaseModel):
    message: str = Field(..., description="The status message")
    task_uri: str = Field(..., description="The URI for tracking the task")


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

    if action_type in [DatasetActionType.INGEST, DatasetActionType.PROCESS]:
        task = process_dataset_task.delay(namespace, project, dataset)
        return {
            "message": "Accepted",
            "task_uri": task_uri(task.id),
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
        "Use the dataset actions endpoint with the 'ingest' action_type to process the file into the vector database)"
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
    metadata_file_content = await DataService.add_data_file(
        namespace=namespace,
        project_id=project,
        file=file,
    )

    was_added = DatasetService.add_file_to_dataset(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file=metadata_file_content,
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
    message: str
    processed_files: int
    skipped_files: int
    failed_files: int
    strategy: str | None = None
    database: str | None = None
    details: list[FileProcessingDetail]
    task_id: str | None = None  # For async processing


class CleanupError(BaseModel):
    file_hash: str
    error: str


class CancelProcessingResponse(BaseModel):
    message: str
    task_id: str
    cancelled: bool
    pending_tasks_cancelled: int
    running_tasks_at_cancel: int
    files_reverted: int = 0
    files_failed_to_revert: int = 0
    errors: list[CleanupError] | None = None


@router.post("/{dataset}/process", response_model=ProcessDatasetResponse)
async def process_dataset(
    namespace: str,
    project: str,
    dataset: str,
    async_processing: bool = False,
):
    """Process all unprocessed files in the dataset into the vector database

    Args:
        async_processing: If True, use task chaining and return immediately with task info
    """
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

    # If async processing is requested, use task chaining
    if async_processing:
        from core.celery.tasks import create_dataset_processing_chain

        # Create task chain for all files
        task_chain = create_dataset_processing_chain(
            namespace=namespace,
            project=project,
            dataset=dataset,
            data_processing_strategy_name=data_processing_strategy_name,
            database_name=database_name,
            file_hashes=dataset_config.files or [],
        )

        # Execute the chain asynchronously
        result = task_chain.apply_async()

        # Save the group result so it can be queried later
        result.save()

        # Store child task IDs in the backend for tracking
        # This is needed because GroupResult.restore() doesn't always work with filesystem backend
        try:
            child_task_ids = [child.id for child in result.results]
        except Exception as e:
            logger.error(f"Error accessing group result children: {e}")
            # Fallback: create task IDs from the file list
            child_task_ids = []

        # Store metadata about this group task
        from core.celery import app as celery_app

        celery_app.backend.store_result(
            result.id,
            {
                "type": "group",
                "children": child_task_ids,
                "total_files": len(child_task_ids),
                "file_hashes": dataset_config.files or [],
            },
            "PENDING",  # Initial state
        )

        logger.info(
            "Started async dataset processing",
            task_id=result.id,
            file_count=len(dataset_config.files or []),
            child_task_ids=child_task_ids[:3],  # Log first 3 for debugging
        )

        # Return immediately with task information
        return ProcessDatasetResponse(
            message="Dataset processing started asynchronously",
            processed_files=0,
            skipped_files=0,
            failed_files=0,
            strategy=data_processing_strategy_name,
            database=database_name,
            details=[
                FileProcessingDetail(
                    hash=file_hash,
                    filename=None,
                    status="pending",
                    error=None,
                )
                for file_hash in dataset_config.files or []
            ],
            task_id=result.id,  # Add task ID for tracking
        )

    # Synchronous processing (existing behavior)
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
        except Exception:
            filename = os.path.basename(data_path)
            # Get file size (data_path already validated above)
            file_size = os.path.getsize(data_path)

        logger.info(
            f"Processing file: {filename} ({file_hash[:8]}...) - {file_size} bytes"
        )

        # Process the file using task chaining instead of direct call
        from core.celery.tasks import process_single_file_task

        # Use the file hash as the identifier
        task = process_single_file_task.delay(
            namespace=namespace,
            project=project,
            dataset=dataset,
            file_hash=file_hash,
            data_processing_strategy_name=data_processing_strategy_name,
            database_name=database_name,
        )

        # Wait for the task to complete using polling to avoid result.get() error
        timeout = 600  # 10 minutes
        poll_interval = 5  # seconds
        waited = 0

        try:
            while waited < timeout:
                try:
                    status = task.status
                    if status not in ("PENDING", "STARTED"):
                        break
                except Exception as e:
                    logger.error(
                        f"Error checking task status for file {file_hash}: {e}",
                        exc_info=True,
                    )
                    await asyncio.sleep(poll_interval)
                    waited += poll_interval
                    continue

                await asyncio.sleep(poll_interval)
                waited += poll_interval

            # Get final status with error handling
            try:
                final_status = task.status
            except Exception as e:
                logger.error(
                    f"Error getting final task status for file {file_hash}: {e}",
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get task status for file {file_hash}: {str(e)}",
                ) from e

            if final_status == "SUCCESS":
                try:
                    result = task.result
                    ok = result["success"]
                    file_details = result["details"]
                except Exception as e:
                    logger.error(
                        f"Error getting task result for file {file_hash}: {e}",
                        exc_info=True,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to get task result for file {file_hash}: {str(e)}",
                    ) from e
            elif final_status == "FAILURE":
                # Handle task failure
                try:
                    error_result = task.result
                    logger.error(f"Task failed for file {file_hash}: {error_result}")
                    error_message = str(error_result)
                except Exception as e:
                    logger.error(
                        f"Error getting failure details for file {file_hash}: {e}",
                        exc_info=True,
                    )
                    error_message = "Unknown error (couldn't access failure details)"

                ok = False
                file_details = {
                    "filename": filename,
                    "error": error_message,
                    "parser": None,
                    "extractors": [],
                    "chunks": None,
                    "chunk_size": None,
                    "embedder": None,
                    "reason": None,
                    "result": None,
                }
            else:
                # Timeout or other status
                logger.error(
                    f"Task timed out or failed for file {file_hash}: status={final_status}"
                )
                ok = False
                file_details = {
                    "filename": filename,
                    "error": f"Task timed out or failed with status: {final_status}",
                    "parser": None,
                    "extractors": [],
                    "chunks": None,
                    "chunk_size": None,
                    "embedder": None,
                    "reason": None,
                    "result": None,
                }
        except Exception as e:
            logger.error(f"Unexpected error for file {file_hash}: {e}")
            ok = False
            file_details = {
                "filename": filename,
                "error": str(e),
                "parser": None,
                "extractors": [],
                "chunks": None,
                "chunk_size": None,
                "embedder": None,
                "reason": None,
                "result": None,
            }

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
        processed_files=processed,
        skipped_files=skipped,
        failed_files=failed,
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
        message="Dataset processing completed",
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


@router.post(
    "/{dataset}/process/cancel",
    response_model=CancelProcessingResponse,
    operation_id="dataset_process_cancel",
    tags=["datasets", "mcp"],
    summary="Cancel ongoing dataset processing",
    description="Cancel processing for a dataset. This will stop pending tasks and prevent new tasks from starting.",
)
async def cancel_dataset_processing(
    namespace: str,
    project: str,
    dataset: str,
    task_id: str | None = None,
) -> CancelProcessingResponse:
    """
    Cancel ongoing dataset processing.

    Args:
        namespace: Project namespace
        project: Project name
        dataset: Dataset name
        task_id: Optional task ID. If not provided, finds active task for dataset.

    Returns:
        Cancellation status with task details

    Raises:
        HTTPException: 404 if no active task found, 400 if task already completed, 500 for other errors
    """
    from datetime import datetime
    from celery.result import AsyncResult

    logger.bind(namespace=namespace, project=project, dataset=dataset)

    try:
        # Import celery app
        from core.celery import app as celery_app

        # For Phase 1, task_id is required (frontend already has it)
        # In future phases, we could look up active tasks per dataset
        if not task_id:
            raise HTTPException(
                status_code=404, detail="No active processing task found. task_id is required."
            )

        # Get group task
        group_result: AsyncResult = celery_app.AsyncResult(task_id)

        # Get stored metadata - try multiple approaches
        result_meta = None

        # Approach 1: Try to get from result when state is PENDING
        # This is how metadata is stored in the existing code
        try:
            if (
                group_result.state == "PENDING"
                and isinstance(group_result.result, dict)
                and group_result.result.get("type") == "group"
            ):
                result_meta = group_result.result
        except Exception as e:
            logger.debug(f"Could not get metadata from PENDING result: {e}")

        # Approach 2: Try accessing result property directly for other states
        # Sometimes the metadata is still accessible even if state changed
        if not result_meta:
            try:
                result_value = group_result.result
                if isinstance(result_value, dict) and result_value.get("type") == "group":
                    result_meta = result_value
            except Exception as e:
                logger.debug(f"Could not get metadata from result property: {e}")

        # Approach 3: Try to access via backend's _get_task_meta_for method if available
        # This is a fallback for filesystem backend
        if not result_meta:
            try:
                # Some backends support this method
                if (
                    hasattr(celery_app.backend, "_get_task_meta_for")
                    and (meta := celery_app.backend._get_task_meta_for(task_id))
                    and isinstance(meta.get("result"), dict)
                    and meta["result"].get("type") == "group"
                ):
                    result_meta = meta["result"]
            except Exception as e:
                logger.debug(f"Could not get metadata via backend method: {e}")

        if not result_meta or result_meta.get("type") != "group":
            raise HTTPException(
                status_code=404, detail="Task not found or not a group task"
            )

        # Check if already cancelled
        if result_meta.get("cancelled"):
            # Already cancelled, return current state
            child_task_ids = result_meta.get("children", [])
            pending_count = 0
            running_count = 0

            # Count current states
            for child_id in child_task_ids:
                child_result = celery_app.AsyncResult(child_id)
                try:
                    if child_result.state == "PENDING":
                        pending_count += 1
                    elif child_result.state == "STARTED":
                        running_count += 1
                except Exception:
                    pass

            return CancelProcessingResponse(
                message="Processing already cancelled",
                task_id=task_id,
                cancelled=True,
                pending_tasks_cancelled=pending_count,
                running_tasks_at_cancel=running_count,
            )

        # Check if already completed
        if group_result.state in ("SUCCESS", "FAILURE"):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel: processing already {group_result.state.lower()}",
            )

        # Get child task IDs
        child_task_ids = result_meta.get("children", [])

        # Revoke child tasks
        pending_cancelled = 0
        running_count = 0

        for child_id in child_task_ids:
            try:
                child_result = celery_app.AsyncResult(child_id)

                # Get current state (may raise exception if task doesn't exist)
                try:
                    child_state = child_result.state
                except Exception:
                    # Task may not exist or be inaccessible, skip it
                    continue

                if child_state == "PENDING":
                    # Revoke pending tasks (prevent from starting)
                    celery_app.control.revoke(child_id, terminate=False)
                    pending_cancelled += 1
                    logger.info(f"Revoked pending task: {child_id}")
                elif child_state == "STARTED":
                    # Revoke running tasks (graceful - let current work finish)
                    celery_app.control.revoke(child_id, terminate=False)
                    running_count += 1
                    logger.info(f"Revoked running task: {child_id}")
            except Exception as e:
                logger.warning(f"Error revoking child task {child_id}: {e}")
                # Continue with other tasks

        # Update group metadata with cancellation flag
        result_meta["cancelled"] = True
        result_meta["cancelled_at"] = datetime.now().isoformat()

        # Store updated metadata
        # Use the current state or "CANCELLED" if backend supports it
        current_state = group_result.state if hasattr(group_result, "state") else "PENDING"
        celery_app.backend.store_result(
            task_id,
            result_meta,
            current_state,  # Keep original state, cancellation is tracked in metadata
        )

        logger.info(
            "Processing cancellation requested",
            task_id=task_id,
            pending_cancelled=pending_cancelled,
            running_count=running_count,
        )

        # Trigger cleanup for successfully processed files
        cleanup_result = {
            "files_reverted": 0,
            "files_failed_to_revert": 0,
            "errors": None,
        }

        try:
            from services.dataset_cleanup_service import DatasetCleanupService

            cleanup_service = DatasetCleanupService()
            cleanup_result = cleanup_service.cleanup_processed_files(
                namespace, project, dataset, task_id
            )

            # Update metadata with cleanup results
            result_meta["cleanup_status"] = cleanup_result

        except Exception as e:
            logger.error(
                f"Error during cleanup (cancellation still succeeded): {e}",
                exc_info=True,
            )
            # Don't fail cancellation if cleanup fails
            cleanup_result["errors"] = [{"file_hash": "unknown", "error": str(e)}]

        # Store updated metadata with cleanup status
        current_state = group_result.state if hasattr(group_result, "state") else "PENDING"
        celery_app.backend.store_result(
            task_id,
            result_meta,
            current_state,
        )

        # Build response message
        if cleanup_result["files_failed_to_revert"] == 0:
            message = (
                f"Processing cancelled and {cleanup_result['files_reverted']} file(s) reverted"
                if cleanup_result["files_reverted"] > 0
                else "Processing cancelled (no files to revert)"
            )
        else:
            message = (
                f"Processing cancelled with cleanup issues: "
                f"{cleanup_result['files_reverted']} reverted, "
                f"{cleanup_result['files_failed_to_revert']} failed"
            )

        # Convert errors to CleanupError objects if present
        cleanup_errors = None
        if cleanup_result.get("errors"):
            cleanup_errors = [
                CleanupError(file_hash=e["file_hash"], error=e["error"])
                for e in cleanup_result["errors"]
            ]

        return CancelProcessingResponse(
            message=message,
            task_id=task_id,
            cancelled=True,
            pending_tasks_cancelled=pending_cancelled,
            running_tasks_at_cancel=running_count,
            files_reverted=cleanup_result["files_reverted"],
            files_failed_to_revert=cleanup_result["files_failed_to_revert"],
            errors=cleanup_errors,
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(
            f"Error cancelling processing for {dataset}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel processing: {str(e)}"
        ) from e


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
