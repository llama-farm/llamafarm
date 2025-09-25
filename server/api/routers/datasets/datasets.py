from config.datamodel import Dataset
from fastapi import APIRouter, HTTPException, Query, UploadFile, Form, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import glob
import json
import time
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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


# New models for smart ingest endpoint
class SmartIngestRequest(BaseModel):
    """Universal request model for all ingestion types"""
    paths: Optional[List[str]] = None  # Can be files, dirs, or patterns
    recursive: bool = False
    pattern: Optional[str] = None  # Additional filter for directories
    batch_size: int = 10
    parallel: bool = True


class IngestItem(BaseModel):
    """Represents a single item to ingest"""
    type: str  # "file", "directory", "pattern", "url", "upload"
    value: Any  # The actual path/pattern/url/UploadFile
    options: Dict[str, Any] = {}  # Type-specific options


class BatchIngestResponse(BaseModel):
    """Response model for batch operations"""
    total: int
    successful: int
    failed: int
    skipped: int
    results: List[Dict[str, Any]]
    processing_time: float
    detected_types: Dict[str, int]  # Count of each type processed


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


# ============================================================================
# NEW: Smart Unified Ingestion Endpoint
# ============================================================================

@router.post("/{dataset}/ingest", response_model=BatchIngestResponse)
async def smart_ingest(
    namespace: str,
    project: str,
    dataset: str,
    # Accept either files OR JSON request body
    files: Optional[List[UploadFile]] = File(None),
    request_body: Optional[str] = Form(None),  # JSON string with paths/patterns
    # Direct parameters for simple cases
    paths: Optional[str] = Form(None),  # Comma-separated paths
    recursive: bool = Form(False),
    pattern: Optional[str] = Form(None),
    parallel: bool = Form(True)
):
    """
    Smart unified endpoint that automatically detects and handles:
    - Direct file uploads (multipart/form-data)
    - File paths (local files on server)
    - Directories (with optional recursion and filtering)
    - Glob patterns (*.pdf, **/*.md, etc.)
    - Mixed inputs (combination of above)
    - URLs (future: for remote file ingestion)
    
    This endpoint processes files in two steps:
    1. Upload/identify files to process
    2. Automatically process them into the vector database
    """
    start_time = time.time()
    
    logger.bind(namespace=namespace, project=project, dataset=dataset)
    logger.info(f"Smart ingest started for dataset '{dataset}'")
    
    # Parse input based on what was provided
    items_to_process = []
    detected_types = {"files": 0, "directories": 0, "patterns": 0, "urls": 0}
    
    # Case 1: Direct file uploads from UI/CLI
    if files:
        detected_types["files"] = len(files)
        items_to_process.extend([
            IngestItem(type="upload", value=f, options={})
            for f in files
        ])
        logger.info(f"Detected {len(files)} file uploads")
    
    # Case 2: JSON request with paths/patterns
    elif request_body:
        try:
            data = json.loads(request_body)
            if isinstance(data, dict):
                # Structured request
                request = SmartIngestRequest(**data)
                paths_to_analyze = request.paths or []
                recursive = request.recursive
                pattern = request.pattern
            else:
                # Simple array of paths
                paths_to_analyze = data
        except:
            # Fallback to treating as comma-separated paths
            paths_to_analyze = request_body.split(',')
    
    # Case 3: Form data with paths (comma-separated)
    elif paths:
        paths_to_analyze = [p.strip() for p in paths.split(',')]
    else:
        # No input provided
        return BatchIngestResponse(
            total=0, successful=0, failed=0, skipped=0,
            results=[], processing_time=0, detected_types={}
        )
    
    # Analyze each path to determine its type
    if 'paths_to_analyze' in locals():
        for path_str in paths_to_analyze:
            item_type, expanded = await detect_and_expand_path(
                path_str, recursive, pattern
            )
            detected_types[item_type] += len(expanded) if isinstance(expanded, list) else 1
            
            if item_type == "pattern":
                # Glob pattern - expand on server
                items_to_process.extend([
                    IngestItem(type="file", value=f, options={"source": "pattern"})
                    for f in expanded
                ])
            elif item_type == "directory":
                # Directory - expand with filters
                items_to_process.extend([
                    IngestItem(type="file", value=f, options={"source": "directory"})
                    for f in expanded
                ])
            elif item_type == "url":
                # URL for remote ingestion (future feature)
                items_to_process.append(
                    IngestItem(type="url", value=path_str, options={})
                )
            else:
                # Regular file
                items_to_process.append(
                    IngestItem(type="file", value=path_str, options={})
                )
    
    logger.info(f"Total items to process: {len(items_to_process)}")
    logger.info(f"Detected types: {detected_types}")
    
    # Get dataset configuration
    project_obj = ProjectService.get_project(namespace, project)
    project_dir = ProjectService.get_project_dir(namespace, project)
    
    dataset_config = next(
        (ds for ds in (project_obj.config.datasets or []) if ds.name == dataset),
        None
    )
    
    if dataset_config is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")
    
    data_processing_strategy_name = dataset_config.data_processing_strategy
    database_name = dataset_config.database
    
    if not data_processing_strategy_name or not database_name:
        raise HTTPException(
            status_code=400,
            detail="Dataset missing data_processing_strategy or database configuration"
        )
    
    # Process all items
    results = []
    
    if parallel and len(items_to_process) > 1:
        # Parallel processing for multiple items
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for item in items_to_process:
                future = executor.submit(
                    process_single_item,
                    namespace, project, dataset,
                    item,
                    project_dir, project_obj.config,
                    data_processing_strategy_name,
                    database_name
                )
                futures.append(future)
            results = [future.result() for future in futures]
    else:
        # Sequential processing
        for item in items_to_process:
            result = await process_single_item_async(
                namespace, project, dataset,
                item,
                project_dir, project_obj.config,
                data_processing_strategy_name,
                database_name
            )
            results.append(result)
    
    # Aggregate results
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    
    processing_time = time.time() - start_time
    
    logger.info(
        f"Smart ingest complete - Total: {len(items_to_process)}, "
        f"Successful: {successful}, Failed: {failed}, Skipped: {skipped}, "
        f"Time: {processing_time:.2f}s"
    )
    
    return BatchIngestResponse(
        total=len(items_to_process),
        successful=successful,
        failed=failed,
        skipped=skipped,
        results=results,
        processing_time=processing_time,
        detected_types=detected_types
    )


# Helper functions for smart ingest

async def detect_and_expand_path(
    path_str: str,
    recursive: bool = False,
    filter_pattern: Optional[str] = None
) -> Tuple[str, List[str]]:
    """
    Detect the type of path and expand it to actual files.
    Returns: (type, list_of_files)
    """
    # Check if URL
    if path_str.startswith(('http://', 'https://', 'ftp://')):
        return ("urls", [path_str])
    
    # Check for glob patterns
    if any(char in path_str for char in ['*', '?', '[', ']']):
        # It's a glob pattern
        matches = glob.glob(path_str, recursive=recursive)
        files = [f for f in matches if Path(f).is_file()]
        return ("patterns", files)
    
    # Check if path exists
    path = Path(path_str)
    if path.exists():
        if path.is_file():
            return ("files", [str(path)])
        elif path.is_dir():
            # Directory - expand based on options
            if recursive:
                pattern = f"**/{filter_pattern or '*'}"
            else:
                pattern = filter_pattern or "*"
            
            matches = list(path.glob(pattern))
            files = [str(f) for f in matches if f.is_file()]
            return ("directories", files)
    
    # Path doesn't exist - might be a pattern without wildcards
    # or a file that will be created
    return ("files", [path_str])


def process_single_item(
    namespace: str, project: str, dataset: str,
    item: IngestItem,
    project_dir: str, project_config,
    data_processing_strategy_name: str,
    database_name: str
) -> Dict[str, Any]:
    """Process a single item (synchronous version for thread pool)"""
    try:
        if item.type == "upload":
            # Direct upload - need to save file first
            file = item.value
            # Read file content - handle both sync and async contexts
            import asyncio
            try:
                # Try to get the current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        content = pool.submit(lambda: asyncio.run(file.read())).result()
                else:
                    # We can run directly
                    content = asyncio.run(file.read())
            except RuntimeError:
                # No event loop, we can run directly
                content = asyncio.run(file.read())
            
            # Save to data directory using the existing synchronous method
            # Create a temporary UploadFile-like object with the content
            import io
            from fastapi import UploadFile
            
            # Create in-memory file
            file_like = io.BytesIO(content)
            temp_file = UploadFile(
                filename=file.filename,
                file=file_like
            )
            
            # Use the existing synchronous add_data_file method
            metadata_file_content = asyncio.run(DataService.add_data_file(
                namespace=namespace,
                project_id=project,
                file=temp_file
            ))
            
            # Add to dataset
            DatasetService.add_file_to_dataset(
                namespace=namespace,
                project=project,
                dataset=dataset,
                file=metadata_file_content
            )
            
            # Process into vector database
            import os
            project_dir_path = ProjectService.get_project_dir(namespace, project)
            data_path = os.path.join(project_dir_path, "lf_data", "raw", metadata_file_content.hash)
            ok, file_details = ingest_file_with_rag(
                project_dir=project_dir,
                project_config=project_config,
                data_processing_strategy_name=data_processing_strategy_name,
                database_name=database_name,
                source_path=data_path,
                filename=file.filename,
                dataset_name=dataset
            )
            
            status = "success" if ok else "error"
            if file_details.get("reason") == "duplicate":
                status = "skipped"
            
            return {
                "status": status,
                "filename": file.filename,
                "hash": metadata_file_content.hash,
                **file_details
            }
            
        elif item.type == "file":
            # Local file path - need to upload first
            path = Path(item.value)
            if not path.exists():
                return {
                    "status": "error",
                    "filename": path.name,
                    "error": "File not found"
                }
            
            with open(path, 'rb') as f:
                content = f.read()
            
            # Save to data directory using the existing method
            import io
            import asyncio
            from fastapi import UploadFile
            
            # Create in-memory file
            file_like = io.BytesIO(content)
            temp_file = UploadFile(
                filename=path.name,
                file=file_like
            )
            
            # Use the existing add_data_file method
            metadata_file_content = asyncio.run(DataService.add_data_file(
                namespace=namespace,
                project_id=project,
                file=temp_file
            ))
            
            # Add to dataset
            DatasetService.add_file_to_dataset(
                namespace=namespace,
                project=project,
                dataset=dataset,
                file=metadata_file_content
            )
            
            # Process into vector database
            import os
            project_dir_path = ProjectService.get_project_dir(namespace, project)
            data_path = os.path.join(project_dir_path, "lf_data", "raw", metadata_file_content.hash)
            ok, file_details = ingest_file_with_rag(
                project_dir=project_dir,
                project_config=project_config,
                data_processing_strategy_name=data_processing_strategy_name,
                database_name=database_name,
                source_path=data_path,
                filename=path.name,
                dataset_name=dataset
            )
            
            status = "success" if ok else "error"
            if file_details.get("reason") == "duplicate":
                status = "skipped"
            
            return {
                "status": status,
                "filename": path.name,
                "hash": metadata_file_content.hash,
                **file_details
            }
            
        elif item.type == "url":
            # Future: Download from URL
            return {"status": "error", "error": "URL ingestion not yet implemented"}
            
        else:
            return {"status": "error", "error": f"Unknown item type: {item.type}"}
            
    except Exception as e:
        logger.error(f"Error processing item {item.value}: {e}")
        filename = "unknown"
        if item.type == "file":
            filename = Path(item.value).name
        elif item.type == "upload":
            filename = item.value.filename
        return {
            "status": "error",
            "filename": filename,
            "error": str(e)
        }


async def process_single_item_async(
    namespace: str, project: str, dataset: str,
    item: IngestItem,
    project_dir: str, project_config,
    data_processing_strategy_name: str,
    database_name: str
) -> Dict[str, Any]:
    """Process a single item (async version)"""
    # For now, just wrap the sync version
    # In future, this could be optimized with async I/O
    return process_single_item(
        namespace, project, dataset,
        item,
        project_dir, project_config,
        data_processing_strategy_name,
        database_name
    )
