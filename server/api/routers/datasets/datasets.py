from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from core.celery.tasks import process_dataset_task
from core.logging import FastAPIStructLogger
from services.data_service import DataService, FileExistsInAnotherDatasetError
from services.dataset_service import Dataset, DatasetService
from services.project_service import ProjectService
from services.rag_subprocess import ingest_file_with_rag

logger = FastAPIStructLogger()

router = APIRouter(
    prefix="/projects/{namespace}/{project}/datasets",
    tags=["datasets"],
)


class ListDatasetsResponse(BaseModel):
    total: int
    datasets: list[Dataset]


@router.get("/", response_model=ListDatasetsResponse)
async def list_datasets(namespace: str, project: str):
    logger.bind(namespace=namespace, project=project)
    datasets = DatasetService.list_datasets(namespace, project)
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
        hash=metadata_file_content.hash
    )
    
    return {"filename": file.filename, "hash": metadata_file_content.hash, "processed": False}


class ProcessDatasetResponse(BaseModel):
    processed_files: int
    skipped_files: int
    failed_files: int
    details: list[dict]


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
            detail="Dataset missing data_processing_strategy or database configuration"
        )
    
    # Process each file in the dataset
    processed = 0
    skipped = 0
    failed = 0
    details = []
    
    for file_hash in dataset_config.files or []:
        data_path = f"{project_dir}/lf_data/raw/{file_hash}"
        
        # Check if file exists
        import os
        if not os.path.exists(data_path):
            logger.warning("File not found", hash=file_hash, path=data_path)
            failed += 1
            details.append({
                "hash": file_hash,
                "status": "failed",
                "error": "File not found"
            })
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
        
        # Process the file
        ok = ingest_file_with_rag(
            project_dir=project_dir,
            project_config=project_obj.config,
            data_processing_strategy_name=data_processing_strategy_name,
            database_name=database_name,
            source_path=data_path,
        )
        
        if ok:
            processed += 1
            details.append({
                "hash": file_hash,
                "status": "processed"
            })
        else:
            failed += 1
            details.append({
                "hash": file_hash,
                "status": "failed",
                "error": "Ingestion failed"
            })
    
    logger.info(
        "Dataset processing complete",
        dataset=dataset,
        processed=processed,
        skipped=skipped,
        failed=failed
    )
    
    return ProcessDatasetResponse(
        processed_files=processed,
        skipped_files=skipped,
        failed_files=failed,
        details=details
    )


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
