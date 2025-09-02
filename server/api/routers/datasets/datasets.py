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


class CreateDatasetRequest(BaseModel):
    name: str
    rag_strategy: str


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
            rag_strategy=request.rag_strategy,
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
async def ingest_data(
    namespace: str,
    project: str,
    dataset: str,
    file: UploadFile,
):
    logger.bind(namespace=namespace, project=project, dataset=dataset)
    metadata_file_content = await DataService.add_data_file(
        namespace=namespace,
        project_id=project,
        file=file,
    )

    # Call the rag subsystem to ingest the file into the vector store
    # Resolve on-disk path for the newly saved file
    project_dir = ProjectService.get_project_dir(namespace, project)
    # replicate DataService.get_data_dir() path assembly; avoid class constants
    data_path = f"{project_dir}/lf_data/raw/{metadata_file_content.hash}"

    # Load project config and pick dataset strategy (default to first strategy)
    project_obj = ProjectService.get_project(namespace, project)
    strategy_name = (
        next(
            (
                ds.rag_strategy
                for ds in project_obj.config.datasets
                if ds.name == dataset
            ),
            None,
        )
        or "default"
    )
    strategy = next(
        (s for s in project_obj.config.rag.strategies if s.name == strategy_name),
        project_obj.config.rag.strategies[0],
    )

    logger.info(
        "Ingesting file into RAG",
        path=data_path,
        dataset=dataset,
        strategy=strategy.name,
    )
    ok = ingest_file_with_rag(strategy, data_path)
    if not ok:
        logger.error("RAG ingest failed", path=data_path)

    DatasetService.add_file_to_dataset(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file=metadata_file_content,
    )
    return {"filename": file.filename}


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
