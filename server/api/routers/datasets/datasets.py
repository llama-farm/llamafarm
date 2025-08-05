from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.errors import NotFoundError
from core.logging import FastAPIStructLogger
from services.data_service import DataService, FileExistsInAnotherDatasetError
from services.dataset_service import Dataset, DatasetService

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
    parser: str

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
            parser=request.parser
        )
        return CreateDatasetResponse(dataset=dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

class DeleteDatasetRequest(BaseModel):
    name: str

class DeleteDatasetResponse(BaseModel):
    dataset: Dataset

@router.delete("/", response_model=DeleteDatasetResponse)
async def delete_dataset(namespace: str, project: str, request: DeleteDatasetRequest):
    logger.bind(namespace=namespace, project=project)
    try:
        deleted_dataset = DatasetService.delete_dataset(
        namespace=namespace,
        project=project,
        name=request.name
        )
        return DeleteDatasetResponse(dataset=deleted_dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

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
    DatasetService.add_file_to_dataset(
        namespace=namespace,
        project=project,
        dataset=dataset,
        file=metadata_file_content,
    )
    return { "filename": file.filename }

@router.delete("/{dataset}/data/{file_hash}")
async def delete_data(
    namespace: str,
    project: str,
    dataset: str,
    file_hash: str,
    remove_from_disk: bool = False,
):
    logger.bind(namespace=namespace, project=project, dataset=dataset, file_hash=file_hash)
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

    return { "file_hash": file_hash }