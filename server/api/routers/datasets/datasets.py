from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.dataset_service import DatasetService, Dataset

router = APIRouter(
  prefix="/projects/{namespace}/{project}/datasets",
  tags=["datasets"],
)

class ListDatasetsResponse(BaseModel):
    total: int
    datasets: list[Dataset]

@router.get("/", response_model=ListDatasetsResponse)
async def list_datasets(namespace: str, project: str):
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
    try:
        dataset = DatasetService.create_dataset(
            namespace=namespace,
            project=project,
            name=request.name,
            parser=request.parser
        )
        return CreateDatasetResponse(dataset=dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

class DeleteDatasetRequest(BaseModel):
    name: str

class DeleteDatasetResponse(BaseModel):
    dataset: Dataset

@router.delete("/", response_model=DeleteDatasetResponse)
async def delete_dataset(namespace: str, project: str, request: DeleteDatasetRequest):
    try:
        deleted_dataset = DatasetService.delete_dataset(
        namespace=namespace,
        project=project,
        name=request.name
        )
        return DeleteDatasetResponse(dataset=deleted_dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
