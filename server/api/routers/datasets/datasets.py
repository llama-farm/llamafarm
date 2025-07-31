from fastapi import APIRouter
from pydantic import BaseModel
from services.project_service import ProjectService

router = APIRouter(
  prefix="/projects/{namespace}/{project}/datasets",
  tags=["datasets"],
)

class Dataset(BaseModel):
    name: str
    parser: str
    files: list[str]

class ListDatasetsResponse(BaseModel):
    total: int
    datasets: list[Dataset]

@router.get("/", response_model=ListDatasetsResponse)
async def list_datasets(namespace: str, project: str):
    project_config = ProjectService.load_config(namespace, project)
    datasets_config = project_config.get("datasets", [])
    datasets = [
      Dataset(
        name=dataset["name"],
        parser=dataset["parser"],
        files=dataset["files"],
      )
      for i, dataset in enumerate(datasets_config)
    ]

    return ListDatasetsResponse(
      total=len(datasets),
      datasets=datasets,
    )