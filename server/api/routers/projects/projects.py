from fastapi import APIRouter
from pydantic import BaseModel
from services.project_service import ProjectService
from config import LlamaFarmConfig

class Project(BaseModel):
    namespace: str;
    name: str;
    config: LlamaFarmConfig;

class ListProjectsResponse(BaseModel):
    total: int
    projects: list[Project]

class CreateProjectRequest(BaseModel):
    name: str

class CreateProjectResponse(BaseModel):
    project: Project

class GetProjectResponse(BaseModel):
    project: Project

class DeleteProjectResponse(BaseModel):
    project: Project

router = APIRouter(
  prefix="/projects",
  tags=["projects"],
)

@router.get("/{namespace}", response_model=ListProjectsResponse)
async def list_projects(namespace: str):
    projects = ProjectService.list_projects(namespace)
    return ListProjectsResponse(
      total=len(projects),
      projects=[Project(
        namespace=namespace,
        name=project.name,
        config=project.config,
      ) for project in projects],
    )

@router.post("/{namespace}", response_model=CreateProjectResponse)
async def create_project(namespace: str, request: CreateProjectRequest):
    project = ProjectService.create_project(namespace, request.name)
    return CreateProjectResponse(
      project=Project(
        namespace=namespace,
        name=request.name,
        config=project,
      ),
    )

@router.get("/{namespace}/{project_id}", response_model=GetProjectResponse)
async def get_project(namespace: str, project_id: str):
    project = ProjectService.get_project(namespace, project_id)
    return GetProjectResponse(
      project=Project(
        namespace=project.namespace,
        name=project.name,
        config=project.config,
      ),
    )

@router.delete("/{namespace}/{project_id}", response_model=DeleteProjectResponse)
async def delete_project(namespace: str, project_id: str):
    project = Project(
        id=project_id,
        name="test",
        description="test",
        namespace=namespace,
    )
    return DeleteProjectResponse(
      project=project,
    )