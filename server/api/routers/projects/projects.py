from fastapi import APIRouter
from pydantic import BaseModel

class Project(BaseModel):
    id: int
    name: str
    description: str
    namespace: str

class ListProjectsResponse(BaseModel):
    total: int
    projects: list[Project]

class CreateProjectRequest(BaseModel):
    name: str
    description: str

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
    return ListProjectsResponse(
      total=0,
      projects=[],
    )

@router.post("/{namespace}", response_model=CreateProjectResponse)
async def create_project(namespace: str, request: CreateProjectRequest):
    project = Project(
        id=1,
        name=request.name,
        description=request.description,
        namespace=namespace,
    )
    return CreateProjectResponse(
      project=project,
    )

@router.get("/{namespace}/{project_id}", response_model=GetProjectResponse)
async def get_project(namespace: str, project_id: int):
    project = Project(
        id=project_id,
        name="test",
        description="test",
        namespace=namespace,
    )
    return GetProjectResponse(
      project=project,
    )

@router.delete("/{namespace}/{project_id}", response_model=DeleteProjectResponse)
async def delete_project(namespace: str, project_id: int):
    project = Project(
        id=project_id,
        name="test",
        description="test",
        namespace=namespace,
    )
    return DeleteProjectResponse(
      project=project,
    )