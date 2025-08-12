import sys
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from services.project_service import ProjectService

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
from config.datamodel import LlamaFarmConfig  # noqa: E402


class Project(BaseModel):
    namespace: str
    name: str
    config: LlamaFarmConfig

class ListProjectsResponse(BaseModel):
    total: int
    projects: list[Project]

class CreateProjectRequest(BaseModel):
    name: str
    schema_template: str | None = "default"

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
    # Optionally override the template for this request by temporarily
    # setting on settings
    from core.settings import settings as global_settings
    original_template = getattr(
        global_settings,
        "lf_schema_template",
        "default",
    )
    try:
        if request.schema_template:
            global_settings.lf_schema_template = request.schema_template
        project = ProjectService.create_project(namespace, request.name)
    finally:
        # restore
        global_settings.lf_schema_template = original_template
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
    # TODO: Implement actual delete in ProjectService; placeholder response for now
    project = Project(
        namespace=namespace,
        name=project_id,
        config=ProjectService.load_config(namespace, project_id),
    )
    return DeleteProjectResponse(
      project=project,
    )