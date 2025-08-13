import sys
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from api.errors import ErrorResponse
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
    config_template: str | None = None

class CreateProjectResponse(BaseModel):
    project: Project

class GetProjectResponse(BaseModel):
    project: Project

class DeleteProjectResponse(BaseModel):
    project: Project

class UpdateProjectRequest(BaseModel):
    # Full replacement update of the project's configuration
    config: LlamaFarmConfig

class UpdateProjectResponse(BaseModel):
    project: Project

router = APIRouter(
  prefix="/projects",
  tags=["projects"],
)

@router.get(
    "/{namespace}",
    response_model=ListProjectsResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
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

@router.post(
    "/{namespace}",
    response_model=CreateProjectResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_project(namespace: str, request: CreateProjectRequest):
    cfg = ProjectService.create_project(
        namespace, request.name, request.config_template
    )
    return CreateProjectResponse(
      project=Project(
        namespace=namespace,
        name=request.name,
        config=cfg,
      ),
    )

@router.get(
    "/{namespace}/{project_id}",
    response_model=GetProjectResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_project(namespace: str, project_id: str):
    project = ProjectService.get_project(namespace, project_id)
    return GetProjectResponse(
      project=Project(
        namespace=project.namespace,
        name=project.name,
        config=project.config,
      ),
    )

@router.put(
    "/{namespace}/{project_id}",
    response_model=UpdateProjectResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def update_project(
    namespace: str,
    project_id: str,
    request: UpdateProjectRequest,
):
    updated_config = ProjectService.update_project(
        namespace,
        project_id,
        request.config,
    )
    return UpdateProjectResponse(
        project=Project(
            namespace=namespace,
            name=project_id,
            config=updated_config,
        )
    )

@router.delete(
    "/{namespace}/{project_id}",
    response_model=DeleteProjectResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
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