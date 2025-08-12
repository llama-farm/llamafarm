from contextlib import contextmanager

from config.datamodel import LlamaFarmConfig
from fastapi import APIRouter
from pydantic import BaseModel

from api.models import ErrorResponse
from core.settings import settings as global_settings
from services.project_service import ProjectService


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

@contextmanager
def override_schema_template(template: str | None):
    original = getattr(global_settings, "lf_schema_template", "default")
    try:
        if template:
            global_settings.lf_schema_template = template
        yield
    finally:
        global_settings.lf_schema_template = original

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
    with override_schema_template(request.schema_template):
        project = ProjectService.create_project(namespace, request.name)
    return CreateProjectResponse(
      project=Project(
        namespace=namespace,
        name=request.name,
        config=project,
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