import builtins
import sys
import threading
import time
import uuid
from pathlib import Path

from atomic_agents import AtomicAgent
from fastapi import APIRouter, Header, HTTPException, Response
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

<<<<<<< HEAD
from agents.project_chat_orchestrator import (
    ProjectChatOrchestratorAgentFactory,
    ProjectChatOrchestratorAgentInputSchema,
)
from api.routers.inference.models import ChatRequest
from api.routers.shared.response_utils import set_session_header
from services.project_chat_service import project_chat_service
=======
from api.errors import ErrorResponse
>>>>>>> main
from services.project_service import ProjectService

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
from config.datamodel import LlamaFarmConfig  # noqa: E402


class Project(BaseModel):
    namespace: str
    name: str
    config: LlamaFarmConfig
<<<<<<< HEAD

=======
>>>>>>> main

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

<<<<<<< HEAD
=======
class UpdateProjectRequest(BaseModel):
    # Full replacement update of the project's configuration
    config: LlamaFarmConfig

class UpdateProjectResponse(BaseModel):
    project: Project
>>>>>>> main

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
)

<<<<<<< HEAD

@router.get("/{namespace}", response_model=ListProjectsResponse)
=======
@router.get(
    "/{namespace}",
    response_model=ListProjectsResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
>>>>>>> main
async def list_projects(namespace: str):
    projects = ProjectService.list_projects(namespace)
    return ListProjectsResponse(
        total=len(projects),
        projects=[
            Project(
                namespace=namespace,
                name=project.name,
                config=project.config,
            )
            for project in projects
        ],
    )

<<<<<<< HEAD

@router.post("/{namespace}", response_model=CreateProjectResponse)
=======
@router.post(
    "/{namespace}",
    response_model=CreateProjectResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
>>>>>>> main
async def create_project(namespace: str, request: CreateProjectRequest):
    cfg = ProjectService.create_project(
        namespace, request.name, request.config_template
    )
    return CreateProjectResponse(
<<<<<<< HEAD
        project=Project(
            namespace=namespace,
            name=request.name,
            config=project,
        ),
    )


@router.get("/{namespace}/{project_id}", response_model=GetProjectResponse)
=======
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
>>>>>>> main
async def get_project(namespace: str, project_id: str):
    project = ProjectService.get_project(namespace, project_id)
    return GetProjectResponse(
        project=Project(
            namespace=project.namespace,
            name=project.name,
            config=project.config,
        ),
    )

<<<<<<< HEAD

@router.delete("/{namespace}/{project_id}", response_model=DeleteProjectResponse)
=======
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
>>>>>>> main
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


agent_sessions: builtins.dict[str, AtomicAgent] = {}
_agent_sessions_lock = threading.RLock()


@router.post(
    "/{namespace}/{project_id}/chat/completions", response_model=ChatCompletion
)
async def chat(
    request: ChatRequest,
    namespace: str,
    project_id: str,
    response: Response,
    session_id: str | None = Header(None, alias="X-Session-ID"),
):
    """Send a message to the chat agent"""
    project_config = ProjectService.load_config(namespace, project_id)

    # If no session ID provided, create a new one and ensure thread-safe session map access
    with _agent_sessions_lock:
        if not session_id or session_id not in agent_sessions:
            if not session_id:
                session_id = str(uuid.uuid4())
            agent = ProjectChatOrchestratorAgentFactory.create_agent(project_config)
            agent_sessions[session_id] = agent
        else:
            # Use existing agent to maintain conversation context
            agent = agent_sessions[session_id]

    # Extract the latest user message
    latest_user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user" and msg.content:
            latest_user_message = msg.content
            break
    if latest_user_message is None:
        raise HTTPException(status_code=400, detail="No user message provided")  # noqa: F821

    completion = project_chat_service.chat(
        project_config=project_config,
        chat_agent=agent,
        message=latest_user_message,
    )

    # Set session header
    set_session_header(response, session_id)

    return completion
