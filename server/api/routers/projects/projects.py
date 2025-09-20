import builtins
import sys
import threading
import uuid
from pathlib import Path

import celery.result
from atomic_agents import AtomicAgent
from fastapi import APIRouter, Header, HTTPException, Response
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from agents.project_chat_orchestrator import ProjectChatOrchestratorAgentFactory
from api.errors import ErrorResponse
from api.routers.inference.models import ChatRequest

# RAG imports moved to function level to avoid circular imports
from api.routers.shared.response_utils import (
    create_streaming_response_from_iterator,
    set_session_header,
)
from core.celery import app
from services.project_chat_service import project_chat_service
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
        projects=[
            Project(
                namespace=namespace,
                name=project.name,
                config=project.config,
            )
            for project in projects
        ],
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
    project_dir = ProjectService.get_project_dir(namespace, project_id)
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

    if request.stream:
        return create_streaming_response_from_iterator(
            request,
            project_chat_service.stream_chat(
                project_dir=project_dir,
                project_config=project_config,
                chat_agent=agent,
                message=latest_user_message,
                rag_enabled=request.rag_enabled,
                database=request.database,
                rag_top_k=request.rag_top_k,
                rag_score_threshold=request.rag_score_threshold,
            ),
            session_id,
        )

    try:
        completion = await project_chat_service.chat(
            project_dir=project_dir,
            project_config=project_config,
            chat_agent=agent,
            message=latest_user_message,
            rag_enabled=request.rag_enabled,
            database=request.database,
            rag_top_k=request.rag_top_k,
            rag_score_threshold=request.rag_score_threshold,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat service failed to generate a response: {e}",
        ) from e

    set_session_header(response, session_id)
    return completion


@router.post(
    "/{namespace}/{project_id}/rag/query",
    responses={
        404: {"model": ErrorResponse, "description": "Database or strategy not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def rag_query(
    namespace: str,
    project_id: str,
    request: dict,  # Using dict to avoid circular import, will validate inside function
):
    """Perform a RAG query on the project's configured databases."""
    # Import here to avoid circular import
    from api.routers.rag.rag_query import QueryRequest, QueryResponse, handle_rag_query

    # Validate request
    request = QueryRequest(**request)
    # Get project configuration
    project_service = ProjectService()
    project_dir = project_service.get_project_dir(namespace, project_id)

    if not Path(project_dir).exists():
        raise HTTPException(
            status_code=404, detail=f"Project {namespace}/{project_id} not found"
        )

    project_config = ProjectService.load_config(namespace, project_id)

    if not project_config:
        raise HTTPException(
            status_code=500, detail="Failed to load project configuration"
        )

    # Handle the RAG query
    response = await handle_rag_query(request, project_config, str(project_dir))

    return response


@router.get("/{namespace}/{project_id}/tasks/{task_id}")
async def get_task(namespace: str, project_id: str, task_id: str):
    """Return state, progress meta, and result/error if available."""
    res: celery.result.AsyncResult = app.AsyncResult(task_id)

    payload = {
        "task_id": task_id,
        "state": res.state,
        "meta": None,
        "result": None,
        "error": None,
        "traceback": None,
    }

    if res.info:
        payload["meta"] = res.info

    if res.state == "SUCCESS":
        payload["result"] = res.result
    elif res.state == "FAILURE":
        payload["error"] = str(res.result)
        payload["traceback"] = res.traceback

    return payload
