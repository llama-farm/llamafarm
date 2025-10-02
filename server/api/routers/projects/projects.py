import builtins
import contextlib
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import celery.result  # type: ignore
from atomic_agents import AtomicAgent  # type: ignore
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
from core.settings import settings
from services.project_chat_service import (
    FALLBACK_ECHO_RESPONSE,
    project_chat_service,
)
from services.project_service import ProjectService
from services.docs_context_service import get_docs_service

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


SESSION_TTL_SECONDS = 30 * 60


@dataclass
class SessionRecord:
    namespace: str
    project_id: str
    agent: AtomicAgent
    created_at: float
    last_used: float
    request_count: int


agent_sessions: builtins.dict[str, SessionRecord] = {}
_agent_sessions_lock = threading.RLock()


def _session_key(namespace: str, project_id: str, session_id: str) -> str:
    return f"{namespace}:{project_id}:{session_id}"


def _cleanup_expired_sessions(now: float | None = None) -> None:
    timestamp = now or time.time()
    to_delete: list[str] = []
    for key, record in agent_sessions.items():
        if timestamp - record.last_used > SESSION_TTL_SECONDS:
            to_delete.append(key)
    for key in to_delete:
        agent_sessions.pop(key, None)


def _delete_session(namespace: str, project_id: str, session_id: str) -> bool:
    key = _session_key(namespace, project_id, session_id)
    record = agent_sessions.pop(key, None)
    if record is None:
        return False
    if hasattr(record.agent, "reset_history"):
        with contextlib.suppress(Exception):
            record.agent.reset_history()
    return True


def _delete_all_sessions(namespace: str, project_id: str) -> int:
    to_delete = [
        key
        for key, record in agent_sessions.items()
        if record.namespace == namespace and record.project_id == project_id
    ]
    for key in to_delete:
        record = agent_sessions.pop(key, None)
        if record and hasattr(record.agent, "reset_history"):
            with contextlib.suppress(Exception):
                record.agent.reset_history()
    return len(to_delete)


@router.post(
    "/{namespace}/{project_id}/chat/completions", response_model=ChatCompletion
)
async def chat(
    request: ChatRequest,
    namespace: str,
    project_id: str,
    response: Response,
    session_id: str | None = Header(None, alias="X-Session-ID"),
    x_no_session: str | None = Header(None, alias="X-No-Session"),
):
    """Send a message to the chat agent"""
    project_dir = ProjectService.get_project_dir(namespace, project_id)
    project_config = ProjectService.load_config(namespace, project_id)

    now = time.time()
    stateless = x_no_session is not None

    if stateless:
        # Stateless mode: create throwaway agent without session or persistence
        agent = ProjectChatOrchestratorAgentFactory.create_agent(
            project_config, project_dir=project_dir
        )
    else:
        # Stateful mode: use or create cached agent with disk-persisted history
        if not session_id:
            session_id = str(uuid.uuid4())

        key = _session_key(namespace, project_id, session_id)
        with _agent_sessions_lock:
            # Clean up expired sessions before checking cache
            _cleanup_expired_sessions(now)

            record = agent_sessions.get(key)
            if record is not None and (now - record.last_used > SESSION_TTL_SECONDS):
                # Session expired, remove it and create fresh
                agent_sessions.pop(key, None)
                record = None

            if record is None:
                # Create new agent and enable persistence
                agent = ProjectChatOrchestratorAgentFactory.create_agent(
                    project_config, project_dir=project_dir, session_id=session_id
                )
                # Cache the agent in memory
                agent_sessions[key] = SessionRecord(
                    namespace=namespace,
                    project_id=project_id,
                    agent=agent,
                    created_at=now,
                    last_used=now,
                    request_count=1,
                )
            else:
                # Reuse cached agent and update stats
                record.last_used = now
                record.request_count += 1
                agent = record.agent

        set_session_header(response, session_id)

    # Extract the latest user message
    latest_user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user" and msg.content:
            latest_user_message = msg.content
            break

    # If no user message, check if this is a greeting request (new session)
    if latest_user_message is None:
        raise HTTPException(status_code=400, detail="No user message provided")  # noqa: F821

    # Inject relevant documentation based on user query (dev mode only)
    if (
        settings.lf_dev_mode_docs_enabled
        and project_id == "project_seed"
        and hasattr(agent, "docs_context_provider")
    ):
        docs_service = get_docs_service()
        matched_docs = docs_service.match_docs_for_query(latest_user_message)
        agent.docs_context_provider.set_docs(matched_docs)

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
            session_id if not stateless else "",
            default_message=FALLBACK_ECHO_RESPONSE,
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

    if not stateless:
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
    from api.routers.rag.rag_query import QueryRequest, handle_rag_query

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


@router.get(
    "/{namespace}/{project_id}/chat/sessions/{session_id}/history",
    responses={
        200: {"model": dict},
        404: {"model": ErrorResponse},
    },
)
async def get_chat_session_history(namespace: str, project_id: str, session_id: str):
    """Retrieve the chat history for a specific session."""
    try:
        project_dir = ProjectService.get_project_dir(namespace, project_id)
        project_config = ProjectService.load_config(namespace, project_id)

        agent = ProjectChatOrchestratorAgentFactory.create_agent(
            project_config, project_dir=project_dir, session_id=session_id
        )
        history = agent.history.get_history()

        return {"messages": history}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load session history: {e}"
        ) from e


@router.delete(
    "/{namespace}/{project_id}/chat/sessions/{session_id}",
    responses={
        200: {"model": dict},
        404: {"model": ErrorResponse},
    },
)
async def delete_chat_session(namespace: str, project_id: str, session_id: str):
    # Delete in-memory record if present
    with _agent_sessions_lock:
        _delete_session(namespace, project_id, session_id)
    # Delete on-disk history directory
    with contextlib.suppress(Exception):
        project_dir = ProjectService.get_project_dir(namespace, project_id)
        sessions_dir = Path(project_dir) / "sessions" / session_id
        if sessions_dir.exists():
            shutil.rmtree(sessions_dir, ignore_errors=True)
    return {"message": f"Session {session_id} deleted"}


@router.delete(
    "/{namespace}/{project_id}/chat/sessions",
    responses={200: {"model": dict}},
)
async def delete_all_chat_sessions(namespace: str, project_id: str):
    with _agent_sessions_lock:
        count = _delete_all_sessions(namespace, project_id)
    return {"message": f"Deleted {count} session(s)", "count": count}
