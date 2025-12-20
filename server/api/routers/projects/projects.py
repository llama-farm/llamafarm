import builtins
import contextlib
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import celery.result
from config.datamodel import LlamaFarmConfig, Model  # noqa: E402
from fastapi import APIRouter, Header, HTTPException, Response
from fastapi import Path as FastAPIPath
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, Field

from agents.base.types import ToolDefinition
from agents.chat_orchestrator import ChatOrchestratorAgent, ChatOrchestratorAgentFactory
from api.errors import ErrorResponse, NamespaceNotFoundError, ProjectNotFoundError

# RAG imports moved to function level to avoid circular imports
from api.routers.shared.response_utils import (
    create_streaming_response_from_iterator,
    set_session_header,
)
from core.celery import app
from core.settings import settings
from services.docs_context_service import get_docs_service
from services.project_chat_service import (
    FALLBACK_ECHO_RESPONSE,
    project_chat_service,
)
from services.project_service import ProjectService


class Project(BaseModel):
    namespace: str = Field(..., description="The namespace of the project")
    name: str = Field(..., description="The name of the project")
    config: LlamaFarmConfig | dict = Field(
        ..., description="The configuration of the project"
    )
    validation_error: str | None = Field(
        None, description="Validation error message if config has issues"
    )
    last_modified: datetime | None = Field(
        None, description="Last modified timestamp of the project config"
    )


class ListProjectsResponse(BaseModel):
    total: int = Field(..., description="The total number of projects")
    projects: list[Project] = Field(..., description="The list of projects")


class CreateProjectRequest(BaseModel):
    name: str = Field(..., description="The name of the project")
    config_template: str | None = Field(
        None, description="The config template to use for the project"
    )


class CreateProjectResponse(BaseModel):
    project: Project = Field(..., description="The created project")


class GetProjectResponse(BaseModel):
    project: Project = Field(..., description="The project")


class DeleteProjectResponse(BaseModel):
    project: Project = Field(..., description="The deleted project")


class UpdateProjectRequest(BaseModel):
    config: LlamaFarmConfig = Field(
        ..., description="The full updated configuration of the project"
    )


class UpdateProjectResponse(BaseModel):
    project: Project = Field(..., description="The updated project")


class ModelResponse(Model):
    """Response for model API endpoint."""

    default: bool = Field(
        False,
        description="Whether this model is the default model in the runtime config",
    )


class ListModelsResponse(BaseModel):
    """Response for list models API endpoint."""

    total: int = Field(..., description="Total number of models")
    models: list[ModelResponse] = Field(..., description="List of models")


router = APIRouter(
    prefix="/projects",
    tags=["projects"],
)


@router.get(
    "/{namespace}",
    operation_id="projects_list",
    summary="List projects for a namespace",
    tags=["projects", "mcp"],
    responses={
        200: {"model": ListProjectsResponse},
    },
)
async def list_projects(
    namespace: str = FastAPIPath(
        ...,
        description='The namespace to list projects for. Use "default" for the default namespace.',
    ),
):
    # Use safe method to handle projects with validation errors
    # If namespace doesn't exist, return empty list instead of 404
    try:
        safe_projects = ProjectService.list_projects_safe(namespace)
    except NamespaceNotFoundError:
        # Namespace doesn't exist yet - return empty list (not an error)
        return ListProjectsResponse(total=0, projects=[])

    return ListProjectsResponse(
        total=len(safe_projects),
        projects=[
            Project(
                namespace=namespace,
                name=project.name,
                # Use validated config if available, otherwise use raw dict
                config=project.config
                if project.config is not None
                else project.config_dict,
                validation_error=project.validation_error,
                last_modified=project.last_modified,
            )
            for project in safe_projects
        ],
    )


@router.post(
    "/{namespace}",
    operation_id="project_create",
    summary="Create a project",
    tags=["projects", "mcp"],
    responses={
        200: {"model": CreateProjectResponse},
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
    operation_id="project_get",
    summary="Get a project",
    tags=["projects", "mcp"],
    responses={
        200: {"model": GetProjectResponse},
    },
)
async def get_project(namespace: str, project_id: str):
    # Use safe method to handle projects with validation errors
    safe_project = ProjectService.get_project_safe(namespace, project_id)
    return GetProjectResponse(
        project=Project(
            namespace=safe_project.namespace,
            name=safe_project.name,
            # Use validated config if available, otherwise use raw dict
            config=safe_project.config
            if safe_project.config is not None
            else safe_project.config_dict,
            validation_error=safe_project.validation_error,
            last_modified=safe_project.last_modified,
        ),
    )


@router.put(
    "/{namespace}/{project_id}",
    operation_id="project_update",
    summary="Update a project",
    tags=["projects", "mcp"],
    response_model=UpdateProjectResponse,
    responses={
        200: {"model": UpdateProjectResponse},
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
    operation_id="project_delete",
    summary="Delete a project",
    tags=["projects", "mcp"],
    responses={
        200: {"model": DeleteProjectResponse},
        404: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def delete_project(namespace: str, project_id: str):
    """
    Delete a project and all its associated resources.

    This endpoint performs a complete cleanup including:
    - All datasets associated with the project
    - All chat sessions
    - All data files (raw, metadata, and indexes)
    - Pending Celery tasks
    - The entire project directory

    Warning: This operation is irreversible.
    """
    try:
        from core.logging import FastAPIStructLogger

        logger = FastAPIStructLogger()

        # Call the delete_project method in ProjectService
        deleted_project = ProjectService.delete_project(namespace, project_id)

        # Clean up in-memory chat sessions to prevent memory leak
        with _agent_sessions_lock:
            session_count = _delete_all_sessions(namespace, project_id)
            if session_count > 0:
                logger.info(
                    "Cleared in-memory chat sessions during project deletion",
                    namespace=namespace,
                    project_id=project_id,
                    session_count=session_count,
                )

        # Revoke any pending Celery tasks for this project to prevent
        # processing failures on deleted projects
        try:
            # Note: This is a best-effort cleanup. Celery doesn't natively support
            # querying tasks by project, so we log this attempt for monitoring.
            logger.info(
                "Attempted to clean up Celery tasks for deleted project",
                namespace=namespace,
                project_id=project_id,
            )
            # If you have task IDs stored somewhere, you could revoke them here:
            # for task_id in stored_task_ids:
            #     AsyncResult(task_id, app=celery_app).revoke(terminate=True)
        except Exception as e:
            logger.warning(
                "Failed to clean up Celery tasks during project deletion",
                namespace=namespace,
                project_id=project_id,
                error=str(e),
            )

        # Convert the Project object to the API response format
        project = Project(
            namespace=deleted_project.namespace,
            name=deleted_project.name,
            config=deleted_project.config,
            validation_error=deleted_project.validation_error,
            last_modified=deleted_project.last_modified,
        )

        return DeleteProjectResponse(project=project)

    except ProjectNotFoundError as e:
        # Return 404 if project doesn't exist
        raise HTTPException(
            status_code=404, detail=f"Project {namespace}/{project_id} not found"
        ) from e
    except PermissionError as e:
        # Return 403 for permission issues
        raise HTTPException(
            status_code=403, detail=f"Permission denied: {str(e)}"
        ) from e
    except Exception as e:
        # Log the full error for debugging
        from core.logging import FastAPIStructLogger

        logger = FastAPIStructLogger()
        logger.error(
            "Failed to delete project",
            namespace=namespace,
            project_id=project_id,
            error=str(e),
            exc_info=True,
        )
        # Return 500 for other failures
        raise HTTPException(
            status_code=500, detail=f"Failed to delete project: {str(e)}"
        ) from e


SESSION_TTL_SECONDS = 30 * 60


@dataclass
class SessionRecord:
    namespace: str
    project_id: str
    agent: ChatOrchestratorAgent
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


class ChatRequest(BaseModel):
    messages: list[ChatCompletionMessageParam]
    model: str | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    metadata: dict | None = None
    n: int | None = None
    parallel_tool_calls: bool | None = None
    presence_penalty: float | None = None
    response_format: dict | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False  # Enable Server-Sent Events streaming
    stream_options: dict | None = None
    temperature: float | None = None
    tool_choice: str | dict | None = None
    tools: list[ChatCompletionToolParam] | None = None
    # tools: dict | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    user: str | None = None

    # LlamaFarm-specific extensions (not part of OpenAI API)
    rag_enabled: bool | None = None
    database: str | None = None
    rag_retrieval_strategy: str | None = None
    rag_top_k: int | None = None
    rag_score_threshold: float | None = None
    n_ctx: int | None = None  # Context window size for GGUF models (universal runtime)

    # Custom RAG query override
    rag_queries: list[str] | None = Field(
        default=None,
        description="Custom queries for RAG retrieval. Overrides using the user message. Can be a single query or multiple queries - results are merged and deduplicated.",
    )

    # Thinking/reasoning model parameters (Ollama-compatible, universal runtime)
    # Controls whether thinking models show their reasoning process
    think: bool | None = (
        None  # None = runtime default (off), True = enable, False = disable
    )
    # Maximum tokens for thinking process (default: 1024 when thinking enabled)
    # max_tokens is used for the final answer, thinking_budget is additional
    thinking_budget: int | None = None


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
    x_active_project: str | None = Header(None, alias="X-Active-Project"),
):
    """Send a message to the chat agent"""
    project_dir = ProjectService.get_project_dir(namespace, project_id)
    project_config = ProjectService.load_config(namespace, project_id)

    # Parse active project from header (format: "namespace/project")
    active_project_namespace = None
    active_project_name = None
    if x_active_project:
        parts = x_active_project.split("/", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            active_project_namespace, active_project_name = parts

    now = time.time()
    stateless = x_no_session is not None

    if stateless:
        agent = await ChatOrchestratorAgentFactory.create_agent(
            project_config=project_config,
            project_dir=project_dir,
            model_name=request.model,
            active_project_namespace=active_project_namespace,
            active_project_name=active_project_name,
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

            if record is None or request.model != record.agent.model_name:
                agent = await ChatOrchestratorAgentFactory.create_agent(
                    project_config=project_config,
                    project_dir=project_dir,
                    model_name=request.model,
                    session_id=session_id,
                    active_project_namespace=active_project_namespace,
                    active_project_name=active_project_name,
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
    latest_user_message = next(
        (
            str(msg.get("content", ""))
            for msg in reversed(request.messages)
            if msg.get("role", None) == "user" and msg.get("content", None)
        ),
        None,
    )

    # Inject relevant documentation based on user query (dev mode only)
    if (
        settings.lf_dev_mode_docs_enabled
        and project_id == "project_seed"
        and hasattr(agent, "docs_context_provider")
    ):
        docs_service = get_docs_service()
        matched_docs = docs_service.match_docs_for_query(latest_user_message)
        agent.docs_context_provider.set_docs(matched_docs)

    tools = [ToolDefinition.from_openai_tool_dict(t) for t in request.tools or []]

    if request.stream:
        return create_streaming_response_from_iterator(
            request,
            project_chat_service.stream_chat(
                project_dir=project_dir,
                project_config=project_config,
                chat_agent=agent,
                messages=request.messages,
                tools=tools,
                rag_enabled=request.rag_enabled,
                database=request.database,
                retrieval_strategy=request.rag_retrieval_strategy,
                rag_top_k=request.rag_top_k,
                rag_score_threshold=request.rag_score_threshold,
                n_ctx=request.n_ctx,
                rag_queries=request.rag_queries,
                think=request.think,
                thinking_budget=request.thinking_budget,
                max_tokens=request.max_tokens,
            ),
            session_id if not stateless else "",
            default_message=FALLBACK_ECHO_RESPONSE,
        )

    try:
        completion = await project_chat_service.chat(
            project_dir=project_dir,
            project_config=project_config,
            chat_agent=agent,
            messages=request.messages,
            tools=tools,
            rag_enabled=request.rag_enabled,
            database=request.database,
            retrieval_strategy=request.rag_retrieval_strategy,
            rag_top_k=request.rag_top_k,
            n_ctx=request.n_ctx,
            rag_score_threshold=request.rag_score_threshold,
            rag_queries=request.rag_queries,
            think=request.think,
            thinking_budget=request.thinking_budget,
            max_tokens=request.max_tokens,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat service failed to generate a response: {e}",
        ) from e

    if not stateless:
        set_session_header(response, session_id)
    return completion


class GetTaskResponse(BaseModel):
    task_id: str = Field(..., description="The ID of the asynchronous task")
    state: str = Field(
        ...,
        description="Current state of the task (e.g., PENDING, STARTED, SUCCESS, FAILURE)",
    )
    meta: dict | None = Field(
        None, description="Progress metadata or intermediate results, if available"
    )
    result: dict | str | None = Field(
        None, description="Result of the task if completed successfully"
    )
    error: str | None = Field(None, description="Error message if the task failed")
    traceback: str | None = Field(
        None, description="Traceback information if the task failed"
    )
    cancelled: bool = Field(
        default=False, description="Whether the task has been cancelled"
    )


class CleanupError(BaseModel):
    """Error details for a file cleanup failure."""

    file_hash: str = Field(..., description="Hash of the file that failed to clean up")
    error: str = Field(..., description="Error message describing the cleanup failure")


class CancelTaskResponse(BaseModel):
    """Response from cancelling a task."""

    message: str = Field(
        ...,
        description="Human-readable message about the cancellation",
        example="Task cancelled successfully",
    )

    task_id: str = Field(..., description="The ID of the cancelled task")

    cancelled: bool = Field(
        ..., description="Whether the task was successfully cancelled"
    )

    pending_tasks_cancelled: int = Field(
        default=0, description="Number of pending tasks that were cancelled"
    )

    running_tasks_at_cancel: int = Field(
        default=0, description="Number of running tasks at the time of cancellation"
    )

    files_reverted: int = Field(
        default=0, description="Number of files that were successfully reverted"
    )

    files_failed_to_revert: int = Field(
        default=0, description="Number of files that failed to revert"
    )

    errors: list[CleanupError] | None = Field(
        None, description="List of errors encountered during cleanup"
    )

    already_completed: bool = Field(
        default=False,
        description="True if the task had already completed before cancellation was requested",
    )

    already_cancelled: bool = Field(
        default=False,
        description="True if the task was already cancelled before this request",
    )


def _process_group_children(
    children: list, file_hashes: list, task_id: str, logger
) -> dict:
    """
    Process a list of Celery child tasks and return aggregated progress information.

    Args:
        children: List of AsyncResult objects
        file_hashes: List of file hashes corresponding to children
        task_id: Parent task ID for logging
        logger: Logger instance

    Returns:
        Dict with keys: total, completed, failed, successful, file_statuses
    """
    total = len(children)
    completed = sum(child.ready() for child in children)
    failed = sum(child.failed() for child in children)
    successful = sum(child.successful() for child in children)

    logger.info(
        "Group progress",
        task_id=task_id,
        total=total,
        completed=completed,
        failed=failed,
        successful=successful,
    )

    # Build per-file status details
    file_statuses = []

    # Validate file_hashes and children lengths match
    if len(file_hashes) != len(children):
        logger.warning(
            "Mismatch between file_hashes and children lengths",
            file_hashes_len=len(file_hashes),
            children_len=len(children),
            task_id=task_id,
        )

    # Aggregate counts across all files
    total_stored_count = 0
    total_skipped_count = 0

    for i, child in enumerate(children):
        # Use clear fallback filename if hash is not available
        if i < len(file_hashes) and file_hashes[i]:
            filename = file_hashes[i]
        else:
            filename = f"unknown_filename_{i}"

        file_status = {
            "index": i,
            "task_id": child.id,
            "state": "pending",
            "filename": filename,
            "error": None,
            "stored_count": None,
            "skipped_count": None,
        }

        if child.successful():
            file_status["state"] = "success"
            try:
                result_data = child.result
                # Handle tuple/list format: (success, details)
                if isinstance(result_data, (list, tuple)) and len(result_data) >= 2:
                    _success, details = result_data[0], result_data[1]
                    if isinstance(details, dict):
                        file_status["filename"] = details.get("filename", filename)
                        file_status["chunks"] = details.get("chunks")
                        # Extract stored_count and skipped_count from details or result
                        result_obj = details.get("result", {})
                        stored = details.get("stored_count") or (
                            result_obj.get("stored_count")
                            if isinstance(result_obj, dict)
                            else None
                        )
                        skipped = details.get("skipped_count") or (
                            result_obj.get("skipped_count")
                            if isinstance(result_obj, dict)
                            else None
                        )
                        file_status["stored_count"] = stored
                        file_status["skipped_count"] = skipped
                        if stored is not None:
                            total_stored_count += stored
                        if skipped is not None:
                            total_skipped_count += skipped
                elif isinstance(result_data, dict):
                    file_status["filename"] = result_data.get("file_hash", filename)
                    file_status["chunks"] = result_data.get("chunks_created")
                    # Extract stored_count and skipped_count from dict format
                    file_status["stored_count"] = result_data.get("stored_count")
                    file_status["skipped_count"] = result_data.get("skipped_count")
                    if file_status["stored_count"] is not None:
                        total_stored_count += file_status["stored_count"]
                    if file_status["skipped_count"] is not None:
                        total_skipped_count += file_status["skipped_count"]
            except Exception:
                pass
        elif child.failed():
            file_status["state"] = "failure"
            try:
                file_status["error"] = str(child.result)
            except Exception:
                file_status["error"] = "Unknown error"
        elif child.state == "STARTED":
            file_status["state"] = "processing"
        else:
            file_status["state"] = "pending"

        file_statuses.append(file_status)

    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "successful": successful,
        "file_statuses": file_statuses,
        "total_stored_count": total_stored_count,
        "total_skipped_count": total_skipped_count,
    }


@router.get(
    "/{namespace}/{project_id}/tasks/{task_id}",
    operation_id="task_get",
    tags=["tasks", "mcp"],
    summary="Get the status of an asynchronous task",
    description="Return state, progress meta, and result/error if available.",
    responses={200: {"model": GetTaskResponse}},
)
async def get_task(namespace: str, project_id: str, task_id: str):
    """Return state, progress meta, and result/error if available."""
    from celery.result import GroupResult

    from core.logging import FastAPIStructLogger

    logger = FastAPIStructLogger(__name__)
    logger.info("Checking task status", task_id=task_id)

    res: celery.result.AsyncResult = app.AsyncResult(task_id)

    logger.info("Task status", task_id=task_id, state=res.state, ready=res.ready())

    # Check for cancelled flag in metadata
    is_cancelled = False
    try:
        # Try to get metadata from result (works when state is PENDING)
        if res.state == "PENDING" and isinstance(res.result, dict):
            is_cancelled = res.result.get("cancelled", False)
        # Also try result property for other states
        if not is_cancelled:
            try:
                result_value = res.result
                if (
                    isinstance(result_value, dict)
                    and result_value.get("type") == "group"
                ):
                    is_cancelled = result_value.get("cancelled", False)
            except Exception:
                pass
        # Try backend's _get_task_meta_for method if available
        if not is_cancelled:
            try:
                if hasattr(app.backend, "_get_task_meta_for"):
                    meta = app.backend._get_task_meta_for(task_id)
                    if meta and isinstance(meta.get("result"), dict):
                        is_cancelled = meta["result"].get("cancelled", False)
            except Exception:
                pass
    except Exception:
        pass

    response = GetTaskResponse(
        task_id=task_id,
        state=res.state,
        meta=None,
        result=None,
        error=None,
        traceback=None,
        cancelled=is_cancelled,
    )

    # Check if this is a group result (parallel tasks)
    try:
        # First check if we stored group metadata manually
        # This is needed because GroupResult.restore() doesn't work well with filesystem backend
        group_info = (
            res.result
            if res.state == "PENDING"
            and isinstance(res.result, dict)
            and res.result.get("type") == "group"
            else None
        )

        # Also check for cancelled flag in group_info
        if group_info and group_info.get("cancelled"):
            is_cancelled = True
            response.cancelled = True

        if group_info and "children" in group_info:
            # SECURITY: Verify task belongs to the requested namespace/project
            task_namespace = group_info.get("namespace")
            task_project = group_info.get("project")

            if (
                task_namespace
                and task_project
                and (task_namespace != namespace or task_project != project_id)
            ):
                logger.warning(
                    "Authorization failed: task does not belong to requested namespace/project",
                    task_id=task_id,
                    task_namespace=task_namespace,
                    task_project=task_project,
                    requested_namespace=namespace,
                    requested_project=project_id,
                )
                raise HTTPException(status_code=404, detail="Task not found")

            # We have stored group metadata - query child tasks directly
            logger.info(
                "Found stored group metadata",
                task_id=task_id,
                child_count=len(group_info["children"]),
            )

            children = [
                app.AsyncResult(child_id) for child_id in group_info["children"]
            ]
            file_hashes = group_info.get("file_hashes", [])

            # Process children using helper function
            progress = _process_group_children(children, file_hashes, task_id, logger)
            total = progress["total"]
            completed = progress["completed"]
            file_statuses = progress["file_statuses"]

            # Determine overall state
            if completed == total:
                # Processing is complete - aggregate results from all tasks (successful AND failed)
                results = []
                for i, child in enumerate(children):
                    # Get file hash from file_hashes if available
                    file_hash = (
                        file_hashes[i] if i < len(file_hashes) else f"unknown_file_{i}"
                    )

                    if child.successful():
                        try:
                            result_data = child.result
                            # Inject file_hash into the result for frontend consumption
                            if (
                                isinstance(result_data, (list, tuple))
                                and len(result_data) >= 2
                            ):
                                # Format: [success, details] - inject hash into details
                                success_flag, details = result_data[0], result_data[1]
                                if isinstance(details, dict):
                                    details["file_hash"] = file_hash
                                results.append([success_flag, details])
                            elif isinstance(result_data, dict):
                                result_data["file_hash"] = file_hash
                                results.append(result_data)
                            else:
                                results.append(result_data)
                        except Exception:
                            pass
                    elif child.failed():
                        # Also collect failed task results
                        try:
                            error_info = str(child.result)
                            # Create a failed result entry
                            failed_result = {
                                "file_hash": file_hash,
                                "success": False,
                                "error": error_info,
                                "details": {
                                    "status": "failed",
                                    "filename": file_hash,
                                    "error": error_info,
                                },
                            }
                            results.append(failed_result)
                        except Exception as e:
                            logger.error(f"Error processing failed child result: {e}")

                # Count actual statuses from the results (don't rely on Celery's successful count)
                processed_count = 0
                skipped_count = 0
                failed_count = 0

                for result in results:
                    # New format: [success, info]
                    success, info = result[0], result[1]
                    if not success:
                        failed_count += 1
                        continue

                    status = (
                        info.get("status") or info.get("result", {}).get("status")
                        if isinstance(info, dict)
                        else None
                    )

                    if status == "skipped":
                        skipped_count += 1
                    else:
                        processed_count += 1

                # Set result with all file details (successful, skipped, and failed)
                response.result = {
                    "processed_files": processed_count,
                    "failed_files": failed_count,
                    "skipped_files": skipped_count,
                    "stored_count": progress.get("total_stored_count", 0),
                    "skipped_count": progress.get("total_skipped_count", 0),
                    "details": results,
                }

                # Set state based on whether any files failed
                if failed_count > 0:
                    response.state = "FAILURE"
                    response.error = f"{failed_count} of {total} tasks failed"
                else:
                    response.state = "SUCCESS"

                # Persist the final state to the backend so subsequent polls see the correct state
                # This is necessary because we manually stored group metadata with PENDING state
                app.backend.store_result(task_id, response.result, response.state)
                logger.info(
                    "Persisted final group task state",
                    task_id=task_id,
                    state=response.state,
                    processed=processed_count,
                    failed=failed_count,
                    skipped=skipped_count,
                )
            else:
                response.meta = {
                    "current": completed,
                    "total": total,
                    "progress": int((completed / total) * 100) if total > 0 else 0,
                    "message": f"Processing {completed}/{total} files",
                    "files": file_statuses,  # Include per-file details
                    "stored_count": progress.get("total_stored_count", 0),
                    "skipped_count": progress.get("total_skipped_count", 0),
                }
            return response

        # Fallback: Try to restore the group from the result backend
        group_res = GroupResult.restore(task_id, app=app)
        logger.info(
            "GroupResult.restore attempt", task_id=task_id, found=group_res is not None
        )

        if group_res is not None:
            # This is a group - aggregate children's states and track per-file progress
            children = list(group_res.results)
            logger.info(
                "Group children found", task_id=task_id, child_count=len(children)
            )

            # Process children using helper function (no file_hashes available from GroupResult)
            file_hashes = []  # Initialize empty list since GroupResult doesn't provide file hashes
            progress = _process_group_children(children, file_hashes, task_id, logger)
            total = progress["total"]
            completed = progress["completed"]
            file_statuses = progress["file_statuses"]

            # Determine overall state
            if completed == total:
                # Processing is complete - aggregate results from all tasks (successful AND failed)
                results = []
                for i, child in enumerate(children):
                    # Get file hash from file_hashes if available (may be empty in this fallback path)
                    file_hash = (
                        file_hashes[i] if i < len(file_hashes) else f"unknown_file_{i}"
                    )

                    if child.successful():
                        try:
                            result_data = child.result
                            # Inject file_hash into the result for frontend consumption
                            if (
                                isinstance(result_data, (list, tuple))
                                and len(result_data) >= 2
                            ):
                                # Format: [success, details] - inject hash into details
                                success_flag, details = result_data[0], result_data[1]
                                if isinstance(details, dict):
                                    details["file_hash"] = file_hash
                                results.append([success_flag, details])
                            elif isinstance(result_data, dict):
                                result_data["file_hash"] = file_hash
                                results.append(result_data)
                            else:
                                results.append(result_data)
                        except Exception:
                            pass
                    elif child.failed():
                        # Also collect failed task results
                        try:
                            error_info = str(child.result)
                            # Create a failed result entry
                            failed_result = {
                                "file_hash": file_hash,
                                "success": False,
                                "error": error_info,
                                "details": {
                                    "status": "failed",
                                    "filename": file_hash,
                                    "error": error_info,
                                },
                            }
                            results.append(failed_result)
                        except Exception as e:
                            logger.error(f"Error processing failed child result: {e}")

                # Count actual statuses from the results (don't rely on Celery's successful count)
                processed_count = 0
                skipped_count = 0
                failed_count = 0

                for result in results:
                    if isinstance(result, list) and len(result) >= 2:
                        # New format: [success, info]
                        success, info = result[0], result[1]
                        status = (
                            info.get("status") or info.get("result", {}).get("status")
                            if isinstance(info, dict)
                            else None
                        )

                        if status == "skipped":
                            skipped_count += 1
                        elif not success:
                            failed_count += 1
                        else:
                            processed_count += 1
                    elif isinstance(result, dict):
                        # Old format or failed task result
                        details = result.get("details", {})
                        status = (
                            details.get("status")
                            or details.get("result", {}).get("status")
                            if isinstance(details, dict)
                            else None
                        )

                        if status == "skipped":
                            skipped_count += 1
                        elif not result.get("success", True):
                            failed_count += 1
                        else:
                            processed_count += 1

                # Set result with all file details (successful, skipped, and failed)
                response.result = {
                    "processed_files": processed_count,
                    "failed_files": failed_count,
                    "skipped_files": skipped_count,
                    "stored_count": progress.get("total_stored_count", 0),
                    "skipped_count": progress.get("total_skipped_count", 0),
                    "details": results,
                }

                # Set state based on whether any files failed
                if failed_count > 0:
                    response.state = "FAILURE"
                    response.error = f"{failed_count} of {total} tasks failed"
                else:
                    response.state = "SUCCESS"

                # Persist the final state to the backend so subsequent polls see the correct state
                app.backend.store_result(task_id, response.result, response.state)
                logger.info(
                    "Persisted final group task state (GroupResult fallback)",
                    task_id=task_id,
                    state=response.state,
                    processed=processed_count,
                    failed=failed_count,
                    skipped=skipped_count,
                )
            else:
                response.state = "PROGRESS"
                response.meta = {
                    "current": completed,
                    "total": total,
                    "progress": int((completed / total) * 100) if total > 0 else 0,
                    "message": f"Processing {completed}/{total} files",
                    "files": file_statuses,  # Include per-file details
                    "stored_count": progress.get("total_stored_count", 0),
                    "skipped_count": progress.get("total_skipped_count", 0),
                }
            return response
        else:
            logger.info("No group found via GroupResult.restore", task_id=task_id)
    except Exception as e:
        # Not a group, handle as normal task
        logger.warning(
            "Error checking for group task",
            task_id=task_id,
            error=str(e),
            exc_info=True,
        )

    # Ensure cancelled flag is checked even for non-group tasks
    if not response.cancelled:
        try:
            # Try to get from result one more time
            result_value = res.result
            if isinstance(result_value, dict) and result_value.get("type") == "group":
                response.cancelled = result_value.get("cancelled", False)
        except Exception:
            pass

    if res.info:
        if isinstance(res.info, (dict, list, str, int, float, bool, type(None))):
            response.meta = res.info
        else:
            response.meta = {"message": str(res.info)}

    if res.state == "SUCCESS":
        if isinstance(
            res.result, (dict | list | str | int | float | bool | type(None))
        ):
            response.result = res.result
        else:
            response.result = {"message": str(res.result)}
    elif res.state == "FAILURE":
        response.error = str(res.result)
        response.traceback = res.traceback

    return response


@router.delete(
    "/{namespace}/{project_id}/tasks/{task_id}",
    operation_id="task_cancel",
    tags=["tasks", "mcp"],
    summary="Cancel a running task",
    description="Cancel a running task and revert any files that were successfully processed. This is useful for cancelling dataset processing operations.",
    response_model=CancelTaskResponse,
)
async def cancel_task(
    namespace: str,
    project_id: str,
    task_id: str,
) -> CancelTaskResponse:
    """
    Cancel a running task and revert any files that were successfully processed.

    This endpoint is primarily used for cancelling dataset processing operations.
    It will:
    1. Revoke the Celery task(s) to stop processing
    2. Mark the task as cancelled in the backend
    3. Clean up any files that were successfully processed (remove chunks from vector store)

    Args:
        namespace: Project namespace
        project_id: Project name
        task_id: The Celery task ID to cancel

    Returns:
        Cancellation status with cleanup details

    Raises:
        HTTPException: 404 if task not found, 500 for other errors
    """
    from datetime import datetime

    from celery.result import AsyncResult  # type: ignore[import-untyped]

    from core.celery import app as celery_app
    from core.logging import FastAPIStructLogger

    logger = FastAPIStructLogger(__name__)
    logger.bind(namespace=namespace, project=project_id, task_id=task_id)

    try:
        # Get task result
        group_result: AsyncResult = celery_app.AsyncResult(task_id)

        # Get stored metadata - try multiple approaches
        result_meta = None

        # Approach 1: Try to get from result when state is PENDING
        # This is how metadata is stored in the existing code
        try:
            if (
                group_result.state == "PENDING"
                and isinstance(group_result.result, dict)
                and group_result.result.get("type") == "group"
            ):
                result_meta = group_result.result
        except Exception as e:
            logger.debug(f"Could not get metadata from PENDING result: {e}")

        # Approach 2: Try accessing result property directly for other states
        # Sometimes the metadata is still accessible even if state changed
        if not result_meta:
            try:
                result_value = group_result.result
                if (
                    isinstance(result_value, dict)
                    and result_value.get("type") == "group"
                ):
                    result_meta = result_value
            except Exception as e:
                logger.debug(f"Could not get metadata from result property: {e}")

        # Approach 3: Try to access via backend's _get_task_meta_for method if available
        # This is a fallback for filesystem backend
        if not result_meta:
            try:
                # Some backends support this method
                if (
                    hasattr(celery_app.backend, "_get_task_meta_for")
                    and (meta := celery_app.backend._get_task_meta_for(task_id))
                    and isinstance(meta.get("result"), dict)
                    and meta["result"].get("type") == "group"
                ):
                    result_meta = meta["result"]
            except Exception as e:
                logger.debug(f"Could not get metadata via backend method: {e}")

        if not result_meta or result_meta.get("type") != "group":
            raise HTTPException(
                status_code=404,
                detail="Task not found or not a group task. Only group tasks (dataset processing) can be cancelled.",
            )

        # SECURITY: Verify task belongs to the requested namespace/project
        task_namespace = result_meta.get("namespace")
        task_project = result_meta.get("project")

        if task_namespace != namespace or task_project != project_id:
            logger.warning(
                "Authorization failed: task does not belong to requested namespace/project",
                task_id=task_id,
                task_namespace=task_namespace,
                task_project=task_project,
                requested_namespace=namespace,
                requested_project=project_id,
            )
            raise HTTPException(
                status_code=404,
                detail="Task not found or not a group task. Only group tasks (dataset processing) can be cancelled.",
            )

        # Check if already cancelled
        if result_meta.get("cancelled"):
            # Already cancelled, return current state
            child_task_ids = result_meta.get("children", [])
            pending_count = 0
            running_count = 0

            # Count current states
            for child_id in child_task_ids:
                child_result = celery_app.AsyncResult(child_id)
                try:
                    if child_result.state == "PENDING":
                        pending_count += 1
                    elif child_result.state == "STARTED":
                        running_count += 1
                except Exception:
                    pass

            logger.info("Task already cancelled", task_id=task_id)

            # Get previous cleanup results if available
            cleanup_status = result_meta.get("cleanup_status", {})

            # Convert errors to CleanupError objects if present
            cleanup_errors = None
            if cleanup_status.get("errors"):
                cleanup_errors = [
                    CleanupError(file_hash=e["file_hash"], error=e["error"])
                    for e in cleanup_status["errors"]
                ]

            return CancelTaskResponse(
                message="Task already cancelled",
                task_id=task_id,
                cancelled=True,
                pending_tasks_cancelled=pending_count,
                running_tasks_at_cancel=running_count,
                files_reverted=cleanup_status.get("files_reverted", 0),
                files_failed_to_revert=cleanup_status.get("files_failed_to_revert", 0),
                errors=cleanup_errors,
                already_cancelled=True,
            )

        # Check if already completed
        if group_result.state in ("SUCCESS", "FAILURE"):
            logger.info(
                "Task already completed",
                task_id=task_id,
                state=group_result.state,
            )
            return CancelTaskResponse(
                message=f"Task already {group_result.state.lower()}",
                task_id=task_id,
                cancelled=False,
                already_completed=True,
            )

        # Get child task IDs
        child_task_ids = result_meta.get("children", [])

        # Revoke child tasks
        pending_cancelled = 0
        running_count = 0

        for child_id in child_task_ids:
            try:
                child_result = celery_app.AsyncResult(child_id)

                # Get current state (may raise exception if task doesn't exist)
                try:
                    child_state = child_result.state
                except Exception:
                    # Task may not exist or be inaccessible, skip it
                    continue

                if child_state == "PENDING":
                    # Revoke pending tasks (prevent from starting)
                    celery_app.control.revoke(child_id, terminate=False)
                    pending_cancelled += 1
                    logger.info(f"Revoked pending task: {child_id}")
                elif child_state == "STARTED":
                    # Revoke running tasks (graceful - let current work finish)
                    celery_app.control.revoke(child_id, terminate=False)
                    running_count += 1
                    logger.info(f"Revoked running task: {child_id}")
            except Exception as e:
                logger.warning(f"Error revoking child task {child_id}: {e}")
                # Continue with other tasks

        # Update group metadata with cancellation flag
        result_meta["cancelled"] = True
        result_meta["cancelled_at"] = datetime.now().isoformat()

        # Store updated metadata
        # Use the current state or "CANCELLED" if backend supports it
        current_state = (
            group_result.state if hasattr(group_result, "state") else "PENDING"
        )
        celery_app.backend.store_result(
            task_id,
            result_meta,
            current_state,  # Keep original state, cancellation is tracked in metadata
        )

        logger.info(
            "Task cancellation requested",
            task_id=task_id,
            pending_cancelled=pending_cancelled,
            running_count=running_count,
        )

        # Trigger cleanup for successfully processed files
        cleanup_result = {
            "files_reverted": 0,
            "files_failed_to_revert": 0,
            "errors": None,
        }

        try:
            from services.dataset_cleanup_service import DatasetCleanupService

            # Extract dataset name from metadata if available
            # Task metadata stores dataset name as "dataset" key
            dataset_name = result_meta.get("dataset")

            if dataset_name:
                cleanup_service = DatasetCleanupService()
                cleanup_result = await cleanup_service.cleanup_processed_files(
                    namespace, project_id, dataset_name, task_id
                )

                # Update metadata with cleanup results
                result_meta["cleanup_status"] = cleanup_result
            else:
                logger.warning(
                    "No dataset name in task metadata, skipping cleanup",
                    task_id=task_id,
                )

        except Exception as e:
            logger.error(
                f"Error during cleanup (cancellation still succeeded): {e}",
                exc_info=True,
            )
            # Don't fail cancellation if cleanup fails
            cleanup_result["errors"] = [{"file_hash": "unknown", "error": str(e)}]

        # Store updated metadata with cleanup status
        current_state = (
            group_result.state if hasattr(group_result, "state") else "PENDING"
        )
        celery_app.backend.store_result(
            task_id,
            result_meta,
            current_state,
        )

        # Build response message
        if cleanup_result["files_failed_to_revert"] == 0:
            message = (
                f"Task cancelled and {cleanup_result['files_reverted']} file(s) reverted"
                if cleanup_result["files_reverted"] > 0
                else "Task cancelled (no files to revert)"
            )
        else:
            message = (
                f"Task cancelled with cleanup issues: "
                f"{cleanup_result['files_reverted']} reverted, "
                f"{cleanup_result['files_failed_to_revert']} failed"
            )

        # Convert errors to CleanupError objects if present
        cleanup_errors = None
        if cleanup_result.get("errors"):
            cleanup_errors = [
                CleanupError(file_hash=e["file_hash"], error=e["error"])
                for e in cleanup_result["errors"]
            ]

        logger.info(
            "Task cancellation completed",
            task_id=task_id,
            message=message,
        )

        return CancelTaskResponse(
            message=message,
            task_id=task_id,
            cancelled=True,
            pending_tasks_cancelled=pending_cancelled,
            running_tasks_at_cancel=running_count,
            files_reverted=cleanup_result["files_reverted"],
            files_failed_to_revert=cleanup_result["files_failed_to_revert"],
            errors=cleanup_errors,
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(
            f"Error cancelling task: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel task: {str(e)}"
        ) from e


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

        agent = await ChatOrchestratorAgentFactory.create_agent(
            project_config=project_config,
            project_dir=project_dir,
            session_id=session_id,
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


@router.get(
    "/{namespace}/{project_id}/models",
    operation_id="models_list",
    tags=["models", "mcp"],
    summary="List all available models for this project",
    description="List all available models for this project",
    responses={
        200: {"model": ListModelsResponse},
        404: {"model": ErrorResponse},
    },
)
async def list_models(namespace: str, project_id: str):
    """List all available models for this project."""
    from services.model_service import ModelService

    project_config = ProjectService.load_config(namespace, project_id)
    models = ModelService.list_models(project_config)

    default_model = project_config.runtime.default_model

    return ListModelsResponse(
        total=len(models),
        models=[
            ModelResponse(
                **model.model_dump(mode="json", exclude_none=True),
                default=model.name == default_model,
            )
            for model in models
        ],
    )
