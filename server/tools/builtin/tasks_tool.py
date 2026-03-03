"""
Task Tools for managing tasks within chat sessions.

This module provides four separate tool classes for task management:
- TaskCreateTool: Create new tasks
- TaskUpdateTool: Update task status, dependencies, and metadata
- TaskListTool: List all tasks for the session
- TaskGetTool: Get full details of a specific task

Each tool delegates to TasksService for persistence and validation.
"""

import asyncio
import concurrent.futures
import json
from collections.abc import Coroutine
from typing import Any, Literal

from atomic_agents import BaseTool
from atomic_agents.base.base_io_schema import BaseIOSchema
from pydantic import Field

from services.tasks_service import (
    CycleDetectedError,
    InvalidStatusTransitionError,
    TaskNotFoundError,
    TasksService,
)


def _run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine synchronously, safe to call from within a running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


class TaskToolOutput(BaseIOSchema):
    """Output schema for all task tools."""

    result: str = Field(
        ...,
        description="JSON string containing the operation result",
    )


# -----------------------------------------------------------------------------
# TaskCreateTool
# -----------------------------------------------------------------------------


class TaskCreateInput(BaseIOSchema):
    """Input schema for task_create tool."""

    subject: str = Field(
        ...,
        description="Brief title for the task in imperative form (e.g., 'Fix authentication bug')",
    )
    description: str = Field(
        ...,
        description="Detailed description of what needs to be done, including context and acceptance criteria",
    )
    activeForm: str | None = Field(
        None,
        description="Present continuous form shown in spinner when in_progress (e.g., 'Fixing authentication bug')",
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description="Arbitrary key-value pairs to attach to the task",
    )


class TaskCreateTool(BaseTool):
    """Create a new task to track work.

    Creates a single task and returns its ID. To create multiple related tasks
    with dependencies, call task_create for each task, then use task_update
    to set up dependencies.

    Example workflow:
        # Create tasks
        task_create(subject="Research options", description="...")  # Returns ID "1"
        task_create(subject="Implement solution", description="...")  # Returns ID "2"
        task_create(subject="Write tests", description="...")  # Returns ID "3"

        # Set up dependencies (task 2 waits for task 1, task 3 waits for task 2)
        task_update(taskId="2", addBlockedBy=["1"])
        task_update(taskId="3", addBlockedBy=["2"])

        # Work on tasks
        task_update(taskId="1", status="in_progress")
        task_update(taskId="1", status="completed")  # Auto-unblocks task 2
    """

    input_schema = TaskCreateInput
    output_schema = TaskToolOutput

    # Class-level attributes for context injection
    _project_dir: str = ""
    _session_id: str = ""

    # Tool name for identification
    tool_name: str = "task_create"

    def run(self, params: TaskCreateInput) -> TaskToolOutput:
        """Execute the task creation synchronously."""
        return _run_sync(self.arun(params))

    async def arun(self, params: TaskCreateInput) -> TaskToolOutput:
        """Create a new task."""
        try:
            task = TasksService.create_task(
                project_dir=self._project_dir,
                session_id=self._session_id,
                subject=params.subject,
                description=params.description,
                activeForm=params.activeForm or "",
                metadata=params.metadata,
            )

            return TaskToolOutput(
                result=json.dumps(
                    {
                        "success": True,
                        "task": task.model_dump(),
                    }
                )
            )
        except TaskNotFoundError as e:
            return TaskToolOutput(result=json.dumps({"error": str(e)}))
        except CycleDetectedError as e:
            return TaskToolOutput(result=json.dumps({"error": str(e)}))
        except Exception as e:
            return TaskToolOutput(
                result=json.dumps({"error": f"Operation failed: {str(e)}"})
            )


# -----------------------------------------------------------------------------
# TaskUpdateTool
# -----------------------------------------------------------------------------


class TaskUpdateInput(BaseIOSchema):
    """Input schema for task_update tool."""

    taskId: str = Field(
        ...,
        description="The ID of the task to update",
    )
    status: Literal["pending", "in_progress", "completed", "deleted"] | None = Field(
        None,
        description="New status. Use 'deleted' to remove the task permanently.",
    )
    owner: str | None = Field(
        None,
        description="Agent/worker ID claiming this task. Set to claim before working.",
    )
    subject: str | None = Field(
        None,
        description="New subject/title for the task",
    )
    description: str | None = Field(
        None,
        description="New description for the task",
    )
    activeForm: str | None = Field(
        None,
        description="New present continuous form for spinner display",
    )
    addBlockedBy: list[str] | None = Field(
        None,
        description="Task IDs to add as blockers (this task waits for these tasks)",
    )
    addBlocks: list[str] | None = Field(
        None,
        description="Task IDs that this task will block (these tasks wait for this task)",
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description="Key-value pairs to merge into task metadata. Set a key to null to delete it.",
    )


class TaskUpdateTool(BaseTool):
    """Update an existing task's status, dependencies, or metadata.

    Status transitions:
        pending -> in_progress -> completed
        Any status -> deleted (removes task)

    Dependency management:
        - addBlockedBy: "This task waits for these tasks"
        - addBlocks: "These tasks wait for this task"
        - When a task completes, it's automatically removed from blockedBy of dependent tasks

    Multi-agent coordination:
        - Set owner to claim a task before working on it
        - Check task_list to find available tasks (pending, no owner, not blocked)

    Examples:
        # Start working on a task
        task_update(taskId="1", status="in_progress")

        # Complete a task (unblocks dependents)
        task_update(taskId="1", status="completed")

        # Delete a task
        task_update(taskId="2", status="deleted")

        # Claim a task for an agent
        task_update(taskId="3", owner="agent-abc")

        # Add a dependency
        task_update(taskId="2", addBlockedBy=["1"])
    """

    input_schema = TaskUpdateInput
    output_schema = TaskToolOutput

    # Class-level attributes for context injection
    _project_dir: str = ""
    _session_id: str = ""

    # Tool name for identification
    tool_name: str = "task_update"

    def run(self, params: TaskUpdateInput) -> TaskToolOutput:
        """Execute the task update synchronously."""
        return _run_sync(self.arun(params))

    async def arun(self, params: TaskUpdateInput) -> TaskToolOutput:
        """Update an existing task."""
        try:
            # Handle deletion via status="deleted"
            if params.status == "deleted":
                task = TasksService.delete_task(
                    project_dir=self._project_dir,
                    session_id=self._session_id,
                    task_id=params.taskId,
                )
                return TaskToolOutput(
                    result=json.dumps(
                        {
                            "success": True,
                            "deleted": True,
                            "task": task.model_dump(),
                        }
                    )
                )

            # Regular update
            task = TasksService.update_task(
                project_dir=self._project_dir,
                session_id=self._session_id,
                task_id=params.taskId,
                status=params.status,
                subject=params.subject,
                description=params.description,
                activeForm=params.activeForm,
                addBlocks=params.addBlocks,
                addBlockedBy=params.addBlockedBy,
                owner=params.owner,
                metadata=params.metadata,
            )

            return TaskToolOutput(
                result=json.dumps(
                    {
                        "success": True,
                        "task": task.model_dump(),
                    }
                )
            )
        except TaskNotFoundError as e:
            return TaskToolOutput(result=json.dumps({"error": str(e)}))
        except CycleDetectedError as e:
            return TaskToolOutput(result=json.dumps({"error": str(e)}))
        except InvalidStatusTransitionError as e:
            return TaskToolOutput(result=json.dumps({"error": str(e)}))
        except Exception as e:
            return TaskToolOutput(
                result=json.dumps({"error": f"Operation failed: {str(e)}"})
            )


# -----------------------------------------------------------------------------
# TaskListTool
# -----------------------------------------------------------------------------


class TaskListInput(BaseIOSchema):
    """Input schema for task_list tool. No parameters required."""

    pass


class TaskListTool(BaseTool):
    """List all tasks for the current session.

    Returns tasks with: id, subject, status, owner, blockedBy.

    A task is "available" when:
        - status is "pending"
        - owner is empty (not claimed)
        - blockedBy is empty (no pending blockers)

    Use this to:
        - See overall progress
        - Find available tasks to work on
        - Check which tasks are blocked

    Example response:
        {
            "tasks": [
                {"id": "1", "subject": "Research", "status": "completed", "owner": "", "blockedBy": []},
                {"id": "2", "subject": "Implement", "status": "pending", "owner": "", "blockedBy": []},
                {"id": "3", "subject": "Test", "status": "pending", "owner": "", "blockedBy": ["2"]}
            ]
        }
    """

    input_schema = TaskListInput
    output_schema = TaskToolOutput

    # Class-level attributes for context injection
    _project_dir: str = ""
    _session_id: str = ""

    # Tool name for identification
    tool_name: str = "task_list"

    def run(self, params: TaskListInput) -> TaskToolOutput:
        """Execute the task list synchronously."""
        return _run_sync(self.arun(params))

    async def arun(self, params: TaskListInput) -> TaskToolOutput:
        """List all tasks for the session."""
        try:
            tasks = TasksService.list_tasks(
                project_dir=self._project_dir,
                session_id=self._session_id,
            )

            if not tasks:
                return TaskToolOutput(
                    result=json.dumps(
                        {
                            "success": True,
                            "message": "No tasks found",
                            "tasks": [],
                        }
                    )
                )

            # Return summary fields for each task
            task_summaries = [
                {
                    "id": task.id,
                    "subject": task.subject,
                    "status": task.status,
                    "owner": task.owner,
                    "blockedBy": task.blockedBy,
                }
                for task in tasks
            ]

            return TaskToolOutput(
                result=json.dumps(
                    {
                        "success": True,
                        "tasks": task_summaries,
                    }
                )
            )
        except Exception as e:
            return TaskToolOutput(
                result=json.dumps({"error": f"Operation failed: {str(e)}"})
            )


# -----------------------------------------------------------------------------
# TaskGetTool
# -----------------------------------------------------------------------------


class TaskGetInput(BaseIOSchema):
    """Input schema for task_get tool."""

    taskId: str = Field(
        ...,
        description="The ID of the task to retrieve",
    )


class TaskGetTool(BaseTool):
    """Get full details of a specific task.

    Returns complete task including:
        - id, subject, description, activeForm
        - status, owner
        - blocks (tasks waiting for this one)
        - blockedBy (tasks this one waits for)
        - metadata

    Use task_list for summaries, task_get for full details before working on a task.
    """

    input_schema = TaskGetInput
    output_schema = TaskToolOutput

    # Class-level attributes for context injection
    _project_dir: str = ""
    _session_id: str = ""

    # Tool name for identification
    tool_name: str = "task_get"

    def run(self, params: TaskGetInput) -> TaskToolOutput:
        """Execute the task get synchronously."""
        return _run_sync(self.arun(params))

    async def arun(self, params: TaskGetInput) -> TaskToolOutput:
        """Get a task by ID."""
        try:
            task = TasksService.get_task(
                project_dir=self._project_dir,
                session_id=self._session_id,
                task_id=params.taskId,
            )

            return TaskToolOutput(
                result=json.dumps(
                    {
                        "success": True,
                        "task": task.model_dump(),
                    }
                )
            )
        except TaskNotFoundError as e:
            return TaskToolOutput(result=json.dumps({"error": str(e)}))
        except Exception as e:
            return TaskToolOutput(
                result=json.dumps({"error": f"Operation failed: {str(e)}"})
            )


# -----------------------------------------------------------------------------
# Legacy export for backwards compatibility
# -----------------------------------------------------------------------------

# Old classes for backwards compatibility during transition
# TODO: Remove after updating all imports

TasksToolInput = TaskCreateInput  # Closest match for create operations
TasksToolOutput = TaskToolOutput


class TasksTool(TaskCreateTool):
    """DEPRECATED: Use TaskCreateTool, TaskUpdateTool, TaskListTool, or TaskGetTool instead.

    This class is kept for backwards compatibility during transition.
    """

    tool_name: str = "tasks"
