"""
Tasks Tool for managing tasks within chat sessions.

This tool provides operations to create, update, list, and get tasks
with dependency tracking (blockedBy/blocks).
"""

import asyncio
import json
from typing import Literal, Optional

from atomic_agents import BaseTool
from atomic_agents.base.base_io_schema import BaseIOSchema
from pydantic import Field

from services.tasks_service import (
    CycleDetectedError,
    TaskNotFoundError,
    TasksService,
)


class TasksToolInput(BaseIOSchema):
    """Input schema for the tasks tool."""

    operation: Literal["create", "update", "list", "get"] = Field(
        ...,
        description="The operation to perform on tasks",
    )
    taskId: Optional[str] = Field(
        None,
        description="Task ID (required for get, update operations)",
    )
    subject: Optional[str] = Field(
        None,
        description="Task subject/title (for create, update)",
    )
    description: Optional[str] = Field(
        None,
        description="Task description (for create, update)",
    )
    activeForm: Optional[str] = Field(
        None,
        description="Present continuous form for display (e.g., 'Running tests')",
    )
    status: Optional[Literal["pending", "in_progress", "completed", "deleted"]] = Field(
        None,
        description="Task status (for update). Use 'deleted' to remove.",
    )
    blockedBy: Optional[list[str]] = Field(
        None,
        description="Task IDs that must complete before this task (create)",
    )
    addBlockedBy: Optional[list[str]] = Field(
        None,
        description="Task IDs to add to blockedBy list (update)",
    )
    addBlocks: Optional[list[str]] = Field(
        None,
        description="Task IDs that this task blocks (update)",
    )


class TasksToolOutput(BaseIOSchema):
    """Output schema for the tasks tool."""

    result: str = Field(
        ...,
        description="JSON string containing the operation result",
    )


class TasksTool(BaseTool):
    """Manage tasks for the current chat session.

    Create, update, list, and get tasks to track work items.
    Tasks can have dependencies (blockedBy/blocks) and status
    (pending, in_progress, completed). Use status='deleted' to remove a task.
    """

    input_schema = TasksToolInput
    output_schema = TasksToolOutput

    # Class-level attributes for context injection
    _project_dir: str = ""
    _session_id: str = ""

    # MCP-compatible name for tool identification
    mcp_tool_name: str = "tasks"

    def run(self, params: TasksToolInput) -> TasksToolOutput:
        """Execute the task operation synchronously."""
        return asyncio.get_event_loop().run_until_complete(self.arun(params))

    async def arun(self, params: TasksToolInput) -> TasksToolOutput:
        """Execute the task operation based on the operation type."""
        try:
            if params.operation == "create":
                return await self._create_task(params)
            elif params.operation == "update":
                return await self._update_task(params)
            elif params.operation == "list":
                return await self._list_tasks()
            elif params.operation == "get":
                return await self._get_task(params)
            else:
                return TasksToolOutput(
                    result=json.dumps(
                        {"error": f"Unknown operation: {params.operation}"}
                    )
                )
        except TaskNotFoundError as e:
            return TasksToolOutput(result=json.dumps({"error": str(e)}))
        except CycleDetectedError as e:
            return TasksToolOutput(result=json.dumps({"error": str(e)}))
        except Exception as e:
            return TasksToolOutput(
                result=json.dumps({"error": f"Operation failed: {str(e)}"})
            )

    async def _create_task(self, params: TasksToolInput) -> TasksToolOutput:
        """Create a new task."""
        if not params.subject:
            return TasksToolOutput(
                result=json.dumps({"error": "subject is required for create operation"})
            )

        task = TasksService.create_task(
            project_dir=self._project_dir,
            session_id=self._session_id,
            subject=params.subject,
            description=params.description or "",
            activeForm=params.activeForm or "",
            blockedBy=params.blockedBy,
        )

        return TasksToolOutput(
            result=json.dumps(
                {
                    "success": True,
                    "task": task.model_dump(),
                }
            )
        )

    async def _update_task(self, params: TasksToolInput) -> TasksToolOutput:
        """Update an existing task."""
        if not params.taskId:
            return TasksToolOutput(
                result=json.dumps({"error": "taskId is required for update operation"})
            )

        # Handle deletion via status="deleted"
        if params.status == "deleted":
            task = TasksService.delete_task(
                project_dir=self._project_dir,
                session_id=self._session_id,
                task_id=params.taskId,
            )
            return TasksToolOutput(
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
        )

        return TasksToolOutput(
            result=json.dumps(
                {
                    "success": True,
                    "task": task.model_dump(),
                }
            )
        )

    async def _list_tasks(self) -> TasksToolOutput:
        """List all tasks for the session."""
        tasks = TasksService.list_tasks(
            project_dir=self._project_dir,
            session_id=self._session_id,
        )

        if not tasks:
            return TasksToolOutput(
                result=json.dumps(
                    {
                        "success": True,
                        "message": "No tasks found",
                        "tasks": [],
                    }
                )
            )

        return TasksToolOutput(
            result=json.dumps(
                {
                    "success": True,
                    "tasks": [task.model_dump() for task in tasks],
                }
            )
        )

    async def _get_task(self, params: TasksToolInput) -> TasksToolOutput:
        """Get a task by ID."""
        if not params.taskId:
            return TasksToolOutput(
                result=json.dumps({"error": "taskId is required for get operation"})
            )

        task = TasksService.get_task(
            project_dir=self._project_dir,
            session_id=self._session_id,
            task_id=params.taskId,
        )

        return TasksToolOutput(
            result=json.dumps(
                {
                    "success": True,
                    "task": task.model_dump(),
                }
            )
        )
