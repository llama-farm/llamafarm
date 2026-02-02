"""
Tests for TasksTool.

This module tests the TasksTool builtin tool integration, including
input/output schemas, operation dispatch, and error handling.
"""

import json
import os
import shutil
import tempfile

import pytest
from config.datamodel import BuiltinTools, Model

from tools.builtin.factory import BuiltinToolFactory
from tools.builtin.registry import BUILTIN_TOOL_NAMES, get_enabled_builtin_tool_names
from tools.builtin.tasks_tool import TasksTool, TasksToolInput, TasksToolOutput


class TestTasksToolSchema:
    """Tests for TasksTool schema definitions."""

    def test_tasks_tool_input_schema_requires_operation(self):
        """Test that operation field is required."""
        with pytest.raises(ValueError):
            TasksToolInput()  # Missing required operation

    def test_tasks_tool_input_schema_accepts_valid_operations(self):
        """Test that valid operations are accepted."""
        for operation in ["create", "update", "list", "get"]:
            input_obj = TasksToolInput(operation=operation)
            assert input_obj.operation == operation

    def test_tasks_tool_input_schema_optional_fields(self):
        """Test that optional fields default to None."""
        input_obj = TasksToolInput(operation="list")
        assert input_obj.taskId is None
        assert input_obj.subject is None
        assert input_obj.description is None
        assert input_obj.activeForm is None
        assert input_obj.status is None
        assert input_obj.blockedBy is None
        assert input_obj.addBlockedBy is None
        assert input_obj.addBlocks is None

    def test_tasks_tool_output_schema_requires_result(self):
        """Test that result field is required."""
        with pytest.raises(ValueError):
            TasksToolOutput()  # Missing required result

    def test_tasks_tool_output_schema_accepts_json_string(self):
        """Test that output schema accepts JSON string."""
        output = TasksToolOutput(result='{"success": true}')
        assert output.result == '{"success": true}'


class TestTasksToolOperations:
    """Tests for TasksTool operations."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        return "test-session-tool"

    @pytest.fixture
    def tasks_tool(self, temp_project_dir, session_id):
        """Create a TasksTool instance with injected context."""
        # Create injected tool class similar to BuiltinToolFactory
        class InjectedTasksTool(TasksTool):
            _project_dir = temp_project_dir
            _session_id = session_id

        return InjectedTasksTool()

    @pytest.mark.asyncio
    async def test_create_operation(self, tasks_tool):
        """Test create operation creates a task."""
        input_obj = TasksToolInput(
            operation="create",
            subject="Test task",
            description="Test description",
            activeForm="Testing",
        )

        output = await tasks_tool.arun(input_obj)
        result = json.loads(output.result)

        assert result["success"] is True
        assert result["task"]["subject"] == "Test task"
        assert result["task"]["description"] == "Test description"
        assert result["task"]["activeForm"] == "Testing"
        assert result["task"]["status"] == "pending"
        assert result["task"]["id"] == "1"

    @pytest.mark.asyncio
    async def test_create_operation_requires_subject(self, tasks_tool):
        """Test create operation requires subject field."""
        input_obj = TasksToolInput(
            operation="create",
            description="No subject provided",
        )

        output = await tasks_tool.arun(input_obj)
        result = json.loads(output.result)

        assert "error" in result
        assert "subject" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_list_operation_empty(self, tasks_tool):
        """Test list operation returns empty list when no tasks exist."""
        input_obj = TasksToolInput(operation="list")

        output = await tasks_tool.arun(input_obj)
        result = json.loads(output.result)

        assert result["success"] is True
        assert result["tasks"] == []

    @pytest.mark.asyncio
    async def test_list_operation_with_tasks(self, tasks_tool):
        """Test list operation returns all tasks."""
        # Create some tasks
        await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Task 1",
            description="First task",
        ))
        await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Task 2",
            description="Second task",
        ))

        # List tasks
        output = await tasks_tool.arun(TasksToolInput(operation="list"))
        result = json.loads(output.result)

        assert result["success"] is True
        assert len(result["tasks"]) == 2

    @pytest.mark.asyncio
    async def test_get_operation(self, tasks_tool):
        """Test get operation retrieves a specific task."""
        # Create a task
        create_output = await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Test task",
            description="Test description",
        ))
        create_result = json.loads(create_output.result)
        task_id = create_result["task"]["id"]

        # Get the task
        output = await tasks_tool.arun(TasksToolInput(
            operation="get",
            taskId=task_id,
        ))
        result = json.loads(output.result)

        assert result["success"] is True
        assert result["task"]["id"] == task_id
        assert result["task"]["subject"] == "Test task"

    @pytest.mark.asyncio
    async def test_get_operation_requires_task_id(self, tasks_tool):
        """Test get operation requires taskId field."""
        output = await tasks_tool.arun(TasksToolInput(operation="get"))
        result = json.loads(output.result)

        assert "error" in result
        assert "taskId" in result["error"]

    @pytest.mark.asyncio
    async def test_get_operation_nonexistent_task(self, tasks_tool):
        """Test get operation returns error for nonexistent task."""
        output = await tasks_tool.arun(TasksToolInput(
            operation="get",
            taskId="999",
        ))
        result = json.loads(output.result)

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_update_operation_status(self, tasks_tool):
        """Test update operation changes task status."""
        # Create a task
        await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Test task",
            description="Test description",
        ))

        # Update status
        output = await tasks_tool.arun(TasksToolInput(
            operation="update",
            taskId="1",
            status="in_progress",
        ))
        result = json.loads(output.result)

        assert result["success"] is True
        assert result["task"]["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_update_operation_requires_task_id(self, tasks_tool):
        """Test update operation requires taskId field."""
        output = await tasks_tool.arun(TasksToolInput(
            operation="update",
            status="in_progress",
        ))
        result = json.loads(output.result)

        assert "error" in result
        assert "taskId" in result["error"]

    @pytest.mark.asyncio
    async def test_delete_via_update_status(self, tasks_tool):
        """Test deletion via update with status='deleted'."""
        # Create a task
        await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Task to delete",
            description="Will be deleted",
        ))

        # Delete via status update
        output = await tasks_tool.arun(TasksToolInput(
            operation="update",
            taskId="1",
            status="deleted",
        ))
        result = json.loads(output.result)

        assert result["success"] is True
        assert result["deleted"] is True

        # Verify task is gone
        list_output = await tasks_tool.arun(TasksToolInput(operation="list"))
        list_result = json.loads(list_output.result)
        assert len(list_result["tasks"]) == 0

    @pytest.mark.asyncio
    async def test_create_with_blocked_by(self, tasks_tool):
        """Test create operation with blockedBy dependencies."""
        # Create first task
        await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Task 1",
            description="First task",
        ))

        # Create second task blocked by first
        output = await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Task 2",
            description="Blocked by task 1",
            blockedBy=["1"],
        ))
        result = json.loads(output.result)

        assert result["success"] is True
        assert "1" in result["task"]["blockedBy"]

    @pytest.mark.asyncio
    async def test_update_add_blocked_by(self, tasks_tool):
        """Test update operation with addBlockedBy."""
        # Create two tasks
        await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Task 1",
            description="First task",
        ))
        await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Task 2",
            description="Second task",
        ))

        # Add blockedBy relationship
        output = await tasks_tool.arun(TasksToolInput(
            operation="update",
            taskId="2",
            addBlockedBy=["1"],
        ))
        result = json.loads(output.result)

        assert result["success"] is True
        assert "1" in result["task"]["blockedBy"]


class TestBuiltinToolFactory:
    """Tests for BuiltinToolFactory."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def test_factory_creates_tasks_tool_with_session(self, temp_project_dir):
        """Test factory creates tasks tool when session_id is provided."""
        factory = BuiltinToolFactory(temp_project_dir, "test-session")
        tool = factory.create_tasks_tool()

        assert tool is not None
        assert hasattr(tool, "_project_dir")
        assert hasattr(tool, "_session_id")
        assert tool._project_dir == temp_project_dir
        assert tool._session_id == "test-session"

    def test_factory_returns_none_without_session(self, temp_project_dir):
        """Test factory returns None for tasks tool without session_id."""
        factory = BuiltinToolFactory(temp_project_dir, None)
        tool = factory.create_tasks_tool()

        assert tool is None

    def test_factory_create_all_tools_with_session(self, temp_project_dir):
        """Test factory creates all tools when session_id is provided."""
        factory = BuiltinToolFactory(temp_project_dir, "test-session")
        tools = factory.create_all_tools()

        assert len(tools) >= 1
        # Verify tasks tool is included
        tool_names = [getattr(t, "tool_name", None) for t in tools]
        assert "tasks" in tool_names

    def test_factory_create_all_tools_without_session(self, temp_project_dir):
        """Test factory creates no session-dependent tools without session_id."""
        factory = BuiltinToolFactory(temp_project_dir, None)
        tools = factory.create_all_tools()

        # Tasks tool requires session, so should be empty
        assert len(tools) == 0


class TestBuiltinToolsRegistry:
    """Tests for builtin tools registry and filtering."""

    def test_registry_contains_tasks_tool(self):
        """Test registry contains the tasks tool name."""
        assert "tasks" in BUILTIN_TOOL_NAMES

    def test_get_enabled_builtin_tool_names_none_by_default(self):
        """Test no builtin tools are enabled by default."""
        model_config = Model(
            name="test",
            provider="universal",
            model="test-model",
        )

        enabled = get_enabled_builtin_tool_names(model_config)

        # No tools should be enabled by default
        assert len(enabled) == 0

    def test_get_enabled_builtin_tool_names_with_include(self):
        """Test including specific tools via config."""
        model_config = Model(
            name="test",
            provider="universal",
            model="test-model",
            builtin_tools=BuiltinTools(include=["tasks"]),
        )

        enabled = get_enabled_builtin_tool_names(model_config)

        # Tasks should be included
        assert "tasks" in enabled
        assert len(enabled) == 1

    def test_get_enabled_builtin_tool_names_empty_include(self):
        """Test empty include list disables all builtin tools."""
        model_config = Model(
            name="test",
            provider="universal",
            model="test-model",
            builtin_tools=BuiltinTools(include=[]),
        )

        enabled = get_enabled_builtin_tool_names(model_config)

        assert len(enabled) == 0

    def test_get_enabled_builtin_tool_names_unknown_tool_ignored(self):
        """Test unknown tool names in include list are ignored."""
        model_config = Model(
            name="test",
            provider="universal",
            model="test-model",
            builtin_tools=BuiltinTools(include=["unknown_tool"]),
        )

        enabled = get_enabled_builtin_tool_names(model_config)

        # Unknown tool is ignored, so no tools returned
        assert len(enabled) == 0


class TestTasksToolIntegration:
    """Integration tests for TasksTool with dependencies."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        return "test-session-integration"

    @pytest.fixture
    def tasks_tool(self, temp_project_dir, session_id):
        """Create a TasksTool instance with injected context."""
        class InjectedTasksTool(TasksTool):
            _project_dir = temp_project_dir
            _session_id = session_id

        return InjectedTasksTool()

    @pytest.mark.asyncio
    async def test_completing_task_unblocks_dependents(self, tasks_tool):
        """Test that completing a blocking task unblocks dependent tasks."""
        # Create task 1 (blocker)
        await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Blocking task",
            description="Must complete first",
        ))

        # Create task 2 blocked by task 1
        await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Blocked task",
            description="Waiting on task 1",
            blockedBy=["1"],
        ))

        # Complete task 1
        await tasks_tool.arun(TasksToolInput(
            operation="update",
            taskId="1",
            status="completed",
        ))

        # Get task 2 and verify it's unblocked
        output = await tasks_tool.arun(TasksToolInput(
            operation="get",
            taskId="2",
        ))
        result = json.loads(output.result)

        assert result["success"] is True
        assert "1" not in result["task"]["blockedBy"]

    @pytest.mark.asyncio
    async def test_cycle_detection(self, tasks_tool):
        """Test cycle detection prevents circular dependencies."""
        # Create A
        await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Task A",
            description="First task",
        ))

        # Create B blocked by A
        await tasks_tool.arun(TasksToolInput(
            operation="create",
            subject="Task B",
            description="Blocked by A",
            blockedBy=["1"],
        ))

        # Try to add A blockedBy B (would create cycle)
        output = await tasks_tool.arun(TasksToolInput(
            operation="update",
            taskId="1",
            addBlockedBy=["2"],
        ))
        result = json.loads(output.result)

        # Should fail with cycle error
        assert "error" in result
        assert "cycle" in result["error"].lower()
