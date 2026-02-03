"""
Tests for Builtin Tools Infrastructure.

This module contains comprehensive TDD tests for:
1. Registry (get_enabled_builtin_tool_names) - filtering builtin tools based on config
2. BuiltinToolFactory - creating tool instances with context injection
3. Task Tools - the four task management tool implementations:
   - TaskCreateTool
   - TaskUpdateTool
   - TaskListTool
   - TaskGetTool

Written following TEST-DRIVEN DEVELOPMENT: tests are written before implementation.
"""

import json
import os
import shutil
import tempfile

import pytest
from pydantic import BaseModel

# Import will fail until implementation exists - that's TDD!
# These imports are what we expect the implementation to provide
try:
    from tools.builtin.factory import BuiltinToolFactory
    from tools.builtin.registry import (
        _TOOL_EXPANSION,
        BUILTIN_TOOL_NAMES,
        get_enabled_builtin_tool_names,
    )
    from tools.builtin.tasks_tool import (
        TaskCreateTool,
        TaskGetTool,
        TaskListTool,
        TaskUpdateTool,
    )
except ImportError:
    # Define placeholder classes for type hints in tests
    # These will be replaced by actual imports once implementation exists
    BUILTIN_TOOL_NAMES: set[str] = set()

    def get_enabled_builtin_tool_names(model_config):
        raise NotImplementedError("Registry not implemented yet")

    class BuiltinToolFactory:
        def __init__(self, project_dir: str, session_id: str | None):
            raise NotImplementedError("Factory not implemented yet")

        def create_task_create_tool(self):
            raise NotImplementedError("Factory not implemented yet")

        def create_task_update_tool(self):
            raise NotImplementedError("Factory not implemented yet")

        def create_task_list_tool(self):
            raise NotImplementedError("Factory not implemented yet")

        def create_task_get_tool(self):
            raise NotImplementedError("Factory not implemented yet")

        def create_all_tools(self):
            raise NotImplementedError("Factory not implemented yet")

    class TaskCreateTool:
        pass

    class TaskUpdateTool:
        pass

    class TaskListTool:
        pass

    class TaskGetTool:
        pass


# Import config models - these should already exist
from config.datamodel import BuiltinTools, Model, Provider

# ==============================================================================
# REGISTRY TESTS (get_enabled_builtin_tool_names)
# ==============================================================================


class TestGetEnabledBuiltinToolNames:
    """Test cases for the get_enabled_builtin_tool_names registry function."""

    @pytest.fixture
    def base_model_config(self):
        """Create a minimal model config without builtin_tools specified."""
        return Model(
            name="test-model",
            provider=Provider.ollama,
            model="llama3.2:latest",
        )

    @pytest.fixture
    def model_config_include_all(self):
        """Create model config with all builtin tools included."""
        return Model(
            name="test-model",
            provider=Provider.ollama,
            model="llama3.2:latest",
            builtin_tools=BuiltinTools(include=["tasks"]),
        )

    @pytest.fixture
    def model_config_include_none(self):
        """Create model config with empty include list."""
        return Model(
            name="test-model",
            provider=Provider.ollama,
            model="llama3.2:latest",
            builtin_tools=BuiltinTools(include=[]),
        )

    def test_no_tools_returned_when_no_builtin_tools_config(self, base_model_config):
        """Test that no tools are returned when no builtin_tools config is specified."""
        tools = get_enabled_builtin_tool_names(base_model_config)

        # No builtin tools should be returned (default is disabled)
        assert len(tools) == 0

    def test_tools_returned_when_included(self, model_config_include_all):
        """Test that tools are returned when explicitly included."""
        tools = get_enabled_builtin_tool_names(model_config_include_all)

        # "tasks" should expand to all four task tool names
        assert "task_create" in tools
        assert "task_update" in tools
        assert "task_list" in tools
        assert "task_get" in tools

    def test_no_tools_returned_when_empty_include(self, model_config_include_none):
        """Test that no tools are returned when include list is empty."""
        tools = get_enabled_builtin_tool_names(model_config_include_none)

        # No tools should be returned
        assert len(tools) == 0

    def test_specific_tools_included(self):
        """Test that only tools in the include list are returned."""
        model_config = Model(
            name="test-model",
            provider=Provider.ollama,
            model="llama3.2:latest",
            builtin_tools=BuiltinTools(include=["tasks"]),
        )

        tools = get_enabled_builtin_tool_names(model_config)

        # "tasks" expands to all four task tool names
        assert "task_create" in tools
        assert "task_update" in tools
        assert "task_list" in tools
        assert "task_get" in tools
        assert len(tools) == 4

    def test_unknown_tool_names_in_include_ignored(self):
        """Test that unknown tool names in include list are silently ignored."""
        model_config = Model(
            name="test-model",
            provider=Provider.ollama,
            model="llama3.2:latest",
            builtin_tools=BuiltinTools(
                include=["nonexistent_tool", "another_fake_tool"],
            ),
        )

        # Should not raise an error
        tools = get_enabled_builtin_tool_names(model_config)

        # No tools should be returned since included ones don't exist
        assert len(tools) == 0

    def test_registry_contains_tasks_tool(self):
        """Test that the BUILTIN_TOOL_NAMES registry contains the tasks tool."""
        assert "tasks" in BUILTIN_TOOL_NAMES

    def test_tool_expansion_mapping(self):
        """Test that _TOOL_EXPANSION correctly maps tasks to four tool names."""
        assert "tasks" in _TOOL_EXPANSION
        expected_tools = {"task_create", "task_update", "task_list", "task_get"}
        assert set(_TOOL_EXPANSION["tasks"]) == expected_tools


# ==============================================================================
# BUILTIN TOOL FACTORY TESTS
# ==============================================================================


class TestBuiltinToolFactory:
    """Test cases for BuiltinToolFactory."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        """Provide a consistent session ID for tests."""
        return "test-session-factory"

    def test_factory_creates_task_create_tool(self, temp_project_dir, session_id):
        """Test that factory creates task_create tool when session_id is provided."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tool_class = factory.create_task_create_tool()

        assert tool_class is not None
        assert hasattr(tool_class, "tool_name")
        assert tool_class.tool_name == "task_create"

    def test_factory_creates_task_update_tool(self, temp_project_dir, session_id):
        """Test that factory creates task_update tool when session_id is provided."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tool_class = factory.create_task_update_tool()

        assert tool_class is not None
        assert hasattr(tool_class, "tool_name")
        assert tool_class.tool_name == "task_update"

    def test_factory_creates_task_list_tool(self, temp_project_dir, session_id):
        """Test that factory creates task_list tool when session_id is provided."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tool_class = factory.create_task_list_tool()

        assert tool_class is not None
        assert hasattr(tool_class, "tool_name")
        assert tool_class.tool_name == "task_list"

    def test_factory_creates_task_get_tool(self, temp_project_dir, session_id):
        """Test that factory creates task_get tool when session_id is provided."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tool_class = factory.create_task_get_tool()

        assert tool_class is not None
        assert hasattr(tool_class, "tool_name")
        assert tool_class.tool_name == "task_get"

    def test_factory_returns_none_for_tools_when_no_session_id(self, temp_project_dir):
        """Test that factory returns None for task tools in stateless mode."""
        factory = BuiltinToolFactory(project_dir=temp_project_dir, session_id=None)

        assert factory.create_task_create_tool() is None
        assert factory.create_task_update_tool() is None
        assert factory.create_task_list_tool() is None
        assert factory.create_task_get_tool() is None

    def test_create_all_tools_returns_list(self, temp_project_dir, session_id):
        """Test that create_all_tools returns a list of tool classes."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tools = factory.create_all_tools()

        assert isinstance(tools, list)

    def test_create_all_tools_includes_all_four_task_tools(
        self, temp_project_dir, session_id
    ):
        """Test that create_all_tools includes all four task tools when session_id provided."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tools = factory.create_all_tools()

        tool_names = [getattr(t, "tool_name", None) for t in tools]
        assert "task_create" in tool_names
        assert "task_update" in tool_names
        assert "task_list" in tool_names
        assert "task_get" in tool_names
        assert len(tools) == 4

    def test_create_all_tools_excludes_task_tools_without_session(
        self, temp_project_dir
    ):
        """Test that create_all_tools excludes task tools when no session_id."""
        factory = BuiltinToolFactory(project_dir=temp_project_dir, session_id=None)

        tools = factory.create_all_tools()

        tool_names = [getattr(t, "tool_name", None) for t in tools]
        assert "task_create" not in tool_names
        assert "task_update" not in tool_names
        assert "task_list" not in tool_names
        assert "task_get" not in tool_names
        assert len(tools) == 0

    def test_tool_classes_have_tool_name_attribute(self, temp_project_dir, session_id):
        """Test that tool classes have the correct tool_name attribute."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tools = factory.create_all_tools()

        for tool in tools:
            assert hasattr(tool, "tool_name"), f"Tool {tool} missing tool_name"
            assert isinstance(tool.tool_name, str)

    def test_factory_injects_project_dir_into_tools(self, temp_project_dir, session_id):
        """Test that factory injects project_dir into all task tools."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        for creator in [
            factory.create_task_create_tool,
            factory.create_task_update_tool,
            factory.create_task_list_tool,
            factory.create_task_get_tool,
        ]:
            tool_class = creator()
            assert hasattr(tool_class, "_project_dir")
            assert tool_class._project_dir == temp_project_dir

    def test_factory_injects_session_id_into_tools(self, temp_project_dir, session_id):
        """Test that factory injects session_id into all task tools."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        for creator in [
            factory.create_task_create_tool,
            factory.create_task_update_tool,
            factory.create_task_list_tool,
            factory.create_task_get_tool,
        ]:
            tool_class = creator()
            assert hasattr(tool_class, "_session_id")
            assert tool_class._session_id == session_id


# ==============================================================================
# TASK CREATE TOOL TESTS
# ==============================================================================


class TestTaskCreateToolSchema:
    """Test cases for TaskCreateTool input/output schemas."""

    def test_has_input_schema(self):
        """Test that TaskCreateTool has an input_schema class attribute."""
        assert hasattr(TaskCreateTool, "input_schema")

        input_schema = TaskCreateTool.input_schema
        assert issubclass(input_schema, BaseModel)

    def test_input_schema_has_required_fields(self):
        """Test that input schema has subject and description as required fields."""
        input_schema = TaskCreateTool.input_schema
        schema_dict = input_schema.model_json_schema()

        assert "properties" in schema_dict
        assert "subject" in schema_dict["properties"]
        assert "description" in schema_dict["properties"]

        # Check required fields
        required = schema_dict.get("required", [])
        assert "subject" in required
        assert "description" in required

    def test_input_schema_has_optional_fields(self):
        """Test that input schema has activeForm and metadata as optional fields."""
        input_schema = TaskCreateTool.input_schema
        schema_dict = input_schema.model_json_schema()

        assert "activeForm" in schema_dict["properties"]
        assert "metadata" in schema_dict["properties"]

        # These should not be required
        required = schema_dict.get("required", [])
        assert "activeForm" not in required
        assert "metadata" not in required

    def test_has_output_schema(self):
        """Test that TaskCreateTool has an output_schema class attribute."""
        assert hasattr(TaskCreateTool, "output_schema")

        output_schema = TaskCreateTool.output_schema
        assert issubclass(output_schema, BaseModel)

    def test_has_tool_name(self):
        """Test that TaskCreateTool has tool_name attribute."""
        assert hasattr(TaskCreateTool, "tool_name")
        assert TaskCreateTool.tool_name == "task_create"


class TestTaskCreateToolOperations:
    """Test cases for TaskCreateTool operation execution."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        """Provide a consistent session ID for tests."""
        return "test-session-create"

    @pytest.fixture
    def create_tool(self, temp_project_dir, session_id):
        """Create a TaskCreateTool instance with injected context."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )
        tool_class = factory.create_task_create_tool()
        return tool_class()

    @pytest.mark.asyncio
    async def test_create_task_minimal(self, create_tool):
        """Test creating a task with only required fields."""
        input_schema = create_tool.input_schema
        tool_input = input_schema(
            subject="Test task",
            description="A test task description",
        )

        result = await create_tool.arun(tool_input)

        assert result is not None
        result_data = json.loads(result.result)
        assert result_data["success"] is True
        assert "task" in result_data
        assert result_data["task"]["subject"] == "Test task"
        assert result_data["task"]["description"] == "A test task description"
        assert result_data["task"]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_create_task_full(self, create_tool):
        """Test creating a task with all fields including metadata."""
        input_schema = create_tool.input_schema
        tool_input = input_schema(
            subject="Full task",
            description="Task with all fields",
            activeForm="Creating full task",
            metadata={"priority": "high", "estimate": 4},
        )

        result = await create_tool.arun(tool_input)

        assert result is not None
        result_data = json.loads(result.result)
        assert result_data["success"] is True
        assert result_data["task"]["activeForm"] == "Creating full task"
        assert result_data["task"]["metadata"] == {"priority": "high", "estimate": 4}

    @pytest.mark.asyncio
    async def test_create_task_returns_id(self, create_tool):
        """Test that created task has a valid ID."""
        input_schema = create_tool.input_schema
        tool_input = input_schema(
            subject="Task for ID test",
            description="Testing ID generation",
        )

        result = await create_tool.arun(tool_input)

        result_data = json.loads(result.result)
        assert "task" in result_data
        assert "id" in result_data["task"]
        assert result_data["task"]["id"] == "1"


# ==============================================================================
# TASK UPDATE TOOL TESTS
# ==============================================================================


class TestTaskUpdateToolSchema:
    """Test cases for TaskUpdateTool input/output schemas."""

    def test_has_input_schema(self):
        """Test that TaskUpdateTool has an input_schema class attribute."""
        assert hasattr(TaskUpdateTool, "input_schema")

        input_schema = TaskUpdateTool.input_schema
        assert issubclass(input_schema, BaseModel)

    def test_input_schema_has_task_id_required(self):
        """Test that input schema has taskId as required field."""
        input_schema = TaskUpdateTool.input_schema
        schema_dict = input_schema.model_json_schema()

        assert "taskId" in schema_dict["properties"]
        required = schema_dict.get("required", [])
        assert "taskId" in required

    def test_input_schema_has_optional_update_fields(self):
        """Test that input schema has optional fields for updates."""
        input_schema = TaskUpdateTool.input_schema
        schema_dict = input_schema.model_json_schema()

        # All update fields should be present
        assert "status" in schema_dict["properties"]
        assert "owner" in schema_dict["properties"]
        assert "subject" in schema_dict["properties"]
        assert "description" in schema_dict["properties"]
        assert "activeForm" in schema_dict["properties"]
        assert "addBlockedBy" in schema_dict["properties"]
        assert "addBlocks" in schema_dict["properties"]
        assert "metadata" in schema_dict["properties"]

    def test_has_tool_name(self):
        """Test that TaskUpdateTool has tool_name attribute."""
        assert hasattr(TaskUpdateTool, "tool_name")
        assert TaskUpdateTool.tool_name == "task_update"


class TestTaskUpdateToolOperations:
    """Test cases for TaskUpdateTool operation execution."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        """Provide a consistent session ID for tests."""
        return "test-session-update"

    @pytest.fixture
    def tools(self, temp_project_dir, session_id):
        """Create all task tools with injected context."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )
        return {
            "create": factory.create_task_create_tool()(),
            "update": factory.create_task_update_tool()(),
            "list": factory.create_task_list_tool()(),
            "get": factory.create_task_get_tool()(),
        }

    @pytest.mark.asyncio
    async def test_update_status_transitions(self, tools):
        """Test status transitions: pending -> in_progress -> completed."""
        create_tool = tools["create"]
        update_tool = tools["update"]

        # Create a task
        create_result = await create_tool.arun(
            create_tool.input_schema(
                subject="Status test task",
                description="Testing status transitions",
            )
        )
        task_data = json.loads(create_result.result)["task"]
        assert task_data["status"] == "pending"

        # Update to in_progress
        update_result = await update_tool.arun(
            update_tool.input_schema(
                taskId=task_data["id"],
                status="in_progress",
            )
        )
        updated_data = json.loads(update_result.result)
        assert updated_data["success"] is True
        assert updated_data["task"]["status"] == "in_progress"

        # Update to completed
        update_result = await update_tool.arun(
            update_tool.input_schema(
                taskId=task_data["id"],
                status="completed",
            )
        )
        updated_data = json.loads(update_result.result)
        assert updated_data["success"] is True
        assert updated_data["task"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_update_owner(self, tools):
        """Test updating task owner."""
        create_tool = tools["create"]
        update_tool = tools["update"]

        # Create a task
        create_result = await create_tool.arun(
            create_tool.input_schema(
                subject="Owner test task",
                description="Testing owner updates",
            )
        )
        task_data = json.loads(create_result.result)["task"]

        # Update owner
        update_result = await update_tool.arun(
            update_tool.input_schema(
                taskId=task_data["id"],
                owner="agent-123",
            )
        )
        updated_data = json.loads(update_result.result)
        assert updated_data["success"] is True
        assert updated_data["task"]["owner"] == "agent-123"

    @pytest.mark.asyncio
    async def test_update_adds_blocked_by(self, tools):
        """Test adding blockedBy dependencies."""
        create_tool = tools["create"]
        update_tool = tools["update"]

        # Create two tasks
        await create_tool.arun(
            create_tool.input_schema(subject="Task 1", description="First")
        )
        create_result2 = await create_tool.arun(
            create_tool.input_schema(subject="Task 2", description="Second")
        )
        task2_data = json.loads(create_result2.result)["task"]

        # Update task 2 to be blocked by task 1
        update_result = await update_tool.arun(
            update_tool.input_schema(
                taskId=task2_data["id"],
                addBlockedBy=["1"],
            )
        )
        updated_data = json.loads(update_result.result)
        assert updated_data["success"] is True
        assert "1" in updated_data["task"]["blockedBy"]

    @pytest.mark.asyncio
    async def test_update_adds_blocks(self, tools):
        """Test adding blocks dependencies."""
        create_tool = tools["create"]
        update_tool = tools["update"]

        # Create two tasks
        create_result1 = await create_tool.arun(
            create_tool.input_schema(subject="Task 1", description="First")
        )
        task1_data = json.loads(create_result1.result)["task"]
        await create_tool.arun(
            create_tool.input_schema(subject="Task 2", description="Second")
        )

        # Update task 1 to block task 2
        update_result = await update_tool.arun(
            update_tool.input_schema(
                taskId=task1_data["id"],
                addBlocks=["2"],
            )
        )
        updated_data = json.loads(update_result.result)
        assert updated_data["success"] is True
        assert "2" in updated_data["task"]["blocks"]

    @pytest.mark.asyncio
    async def test_update_completion_unblocks(self, tools):
        """Test that completing a task unblocks dependent tasks."""
        create_tool = tools["create"]
        update_tool = tools["update"]
        get_tool = tools["get"]

        # Create task 1 (blocker)
        await create_tool.arun(
            create_tool.input_schema(subject="Blocker", description="Blocks task 2")
        )
        # Create task 2 blocked by task 1
        await create_tool.arun(
            create_tool.input_schema(subject="Blocked", description="Blocked by task 1")
        )
        await update_tool.arun(update_tool.input_schema(taskId="2", addBlockedBy=["1"]))

        # Complete task 1
        await update_tool.arun(update_tool.input_schema(taskId="1", status="completed"))

        # Check task 2 is no longer blocked
        get_result = await get_tool.arun(get_tool.input_schema(taskId="2"))
        task2_data = json.loads(get_result.result)["task"]
        assert "1" not in task2_data["blockedBy"]

    @pytest.mark.asyncio
    async def test_update_delete_status(self, tools):
        """Test that status='deleted' removes the task."""
        create_tool = tools["create"]
        update_tool = tools["update"]
        list_tool = tools["list"]

        # Create a task
        await create_tool.arun(
            create_tool.input_schema(subject="To delete", description="Will be deleted")
        )

        # Delete via update
        update_result = await update_tool.arun(
            update_tool.input_schema(taskId="1", status="deleted")
        )
        result_data = json.loads(update_result.result)
        assert result_data["success"] is True
        assert result_data["deleted"] is True

        # Verify task is gone
        list_result = await list_tool.arun(list_tool.input_schema())
        list_data = json.loads(list_result.result)
        assert list_data["tasks"] == []

    @pytest.mark.asyncio
    async def test_update_metadata(self, tools):
        """Test merging and deleting metadata keys."""
        create_tool = tools["create"]
        update_tool = tools["update"]

        # Create task with initial metadata
        create_result = await create_tool.arun(
            create_tool.input_schema(
                subject="Metadata test",
                description="Testing metadata",
                metadata={"key1": "value1", "key2": "value2"},
            )
        )
        task_data = json.loads(create_result.result)["task"]

        # Update metadata: update key2, add key3, delete key1
        update_result = await update_tool.arun(
            update_tool.input_schema(
                taskId=task_data["id"],
                metadata={"key2": "updated", "key3": "new", "key1": None},
            )
        )
        updated_data = json.loads(update_result.result)
        assert updated_data["task"]["metadata"] == {"key2": "updated", "key3": "new"}

    @pytest.mark.asyncio
    async def test_update_not_found(self, tools):
        """Test updating a non-existent task returns error."""
        update_tool = tools["update"]

        update_result = await update_tool.arun(
            update_tool.input_schema(taskId="999", subject="Updated")
        )
        result_data = json.loads(update_result.result)
        assert "error" in result_data
        assert "not found" in result_data["error"].lower()

    @pytest.mark.asyncio
    async def test_update_cycle_detection(self, tools):
        """Test that cycle-creating updates are rejected."""
        create_tool = tools["create"]
        update_tool = tools["update"]

        # Create A -> B chain
        await create_tool.arun(
            create_tool.input_schema(subject="Task A", description="Root")
        )
        await create_tool.arun(
            create_tool.input_schema(subject="Task B", description="Blocked by A")
        )
        await update_tool.arun(update_tool.input_schema(taskId="2", addBlockedBy=["1"]))

        # Try to create cycle by making A blocked by B
        update_result = await update_tool.arun(
            update_tool.input_schema(taskId="1", addBlockedBy=["2"])
        )
        result_data = json.loads(update_result.result)
        assert "error" in result_data
        assert "cycle" in result_data["error"].lower()

    @pytest.mark.asyncio
    async def test_update_reopen_completed_rejected(self, tools):
        """Test that reopening a completed task is rejected."""
        create_tool = tools["create"]
        update_tool = tools["update"]

        # Create and complete a task
        await create_tool.arun(
            create_tool.input_schema(
                subject="Completed task", description="Will be completed"
            )
        )
        await update_tool.arun(update_tool.input_schema(taskId="1", status="completed"))

        # Try to reopen
        update_result = await update_tool.arun(
            update_tool.input_schema(taskId="1", status="pending")
        )
        result_data = json.loads(update_result.result)
        assert "error" in result_data
        assert "reopen" in result_data["error"].lower()


# ==============================================================================
# TASK LIST TOOL TESTS
# ==============================================================================


class TestTaskListToolSchema:
    """Test cases for TaskListTool input/output schemas."""

    def test_has_input_schema(self):
        """Test that TaskListTool has an input_schema class attribute."""
        assert hasattr(TaskListTool, "input_schema")

        input_schema = TaskListTool.input_schema
        assert issubclass(input_schema, BaseModel)

    def test_input_schema_has_no_required_fields(self):
        """Test that input schema has no required fields."""
        input_schema = TaskListTool.input_schema
        schema_dict = input_schema.model_json_schema()

        # No required fields
        required = schema_dict.get("required", [])
        assert len(required) == 0

    def test_has_tool_name(self):
        """Test that TaskListTool has tool_name attribute."""
        assert hasattr(TaskListTool, "tool_name")
        assert TaskListTool.tool_name == "task_list"


class TestTaskListToolOperations:
    """Test cases for TaskListTool operation execution."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        """Provide a consistent session ID for tests."""
        return "test-session-list"

    @pytest.fixture
    def tools(self, temp_project_dir, session_id):
        """Create all task tools with injected context."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )
        return {
            "create": factory.create_task_create_tool()(),
            "update": factory.create_task_update_tool()(),
            "list": factory.create_task_list_tool()(),
        }

    @pytest.mark.asyncio
    async def test_list_empty(self, tools):
        """Test listing when no tasks exist."""
        list_tool = tools["list"]

        result = await list_tool.arun(list_tool.input_schema())

        result_data = json.loads(result.result)
        assert result_data["success"] is True
        assert result_data["tasks"] == []
        assert "No tasks found" in result_data.get("message", "")

    @pytest.mark.asyncio
    async def test_list_returns_all(self, tools):
        """Test that list returns all tasks."""
        create_tool = tools["create"]
        list_tool = tools["list"]

        # Create multiple tasks
        await create_tool.arun(
            create_tool.input_schema(subject="Task 1", description="First")
        )
        await create_tool.arun(
            create_tool.input_schema(subject="Task 2", description="Second")
        )
        await create_tool.arun(
            create_tool.input_schema(subject="Task 3", description="Third")
        )

        result = await list_tool.arun(list_tool.input_schema())

        result_data = json.loads(result.result)
        assert result_data["success"] is True
        assert len(result_data["tasks"]) == 3
        subjects = [t["subject"] for t in result_data["tasks"]]
        assert "Task 1" in subjects
        assert "Task 2" in subjects
        assert "Task 3" in subjects

    @pytest.mark.asyncio
    async def test_list_shows_blocked_status(self, tools):
        """Test that list shows blockedBy information."""
        create_tool = tools["create"]
        update_tool = tools["update"]
        list_tool = tools["list"]

        # Create two tasks with dependency
        await create_tool.arun(
            create_tool.input_schema(subject="Blocker", description="Blocks others")
        )
        await create_tool.arun(
            create_tool.input_schema(subject="Blocked", description="Blocked by task 1")
        )
        await update_tool.arun(update_tool.input_schema(taskId="2", addBlockedBy=["1"]))

        result = await list_tool.arun(list_tool.input_schema())

        result_data = json.loads(result.result)
        tasks = result_data["tasks"]
        task2 = next(t for t in tasks if t["subject"] == "Blocked")
        assert "1" in task2["blockedBy"]


# ==============================================================================
# TASK GET TOOL TESTS
# ==============================================================================


class TestTaskGetToolSchema:
    """Test cases for TaskGetTool input/output schemas."""

    def test_has_input_schema(self):
        """Test that TaskGetTool has an input_schema class attribute."""
        assert hasattr(TaskGetTool, "input_schema")

        input_schema = TaskGetTool.input_schema
        assert issubclass(input_schema, BaseModel)

    def test_input_schema_has_task_id_required(self):
        """Test that input schema has taskId as required field."""
        input_schema = TaskGetTool.input_schema
        schema_dict = input_schema.model_json_schema()

        assert "taskId" in schema_dict["properties"]
        required = schema_dict.get("required", [])
        assert "taskId" in required

    def test_has_tool_name(self):
        """Test that TaskGetTool has tool_name attribute."""
        assert hasattr(TaskGetTool, "tool_name")
        assert TaskGetTool.tool_name == "task_get"


class TestTaskGetToolOperations:
    """Test cases for TaskGetTool operation execution."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        """Provide a consistent session ID for tests."""
        return "test-session-get"

    @pytest.fixture
    def tools(self, temp_project_dir, session_id):
        """Create all task tools with injected context."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )
        return {
            "create": factory.create_task_create_tool()(),
            "get": factory.create_task_get_tool()(),
        }

    @pytest.mark.asyncio
    async def test_get_existing(self, tools):
        """Test getting an existing task returns full details."""
        create_tool = tools["create"]
        get_tool = tools["get"]

        # Create a task with all fields
        await create_tool.arun(
            create_tool.input_schema(
                subject="Detailed task",
                description="Full details here",
                activeForm="Getting task details",
                metadata={"key": "value"},
            )
        )

        result = await get_tool.arun(get_tool.input_schema(taskId="1"))

        result_data = json.loads(result.result)
        assert result_data["success"] is True
        task = result_data["task"]
        assert task["id"] == "1"
        assert task["subject"] == "Detailed task"
        assert task["description"] == "Full details here"
        assert task["activeForm"] == "Getting task details"
        assert task["metadata"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_not_found(self, tools):
        """Test getting a non-existent task returns error."""
        get_tool = tools["get"]

        result = await get_tool.arun(get_tool.input_schema(taskId="999"))

        result_data = json.loads(result.result)
        assert "error" in result_data
        assert "not found" in result_data["error"].lower()


# ==============================================================================
# TASKS TOOL WITH DEPENDENCIES TESTS
# ==============================================================================


class TestTaskToolsWithDependencies:
    """Test cases for task tools with task dependencies."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        """Provide a consistent session ID for tests."""
        return "test-session-deps"

    @pytest.fixture
    def tools(self, temp_project_dir, session_id):
        """Create all task tools with injected context."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )
        return {
            "create": factory.create_task_create_tool()(),
            "update": factory.create_task_update_tool()(),
            "list": factory.create_task_list_tool()(),
            "get": factory.create_task_get_tool()(),
        }

    @pytest.mark.asyncio
    async def test_complex_dependency_chain(self, tools):
        """Test a chain of dependencies: A -> B -> C."""
        create_tool = tools["create"]
        update_tool = tools["update"]
        get_tool = tools["get"]

        # Create tasks
        await create_tool.arun(
            create_tool.input_schema(subject="Task A", description="Root")
        )
        await create_tool.arun(
            create_tool.input_schema(subject="Task B", description="Middle")
        )
        await create_tool.arun(
            create_tool.input_schema(subject="Task C", description="End")
        )

        # Set up dependencies
        await update_tool.arun(update_tool.input_schema(taskId="2", addBlockedBy=["1"]))
        await update_tool.arun(update_tool.input_schema(taskId="3", addBlockedBy=["2"]))

        # Verify chain
        result_a = await get_tool.arun(get_tool.input_schema(taskId="1"))
        result_b = await get_tool.arun(get_tool.input_schema(taskId="2"))
        result_c = await get_tool.arun(get_tool.input_schema(taskId="3"))

        task_a = json.loads(result_a.result)["task"]
        task_b = json.loads(result_b.result)["task"]
        task_c = json.loads(result_c.result)["task"]

        assert "2" in task_a["blocks"]
        assert "1" in task_b["blockedBy"]
        assert "3" in task_b["blocks"]
        assert "2" in task_c["blockedBy"]

        # Complete A -> B should be unblocked
        await update_tool.arun(update_tool.input_schema(taskId="1", status="completed"))
        result_b = await get_tool.arun(get_tool.input_schema(taskId="2"))
        task_b = json.loads(result_b.result)["task"]
        assert task_b["blockedBy"] == []

        # Complete B -> C should be unblocked
        await update_tool.arun(update_tool.input_schema(taskId="2", status="completed"))
        result_c = await get_tool.arun(get_tool.input_schema(taskId="3"))
        task_c = json.loads(result_c.result)["task"]
        assert task_c["blockedBy"] == []


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestBuiltinToolsIntegration:
    """Integration tests combining registry, factory, and tool execution."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_id(self):
        """Provide a consistent session ID for tests."""
        return "test-session-integration"

    def test_registry_and_factory_tool_names_match(self, temp_project_dir, session_id):
        """Test that registry tool names match factory-created tool names."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )
        factory_tools = factory.create_all_tools()

        factory_tool_names = {getattr(t, "tool_name", None) for t in factory_tools}

        # Get all expanded tool names from the registry
        all_expanded_names = set()
        for names in _TOOL_EXPANSION.values():
            all_expanded_names.update(names)

        # Factory tools should be a subset of expanded registry tools
        # (some tools may not be created if no session_id, etc.)
        for name in factory_tool_names:
            if name is not None:
                assert name in all_expanded_names

    @pytest.mark.asyncio
    async def test_full_task_workflow(self, temp_project_dir, session_id):
        """Test full workflow: create -> update -> get -> list -> delete."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )
        tools = {
            "create": factory.create_task_create_tool()(),
            "update": factory.create_task_update_tool()(),
            "list": factory.create_task_list_tool()(),
            "get": factory.create_task_get_tool()(),
        }

        # Create
        create_result = await tools["create"].arun(
            tools["create"].input_schema(
                subject="Full workflow task",
                description="Testing complete workflow",
            )
        )
        create_data = json.loads(create_result.result)
        assert create_data["success"] is True
        task_id = create_data["task"]["id"]

        # Update
        update_result = await tools["update"].arun(
            tools["update"].input_schema(
                taskId=task_id,
                status="in_progress",
            )
        )
        update_data = json.loads(update_result.result)
        assert update_data["success"] is True
        assert update_data["task"]["status"] == "in_progress"

        # Get
        get_result = await tools["get"].arun(tools["get"].input_schema(taskId=task_id))
        get_data = json.loads(get_result.result)
        assert get_data["success"] is True
        assert get_data["task"]["status"] == "in_progress"

        # List
        list_result = await tools["list"].arun(tools["list"].input_schema())
        list_data = json.loads(list_result.result)
        assert list_data["success"] is True
        assert len(list_data["tasks"]) == 1
        assert list_data["tasks"][0]["subject"] == "Full workflow task"

        # Delete
        delete_result = await tools["update"].arun(
            tools["update"].input_schema(
                taskId=task_id,
                status="deleted",
            )
        )
        delete_data = json.loads(delete_result.result)
        assert delete_data["success"] is True
        assert delete_data["deleted"] is True

        # Verify deleted
        list_result_after = await tools["list"].arun(tools["list"].input_schema())
        list_data_after = json.loads(list_result_after.result)
        assert list_data_after["tasks"] == []


# ==============================================================================
# SCHEMA INTEGRATION TESTS
# ==============================================================================


class TestBuiltinToolsSchemaIntegration:
    """Tests for builtin_tools schema integration."""

    def test_model_config_accepts_builtin_tools(self):
        """Test Model can be instantiated with builtin_tools config."""
        from config.datamodel import BuiltinTools, Model

        model = Model(
            name="test",
            provider="openai",
            model="gpt-4",
            builtin_tools=BuiltinTools(include=["tasks"]),
        )
        assert model.builtin_tools.include == ["tasks"]

    def test_model_config_builtin_tools_defaults(self):
        """Test builtin_tools has correct defaults when not specified."""
        from config.datamodel import Model

        model = Model(name="test", provider="openai", model="gpt-4")
        # builtin_tools should be None (not specified)
        assert model.builtin_tools is None

    def test_builtin_tools_include_defaults_to_none(self):
        """Test BuiltinTools.include defaults to None."""
        from config.datamodel import BuiltinTools

        bt = BuiltinTools()
        assert bt.include is None

    def test_config_yaml_with_builtin_tools_parses(self, tmp_path):
        """Test loading a YAML config with builtin_tools."""
        from config.helpers.loader import load_config

        config_content = """
version: v1
name: test
namespace: test
runtime:
  models:
    - name: default
      provider: openai
      model: gpt-4
      builtin_tools:
        include:
          - tasks
"""
        config_file = tmp_path / "llamafarm.yaml"
        config_file.write_text(config_content)

        config = load_config(str(config_file))
        model = config.runtime.models[0]
        assert model.builtin_tools is not None
        assert "tasks" in model.builtin_tools.include
