"""
Tests for Builtin Tools Infrastructure.

This module contains comprehensive TDD tests for:
1. Registry (get_enabled_builtin_tools) - filtering builtin tools based on config
2. BuiltinToolFactory - creating tool instances with context injection
3. TasksTool - the tasks management tool implementation

Written following TEST-DRIVEN DEVELOPMENT: tests are written before implementation.
"""

import os
import shutil
import tempfile

import pytest
from pydantic import BaseModel

# Import will fail until implementation exists - that's TDD!
# These imports are what we expect the implementation to provide
try:
    from tools.builtin.factory import BuiltinToolFactory
    from tools.builtin.registry import BUILTIN_TOOLS, get_enabled_builtin_tools
    from tools.builtin.tasks_tool import TasksTool
except ImportError:
    # Define placeholder classes for type hints in tests
    # These will be replaced by actual imports once implementation exists
    BUILTIN_TOOLS = {}

    def get_enabled_builtin_tools(model_config):
        raise NotImplementedError("Registry not implemented yet")

    class BuiltinToolFactory:
        def __init__(self, project_dir: str, session_id: str | None):
            raise NotImplementedError("Factory not implemented yet")

        def create_tasks_tool(self):
            raise NotImplementedError("Factory not implemented yet")

        def create_all_tools(self):
            raise NotImplementedError("Factory not implemented yet")

    class TasksTool:
        pass


# Import config models - these should already exist
from config.datamodel import BuiltinTools, Model, Provider

# ==============================================================================
# REGISTRY TESTS (get_enabled_builtin_tools)
# ==============================================================================


class TestGetEnabledBuiltinTools:
    """Test cases for the get_enabled_builtin_tools registry function."""

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
        tools = get_enabled_builtin_tools(base_model_config)

        # No builtin tools should be returned (default is disabled)
        assert len(tools) == 0

    def test_tools_returned_when_included(self, model_config_include_all):
        """Test that tools are returned when explicitly included."""
        tools = get_enabled_builtin_tools(model_config_include_all)

        # Only included tools should be returned
        tool_names = {t.name for t in tools}
        assert "tasks" in tool_names

    def test_no_tools_returned_when_empty_include(self, model_config_include_none):
        """Test that no tools are returned when include list is empty."""
        tools = get_enabled_builtin_tools(model_config_include_none)

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

        tools = get_enabled_builtin_tools(model_config)

        tool_names = {t.name for t in tools}
        assert "tasks" in tool_names
        assert len(tools) == 1

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
        tools = get_enabled_builtin_tools(model_config)

        # No tools should be returned since included ones don't exist
        assert len(tools) == 0

    def test_registry_contains_tasks_tool(self):
        """Test that the BUILTIN_TOOLS registry contains the tasks tool."""
        assert "tasks" in BUILTIN_TOOLS

        tasks_tool_def = BUILTIN_TOOLS["tasks"]
        assert tasks_tool_def.name == "tasks"
        assert tasks_tool_def.description is not None
        assert "parameters" in dir(tasks_tool_def) or hasattr(
            tasks_tool_def, "parameters"
        )

    def test_tool_definitions_have_required_fields(self):
        """Test that all tool definitions have required name, description, parameters."""
        for name, tool_def in BUILTIN_TOOLS.items():
            assert tool_def.name == name, f"Tool {name} has mismatched name"
            assert tool_def.description, f"Tool {name} missing description"
            assert isinstance(
                tool_def.parameters, dict
            ), f"Tool {name} parameters not a dict"
            assert (
                tool_def.parameters.get("type") == "object"
            ), f"Tool {name} parameters not object type"


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

    def test_factory_creates_tasks_tool_when_session_id_provided(
        self, temp_project_dir, session_id
    ):
        """Test that factory creates tasks tool when session_id is provided."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tasks_tool = factory.create_tasks_tool()

        assert tasks_tool is not None
        # The tool should be a class that can be instantiated
        assert hasattr(tasks_tool, "mcp_tool_name")
        assert tasks_tool.mcp_tool_name == "tasks"

    def test_factory_returns_none_for_tasks_tool_when_no_session_id(
        self, temp_project_dir
    ):
        """Test that factory returns None for tasks tool in stateless mode (no session_id)."""
        factory = BuiltinToolFactory(project_dir=temp_project_dir, session_id=None)

        tasks_tool = factory.create_tasks_tool()

        assert tasks_tool is None

    def test_create_all_tools_returns_list(self, temp_project_dir, session_id):
        """Test that create_all_tools returns a list of tool classes."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tools = factory.create_all_tools()

        assert isinstance(tools, list)

    def test_create_all_tools_includes_tasks_tool_with_session(
        self, temp_project_dir, session_id
    ):
        """Test that create_all_tools includes tasks tool when session_id provided."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tools = factory.create_all_tools()

        tool_names = [getattr(t, "mcp_tool_name", None) for t in tools]
        assert "tasks" in tool_names

    def test_create_all_tools_excludes_tasks_tool_without_session(
        self, temp_project_dir
    ):
        """Test that create_all_tools excludes tasks tool when no session_id."""
        factory = BuiltinToolFactory(project_dir=temp_project_dir, session_id=None)

        tools = factory.create_all_tools()

        tool_names = [getattr(t, "mcp_tool_name", None) for t in tools]
        assert "tasks" not in tool_names

    def test_tool_classes_have_mcp_tool_name_attribute(
        self, temp_project_dir, session_id
    ):
        """Test that tool classes have the correct mcp_tool_name attribute."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tools = factory.create_all_tools()

        for tool in tools:
            assert hasattr(tool, "mcp_tool_name"), f"Tool {tool} missing mcp_tool_name"
            assert isinstance(tool.mcp_tool_name, str)

    def test_factory_injects_project_dir_into_tasks_tool(
        self, temp_project_dir, session_id
    ):
        """Test that factory injects project_dir into tasks tool."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tasks_tool_class = factory.create_tasks_tool()

        # The class should have the project_dir set
        assert hasattr(tasks_tool_class, "_project_dir")
        assert tasks_tool_class._project_dir == temp_project_dir

    def test_factory_injects_session_id_into_tasks_tool(
        self, temp_project_dir, session_id
    ):
        """Test that factory injects session_id into tasks tool."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )

        tasks_tool_class = factory.create_tasks_tool()

        # The class should have the session_id set
        assert hasattr(tasks_tool_class, "_session_id")
        assert tasks_tool_class._session_id == session_id


# ==============================================================================
# TASKS TOOL TESTS
# ==============================================================================


class TestTasksToolSchema:
    """Test cases for TasksTool input/output schemas."""

    def test_tasks_tool_has_input_schema(self):
        """Test that TasksTool has an input_schema class attribute."""
        assert hasattr(TasksTool, "input_schema")

        input_schema = TasksTool.input_schema
        assert issubclass(input_schema, BaseModel)

    def test_tasks_tool_input_schema_has_operation_field(self):
        """Test that input schema has operation field for discriminated operations."""
        input_schema = TasksTool.input_schema
        schema_dict = input_schema.model_json_schema()

        assert "properties" in schema_dict
        assert "operation" in schema_dict["properties"]

    def test_tasks_tool_input_schema_operations(self):
        """Test that input schema supports create, update, list, get operations."""
        input_schema = TasksTool.input_schema
        schema_dict = input_schema.model_json_schema()

        operation_schema = schema_dict["properties"]["operation"]
        # Should be an enum with specific values
        assert "enum" in operation_schema
        operations = operation_schema["enum"]
        assert "create" in operations
        assert "update" in operations
        assert "list" in operations
        assert "get" in operations

    def test_tasks_tool_has_output_schema(self):
        """Test that TasksTool has an output_schema class attribute."""
        assert hasattr(TasksTool, "output_schema")

        output_schema = TasksTool.output_schema
        assert issubclass(output_schema, BaseModel)

    def test_tasks_tool_has_mcp_tool_name(self):
        """Test that TasksTool has mcp_tool_name attribute."""
        assert hasattr(TasksTool, "mcp_tool_name")
        assert TasksTool.mcp_tool_name == "tasks"


class TestTasksToolOperations:
    """Test cases for TasksTool operation execution."""

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
        return "test-session-tool"

    @pytest.fixture
    def tasks_tool(self, temp_project_dir, session_id):
        """Create a TasksTool instance with injected context."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )
        tool_class = factory.create_tasks_tool()
        # Instantiate the tool
        return tool_class()

    @pytest.mark.asyncio
    async def test_create_operation_creates_task(self, tasks_tool):
        """Test that create operation creates a task and returns result."""
        input_schema = tasks_tool.input_schema
        tool_input = input_schema(
            operation="create",
            subject="Test task",
            description="A test task description",
            activeForm="Testing task creation",
        )

        result = await tasks_tool.arun(tool_input)

        # Result should indicate success and include task ID
        assert result is not None
        assert hasattr(result, "result") or isinstance(result, str)
        result_str = result.result if hasattr(result, "result") else str(result)
        assert "1" in result_str  # First task should have ID 1

    @pytest.mark.asyncio
    async def test_list_operation_returns_task_list(self, tasks_tool):
        """Test that list operation returns list of tasks."""
        input_schema = tasks_tool.input_schema

        # Create a task first
        create_input = input_schema(
            operation="create",
            subject="Task 1",
            description="First task",
        )
        await tasks_tool.arun(create_input)

        # Now list tasks
        list_input = input_schema(operation="list")
        result = await tasks_tool.arun(list_input)

        assert result is not None
        result_str = result.result if hasattr(result, "result") else str(result)
        assert "Task 1" in result_str

    @pytest.mark.asyncio
    async def test_get_operation_returns_task_details(self, tasks_tool):
        """Test that get operation returns task details."""
        input_schema = tasks_tool.input_schema

        # Create a task first
        create_input = input_schema(
            operation="create",
            subject="Detailed task",
            description="Task with full details",
            activeForm="Getting task details",
        )
        await tasks_tool.arun(create_input)

        # Get the task
        get_input = input_schema(operation="get", taskId="1")
        result = await tasks_tool.arun(get_input)

        assert result is not None
        result_str = result.result if hasattr(result, "result") else str(result)
        assert "Detailed task" in result_str
        assert "Task with full details" in result_str

    @pytest.mark.asyncio
    async def test_update_operation_updates_task(self, tasks_tool):
        """Test that update operation updates a task."""
        input_schema = tasks_tool.input_schema

        # Create a task first
        create_input = input_schema(
            operation="create",
            subject="Original subject",
            description="Original description",
        )
        await tasks_tool.arun(create_input)

        # Update the task
        update_input = input_schema(
            operation="update",
            taskId="1",
            status="in_progress",
            subject="Updated subject",
        )
        result = await tasks_tool.arun(update_input)

        assert result is not None
        result_str = result.result if hasattr(result, "result") else str(result)
        assert "Updated subject" in result_str or "in_progress" in result_str

    @pytest.mark.asyncio
    async def test_delete_via_update_removes_task(self, tasks_tool):
        """Test that update with status=deleted removes task."""
        input_schema = tasks_tool.input_schema

        # Create a task first
        create_input = input_schema(
            operation="create",
            subject="Task to delete",
            description="Will be deleted",
        )
        await tasks_tool.arun(create_input)

        # Delete via update
        delete_input = input_schema(
            operation="update",
            taskId="1",
            status="deleted",
        )
        result = await tasks_tool.arun(delete_input)

        assert result is not None

        # Task should no longer appear in list
        list_input = input_schema(operation="list")
        list_result = await tasks_tool.arun(list_input)
        result_str = (
            list_result.result if hasattr(list_result, "result") else str(list_result)
        )
        assert "Task to delete" not in result_str


class TestTasksToolErrorHandling:
    """Test cases for TasksTool error handling."""

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
        return "test-session-errors"

    @pytest.fixture
    def tasks_tool(self, temp_project_dir, session_id):
        """Create a TasksTool instance with injected context."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )
        tool_class = factory.create_tasks_tool()
        return tool_class()

    @pytest.mark.asyncio
    async def test_get_missing_task_returns_error(self, tasks_tool):
        """Test that get operation for missing task returns error message."""
        input_schema = tasks_tool.input_schema

        get_input = input_schema(operation="get", taskId="999")
        result = await tasks_tool.arun(get_input)

        assert result is not None
        result_str = result.result if hasattr(result, "result") else str(result)
        # Should contain error indication
        assert "not found" in result_str.lower() or "error" in result_str.lower()

    @pytest.mark.asyncio
    async def test_update_missing_task_returns_error(self, tasks_tool):
        """Test that update operation for missing task returns error message."""
        input_schema = tasks_tool.input_schema

        update_input = input_schema(
            operation="update",
            taskId="999",
            subject="Updated",
        )
        result = await tasks_tool.arun(update_input)

        assert result is not None
        result_str = result.result if hasattr(result, "result") else str(result)
        # Should contain error indication
        assert "not found" in result_str.lower() or "error" in result_str.lower()

    @pytest.mark.asyncio
    async def test_delete_missing_task_returns_error(self, tasks_tool):
        """Test that delete operation for missing task returns error message."""
        input_schema = tasks_tool.input_schema

        delete_input = input_schema(
            operation="update",
            taskId="999",
            status="deleted",
        )
        result = await tasks_tool.arun(delete_input)

        assert result is not None
        result_str = result.result if hasattr(result, "result") else str(result)
        # Should contain error indication
        assert "not found" in result_str.lower() or "error" in result_str.lower()

    @pytest.mark.asyncio
    async def test_list_with_no_tasks_returns_empty(self, tasks_tool):
        """Test that list operation with no tasks returns empty list indication."""
        input_schema = tasks_tool.input_schema

        list_input = input_schema(operation="list")
        result = await tasks_tool.arun(list_input)

        assert result is not None
        # Should indicate no tasks or empty list
        result_str = result.result if hasattr(result, "result") else str(result)
        assert (
            "no tasks" in result_str.lower()
            or "empty" in result_str.lower()
            or "[]" in result_str
        )


class TestTasksToolWithDependencies:
    """Test cases for TasksTool with task dependencies."""

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
    def tasks_tool(self, temp_project_dir, session_id):
        """Create a TasksTool instance with injected context."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )
        tool_class = factory.create_tasks_tool()
        return tool_class()

    @pytest.mark.asyncio
    async def test_create_task_with_blocked_by(self, tasks_tool):
        """Test creating a task with blockedBy dependencies."""
        input_schema = tasks_tool.input_schema

        # Create first task
        create_input1 = input_schema(
            operation="create",
            subject="Task 1",
            description="First task",
        )
        await tasks_tool.arun(create_input1)

        # Create second task blocked by first
        create_input2 = input_schema(
            operation="create",
            subject="Task 2",
            description="Blocked by task 1",
            blockedBy=["1"],
        )
        result = await tasks_tool.arun(create_input2)

        assert result is not None
        result_str = result.result if hasattr(result, "result") else str(result)
        assert "2" in result_str  # Task 2 should be created

    @pytest.mark.asyncio
    async def test_update_task_add_blocked_by(self, tasks_tool):
        """Test updating a task to add blockedBy dependencies."""
        input_schema = tasks_tool.input_schema

        # Create two tasks
        await tasks_tool.arun(
            input_schema(
                operation="create",
                subject="Task 1",
                description="First task",
            )
        )
        await tasks_tool.arun(
            input_schema(
                operation="create",
                subject="Task 2",
                description="Second task",
            )
        )

        # Update task 2 to be blocked by task 1
        update_input = input_schema(
            operation="update",
            taskId="2",
            addBlockedBy=["1"],
        )
        result = await tasks_tool.arun(update_input)

        assert result is not None

    @pytest.mark.asyncio
    async def test_update_task_add_blocks(self, tasks_tool):
        """Test updating a task to add blocks dependencies."""
        input_schema = tasks_tool.input_schema

        # Create two tasks
        await tasks_tool.arun(
            input_schema(
                operation="create",
                subject="Task 1",
                description="First task",
            )
        )
        await tasks_tool.arun(
            input_schema(
                operation="create",
                subject="Task 2",
                description="Second task",
            )
        )

        # Update task 1 to block task 2
        update_input = input_schema(
            operation="update",
            taskId="1",
            addBlocks=["2"],
        )
        result = await tasks_tool.arun(update_input)

        assert result is not None


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

    def test_registry_and_factory_tool_names_match(
        self, temp_project_dir, session_id
    ):
        """Test that registry tool names match factory-created tool names."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )
        factory_tools = factory.create_all_tools()

        factory_tool_names = {getattr(t, "mcp_tool_name", None) for t in factory_tools}
        registry_tool_names = set(BUILTIN_TOOLS.keys())

        # Factory tools should be a subset of registry tools
        # (some tools may not be created if no session_id, etc.)
        for name in factory_tool_names:
            if name is not None:
                assert name in registry_tool_names

    @pytest.mark.asyncio
    async def test_full_task_workflow(self, temp_project_dir, session_id):
        """Test full workflow: create -> update -> get -> list -> delete."""
        factory = BuiltinToolFactory(
            project_dir=temp_project_dir, session_id=session_id
        )
        tool_class = factory.create_tasks_tool()
        tool = tool_class()
        input_schema = tool.input_schema

        # Create
        create_result = await tool.arun(
            input_schema(
                operation="create",
                subject="Full workflow task",
                description="Testing complete workflow",
            )
        )
        assert create_result is not None

        # Update
        update_result = await tool.arun(
            input_schema(
                operation="update",
                taskId="1",
                status="in_progress",
            )
        )
        assert update_result is not None

        # Get
        get_result = await tool.arun(
            input_schema(operation="get", taskId="1")
        )
        result_str = (
            get_result.result if hasattr(get_result, "result") else str(get_result)
        )
        assert "in_progress" in result_str

        # List
        list_result = await tool.arun(input_schema(operation="list"))
        result_str = (
            list_result.result if hasattr(list_result, "result") else str(list_result)
        )
        assert "Full workflow task" in result_str

        # Delete
        delete_result = await tool.arun(
            input_schema(
                operation="update",
                taskId="1",
                status="deleted",
            )
        )
        assert delete_result is not None

        # Verify deleted
        list_result_after = await tool.arun(input_schema(operation="list"))
        result_str = (
            list_result_after.result
            if hasattr(list_result_after, "result")
            else str(list_result_after)
        )
        assert "Full workflow task" not in result_str


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
