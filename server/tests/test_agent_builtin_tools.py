"""
Tests for ChatOrchestratorAgent integration with builtin tools (Phase 5).

This module tests:
1. Builtin Tools Loading - Agent loads builtin tools via _load_builtin_tools()
2. Tool Execution Refactoring - Dispatch to _execute_mcp_tool() or _execute_builtin_tool()
3. Tool Merging - run_async() merges builtin tools with MCP and config tools
4. Tool Detection - _can_execute_tool_call() handles both MCP and builtin tools

Written following TEST-DRIVEN DEVELOPMENT: tests written before implementation.
"""

import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config.datamodel import (
    BuiltinTools,
    LlamaFarmConfig,
    Model,
    PromptMessage,
    PromptSet,
    Provider,
    Runtime,
    Version,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from agents.base.history import LFChatCompletionUserMessageParam
from agents.chat_orchestrator import ChatOrchestratorAgent


def make_completion(content: str, *, tool_calls: list | None = None):
    """Helper to create mock completion response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def make_tool_call(*, name: str, arguments: str, call_id: str = "call_1"):
    """Helper to create mock tool call."""
    return ChoiceDeltaToolCall(
        index=0,
        type="function",
        id=call_id,
        function=ChoiceDeltaToolCallFunction(name=name, arguments=arguments),
    )


@pytest.fixture
def base_config():
    """Create base config without MCP or builtin tool exclusions."""
    return LlamaFarmConfig(
        version=Version.v1,
        name="test-project",
        namespace="test",
        runtime=Runtime(
            default_model="default",
            models=[
                Model(
                    name="default",
                    provider=Provider.ollama,
                    model="llama3.2:latest",
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                )
            ],
        ),
        prompts=[
            PromptSet(
                name="default",
                messages=[PromptMessage(role="system", content="You are helpful")],
            )
        ],
    )


@pytest.fixture
def config_with_builtin_tools_disabled():
    """Create config with builtin tools disabled."""
    return LlamaFarmConfig(
        version=Version.v1,
        name="test-project",
        namespace="test",
        runtime=Runtime(
            default_model="default",
            models=[
                Model(
                    name="default",
                    provider=Provider.ollama,
                    model="llama3.2:latest",
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                    builtin_tools=BuiltinTools(enabled=False),
                )
            ],
        ),
        prompts=[
            PromptSet(
                name="default",
                messages=[PromptMessage(role="system", content="You are helpful")],
            )
        ],
    )


@pytest.fixture
def config_with_tasks_excluded():
    """Create config with tasks tool excluded."""
    return LlamaFarmConfig(
        version=Version.v1,
        name="test-project",
        namespace="test",
        runtime=Runtime(
            default_model="default",
            models=[
                Model(
                    name="default",
                    provider=Provider.ollama,
                    model="llama3.2:latest",
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                    builtin_tools=BuiltinTools(enabled=True, exclude=["tasks"]),
                )
            ],
        ),
        prompts=[
            PromptSet(
                name="default",
                messages=[PromptMessage(role="system", content="You are helpful")],
            )
        ],
    )


# ==============================================================================
# BUILTIN TOOLS LOADING TESTS
# ==============================================================================


class TestBuiltinToolsLoading:
    """Test cases for loading builtin tools in ChatOrchestratorAgent."""

    def test_agent_has_builtin_tools_attribute(self, base_config):
        """Test that agent has _builtin_tools instance variable."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            # Agent should have _builtin_tools attribute initialized as empty list
            assert hasattr(agent, "_builtin_tools")
            assert isinstance(agent._builtin_tools, list)

    @pytest.mark.asyncio
    async def test_load_builtin_tools_method_exists(self, base_config):
        """Test that _load_builtin_tools method exists."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            # Method should exist
            assert hasattr(agent, "_load_builtin_tools")
            assert callable(agent._load_builtin_tools)

    @pytest.mark.asyncio
    async def test_load_builtin_tools_populates_list(self, base_config):
        """Test that _load_builtin_tools populates _builtin_tools list."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent.enable_persistence(session_id="test-session")

            # Mock the factory
            with patch("agents.chat_orchestrator.BuiltinToolFactory") as mock_factory:
                mock_tool_class = MagicMock()
                mock_tool_class.tool_name = "tasks"
                mock_factory_instance = MagicMock()
                mock_factory_instance.create_all_tools.return_value = [mock_tool_class]
                mock_factory.return_value = mock_factory_instance

                await agent._load_builtin_tools()

                # Builtin tools should be loaded
                assert len(agent._builtin_tools) > 0

    @pytest.mark.asyncio
    async def test_setup_tools_calls_load_builtin_tools(self, base_config):
        """Test that setup_tools() calls both enable_mcp() and _load_builtin_tools()."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            # Spy on the methods
            with (
                patch.object(agent, "enable_mcp", new_callable=AsyncMock) as mock_mcp,
                patch.object(
                    agent, "_load_builtin_tools", new_callable=AsyncMock
                ) as mock_builtin,
            ):
                await agent.setup_tools()

                mock_mcp.assert_awaited_once()
                mock_builtin.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_builtin_tools_requires_session_for_tasks(self, base_config):
        """Test that tasks tool is only loaded when session_id is set."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            # Don't set session_id

            with patch("agents.chat_orchestrator.BuiltinToolFactory") as mock_factory:
                mock_factory_instance = MagicMock()
                # Factory returns empty list when no session_id
                mock_factory_instance.create_all_tools.return_value = []
                mock_factory.return_value = mock_factory_instance

                await agent._load_builtin_tools()

                # Factory should be called with None session_id
                mock_factory.assert_called_once_with(
                    agent._project_dir, agent._session_id
                )


# ==============================================================================
# TOOL EXECUTION REFACTORING TESTS
# ==============================================================================


class TestToolExecutionDispatch:
    """Test cases for _execute_tool() dispatch logic."""

    @pytest.mark.asyncio
    async def test_execute_tool_method_exists(self, base_config):
        """Test that _execute_tool method exists."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            assert hasattr(agent, "_execute_tool")
            assert callable(agent._execute_tool)

    @pytest.mark.asyncio
    async def test_execute_tool_dispatches_to_mcp_tool(self, base_config):
        """Test that _execute_tool dispatches to _execute_mcp_tool for MCP tools."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            # Mock an MCP tool
            mock_tool = MagicMock()
            mock_tool.tool_name = "mcp_test_tool"
            agent._mcp_tools = [mock_tool]
            agent._builtin_tools = []

            with patch.object(
                agent, "_execute_mcp_tool", new_callable=AsyncMock
            ) as mock_execute:
                mock_execute.return_value = "mcp result"

                result = await agent._execute_tool(
                    "mcp_test_tool", '{"arg": "value"}'
                )

                mock_execute.assert_awaited_once_with("mcp_test_tool", '{"arg": "value"}')
                assert result == "mcp result"

    @pytest.mark.asyncio
    async def test_execute_tool_dispatches_to_builtin_tool(self, base_config):
        """Test that _execute_tool dispatches to _execute_builtin_tool for builtin tools."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            # Mock a builtin tool
            mock_builtin = MagicMock()
            mock_builtin.tool_name = "tasks"
            agent._mcp_tools = []
            agent._builtin_tools = [mock_builtin]

            # Mock get_enabled_builtin_tool_names to return the "tasks" name
            with (
                patch(
                    "agents.chat_orchestrator.get_enabled_builtin_tool_names"
                ) as mock_registry,
                patch.object(
                    agent, "_execute_builtin_tool", new_callable=AsyncMock
                ) as mock_execute,
            ):
                mock_registry.return_value = {"tasks"}
                mock_execute.return_value = "builtin result"

                result = await agent._execute_tool("tasks", '{"operation": "list"}')

                mock_execute.assert_awaited_once_with(mock_builtin, '{"operation": "list"}')
                assert result == "builtin result"

    @pytest.mark.asyncio
    async def test_execute_tool_raises_for_unknown_tool(self, base_config):
        """Test that _execute_tool raises ValueError for unknown tools."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent._mcp_tools = []
            agent._builtin_tools = []

            with pytest.raises(ValueError, match="not found"):
                await agent._execute_tool("unknown_tool", "{}")

    @pytest.mark.asyncio
    async def test_execute_builtin_tool_method_exists(self, base_config):
        """Test that _execute_builtin_tool method exists."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            assert hasattr(agent, "_execute_builtin_tool")
            assert callable(agent._execute_builtin_tool)

    @pytest.mark.asyncio
    async def test_execute_builtin_tool_instantiates_and_runs(self, base_config):
        """Test that _execute_builtin_tool instantiates and runs the tool correctly."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            # Create a mock tool class
            mock_result = MagicMock()
            mock_result.result = "task created"

            mock_instance = AsyncMock()
            mock_instance.arun = AsyncMock(return_value=mock_result)

            mock_input_schema = MagicMock()

            mock_tool_class = MagicMock()
            mock_tool_class.return_value = mock_instance
            mock_tool_class.input_schema = mock_input_schema

            result = await agent._execute_builtin_tool(
                mock_tool_class, '{"operation": "create", "subject": "Test"}'
            )

            # Tool should be instantiated
            mock_tool_class.assert_called_once()
            # Input schema should be called with parsed arguments
            mock_input_schema.assert_called_once()
            # arun should be called
            mock_instance.arun.assert_awaited_once()
            # Result should be returned
            assert result == "task created"


# ==============================================================================
# TOOL MERGING TESTS
# ==============================================================================


class TestToolMerging:
    """Test cases for merging builtin tools with MCP and config tools."""

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_includes_builtin_tools_in_tool_list(
        self, mock_run_async, base_config
    ):
        """Test that run_async() merges builtin tools into the tool list."""
        mock_run_async.return_value = make_completion("Response")

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent.enable_persistence(session_id="test-session")

            # Mock builtin tool
            mock_builtin = MagicMock()
            mock_builtin.tool_name = "tasks"
            mock_builtin.__doc__ = "Manage tasks"
            mock_builtin.input_schema = MagicMock()
            mock_builtin.input_schema.model_json_schema.return_value = {
                "type": "object",
                "properties": {"operation": {"type": "string"}},
                "required": ["operation"],
            }
            agent._builtin_tools = [mock_builtin]

            # Mock registry to say tasks is enabled
            with patch(
                "agents.chat_orchestrator.get_enabled_builtin_tool_names"
            ) as mock_registry:
                mock_registry.return_value = {"tasks"}

                await agent.run_async(
                    messages=[
                        LFChatCompletionUserMessageParam(role="user", content="Hi")
                    ]
                )

                # Check that tools were passed to parent run_async
                call_kwargs = mock_run_async.call_args.kwargs
                tools = call_kwargs.get("tools", [])

                # Should include builtin tool
                tool_names = [t.name for t in tools]
                assert "tasks" in tool_names

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_excludes_disabled_builtin_tools(
        self, mock_run_async, config_with_builtin_tools_disabled
    ):
        """Test that run_async() excludes builtin tools when disabled in config."""
        mock_run_async.return_value = make_completion("Response")

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=config_with_builtin_tools_disabled,
                project_dir=project_dir,
            )
            agent.enable_persistence(session_id="test-session")

            # Mock builtin tool that would normally be loaded
            mock_builtin = MagicMock()
            mock_builtin.tool_name = "tasks"
            agent._builtin_tools = [mock_builtin]

            # Mock registry to return empty (disabled)
            with patch(
                "agents.chat_orchestrator.get_enabled_builtin_tool_names"
            ) as mock_registry:
                mock_registry.return_value = set()  # All disabled

                await agent.run_async(
                    messages=[
                        LFChatCompletionUserMessageParam(role="user", content="Hi")
                    ]
                )

                # Check that tools were passed to parent run_async
                call_kwargs = mock_run_async.call_args.kwargs
                tools = call_kwargs.get("tools", [])

                # Should NOT include builtin tool
                tool_names = [t.name for t in tools]
                assert "tasks" not in tool_names

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_excludes_specific_tools_from_exclude_list(
        self, mock_run_async, config_with_tasks_excluded
    ):
        """Test that run_async() excludes specific tools based on exclude list."""
        mock_run_async.return_value = make_completion("Response")

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=config_with_tasks_excluded,
                project_dir=project_dir,
            )
            agent.enable_persistence(session_id="test-session")

            # Mock builtin tool
            mock_builtin = MagicMock()
            mock_builtin.tool_name = "tasks"
            agent._builtin_tools = [mock_builtin]

            # Mock registry to return empty for excluded tool
            with patch(
                "agents.chat_orchestrator.get_enabled_builtin_tool_names"
            ) as mock_registry:
                mock_registry.return_value = set()  # Tasks excluded

                await agent.run_async(
                    messages=[
                        LFChatCompletionUserMessageParam(role="user", content="Hi")
                    ]
                )

                # Check that tools were passed to parent run_async
                call_kwargs = mock_run_async.call_args.kwargs
                tools = call_kwargs.get("tools", [])

                # Should NOT include excluded builtin tool
                tool_names = [t.name for t in tools]
                assert "tasks" not in tool_names

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_merges_builtin_mcp_and_config_tools(
        self, mock_run_async, base_config
    ):
        """Test that run_async() merges all tool types together."""
        mock_run_async.return_value = make_completion("Response")

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent.enable_persistence(session_id="test-session")

            # Mock MCP tool
            mock_mcp = MagicMock()
            mock_mcp.tool_name = "mcp_tool"
            mock_mcp.__doc__ = "MCP tool"
            mock_mcp.input_schema = MagicMock()
            mock_mcp.input_schema.model_json_schema.return_value = {
                "type": "object",
                "properties": {},
            }
            agent._mcp_tools = [mock_mcp]

            # Mock builtin tool
            mock_builtin = MagicMock()
            mock_builtin.tool_name = "tasks"
            mock_builtin.__doc__ = "Tasks tool"
            mock_builtin.input_schema = MagicMock()
            mock_builtin.input_schema.model_json_schema.return_value = {
                "type": "object",
                "properties": {},
            }
            agent._builtin_tools = [mock_builtin]

            # Mock registry to return tasks enabled
            with patch(
                "agents.chat_orchestrator.get_enabled_builtin_tool_names"
            ) as mock_registry:
                mock_registry.return_value = {"tasks"}

                # Also pass config tools
                from agents.base.types import ToolDefinition

                config_tool = ToolDefinition(
                    name="config_tool",
                    description="A config tool",
                    parameters={"type": "object", "properties": {}},
                )

                await agent.run_async(
                    messages=[
                        LFChatCompletionUserMessageParam(role="user", content="Hi")
                    ],
                    tools=[config_tool],
                )

                # Check that all tool types are present
                call_kwargs = mock_run_async.call_args.kwargs
                tools = call_kwargs.get("tools", [])
                tool_names = [t.name for t in tools]

                assert "mcp_tool" in tool_names
                assert "tasks" in tool_names
                assert "config_tool" in tool_names


# ==============================================================================
# TOOL DETECTION TESTS
# ==============================================================================


class TestToolDetection:
    """Test cases for _can_execute_tool_call() with MCP and builtin tools."""

    def test_can_execute_returns_true_for_mcp_tools(self, base_config):
        """Test that _can_execute_tool_call returns True for MCP tools."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            # Mock MCP tool
            mock_mcp = MagicMock()
            mock_mcp.tool_name = "mcp_test_tool"
            agent._mcp_tools = [mock_mcp]
            agent._builtin_tools = []

            # Create tool call
            tool_call = make_tool_call(name="mcp_test_tool", arguments="{}")

            result = agent._can_execute_tool_call(tool_call)

            assert result is True

    def test_can_execute_returns_true_for_builtin_tools(self, base_config):
        """Test that _can_execute_tool_call returns True for builtin tools."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            # Mock builtin tool
            mock_builtin = MagicMock()
            mock_builtin.tool_name = "tasks"
            agent._mcp_tools = []
            agent._builtin_tools = [mock_builtin]

            # Mock get_enabled_builtin_tool_names to return the "tasks" name
            with patch(
                "agents.chat_orchestrator.get_enabled_builtin_tool_names"
            ) as mock_registry:
                mock_registry.return_value = {"tasks"}

                # Create tool call
                tool_call = make_tool_call(name="tasks", arguments='{"operation": "list"}')

                result = agent._can_execute_tool_call(tool_call)

                assert result is True

    def test_can_execute_returns_false_for_unknown_tools(self, base_config):
        """Test that _can_execute_tool_call returns False for unknown tools."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            agent._mcp_tools = []
            agent._builtin_tools = []

            # Create tool call for unknown tool
            tool_call = make_tool_call(name="unknown_tool", arguments="{}")

            result = agent._can_execute_tool_call(tool_call)

            assert result is False

    def test_can_execute_checks_both_mcp_and_builtin(self, base_config):
        """Test that _can_execute_tool_call checks both MCP and builtin lists."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            # Mock both MCP and builtin tools
            mock_mcp = MagicMock()
            mock_mcp.tool_name = "mcp_tool"
            mock_builtin = MagicMock()
            mock_builtin.tool_name = "tasks"
            agent._mcp_tools = [mock_mcp]
            agent._builtin_tools = [mock_builtin]

            # Mock get_enabled_builtin_tool_names to return the "tasks" name
            with patch(
                "agents.chat_orchestrator.get_enabled_builtin_tool_names"
            ) as mock_registry:
                mock_registry.return_value = {"tasks"}

                # Both should be found
                mcp_call = make_tool_call(name="mcp_tool", arguments="{}")
                builtin_call = make_tool_call(name="tasks", arguments="{}")
                unknown_call = make_tool_call(name="unknown", arguments="{}")

                assert agent._can_execute_tool_call(mcp_call) is True
                assert agent._can_execute_tool_call(builtin_call) is True
                assert agent._can_execute_tool_call(unknown_call) is False


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestBuiltinToolsAgentIntegration:
    """Integration tests for builtin tools with the agent orchestration loop."""

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_agent_executes_builtin_tool_in_orchestration_loop(
        self, mock_run_async, base_config
    ):
        """Test that the agent executes builtin tools during the orchestration loop."""
        # First call: LLM requests builtin tool
        # Second call: LLM provides final answer
        tool_call = make_tool_call(
            name="tasks", arguments='{"operation": "list"}'
        )
        mock_run_async.side_effect = [
            make_completion("Let me check tasks", tool_calls=[tool_call]),
            make_completion("Here are your tasks: none"),
        ]

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent.enable_persistence(session_id="test-session")

            # Mock builtin tool
            mock_result = MagicMock()
            mock_result.result = "No tasks found"

            mock_instance = AsyncMock()
            mock_instance.arun = AsyncMock(return_value=mock_result)

            mock_input_schema = MagicMock()
            mock_input_schema.return_value = MagicMock()

            mock_builtin = MagicMock()
            mock_builtin.tool_name = "tasks"
            mock_builtin.return_value = mock_instance
            mock_builtin.input_schema = mock_input_schema

            agent._builtin_tools = [mock_builtin]
            agent._mcp_tools = []

            # Mock registry
            with patch(
                "agents.chat_orchestrator.get_enabled_builtin_tool_names"
            ) as mock_registry:
                mock_registry.return_value = {"tasks"}

                # Run the agent
                response = await agent.run_async(
                    messages=[
                        LFChatCompletionUserMessageParam(
                            role="user", content="Show my tasks"
                        )
                    ]
                )

                # Builtin tool should have been executed
                mock_instance.arun.assert_awaited_once()

                # Final response should be from second LLM call
                assert response.choices[0].message.content == "Here are your tasks: none"

    @pytest.mark.asyncio
    async def test_factory_creates_agent_with_builtin_tools(self, base_config):
        """Test that ChatOrchestratorAgentFactory creates agent with builtin tools loaded."""
        from agents.chat_orchestrator import ChatOrchestratorAgentFactory

        with (
            tempfile.TemporaryDirectory() as project_dir,
            patch("agents.chat_orchestrator.MCPToolFactory") as mock_mcp_factory,
            patch(
                "agents.chat_orchestrator.BuiltinToolFactory"
            ) as mock_builtin_factory,
        ):
            # Mock MCP factory
            mock_mcp_instance = AsyncMock()
            mock_mcp_instance.create_all_tools = AsyncMock(return_value=[])
            mock_mcp_factory.return_value = mock_mcp_instance

            # Mock builtin factory
            mock_builtin = MagicMock()
            mock_builtin.tool_name = "tasks"
            mock_builtin_instance = MagicMock()
            mock_builtin_instance.create_all_tools.return_value = [mock_builtin]
            mock_builtin_factory.return_value = mock_builtin_instance

            agent = await ChatOrchestratorAgentFactory.create_agent(
                project_config=base_config,
                project_dir=project_dir,
                session_id="test-session",
            )

            # Factory should have been called
            mock_builtin_factory.assert_called_once()

            # Agent should have builtin tools loaded
            assert hasattr(agent, "_builtin_tools")
