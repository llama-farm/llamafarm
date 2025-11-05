"""
Unit tests for ChatOrchestratorAgent.

Tests the chat orchestrator including:
- Basic chat functionality
- Streaming chat
- Tool calling support
- MCP integration
- History persistence
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.chat_orchestrator import (
    ChatOrchestratorAgent,
    ChatOrchestratorAgentFactory,
)
from agents.base.history import (
    LFChatCompletionAssistantMessageParam,
    LFChatCompletionUserMessageParam,
)
from agents.base.types import StreamEvent, ToolCallRequest
from config.datamodel import (
    LlamaFarmConfig,
    Mcp,
    PromptMessage,
    Model,
    PromptSet,
    Provider,
    Runtime,
    Server,
    Transport,
    Version,
)


def make_completion(content: str, *, tool_calls: list | None = None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def make_tool_call(*, name: str, arguments: str):
    return SimpleNamespace(
        type="function",
        id="call_1",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_chunk(
    *,
    content: str | None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def base_config():
    """Create base config without MCP."""
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
def config_with_mcp():
    """Create config with MCP servers."""
    return LlamaFarmConfig(
        version=Version.v1,
        name="test-project",
        namespace="test",
        runtime=Runtime(
            default_model="default",
            models=[
                Model(
                    name="default",
                    provider=Provider.openai,
                    model="gpt-4",
                    base_url="https://api.openai.com/v1",
                    api_key="test-key",
                )
            ],
        ),
        prompts=[
            PromptSet(
                name="default",
                messages=[PromptMessage(role="system", content="You are helpful")],
            )
        ],
        mcp=Mcp(
            servers=[
                Server(
                    name="test-server",
                    transport=Transport.http,
                    base_url="http://localhost:8080",
                )
            ]
        ),
    )


@pytest.fixture
def config_with_multiple_mcp_servers():
    """Create config with multiple MCP servers and models with server subsets."""
    return LlamaFarmConfig(
        version=Version.v1,
        name="test-project",
        namespace="test",
        runtime=Runtime(
            default_model="default",
            models=[
                Model(
                    name="default",
                    provider=Provider.openai,
                    model="gpt-4",
                    base_url="https://api.openai.com/v1",
                    api_key="test-key",
                    # No mcp_servers specified - should use all servers
                ),
                Model(
                    name="model-with-subset",
                    provider=Provider.openai,
                    model="gpt-4",
                    base_url="https://api.openai.com/v1",
                    api_key="test-key",
                    mcp_servers=["filesystem", "weather"],  # Only these two
                ),
                Model(
                    name="model-with-one-server",
                    provider=Provider.openai,
                    model="gpt-3.5-turbo",
                    base_url="https://api.openai.com/v1",
                    api_key="test-key",
                    mcp_servers=["database"],  # Only database
                ),
                Model(
                    name="model-with-empty-list",
                    provider=Provider.openai,
                    model="gpt-3.5-turbo",
                    base_url="https://api.openai.com/v1",
                    api_key="test-key",
                    mcp_servers=[],  # Explicitly empty
                ),
            ],
        ),
        prompts=[
            PromptSet(
                name="default",
                messages=[PromptMessage(role="system", content="You are helpful")],
            )
        ],
        mcp=Mcp(
            servers=[
                Server(
                    name="filesystem",
                    transport=Transport.http,
                    base_url="http://localhost:8080",
                ),
                Server(
                    name="weather",
                    transport=Transport.http,
                    base_url="http://localhost:8081",
                ),
                Server(
                    name="database",
                    transport=Transport.http,
                    base_url="http://localhost:8082",
                ),
                Server(
                    name="calendar",
                    transport=Transport.http,
                    base_url="http://localhost:8083",
                ),
            ]
        ),
    )


class TestChatOrchestratorAgent:
    """Test suite for ChatOrchestratorAgent."""

    def test_init(self, base_config):
        """Test agent initialization."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            assert agent._project_config == base_config
            assert agent._project_dir == project_dir
            assert agent._session_id is None
            assert not agent._persist_enabled
            assert not agent._mcp_enabled

    def test_init_with_model_name(self, base_config):
        """Test agent initialization with specific model."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
                model_name="default",
            )
            # agent.model_name should be the config name, not the model string
            assert agent.model_name == "default"
            assert agent._model_string == "llama3.2:latest"

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_simple_response(self, mock_run_async, base_config):
        """Test simple chat without tool calling."""
        mock_run_async.return_value = make_completion("Hello there!")

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            user_input = LFChatCompletionUserMessageParam(role="user", content="Hi")
            response = await agent.run_async(user_input=user_input)

            assert response.choices[0].message.content == "Hello there!"
            # Note: Since we're mocking the parent's run_async, history management
            # is bypassed in this test

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_with_tool_call(self, mock_run_async, base_config):
        """Test chat with tool call."""
        # First call: LLM requests a tool
        # Second call: LLM provides final answer
        tool_call = make_tool_call(name="test_tool", arguments='{"arg": "value"}')
        mock_run_async.side_effect = [
            make_completion("Tool call", tool_calls=[tool_call]),
            make_completion("Final answer based on tool result"),
        ]

        # Mock MCP tool
        mock_tool_class = MagicMock()
        mock_tool_class.mcp_tool_name = "test_tool"
        mock_tool_class.__name__ = "TestTool"
        mock_tool_instance = AsyncMock()
        mock_tool_instance.arun = AsyncMock(
            return_value=SimpleNamespace(result="tool result")
        )
        mock_tool_class.return_value = mock_tool_instance
        mock_tool_class.input_schema = MagicMock()
        mock_tool_class.input_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "arg": {"type": "string"},
            },
            "required": ["tool_name", "arg"],
        }
        mock_tool_class.input_schema.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent._mcp_enabled = True
            agent._mcp_tools = [mock_tool_class]

            user_input = LFChatCompletionUserMessageParam(
                role="user", content="Use the tool"
            )
            response = await agent.run_async(user_input=user_input)

            assert (
                response.choices[0].message.content
                == "Final answer based on tool result"
            )
            mock_tool_instance.arun.assert_awaited()

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_max_iterations(self, mock_run_async, base_config):
        """Test that max iterations is enforced."""
        # Keep requesting tools forever
        tool_call = make_tool_call(name="test_tool", arguments='{"arg": "value"}')
        mock_run_async.return_value = make_completion(
            "Tool call", tool_calls=[tool_call]
        )

        mock_tool_class = MagicMock()
        mock_tool_class.mcp_tool_name = "test_tool"
        mock_tool_class.__name__ = "TestTool"
        mock_tool_instance = AsyncMock()
        mock_tool_instance.arun = AsyncMock(
            return_value=SimpleNamespace(result="result")
        )
        mock_tool_class.return_value = mock_tool_instance
        mock_tool_class.input_schema = MagicMock()
        mock_tool_class.input_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "arg": {"type": "string"},
            },
            "required": ["tool_name", "arg"],
        }
        mock_tool_class.input_schema.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent._mcp_enabled = True
            agent._mcp_tools = [mock_tool_class]

            user_input = LFChatCompletionUserMessageParam(role="user", content="Test")
            response = await agent.run_async(user_input=user_input)

            # Should return max iterations message
            assert "maximum number of tool calls" in response.choices[0].message.content

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_tool_not_found(self, mock_run_async, base_config):
        """Test handling of non-existent tool."""
        tool_call = make_tool_call(name="nonexistent_tool", arguments="{}")
        mock_run_async.side_effect = [
            make_completion("Tool call", tool_calls=[tool_call]),
            make_completion("I apologize"),
        ]

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent._mcp_enabled = True
            agent._mcp_tools = []

            user_input = LFChatCompletionUserMessageParam(role="user", content="Test")
            response = await agent.run_async(user_input=user_input)

            # Should handle error gracefully
            assert "apologize" in response.choices[0].message.content.lower()

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_tool_execution_error(self, mock_run_async, base_config):
        """Test handling of tool execution errors."""
        tool_call = make_tool_call(name="test_tool", arguments='{"arg": "value"}')
        mock_run_async.side_effect = [
            make_completion("Tool call", tool_calls=[tool_call]),
            make_completion("Sorry, there was an error"),
        ]

        mock_tool_class = MagicMock()
        mock_tool_class.mcp_tool_name = "test_tool"
        mock_tool_class.__name__ = "TestTool"
        mock_tool_instance = AsyncMock()
        mock_tool_instance.arun = AsyncMock(side_effect=Exception("Tool failed"))
        mock_tool_class.return_value = mock_tool_instance
        mock_tool_class.input_schema = MagicMock()
        mock_tool_class.input_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "arg": {"type": "string"},
            },
            "required": ["tool_name", "arg"],
        }
        mock_tool_class.input_schema.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent._mcp_enabled = True
            agent._mcp_tools = [mock_tool_class]

            user_input = LFChatCompletionUserMessageParam(role="user", content="Test")
            response = await agent.run_async(user_input=user_input)

            # Should handle error gracefully
            message = response.choices[0].message.content.lower()
            assert "error" in message or "sorry" in message

    @pytest.mark.asyncio
    async def test_run_async_stream_no_tools(self, base_config):
        """Test streaming without MCP tools."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            # Mock the parent stream_chat method
            async def chunk_generator():
                yield make_chunk(content="Hello")
                yield make_chunk(content=" world", finish_reason="stop")

            with patch.object(
                agent.__class__.__bases__[0],
                "run_async_stream",
                side_effect=lambda *args, **kwargs: chunk_generator(),
            ):
                user_input = LFChatCompletionUserMessageParam(role="user", content="Hi")
                chunks = []
                async for chunk in agent.run_async_stream(user_input=user_input):
                    chunks.append(chunk)

                assert len(chunks) == 2
                content = "".join(c.choices[0].delta.content for c in chunks)
                assert content == "Hello world"

    @pytest.mark.asyncio
    async def test_run_async_stream_with_tool_call(self, base_config):
        """Test streaming with tool call."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent._mcp_enabled = True

            # Mock tool
            mock_tool_class = MagicMock()
            mock_tool_class.__name__ = "TestTool"
            mock_tool_class.mcp_tool_name = "test_tool"
            mock_tool_instance = AsyncMock()
            mock_tool_instance.arun = AsyncMock(
                return_value=SimpleNamespace(result="Tool result")
            )
            mock_tool_class.return_value = mock_tool_instance
            mock_tool_class.input_schema = MagicMock()
            mock_tool_class.input_schema.model_json_schema.return_value = {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "arg": {"type": "string"},
                },
                "required": ["tool_name", "arg"],
            }
            mock_tool_class.input_schema.return_value = MagicMock()
            agent._mcp_tools = [mock_tool_class]

            tool_call_delta = make_tool_call(
                name="test_tool", arguments='{"arg": "value"}'
            )

            async def first_stream(*args, **kwargs):
                yield make_chunk(content="Let me check...")
                yield make_chunk(
                    content=None,
                    tool_calls=[tool_call_delta],
                    finish_reason="tool_calls",
                )

            async def second_stream(*args, **kwargs):
                yield make_chunk(content="Final answer", finish_reason="stop")

            with patch.object(
                agent._client,
                "stream_chat",
                side_effect=[first_stream(), second_stream()],
            ):
                user_input = LFChatCompletionUserMessageParam(
                    role="user", content="Test"
                )
                chunks = []
                async for chunk in agent.run_async_stream(user_input=user_input):
                    chunks.append(chunk)

                # Should include content and tool call indicator
                assert len(chunks) > 0
                contents = [
                    c.choices[0].delta.content
                    for c in chunks
                    if c.choices[0].delta.content
                ]
                assert any("Let me check" in (content or "") for content in contents)
                final_contents = [
                    c.choices[0].delta.content
                    for c in chunks
                    if c.choices[0].finish_reason == "stop"
                ]
                assert "Final answer" in final_contents
                mock_tool_instance.arun.assert_awaited()

    def test_enable_persistence(self, base_config):
        """Test enabling persistence."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            session_id = "test-session-123"
            agent.enable_persistence(session_id=session_id)

            assert agent._persist_enabled
            assert agent._session_id == session_id

    def test_history_file_path(self, base_config):
        """Test history file path generation."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            # Without persistence enabled
            assert agent._history_file_path is None

            # With persistence enabled
            agent.enable_persistence(session_id="test-session")
            path = agent._history_file_path
            assert path is not None
            assert "test-session" in str(path)
            assert "history.json" in str(path)

    def test_persist_and_restore_history(self, base_config):
        """Test persisting and restoring history."""
        with tempfile.TemporaryDirectory() as project_dir:
            # Create agent and add messages
            agent1 = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent1.enable_persistence(session_id="test-session")

            agent1.history.add_message(
                LFChatCompletionUserMessageParam(role="user", content="Hello")
            )
            agent1.history.add_message(
                LFChatCompletionAssistantMessageParam(
                    role="assistant", content="Hi there"
                )
            )

            # Persist history
            agent1._persist_history()

            # Create new agent and restore
            agent2 = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent2.enable_persistence(session_id="test-session")

            # History should be restored
            assert len(agent2.history.history) == 2
            assert agent2.history.history[0]["content"] == "Hello"
            assert agent2.history.history[1]["content"] == "Hi there"

    def test_reset_history(self, base_config):
        """Test resetting history."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent.enable_persistence(session_id="test-session")

            # Add messages and persist
            agent.history.add_message(
                LFChatCompletionUserMessageParam(role="user", content="Hello")
            )
            agent._persist_history()

            # Reset history
            agent.reset_history()

            # History should be empty and file should be deleted
            assert len(agent.history.history) == 0
            path = agent._history_file_path
            if path:
                assert not path.exists()

    @pytest.mark.asyncio
    async def test_setup_tools(self, config_with_mcp):
        """Test enabling MCP."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=config_with_mcp,
                project_dir=project_dir,
            )

            # Mock MCPToolFactory
            with patch("agents.chat_orchestrator.MCPToolFactory") as mock_factory:
                mock_factory_instance = AsyncMock()
                mock_factory_instance.create_all_tools = AsyncMock(return_value=[])
                mock_factory.return_value = mock_factory_instance

                await agent.setup_tools()

                assert agent._mcp_enabled
                assert agent._mcp_service is not None
                assert agent._mcp_tool_factory is not None

    @pytest.mark.asyncio
    async def test_load_mcp_tools(self, config_with_mcp):
        """Test loading MCP tools."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=config_with_mcp,
                project_dir=project_dir,
            )

            # Mock tools
            mock_tool1 = MagicMock()
            mock_tool1.__name__ = "Tool1"
            mock_tool1.mcp_tool_name = "tool1"
            mock_tool2 = MagicMock()
            mock_tool2.__name__ = "Tool2"
            mock_tool2.mcp_tool_name = "tool2"

            with patch("agents.chat_orchestrator.MCPToolFactory") as mock_factory:
                mock_factory_instance = AsyncMock()
                mock_factory_instance.create_all_tools = AsyncMock(
                    return_value=[mock_tool1, mock_tool2]
                )
                mock_factory.return_value = mock_factory_instance

                await agent.enable_mcp()

                assert len(agent._mcp_tools) == 2
                assert agent._mcp_tools[0].mcp_tool_name == "tool1"
                assert agent._mcp_tools[1].mcp_tool_name == "tool2"

    @pytest.mark.asyncio
    async def test_mcp_servers_subset_selection(self, config_with_multiple_mcp_servers):
        """Test that model can specify subset of MCP servers."""
        with tempfile.TemporaryDirectory() as project_dir:
            # Test model with subset specified
            agent = ChatOrchestratorAgent(
                project_config=config_with_multiple_mcp_servers,
                project_dir=project_dir,
                model_name="model-with-subset",
            )

            with patch("agents.chat_orchestrator.MCPToolFactory") as mock_factory:
                mock_factory_instance = AsyncMock()
                mock_factory_instance.create_all_tools = AsyncMock(return_value=[])
                mock_factory.return_value = mock_factory_instance

                await agent.enable_mcp()

                # MCPService should be initialized
                assert agent._mcp_service is not None

                # Check that only the specified servers are available
                available_servers = agent._mcp_service.list_servers()
                assert set(available_servers) == {"filesystem", "weather"}
                assert "database" not in available_servers
                assert "calendar" not in available_servers

    @pytest.mark.asyncio
    async def test_mcp_servers_all_when_not_specified(
        self, config_with_multiple_mcp_servers
    ):
        """Test that model uses all MCP servers when mcp_servers is not specified."""
        with tempfile.TemporaryDirectory() as project_dir:
            # Test default model (no mcp_servers specified)
            agent = ChatOrchestratorAgent(
                project_config=config_with_multiple_mcp_servers,
                project_dir=project_dir,
                model_name="default",
            )

            with patch("agents.chat_orchestrator.MCPToolFactory") as mock_factory:
                mock_factory_instance = AsyncMock()
                mock_factory_instance.create_all_tools = AsyncMock(return_value=[])
                mock_factory.return_value = mock_factory_instance

                await agent.enable_mcp()

                # MCPService should be initialized
                assert agent._mcp_service is not None

                # Check that all servers are available
                available_servers = agent._mcp_service.list_servers()
                assert set(available_servers) == {
                    "filesystem",
                    "weather",
                    "database",
                    "calendar",
                }

    @pytest.mark.asyncio
    async def test_mcp_servers_single_server(self, config_with_multiple_mcp_servers):
        """Test that model can specify a single MCP server."""
        with tempfile.TemporaryDirectory() as project_dir:
            # Test model with single server
            agent = ChatOrchestratorAgent(
                project_config=config_with_multiple_mcp_servers,
                project_dir=project_dir,
                model_name="model-with-one-server",
            )

            with patch("agents.chat_orchestrator.MCPToolFactory") as mock_factory:
                mock_factory_instance = AsyncMock()
                mock_factory_instance.create_all_tools = AsyncMock(return_value=[])
                mock_factory.return_value = mock_factory_instance

                await agent.enable_mcp()

                # MCPService should be initialized
                assert agent._mcp_service is not None

                # Check that only the database server is available
                available_servers = agent._mcp_service.list_servers()
                assert available_servers == ["database"]

    @pytest.mark.asyncio
    async def test_mcp_servers_empty_list(self, config_with_multiple_mcp_servers):
        """Test that model with empty mcp_servers list has no servers."""
        with tempfile.TemporaryDirectory() as project_dir:
            # Test model with empty list
            agent = ChatOrchestratorAgent(
                project_config=config_with_multiple_mcp_servers,
                project_dir=project_dir,
                model_name="model-with-empty-list",
            )

            with patch("agents.chat_orchestrator.MCPToolFactory") as mock_factory:
                mock_factory_instance = AsyncMock()
                mock_factory_instance.create_all_tools = AsyncMock(return_value=[])
                mock_factory.return_value = mock_factory_instance

                await agent.enable_mcp()

                # MCPService should be initialized
                assert agent._mcp_service is not None

                # Check that no servers are available
                available_servers = agent._mcp_service.list_servers()
                assert available_servers == []

    @pytest.mark.asyncio
    async def test_mcp_servers_invalid_server_name(
        self, config_with_multiple_mcp_servers
    ):
        """Test that non-existent server names are silently filtered out."""
        with tempfile.TemporaryDirectory() as project_dir:
            # Modify config to include a non-existent server name
            config = config_with_multiple_mcp_servers
            config.runtime.models.append(
                Model(
                    name="model-with-invalid-server",
                    provider=Provider.openai,
                    model="gpt-3.5-turbo",
                    base_url="https://api.openai.com/v1",
                    api_key="test-key",
                    mcp_servers=["filesystem", "nonexistent-server", "weather"],
                )
            )

            agent = ChatOrchestratorAgent(
                project_config=config,
                project_dir=project_dir,
                model_name="model-with-invalid-server",
            )

            with patch("agents.chat_orchestrator.MCPToolFactory") as mock_factory:
                mock_factory_instance = AsyncMock()
                mock_factory_instance.create_all_tools = AsyncMock(return_value=[])
                mock_factory.return_value = mock_factory_instance

                await agent.enable_mcp()

                # MCPService should be initialized
                assert agent._mcp_service is not None

                # Check that only valid servers are available (nonexistent-server filtered out)
                available_servers = agent._mcp_service.list_servers()
                assert set(available_servers) == {"filesystem", "weather"}
                assert "nonexistent-server" not in available_servers


class TestChatOrchestratorAgentFactory:
    """Test suite for ChatOrchestratorAgentFactory."""

    @pytest.mark.asyncio
    async def test_create_agent_without_mcp(self, base_config):
        """Test creating agent without MCP configuration."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = await ChatOrchestratorAgentFactory.create_agent(
                project_config=base_config,
                project_dir=project_dir,
            )

            assert isinstance(agent, ChatOrchestratorAgent)
            # MCP is enabled but has no servers/tools when no MCP config
            assert agent._mcp_enabled
            assert len(agent._mcp_tools) == 0

    @pytest.mark.asyncio
    async def test_create_agent_with_session_id(self, base_config):
        """Test creating agent with session ID."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = await ChatOrchestratorAgentFactory.create_agent(
                project_config=base_config,
                project_dir=project_dir,
                session_id="test-session",
            )

            assert isinstance(agent, ChatOrchestratorAgent)
            assert agent._persist_enabled
            assert agent._session_id == "test-session"

    @pytest.mark.asyncio
    async def test_create_agent_with_mcp(self, config_with_mcp):
        """Test creating agent with MCP configuration."""
        with tempfile.TemporaryDirectory() as project_dir:
            with patch("agents.chat_orchestrator.MCPToolFactory") as mock_factory:
                mock_factory_instance = AsyncMock()
                mock_factory_instance.create_all_tools = AsyncMock(return_value=[])
                mock_factory.return_value = mock_factory_instance

                agent = await ChatOrchestratorAgentFactory.create_agent(
                    project_config=config_with_mcp,
                    project_dir=project_dir,
                )

                assert isinstance(agent, ChatOrchestratorAgent)
                assert agent._mcp_enabled

    @pytest.mark.asyncio
    async def test_create_agent_with_model_name(self, base_config):
        """Test creating agent with specific model name."""
        with tempfile.TemporaryDirectory() as project_dir:
            # Use model name (alias) "default" which maps to model "llama3.2:latest"
            agent = await ChatOrchestratorAgentFactory.create_agent(
                project_config=base_config,
                project_dir=project_dir,
                model_name="default",
            )

            assert isinstance(agent, ChatOrchestratorAgent)
            # agent.model_name should be the config name
            assert agent.model_name == "default"
            assert agent._model_string == "llama3.2:latest"
