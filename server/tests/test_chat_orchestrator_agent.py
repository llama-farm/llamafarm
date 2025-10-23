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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.chat_orchestrator import (
    ChatOrchestratorAgent,
    ChatOrchestratorAgentFactory,
)
from agents.base.history import LFAgentChatMessage
from agents.base.types import StreamEvent, ToolCallRequest
from config.datamodel import (
    LlamaFarmConfig,
    Mcp,
    Message,
    Model,
    Prompt,
    Provider,
    Runtime,
    Server,
    Transport,
    Version,
)


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
            Prompt(
                name="default",
                messages=[Message(role="system", content="You are helpful")],
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
            Prompt(
                name="default",
                messages=[Message(role="system", content="You are helpful")],
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
        mock_run_async.return_value = "Hello there!"

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            user_input = LFAgentChatMessage(role="user", content="Hi")
            response = await agent.run_async(user_input=user_input)

            assert response == "Hello there!"
            # Note: Since we're mocking the parent's run_async, history management
            # is bypassed in this test

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_with_tool_call(self, mock_run_async, base_config):
        """Test chat with tool call."""
        # First call: LLM requests a tool
        # Second call: LLM provides final answer
        mock_run_async.side_effect = [
            json.dumps({"tool_name": "test_tool", "tool_parameters": {"arg": "value"}}),
            "Final answer based on tool result",
        ]

        # Mock MCP tool
        mock_tool_class = MagicMock()
        mock_tool_class.mcp_tool_name = "test_tool"
        mock_tool_instance = AsyncMock()
        mock_tool_instance.arun = AsyncMock(
            return_value=MagicMock(result="tool result")
        )
        mock_tool_class.return_value = mock_tool_instance
        mock_tool_class.input_schema = MagicMock()
        mock_tool_class.input_schema.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent._mcp_enabled = True
            agent._mcp_tools = [mock_tool_class]

            user_input = LFAgentChatMessage(role="user", content="Use the tool")
            response = await agent.run_async(user_input=user_input)

            assert response == "Final answer based on tool result"
            assert mock_tool_instance.arun.called

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_max_iterations(self, mock_run_async, base_config):
        """Test that max iterations is enforced."""
        # Keep requesting tools forever
        mock_run_async.return_value = json.dumps(
            {"tool_name": "test_tool", "tool_parameters": {"arg": "value"}}
        )

        mock_tool_class = MagicMock()
        mock_tool_class.mcp_tool_name = "test_tool"
        mock_tool_instance = AsyncMock()
        mock_tool_instance.arun = AsyncMock(return_value=MagicMock(result="result"))
        mock_tool_class.return_value = mock_tool_instance
        mock_tool_class.input_schema = MagicMock()
        mock_tool_class.input_schema.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent._mcp_enabled = True
            agent._mcp_tools = [mock_tool_class]

            user_input = LFAgentChatMessage(role="user", content="Test")
            response = await agent.run_async(user_input=user_input)

            # Should return max iterations message
            assert "maximum number of tool calls" in response

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_tool_not_found(self, mock_run_async, base_config):
        """Test handling of non-existent tool."""
        mock_run_async.side_effect = [
            json.dumps({"tool_name": "nonexistent_tool", "tool_parameters": {}}),
            "I apologize",
        ]

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent._mcp_enabled = True
            agent._mcp_tools = []

            user_input = LFAgentChatMessage(role="user", content="Test")
            response = await agent.run_async(user_input=user_input)

            # Should handle error gracefully
            assert "apologize" in response.lower()

    @pytest.mark.asyncio
    @patch("agents.base.agent.LFAgent.run_async")
    async def test_run_async_tool_execution_error(self, mock_run_async, base_config):
        """Test handling of tool execution errors."""
        mock_run_async.side_effect = [
            json.dumps({"tool_name": "test_tool", "tool_parameters": {"arg": "value"}}),
            "Sorry, there was an error",
        ]

        mock_tool_class = MagicMock()
        mock_tool_class.mcp_tool_name = "test_tool"
        mock_tool_instance = AsyncMock()
        mock_tool_instance.arun = AsyncMock(side_effect=Exception("Tool failed"))
        mock_tool_class.return_value = mock_tool_instance
        mock_tool_class.input_schema = MagicMock()
        mock_tool_class.input_schema.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent._mcp_enabled = True
            agent._mcp_tools = [mock_tool_class]

            user_input = LFAgentChatMessage(role="user", content="Test")
            response = await agent.run_async(user_input=user_input)

            # Should handle error gracefully
            assert "error" in response.lower() or "sorry" in response.lower()

    @pytest.mark.asyncio
    async def test_run_async_stream_no_tools(self, base_config):
        """Test streaming without MCP tools."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )

            # Mock the parent stream_chat method
            async def mock_stream():
                yield "Hello"
                yield " world"

            with patch.object(
                agent.__class__.__bases__[0],
                "run_async_stream",
                return_value=mock_stream(),
            ):
                user_input = LFAgentChatMessage(role="user", content="Hi")
                chunks = []
                async for chunk in agent.run_async_stream(user_input=user_input):
                    chunks.append(chunk)

                assert len(chunks) == 2
                assert "".join(chunks) == "Hello world"

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
                return_value=MagicMock(result="Tool result")
            )
            mock_tool_class.return_value = mock_tool_instance
            mock_tool_class.input_schema = MagicMock()
            mock_tool_class.input_schema.return_value = MagicMock()
            agent._mcp_tools = [mock_tool_class]

            # Mock streaming with tool call
            async def mock_stream_with_tools(*args, **kwargs):
                yield StreamEvent(type="content", content="Let me check...")
                yield StreamEvent(
                    type="tool_call",
                    tool_call=ToolCallRequest(
                        id="call_1", name="test_tool", arguments={"arg": "value"}
                    ),
                )

            with patch.object(
                agent, "stream_chat_with_tools", side_effect=mock_stream_with_tools
            ):
                user_input = LFAgentChatMessage(role="user", content="Test")
                chunks = []
                async for chunk in agent.run_async_stream(user_input=user_input):
                    chunks.append(chunk)

                # Should include content and tool call indicator
                assert len(chunks) > 0
                assert any("Let me check" in str(c) for c in chunks)

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

            agent1.history.add_message(LFAgentChatMessage(role="user", content="Hello"))
            agent1.history.add_message(
                LFAgentChatMessage(role="assistant", content="Hi there")
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
            assert agent2.history.history[0].content == "Hello"
            assert agent2.history.history[1].content == "Hi there"

    def test_reset_history(self, base_config):
        """Test resetting history."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = ChatOrchestratorAgent(
                project_config=base_config,
                project_dir=project_dir,
            )
            agent.enable_persistence(session_id="test-session")

            # Add messages and persist
            agent.history.add_message(LFAgentChatMessage(role="user", content="Hello"))
            agent._persist_history()

            # Reset history
            agent.reset_history()

            # History should be empty and file should be deleted
            assert len(agent.history.history) == 0
            path = agent._history_file_path
            if path:
                assert not path.exists()

    @pytest.mark.asyncio
    async def test_enable_mcp(self, config_with_mcp):
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

                await agent.enable_mcp()

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


class TestChatOrchestratorAgentFactory:
    """Test suite for ChatOrchestratorAgentFactory."""

    @pytest.mark.asyncio
    async def test_create_agent_without_mcp(self, base_config):
        """Test creating agent without MCP."""
        with tempfile.TemporaryDirectory() as project_dir:
            agent = await ChatOrchestratorAgentFactory.create_agent(
                project_config=base_config,
                project_dir=project_dir,
            )

            assert isinstance(agent, ChatOrchestratorAgent)
            assert not agent._mcp_enabled

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
