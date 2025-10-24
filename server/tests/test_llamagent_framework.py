"""
Unit tests for LFAgent framework components.

Tests the core agent framework including:
- LFAgent base class
- LFAgentHistory
- LFAgentSystemPromptGenerator
- LFAgentContextProvider
- ToolDefinition and StreamEvent types
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.base.agent import LFAgent, LFAgentConfig
from agents.base.context_provider import LFAgentContextProvider
from agents.base.history import LFAgentChatMessage, LFAgentHistory
from agents.base.system_prompt_generator import LFAgentSystemPromptGenerator
from agents.base.types import (
    StreamEvent,
    ToolCallRequest,
    ToolDefinition,
)
from config.datamodel import Model, Provider


class TestLFAgentHistory:
    """Test suite for LFAgentHistory."""

    def test_init(self):
        """Test history initialization."""
        history = LFAgentHistory()
        assert history.history == []

    def test_add_message(self):
        """Test adding messages to history."""
        history = LFAgentHistory()
        msg1 = LFAgentChatMessage(role="user", content="Hello")
        msg2 = LFAgentChatMessage(role="assistant", content="Hi there")

        history.add_message(msg1)
        history.add_message(msg2)

        assert len(history.history) == 2
        assert history.history[0] == msg1
        assert history.history[1] == msg2

    def test_get_history(self):
        """Test getting history as dict list."""
        history = LFAgentHistory()
        history.add_message(LFAgentChatMessage(role="user", content="Hello"))
        history.add_message(LFAgentChatMessage(role="assistant", content="Hi"))

        result = history.get_history()
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi"}


class TestLFAgentSystemPromptGenerator:
    """Test suite for LFAgentSystemPromptGenerator."""

    def test_init_no_prompts(self):
        """Test initialization without prompts."""
        generator = LFAgentSystemPromptGenerator(prompts=[])
        assert generator.system_prompts == []
        assert generator.context_providers == {}

    def test_init_with_system_prompts(self):
        """Test initialization with system prompts."""
        prompts = [
            LFAgentChatMessage(role="system", content="You are helpful"),
            LFAgentChatMessage(role="user", content="Hello"),  # Should be filtered
            LFAgentChatMessage(role="system", content="Be concise"),
        ]
        generator = LFAgentSystemPromptGenerator(prompts=prompts)

        # Only system prompts should be stored
        assert len(generator.system_prompts) == 2
        assert generator.system_prompts[0].content == "You are helpful"
        assert generator.system_prompts[1].content == "Be concise"

    def test_generate_prompt_basic(self):
        """Test basic prompt generation."""
        prompts = [LFAgentChatMessage(role="system", content="You are helpful")]
        generator = LFAgentSystemPromptGenerator(prompts=prompts)

        prompt = generator.generate_prompt()
        assert "You are helpful" in prompt

    def test_generate_prompt_with_context_providers(self):
        """Test prompt generation with context providers."""
        prompts = [LFAgentChatMessage(role="system", content="You are helpful")]

        # Create mock context provider
        mock_provider = MagicMock(spec=LFAgentContextProvider)
        mock_provider.title = "Test Context"
        mock_provider.get_info.return_value = "Context information here"

        generator = LFAgentSystemPromptGenerator(
            prompts=prompts, context_providers={"test": mock_provider}
        )

        prompt = generator.generate_prompt()
        assert "You are helpful" in prompt
        assert "EXTRA INFORMATION AND CONTEXT" in prompt
        assert "Test Context" in prompt
        assert "Context information here" in prompt

    def test_generate_prompt_empty_context(self):
        """Test prompt generation when context provider returns empty."""
        prompts = [LFAgentChatMessage(role="system", content="You are helpful")]

        mock_provider = MagicMock(spec=LFAgentContextProvider)
        mock_provider.title = "Empty Context"
        mock_provider.get_info.return_value = ""  # Empty context

        generator = LFAgentSystemPromptGenerator(
            prompts=prompts, context_providers={"test": mock_provider}
        )

        prompt = generator.generate_prompt()
        # Should still have system prompt, but not empty context
        assert "You are helpful" in prompt
        assert "Empty Context" not in prompt


class TestToolDefinition:
    """Test suite for ToolDefinition."""

    def test_tool_definition_creation(self):
        """Test creating a ToolDefinition."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"arg1": {"type": "string"}},
                "required": ["arg1"],
            },
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert "arg1" in tool.parameters["properties"]

    def test_from_mcp_tool_with_schema(self):
        """Test converting MCP tool to ToolDefinition."""

        # Create mock MCP tool class
        class MockMCPTool:
            mcp_tool_name = "mock_tool"
            __doc__ = "Mock tool description"

            class input_schema:
                @staticmethod
                def model_json_schema():
                    return {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string"},
                            "param1": {"type": "string", "description": "First param"},
                        },
                        "required": ["tool_name", "param1"],
                    }

        tool_def = ToolDefinition.from_mcp_tool(MockMCPTool)
        assert tool_def.name == "mock_tool"
        assert tool_def.description == "Mock tool description"
        # tool_name discriminator should be removed
        assert "tool_name" not in tool_def.parameters["properties"]
        assert "param1" in tool_def.parameters["properties"]
        assert tool_def.parameters["required"] == ["param1"]

    def test_from_mcp_tool_no_schema(self):
        """Test converting MCP tool without input schema."""

        class MockMCPTool:
            __name__ = "MockMCPTool"
            __doc__ = "Tool without schema"

        tool_def = ToolDefinition.from_mcp_tool(MockMCPTool)
        assert tool_def.name == "MockMCPTool"
        assert tool_def.description == "Tool without schema"
        assert tool_def.parameters == {"type": "object", "properties": {}}


class TestStreamEvent:
    """Test suite for StreamEvent."""

    def test_content_event(self):
        """Test creating and checking content event."""
        event = StreamEvent(type="content", content="Hello")
        assert event.is_content()
        assert not event.is_tool_call()
        assert event.content == "Hello"
        assert event.tool_call is None

    def test_tool_call_event(self):
        """Test creating and checking tool call event."""
        tool_call = ToolCallRequest(
            id="call_1", name="test_tool", arguments={"arg1": "value1"}
        )
        event = StreamEvent(type="tool_call", tool_call=tool_call)
        assert event.is_tool_call()
        assert not event.is_content()
        assert event.tool_call == tool_call
        assert event.content is None


class TestLFAgent:
    """Test suite for LFAgent base class."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client that inherits from LFAgentClient."""
        from agents.base.clients.client import LFAgentClient
        from config.datamodel import Model, Provider

        # Create a concrete mock that inherits from LFAgentClient
        class MockLFAgentClient(LFAgentClient):
            async def chat(self, *, messages):
                return "Response"

            async def stream_chat(self, *, messages):
                yield "Response"
                yield " text"

            async def stream_chat_with_tools(self, *, messages, tools):
                yield StreamEvent(type="content", content="Response")

            @staticmethod
            def prompt_to_message(prompt):
                from agents.base.history import LFAgentChatMessage

                return LFAgentChatMessage(role="system", content=prompt.content)

        model_config = Model(
            name="test-model",
            provider=Provider.openai,
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        return MockLFAgentClient(model_config=model_config)

    @pytest.fixture
    def agent_config(self, mock_client):
        """Create agent config."""
        history = LFAgentHistory()
        system_prompt_gen = LFAgentSystemPromptGenerator(
            prompts=[LFAgentChatMessage(role="system", content="You are helpful")]
        )
        return LFAgentConfig(
            client=mock_client,
            history=history,
            system_prompt_generator=system_prompt_gen,
        )

    def test_agent_init(self, agent_config, mock_client):
        """Test agent initialization."""
        agent = LFAgent(config=agent_config)
        assert agent.history is not None
        assert agent._client == mock_client
        assert agent._system_prompt_generator is not None

    @pytest.mark.asyncio
    async def test_run_async_with_user_input(self, agent_config):
        """Test running agent with user input."""
        agent = LFAgent(config=agent_config)
        user_msg = LFAgentChatMessage(role="user", content="Hello")

        response = await agent.run_async(user_input=user_msg)

        assert response == "Response"
        assert len(agent.history.history) == 1
        assert agent.history.history[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_run_async_without_user_input(self, agent_config):
        """Test running agent without new user input."""
        agent = LFAgent(config=agent_config)
        # Pre-populate history
        agent.history.add_message(
            LFAgentChatMessage(role="user", content="Previous message")
        )

        response = await agent.run_async()

        assert response == "Response"
        # History should still have the previous message
        assert len(agent.history.history) == 1

    @pytest.mark.asyncio
    async def test_run_async_stream(self, agent_config):
        """Test streaming response."""
        agent = LFAgent(config=agent_config)
        user_msg = LFAgentChatMessage(role="user", content="Hello")

        chunks = []
        async for chunk in agent.run_async_stream(user_input=user_msg):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert len(agent.history.history) == 1

    @pytest.mark.asyncio
    async def test_stream_chat_with_tools(self, agent_config):
        """Test streaming with tools."""
        agent = LFAgent(config=agent_config)
        user_msg = LFAgentChatMessage(role="user", content="Hello")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="Test",
                parameters={"type": "object", "properties": {}},
            )
        ]

        events = []
        async for event in agent.stream_chat_with_tools(
            user_input=user_msg, tools=tools
        ):
            events.append(event)

        assert len(events) > 0
        assert isinstance(events[0], StreamEvent)
        assert len(agent.history.history) == 1

    def test_register_context_provider(self, agent_config):
        """Test registering context provider."""
        agent = LFAgent(config=agent_config)
        provider = MagicMock(spec=LFAgentContextProvider)
        provider.title = "Test"

        agent.register_context_provider("test", provider)

        registered = agent.get_context_provider("test")
        assert registered == provider

    def test_register_duplicate_context_provider(self, agent_config):
        """Test that duplicate registration raises error."""
        agent = LFAgent(config=agent_config)
        provider = MagicMock(spec=LFAgentContextProvider)
        provider.title = "Test"

        agent.register_context_provider("test", provider)

        with pytest.raises(ValueError, match="already registered"):
            agent.register_context_provider("test", provider)

    def test_remove_context_provider(self, agent_config):
        """Test removing context provider."""
        agent = LFAgent(config=agent_config)
        provider = MagicMock(spec=LFAgentContextProvider)
        provider.title = "Test"

        agent.register_context_provider("test", provider)
        agent.remove_context_provider("test")

        assert agent.get_context_provider("test") is None

    def test_prepare_messages(self, agent_config):
        """Test message preparation."""
        agent = LFAgent(config=agent_config)
        agent.history.add_message(LFAgentChatMessage(role="user", content="Hello"))
        agent.history.add_message(
            LFAgentChatMessage(role="assistant", content="Hi there")
        )

        messages = agent._prepare_messages()

        # Should have system prompt + history messages
        assert len(messages) >= 3
        assert messages[0].role == "system"
        assert "helpful" in messages[0].content
        assert messages[1].role == "user"
        assert messages[1].content == "Hello"
        assert messages[2].role == "assistant"
        assert messages[2].content == "Hi there"


class TestToolCallRequest:
    """Test suite for ToolCallRequest."""

    def test_tool_call_request_creation(self):
        """Test creating a ToolCallRequest."""
        request = ToolCallRequest(
            id="call_123", name="test_tool", arguments={"arg1": "value1", "arg2": 42}
        )
        assert request.id == "call_123"
        assert request.name == "test_tool"
        assert request.arguments["arg1"] == "value1"
        assert request.arguments["arg2"] == 42
