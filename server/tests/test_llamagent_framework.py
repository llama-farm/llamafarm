"""
Unit tests for LFAgent framework components.

Tests the core agent framework including:
- LFAgent base class
- LFAgentHistory
- LFAgentSystemPromptGenerator
- LFAgentContextProvider
- ToolDefinition and StreamEvent types
"""

from unittest.mock import MagicMock

import pytest

from agents.base.agent import LFAgent, LFAgentConfig
from agents.base.context_provider import LFAgentContextProvider
from agents.base.history import (
    LFAgentHistory,
    LFChatCompletionAssistantMessageParam,
    LFChatCompletionSystemMessageParam,
    LFChatCompletionUserMessageParam,
)
from agents.base.system_prompt_generator import LFAgentSystemPromptGenerator
from agents.base.types import (
    StreamEvent,
    ToolCallRequest,
    ToolDefinition,
)


class TestLFAgentHistory:
    """Test suite for LFAgentHistory."""

    def test_init(self):
        """Test history initialization."""
        history = LFAgentHistory()
        assert history.history == []

    def test_add_message(self):
        """Test adding messages to history."""
        history = LFAgentHistory()
        msg1 = LFChatCompletionUserMessageParam(role="user", content="Hello")
        msg2 = LFChatCompletionAssistantMessageParam(
            role="assistant", content="Hi there"
        )

        history.add_message(msg1)
        history.add_message(msg2)

        assert len(history.history) == 2
        assert history.history[0] == msg1
        assert history.history[1] == msg2

    def test_get_history(self):
        """Test getting history as dict list."""
        history = LFAgentHistory()
        history.add_message(
            LFChatCompletionUserMessageParam(role="user", content="Hello")
        )
        history.add_message(
            LFChatCompletionAssistantMessageParam(role="assistant", content="Hi")
        )

        result = history.get_history()
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi"}

    def test_serialize_message_with_tool_calls(self):
        """Test serialization of messages with tool_calls using OpenAI types.

        This tests the fix for the ValidatorIterator serialization issue where
        OpenAI SDK Pydantic models weren't being properly converted to dicts.
        """
        from openai.types.chat import ChatCompletionMessageFunctionToolCallParam
        from openai.types.chat.chat_completion_message_tool_call_param import Function

        history = LFAgentHistory()

        # Create message with tool_calls using OpenAI types (mimics chat_orchestrator)
        msg = LFChatCompletionAssistantMessageParam(
            role="assistant",
            content="I'll call the weather function.",
            tool_calls=[
                ChatCompletionMessageFunctionToolCallParam(
                    type="function",
                    id="call_123",
                    function=Function(
                        name="get_weather",
                        arguments='{"location": "NYC"}',
                    ),
                )
            ],
        )
        history.add_message(msg)

        # Get serialized history
        result = history.get_history()

        # Verify tool_calls was serialized correctly as plain dicts
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "I'll call the weather function."
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1

        tool_call = result[0]["tool_calls"][0]
        assert isinstance(tool_call, dict)
        assert tool_call["type"] == "function"
        assert tool_call["id"] == "call_123"
        assert isinstance(tool_call["function"], dict)
        assert tool_call["function"]["name"] == "get_weather"
        assert tool_call["function"]["arguments"] == '{"location": "NYC"}'

    def test_serialize_message_without_tool_calls(self):
        """Test serialization of regular messages without tool_calls."""
        history = LFAgentHistory()
        msg = LFChatCompletionAssistantMessageParam(
            role="assistant",
            content="Hello!",
        )
        history.add_message(msg)

        result = history.get_history()
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hello!"
        # tool_calls should not be present or should be None/falsy
        assert not result[0].get("tool_calls")

    def test_serialize_message_with_dict_tool_calls(self):
        """Test serialization when tool_calls are already plain dicts."""
        history = LFAgentHistory()
        msg = LFChatCompletionAssistantMessageParam(
            role="assistant",
            content="Calling tool",
            tool_calls=[
                {
                    "type": "function",
                    "id": "call_456",
                    "function": {
                        "name": "test_func",
                        "arguments": "{}",
                    },
                }
            ],
        )
        history.add_message(msg)

        result = history.get_history()
        assert len(result) == 1
        tool_call = result[0]["tool_calls"][0]
        assert tool_call["id"] == "call_456"
        assert tool_call["function"]["name"] == "test_func"


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
            LFChatCompletionSystemMessageParam(
                role="system", content="You are helpful"
            ),
            LFChatCompletionUserMessageParam(
                role="user", content="Hello"
            ),  # Should be filtered
            LFChatCompletionSystemMessageParam(role="system", content="Be concise"),
        ]
        generator = LFAgentSystemPromptGenerator(prompts=prompts)

        # Only system prompts should be stored
        assert len(generator.system_prompts) == 2
        assert generator.system_prompts[0].content == "You are helpful"
        assert generator.system_prompts[1].content == "Be concise"

    def test_generate_prompt_basic(self):
        """Test basic prompt generation."""
        prompts = [
            LFChatCompletionSystemMessageParam(role="system", content="You are helpful")
        ]
        generator = LFAgentSystemPromptGenerator(prompts=prompts)

        prompt = generator.generate_prompt()
        assert "You are helpful" in prompt

    def test_generate_prompt_with_context_providers(self):
        """Test prompt generation with context providers."""
        prompts = [
            LFChatCompletionSystemMessageParam(role="system", content="You are helpful")
        ]

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
        prompts = [
            LFChatCompletionSystemMessageParam(role="system", content="You are helpful")
        ]

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
        from config.datamodel import Model, Provider

        from agents.base.clients.client import LFAgentClient

        # Create a concrete mock that inherits from LFAgentClient
        class MockLFAgentClient(LFAgentClient):
            async def chat(
                self,
                *,
                messages,
                tools=None,
                extra_body=None,
            ):
                return "Response"

            async def stream_chat(
                self,
                *,
                messages,
                tools=None,
                extra_body=None,
            ):
                from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

                mock_chunk = MagicMock(spec=ChatCompletionChunk)
                mock_chunk.choices = [MagicMock()]
                mock_chunk.choices[0].delta = MagicMock()
                mock_chunk.choices[0].delta.content = "Response"
                yield mock_chunk

            @staticmethod
            def prompt_to_message(prompt):
                return LFChatCompletionSystemMessageParam(
                    role="system", content=prompt.content
                )

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
            prompts=[
                LFChatCompletionSystemMessageParam(
                    role="system", content="You are helpful"
                )
            ]
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
        user_msg = LFChatCompletionUserMessageParam(role="user", content="Hello")

        response = await agent.run_async(messages=[user_msg])

        assert response == "Response"
        assert len(agent.history.history) == 1
        assert agent.history.history[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_run_async_without_user_input(self, agent_config):
        """Test running agent without new user input."""
        agent = LFAgent(config=agent_config)
        # Pre-populate history
        agent.history.add_message(
            LFChatCompletionUserMessageParam(role="user", content="Previous message")
        )

        response = await agent.run_async()

        assert response == "Response"
        # History should still have the previous message
        assert len(agent.history.history) == 1

    @pytest.mark.asyncio
    async def test_run_async_stream(self, agent_config):
        """Test streaming response."""
        agent = LFAgent(config=agent_config)
        user_msg = LFChatCompletionUserMessageParam(role="user", content="Hello")

        chunks = []
        async for chunk in agent.run_async_stream(messages=[user_msg]):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert len(agent.history.history) == 1

    @pytest.mark.asyncio
    async def test_stream_chat_with_tools(self, agent_config):
        """Test streaming with tools."""
        agent = LFAgent(config=agent_config)
        user_msg = LFChatCompletionUserMessageParam(role="user", content="Hello")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="Test",
                parameters={"type": "object", "properties": {}},
            )
        ]

        chunks = []
        async for chunk in agent.run_async_stream(messages=[user_msg], tools=tools):
            chunks.append(chunk)

        assert len(chunks) > 0
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
        agent.history.add_message(
            LFChatCompletionUserMessageParam(role="user", content="Hello")
        )
        agent.history.add_message(
            LFChatCompletionAssistantMessageParam(role="assistant", content="Hi there")
        )

        messages = agent._prepare_messages()

        # Should have system prompt + history messages
        assert len(messages) >= 3
        assert messages[0]["role"] == "system"
        assert "helpful" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hi there"


class TestSystemPromptOverride:
    """Test API system prompts override config system prompts."""

    @pytest.fixture
    def mock_client(self):
        from config.datamodel import Model, Provider

        from agents.base.clients.client import LFAgentClient

        class MockLFAgentClient(LFAgentClient):
            last_messages = None

            async def chat(self, *, messages, tools=None, extra_body=None):
                self.last_messages = messages
                return "Response"

            async def stream_chat(
                self, *, messages, tools=None, extra_body=None
            ):
                self.last_messages = messages
                mock_chunk = MagicMock()
                mock_chunk.choices = [MagicMock()]
                mock_chunk.choices[0].delta = MagicMock()
                mock_chunk.choices[0].delta.content = "Response"
                yield mock_chunk

            @staticmethod
            def prompt_to_message(prompt):
                return LFChatCompletionSystemMessageParam(
                    role="system", content=prompt.content
                )

        model_config = Model(
            name="test-model",
            provider=Provider.openai,
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        return MockLFAgentClient(model_config=model_config)

    @pytest.fixture
    def agent_with_config_prompt(self, mock_client):
        history = LFAgentHistory()
        system_prompt_gen = LFAgentSystemPromptGenerator(
            prompts=[
                LFChatCompletionSystemMessageParam(
                    role="system",
                    content="Config system prompt",
                )
            ]
        )
        config = LFAgentConfig(
            client=mock_client,
            history=history,
            system_prompt_generator=system_prompt_gen,
        )
        return LFAgent(config=config), mock_client

    @pytest.mark.asyncio
    async def test_api_system_prompt_overrides_config(
        self, agent_with_config_prompt
    ):
        """API system message replaces config system prompt."""
        agent, client = agent_with_config_prompt

        await agent.run_async(
            messages=[
                {"role": "system", "content": "API system prompt"},
                {"role": "user", "content": "Hello"},
            ]
        )

        msgs = client.last_messages
        # First message should be the API system prompt
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "API system prompt"
        # Config system prompt should NOT appear
        assert not any(
            "Config system prompt" in str(m.get("content", ""))
            for m in msgs
        )

    @pytest.mark.asyncio
    async def test_config_prompt_used_when_no_api_system(
        self, agent_with_config_prompt
    ):
        """Config system prompt used when API doesn't send one."""
        agent, client = agent_with_config_prompt

        await agent.run_async(
            messages=[{"role": "user", "content": "Hello"}]
        )

        msgs = client.last_messages
        assert msgs[0]["role"] == "system"
        assert "Config system prompt" in msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_system_prompt_not_persisted_in_history(
        self, agent_with_config_prompt
    ):
        """System messages should not be stored in history."""
        agent, _ = agent_with_config_prompt

        await agent.run_async(
            messages=[
                {"role": "system", "content": "API system prompt"},
                {"role": "user", "content": "Hello"},
            ]
        )

        # History should only have the user message
        assert len(agent.history.history) == 1
        assert agent.history.history[0]["role"] == "user"
        assert agent.history.history[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_context_providers_append_to_api_system_prompt(
        self, agent_with_config_prompt
    ):
        """RAG context providers still work with API system prompts."""
        agent, client = agent_with_config_prompt

        provider = MagicMock(spec=LFAgentContextProvider)
        provider.title = "RAG Context"
        provider.get_info.return_value = "Retrieved document content"
        agent.register_context_provider("rag", provider)

        await agent.run_async(
            messages=[
                {"role": "system", "content": "API system prompt"},
                {"role": "user", "content": "Hello"},
            ]
        )

        msgs = client.last_messages
        system_content = msgs[0]["content"]
        assert "API system prompt" in system_content
        assert "RAG Context" in system_content
        assert "Retrieved document content" in system_content

    @pytest.mark.asyncio
    async def test_successive_requests_different_system_prompts(
        self, agent_with_config_prompt
    ):
        """Each request can send a different system prompt."""
        agent, client = agent_with_config_prompt

        # First request with system prompt A
        await agent.run_async(
            messages=[
                {"role": "system", "content": "System A"},
                {"role": "user", "content": "Hello"},
            ]
        )
        assert "System A" in client.last_messages[0]["content"]

        # Second request with system prompt B
        await agent.run_async(
            messages=[
                {"role": "system", "content": "System B"},
                {"role": "user", "content": "World"},
            ]
        )
        assert "System B" in client.last_messages[0]["content"]
        assert "System A" not in str(client.last_messages)

    @pytest.mark.asyncio
    async def test_request_without_system_falls_back_to_config(
        self, agent_with_config_prompt
    ):
        """After a request with API system, next without falls back to config."""
        agent, client = agent_with_config_prompt

        # First request with API system prompt
        await agent.run_async(
            messages=[
                {"role": "system", "content": "API override"},
                {"role": "user", "content": "Hello"},
            ]
        )
        assert "API override" in client.last_messages[0]["content"]

        # Second request WITHOUT system prompt — should use config
        await agent.run_async(
            messages=[{"role": "user", "content": "World"}]
        )
        assert "Config system prompt" in client.last_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_stream_api_system_prompt_overrides_config(
        self, agent_with_config_prompt
    ):
        """Streaming also respects API system prompt override."""
        agent, client = agent_with_config_prompt

        async for _ in agent.run_async_stream(
            messages=[
                {"role": "system", "content": "Stream API prompt"},
                {"role": "user", "content": "Hello"},
            ]
        ):
            pass

        msgs = client.last_messages
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Stream API prompt"
        assert not any(
            "Config system prompt" in str(m.get("content", ""))
            for m in msgs
        )

    @pytest.mark.asyncio
    async def test_override_persists_across_tool_loop_calls(
        self, agent_with_config_prompt
    ):
        """System prompt override survives messages=None in tool loops."""
        agent, client = agent_with_config_prompt

        # First call sets the override
        await agent.run_async(
            messages=[
                {"role": "system", "content": "Tool loop prompt"},
                {"role": "user", "content": "Use a tool"},
            ]
        )
        assert "Tool loop prompt" in client.last_messages[0]["content"]

        # Subsequent call with messages=None (tool iteration)
        await agent.run_async(messages=None)
        assert "Tool loop prompt" in client.last_messages[0]["content"]
        assert "Config system prompt" not in str(client.last_messages)

    @pytest.mark.asyncio
    async def test_none_content_not_stringified(
        self, agent_with_config_prompt
    ):
        """None content should not become the string 'None'."""
        agent, client = agent_with_config_prompt

        await agent.run_async(
            messages=[
                {"role": "system", "content": None},
                {"role": "user", "content": "Hello"},
            ]
        )

        msgs = client.last_messages
        # Should use empty string, not "None"
        assert msgs[0]["role"] == "system"
        assert "None" not in msgs[0]["content"]


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
