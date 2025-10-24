"""
Unit tests for LFAgentClientOpenAI.

Tests the OpenAI client implementation including:
- Basic chat
- Streaming chat
- Tool calling with native OpenAI function calling
- Message format conversion
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.base.clients.openai import LFAgentClientOpenAI
from agents.base.history import LFAgentChatMessage
from agents.base.types import ToolDefinition
from config.datamodel import Model, Prompt, Provider


@pytest.fixture
def model_config():
    """Create test model config."""
    return Model(
        name="test-model",
        provider=Provider.openai,
        model="gpt-4",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
    )


@pytest.fixture
def client(model_config):
    """Create client instance."""
    return LFAgentClientOpenAI(model_config=model_config)


class TestLFAgentClientOpenAI:
    """Test suite for LFAgentClientOpenAI."""

    def test_init(self, client, model_config):
        """Test client initialization."""
        assert client.model_name == "test-model"
        assert client._model_config == model_config

    def test_prompt_to_message(self):
        """Test converting Prompt to list of LFAgentChatMessage."""
        from config.datamodel import Message

        prompt = Prompt(
            name="test", messages=[Message(role="system", content="You are helpful")]
        )
        messages = LFAgentClientOpenAI.prompt_to_message(prompt)

        assert isinstance(messages, list)
        assert len(messages) == 1
        assert isinstance(messages[0], LFAgentChatMessage)
        assert messages[0].role == "system"
        assert "helpful" in messages[0].content

    def test_message_to_openai_message_system(self, client):
        """Test converting system message."""
        msg = LFAgentChatMessage(role="system", content="System prompt")
        result = client._message_to_openai_message(msg)

        assert result["role"] == "system"
        assert result["content"] == "System prompt"

    def test_message_to_openai_message_user(self, client):
        """Test converting user message."""
        msg = LFAgentChatMessage(role="user", content="Hello")
        result = client._message_to_openai_message(msg)

        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_message_to_openai_message_assistant(self, client):
        """Test converting assistant message."""
        msg = LFAgentChatMessage(role="assistant", content="Hi there")
        result = client._message_to_openai_message(msg)

        assert result["role"] == "assistant"
        assert result["content"] == "Hi there"

    def test_message_to_openai_message_tool(self, client):
        """Test converting tool result message."""
        msg = LFAgentChatMessage(role="tool", content="Tool result")
        result = client._message_to_openai_message(msg)

        # Tool results are converted to user messages
        assert result["role"] == "user"
        assert result["content"] == "Tool result"

    def test_message_to_openai_message_developer_role(self, client):
        """Test converting developer role message."""
        msg = LFAgentChatMessage(role="developer", content="Developer message")

        # Developer role should raise ValueError as it's not handled
        with pytest.raises(ValueError, match="Unknown message role"):
            client._message_to_openai_message(msg)

    def test_tool_to_openai_format(self, client):
        """Test converting ToolDefinition to OpenAI format."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "First argument"}
                },
                "required": ["arg1"],
            },
        )

        result = client._tool_to_openai_format(tool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "test_tool"
        assert result["function"]["description"] == "A test tool"
        assert result["function"]["parameters"] == tool.parameters

    @pytest.mark.asyncio
    @patch("agents.base.clients.openai.AsyncOpenAI")
    async def test_chat_success(self, mock_openai_class, client):
        """Test basic chat without tools."""
        # Mock streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.choices[0].delta.tool_calls = None
        mock_chunk1.choices[0].finish_reason = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"
        mock_chunk2.choices[0].delta.tool_calls = None
        mock_chunk2.choices[0].finish_reason = "stop"

        async def mock_stream():
            yield mock_chunk1
            yield mock_chunk2

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_class.return_value = mock_client

        messages = [LFAgentChatMessage(role="user", content="Hi")]
        response = await client.chat(messages=messages)

        assert response == "Hello world"

    @pytest.mark.asyncio
    @patch("agents.base.clients.openai.AsyncOpenAI")
    async def test_stream_chat(self, mock_openai_class, client):
        """Test streaming chat without tools."""
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.choices[0].delta.tool_calls = None
        mock_chunk1.choices[0].finish_reason = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"
        mock_chunk2.choices[0].delta.tool_calls = None
        mock_chunk2.choices[0].finish_reason = "stop"

        async def mock_stream():
            yield mock_chunk1
            yield mock_chunk2

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_class.return_value = mock_client

        messages = [LFAgentChatMessage(role="user", content="Hi")]
        chunks = []
        async for chunk in client.stream_chat(messages=messages):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0] == "Hello"
        assert chunks[1] == " world"

    @pytest.mark.asyncio
    @patch("agents.base.clients.openai.AsyncOpenAI")
    async def test_stream_chat_with_tools_no_tool_call(self, mock_openai_class, client):
        """Test streaming chat with tools available but no tool call."""
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Response"
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].finish_reason = "stop"

        async def mock_stream():
            yield mock_chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_class.return_value = mock_client

        messages = [LFAgentChatMessage(role="user", content="Hi")]
        tools = [
            ToolDefinition(
                name="test_tool",
                description="Test",
                parameters={"type": "object", "properties": {}},
            )
        ]

        events = []
        async for event in client.stream_chat_with_tools(
            messages=messages, tools=tools
        ):
            events.append(event)

        assert len(events) == 1
        assert events[0].is_content()
        assert events[0].content == "Response"

    @pytest.mark.asyncio
    @patch("agents.base.clients.openai.AsyncOpenAI")
    async def test_stream_chat_with_tools_tool_call(self, mock_openai_class, client):
        """Test streaming chat with tool call."""
        # Create mock tool call delta chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = None
        mock_tc1 = MagicMock()
        mock_tc1.index = 0
        mock_tc1.id = "call_abc123"
        mock_tc1.function = MagicMock()
        mock_tc1.function.name = "test_tool"
        mock_tc1.function.arguments = '{"ar'
        mock_chunk1.choices[0].delta.tool_calls = [mock_tc1]
        mock_chunk1.choices[0].finish_reason = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = None
        mock_tc2 = MagicMock()
        mock_tc2.index = 0
        mock_tc2.id = None
        mock_tc2.function = MagicMock()
        mock_tc2.function.name = None
        mock_tc2.function.arguments = 'g1": "value1"}'
        mock_chunk2.choices[0].delta.tool_calls = [mock_tc2]
        mock_chunk2.choices[0].finish_reason = "tool_calls"

        async def mock_stream():
            yield mock_chunk1
            yield mock_chunk2

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_class.return_value = mock_client

        messages = [LFAgentChatMessage(role="user", content="Use the tool")]
        tools = [
            ToolDefinition(
                name="test_tool",
                description="Test",
                parameters={
                    "type": "object",
                    "properties": {"arg1": {"type": "string"}},
                },
            )
        ]

        events = []
        async for event in client.stream_chat_with_tools(
            messages=messages, tools=tools
        ):
            events.append(event)

        # Should get one tool call event
        assert len(events) == 1
        assert events[0].is_tool_call()
        assert events[0].tool_call is not None
        assert events[0].tool_call.name == "test_tool"
        assert events[0].tool_call.arguments == {"arg1": "value1"}

    @pytest.mark.asyncio
    @patch("agents.base.clients.openai.AsyncOpenAI")
    async def test_stream_chat_with_tools_mixed_content_and_tool_call(
        self, mock_openai_class, client
    ):
        """Test streaming with both content and tool call."""
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Let me check that..."
        mock_chunk1.choices[0].delta.tool_calls = None
        mock_chunk1.choices[0].finish_reason = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = None
        mock_tc = MagicMock()
        mock_tc.index = 0
        mock_tc.id = "call_123"
        mock_tc.function = MagicMock()
        mock_tc.function.name = "search"
        mock_tc.function.arguments = '{"query": "test"}'
        mock_chunk2.choices[0].delta.tool_calls = [mock_tc]
        mock_chunk2.choices[0].finish_reason = "tool_calls"

        async def mock_stream():
            yield mock_chunk1
            yield mock_chunk2

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_class.return_value = mock_client

        messages = [LFAgentChatMessage(role="user", content="Search for test")]
        tools = [
            ToolDefinition(
                name="search",
                description="Search",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            )
        ]

        events = []
        async for event in client.stream_chat_with_tools(
            messages=messages, tools=tools
        ):
            events.append(event)

        assert len(events) == 2
        assert events[0].is_content()
        assert events[0].content == "Let me check that..."
        assert events[1].is_tool_call()
        assert events[1].tool_call.name == "search"

    @pytest.mark.asyncio
    @patch("agents.base.clients.openai.AsyncOpenAI")
    async def test_stream_chat_with_tools_invalid_json_arguments(
        self, mock_openai_class, client
    ):
        """Test handling of invalid JSON in tool call arguments."""
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = None
        mock_tc = MagicMock()
        mock_tc.index = 0
        mock_tc.id = "call_123"
        mock_tc.function = MagicMock()
        mock_tc.function.name = "test_tool"
        mock_tc.function.arguments = '{"invalid json'  # Invalid JSON
        mock_chunk.choices[0].delta.tool_calls = [mock_tc]
        mock_chunk.choices[0].finish_reason = "tool_calls"

        async def mock_stream():
            yield mock_chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_class.return_value = mock_client

        messages = [LFAgentChatMessage(role="user", content="Test")]
        tools = [
            ToolDefinition(
                name="test_tool",
                description="Test",
                parameters={"type": "object", "properties": {}},
            )
        ]

        events = []
        async for event in client.stream_chat_with_tools(
            messages=messages, tools=tools
        ):
            events.append(event)

        # Should not yield any tool call events for invalid JSON
        assert len(events) == 0

    @pytest.mark.asyncio
    @patch("agents.base.clients.openai.AsyncOpenAI")
    async def test_stream_chat_with_multiple_tools(self, mock_openai_class, client):
        """Test streaming with multiple tool calls."""
        # First tool call
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = None
        mock_tc1 = MagicMock()
        mock_tc1.index = 0
        mock_tc1.id = "call_1"
        mock_tc1.function = MagicMock()
        mock_tc1.function.name = "tool1"
        mock_tc1.function.arguments = '{"arg": "val1"}'
        mock_chunk1.choices[0].delta.tool_calls = [mock_tc1]
        mock_chunk1.choices[0].finish_reason = None

        # Second tool call
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = None
        mock_tc2 = MagicMock()
        mock_tc2.index = 1
        mock_tc2.id = "call_2"
        mock_tc2.function = MagicMock()
        mock_tc2.function.name = "tool2"
        mock_tc2.function.arguments = '{"arg": "val2"}'
        mock_chunk2.choices[0].delta.tool_calls = [mock_tc2]
        mock_chunk2.choices[0].finish_reason = "tool_calls"

        async def mock_stream():
            yield mock_chunk1
            yield mock_chunk2

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_class.return_value = mock_client

        messages = [LFAgentChatMessage(role="user", content="Use tools")]
        tools = [
            ToolDefinition(
                name="tool1",
                description="First tool",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="tool2",
                description="Second tool",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        events = []
        async for event in client.stream_chat_with_tools(
            messages=messages, tools=tools
        ):
            events.append(event)

        assert len(events) == 2
        assert all(e.is_tool_call() for e in events)
        assert events[0].tool_call.name == "tool1"
        assert events[1].tool_call.name == "tool2"

    @pytest.mark.asyncio
    @patch("agents.base.clients.openai.AsyncOpenAI")
    async def test_stream_chat_empty_choices(self, mock_openai_class, client):
        """Test handling of chunks with empty choices."""
        mock_chunk = MagicMock()
        mock_chunk.choices = []  # Empty choices

        async def mock_stream():
            yield mock_chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_class.return_value = mock_client

        messages = [LFAgentChatMessage(role="user", content="Test")]

        events = []
        async for event in client.stream_chat_with_tools(messages=messages, tools=[]):
            events.append(event)

        # Should not yield any events for empty choices
        assert len(events) == 0
