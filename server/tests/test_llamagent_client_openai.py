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
from config.datamodel import Model, PromptMessage, PromptSet, Provider, ToolCallStrategy
from pydantic import BaseModel

from agents.base.clients.client import LFAgentClient
from agents.base.clients.openai import LFAgentClientOpenAI
from agents.base.history import LFChatCompletionUserMessageParam
from agents.base.types import ToolDefinition


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

    def test_prompt_message_to_chat_completion_message(self):
        """Test converting PromptMessage to LFAgentChatMessage."""
        prompt_message = PromptMessage(role="system", content="You are helpful")
        message = LFAgentClient.prompt_message_to_chat_completion_message(
            prompt_message
        )

        assert isinstance(message, dict)
        assert message["role"] == "system"
        assert "helpful" in message["content"]

    def test_prompt_set_to_messages(self):
        """Test converting PromptSet to list of LFAgentChatMessage."""
        prompt_set = PromptSet(
            name="test",
            messages=[
                PromptMessage(role="system", content="You are helpful"),
                PromptMessage(role="user", content="Hello"),
            ],
        )
        messages = [
            LFAgentClient.prompt_message_to_chat_completion_message(msg)
            for msg in prompt_set.messages
        ]

        assert isinstance(messages, list)
        assert len(messages) == 2
        assert all(isinstance(m, dict) for m in messages)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

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
        # Mock non-streaming completion response
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Hello world"
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_openai_class.return_value = mock_client

        messages = [LFChatCompletionUserMessageParam(role="user", content="Hi")]
        response = await client.chat(messages=messages)

        assert response.choices[0].message.content == "Hello world"

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

        messages = [LFChatCompletionUserMessageParam(role="user", content="Hi")]
        chunks = []
        async for chunk in client.stream_chat(messages=messages):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " world"

    @pytest.mark.asyncio
    @patch("agents.base.clients.openai.AsyncOpenAI")
    async def test_stream_chat_with_tools(self, mock_openai_class, client):
        """Test streaming chat with tools."""
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

        messages = [LFChatCompletionUserMessageParam(role="user", content="Hi")]
        tools = [
            ToolDefinition(
                name="test_tool",
                description="Test",
                parameters={"type": "object", "properties": {}},
            )
        ]

        chunks = []
        async for chunk in client.stream_chat(messages=messages, tools=tools):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Response"

    @pytest.mark.asyncio
    @patch("agents.base.clients.openai.LFAgentClientOpenAI._wrap_with_instructor")
    @patch("agents.base.clients.openai.AsyncOpenAI")
    async def test_chat_structured_response(
        self, mock_openai_class, mock_wrap_instructor, client
    ):
        """Test structured output path with response model."""

        class StructuredResponse(BaseModel):
            name: str
            age: int

        structured = StructuredResponse(name="Ada", age=37)
        mock_instructor = AsyncMock()
        mock_instructor.chat.completions.create = AsyncMock(return_value=structured)
        mock_wrap_instructor.return_value = mock_instructor
        mock_openai_class.return_value = AsyncMock()

        client.set_response_model(StructuredResponse)
        messages = [LFChatCompletionUserMessageParam(role="user", content="Hello")]
        response = await client.chat(messages=messages)

        content = response.choices[0].message.content
        assert json.loads(content) == {"name": "Ada", "age": 37}

        assert mock_instructor.chat.completions.create.await_count == 1
        assert (
            mock_instructor.chat.completions.create.call_args.kwargs["response_model"]
            is StructuredResponse
        )

    @pytest.mark.asyncio
    @patch("agents.base.clients.openai.LFAgentClientOpenAI._wrap_with_instructor")
    @patch("agents.base.clients.openai.AsyncOpenAI")
    async def test_chat_structured_response_does_not_forward_user_tools(
        self, mock_openai_class, mock_wrap_instructor, client
    ):
        """Instructor manages tools for response_model; user tools must not be forwarded."""

        class StructuredResponse(BaseModel):
            ok: bool

        structured = StructuredResponse(ok=True)
        mock_instructor = AsyncMock()
        mock_instructor.chat.completions.create = AsyncMock(return_value=structured)
        mock_wrap_instructor.return_value = mock_instructor
        mock_openai_class.return_value = AsyncMock()

        client.set_response_model(StructuredResponse)
        messages = [LFChatCompletionUserMessageParam(role="user", content="Hello")]
        tools = [
            ToolDefinition(
                name="external_tool",
                description="Should not be passed to instructor create()",
                parameters={"type": "object", "properties": {}},
            )
        ]

        await client.chat(messages=messages, tools=tools)

        kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        assert kwargs["response_model"] is StructuredResponse
        assert "tools" not in kwargs

    @pytest.mark.asyncio
    @patch("agents.base.clients.openai.LFAgentClientOpenAI._wrap_with_instructor")
    @patch("agents.base.clients.openai.AsyncOpenAI")
    async def test_chat_structured_response_skips_prompt_tool_injection(
        self, mock_openai_class, mock_wrap_instructor, client
    ):
        """Prompt-based tool injection must not alter instructor-bound messages."""

        class StructuredResponse(BaseModel):
            ok: bool

        structured = StructuredResponse(ok=True)
        mock_instructor = AsyncMock()
        mock_instructor.chat.completions.create = AsyncMock(return_value=structured)
        mock_wrap_instructor.return_value = mock_instructor
        mock_openai_class.return_value = AsyncMock()

        client._model_config.tool_call_strategy = ToolCallStrategy.prompt_based
        client.set_response_model(StructuredResponse)

        original_system = "You are a helpful assistant."
        messages = [
            {"role": "system", "content": original_system},
            LFChatCompletionUserMessageParam(role="user", content="Hello"),
        ]
        tools = [
            ToolDefinition(
                name="external_tool",
                description="Should not be injected into system prompt",
                parameters={"type": "object", "properties": {}},
            )
        ]

        await client.chat(messages=messages, tools=tools)

        kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        sent_messages = kwargs["messages"]
        assert sent_messages[0]["content"] == original_system

    def test_wrap_with_instructor_invalid_mode_has_clear_error(self, model_config):
        model_config.instructor_mode = "foo"
        client = LFAgentClientOpenAI(model_config=model_config)

        with pytest.raises(ValueError, match="Invalid instructor_mode 'foo'"):
            client._wrap_with_instructor(MagicMock())
