"""
End-to-end tests for tool calling through the ChatCompletionsService.

These tests verify the full tool calling flow:
1. Request with tools is received
2. Model generates response with tool call
3. Tool call is detected and parsed
4. Response is formatted correctly (OpenAI-compatible)

Tests cover both streaming and non-streaming paths.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from routers.chat_completions.service import ChatCompletionsService
from routers.chat_completions.types import ChatCompletionRequest

# Simple tool definitions for testing
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform a calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    },
}

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"],
        },
    },
}


class TestToolCallingE2E:
    """End-to-end tests for tool calling."""

    @pytest.fixture
    def service(self):
        """Create a ChatCompletionsService instance."""
        return ChatCompletionsService()

    @pytest.fixture
    def mock_gguf_model(self):
        """Create a mock GGUF model."""
        from models.gguf_language_model import GGUFLanguageModel

        model = MagicMock(spec=GGUFLanguageModel)
        model.model_id = "test/model"
        model.context_manager = None  # Disable context management for simplicity
        return model

    def create_request(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> ChatCompletionRequest:
        """Create a ChatCompletionRequest for testing."""
        return ChatCompletionRequest(
            model="test/model",
            messages=messages,
            tools=tools,
            stream=stream,
            max_tokens=100,
        )

    @pytest.mark.asyncio
    async def test_non_streaming_tool_call_detected(self, service, mock_gguf_model):
        """Test that a tool call in model output is detected and returned correctly."""
        # Model generates a response with a tool call
        model_response = (
            "I'll calculate that for you. "
            '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
        )
        mock_gguf_model.generate = AsyncMock(return_value=model_response)

        messages = [{"role": "user", "content": "What is 2+2?"}]
        request = self.create_request(messages, tools=[CALCULATOR_TOOL])

        with patch.object(service, "load_language", return_value=mock_gguf_model):
            response = await service.chat_completions(request)

        # Verify response structure
        assert response["object"] == "chat.completion"
        assert len(response["choices"]) == 1

        choice = response["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["role"] == "assistant"
        assert (
            choice["message"]["content"] is None
        )  # Content is null when tool calls present

        # Verify tool call structure
        tool_calls = choice["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "calculator"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {
            "expression": "2+2"
        }
        assert tool_calls[0]["id"].startswith("call_")

    @pytest.mark.asyncio
    async def test_non_streaming_multiple_tool_calls(self, service, mock_gguf_model):
        """Test that multiple tool calls are detected and returned correctly."""
        # Model generates a response with multiple tool calls
        model_response = (
            "I'll get that information for you. "
            '<tool_call>{"name": "get_weather", "arguments": {"location": "New York, NY"}}</tool_call> '
            '<tool_call>{"name": "get_weather", "arguments": {"location": "Los Angeles, CA"}}</tool_call>'
        )
        mock_gguf_model.generate = AsyncMock(return_value=model_response)

        messages = [{"role": "user", "content": "What's the weather in NY and LA?"}]
        request = self.create_request(messages, tools=[GET_WEATHER_TOOL])

        with patch.object(service, "load_language", return_value=mock_gguf_model):
            response = await service.chat_completions(request)

        # Verify multiple tool calls
        tool_calls = response["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[1]["function"]["name"] == "get_weather"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {
            "location": "New York, NY"
        }
        assert json.loads(tool_calls[1]["function"]["arguments"]) == {
            "location": "Los Angeles, CA"
        }

    @pytest.mark.asyncio
    async def test_non_streaming_no_tool_call_when_not_needed(
        self, service, mock_gguf_model
    ):
        """Test that regular responses without tool calls work correctly."""
        # Model generates a normal response without tool call
        model_response = "2+2 equals 4."
        mock_gguf_model.generate = AsyncMock(return_value=model_response)

        messages = [{"role": "user", "content": "What is 2+2?"}]
        request = self.create_request(messages, tools=[CALCULATOR_TOOL])

        with patch.object(service, "load_language", return_value=mock_gguf_model):
            response = await service.chat_completions(request)

        # Verify normal response (no tool calls)
        choice = response["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == "2+2 equals 4."
        assert "tool_calls" not in choice["message"]

    @pytest.mark.asyncio
    async def test_non_streaming_no_tools_provided(self, service, mock_gguf_model):
        """Test that tool call syntax in response is ignored when no tools provided."""
        # Model generates a response with tool call syntax, but no tools were provided
        model_response = (
            '<tool_call>{"name": "calculator", "arguments": {}}</tool_call>'
        )
        mock_gguf_model.generate = AsyncMock(return_value=model_response)

        messages = [{"role": "user", "content": "Hello"}]
        request = self.create_request(messages, tools=None)  # No tools

        with patch.object(service, "load_language", return_value=mock_gguf_model):
            response = await service.chat_completions(request)

        # Tool call should be ignored and returned as regular content
        choice = response["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == model_response

    @pytest.mark.asyncio
    async def test_streaming_tool_call_emitted(self, service, mock_gguf_model):
        """Test that streaming tool calls emit proper SSE events."""

        # Simulate streaming tokens that form a tool call
        async def generate_tokens(
            messages,
            max_tokens=None,
            temperature=0.7,
            top_p=1.0,
            stop=None,
            thinking_budget=None,
            tools=None,
            tool_choice=None,
        ):
            tokens = [
                "I'll ",
                "calculate ",
                "that. ",
                "<tool_call>",
                '{"name": "calculator", ',
                '"arguments": {"expression": "2+2"}}',
                "</tool_call>",
            ]
            for token in tokens:
                yield token

        mock_gguf_model.generate_stream = generate_tokens

        messages = [{"role": "user", "content": "What is 2+2?"}]
        request = self.create_request(messages, tools=[CALCULATOR_TOOL], stream=True)

        with patch.object(service, "load_language", return_value=mock_gguf_model):
            response = await service.chat_completions(request)

        # Response should be a StreamingResponse
        from fastapi.responses import StreamingResponse

        assert isinstance(response, StreamingResponse)

        # Collect all SSE events
        events = []
        async for chunk in response.body_iterator:
            if chunk.startswith(b"data: ") and not chunk.startswith(b"data: [DONE]"):
                data = chunk.decode().replace("data: ", "").strip()
                if data:
                    events.append(json.loads(data))

        # Verify we got chunks
        assert len(events) > 0

        # Find the tool call chunk (has tool_calls in delta)
        tool_call_chunks = [
            e for e in events if e["choices"][0]["delta"].get("tool_calls")
        ]
        assert len(tool_call_chunks) > 0, "Should have tool call chunks"

        # First tool call chunk should have the name
        first_tool_chunk = tool_call_chunks[0]
        tool_call = first_tool_chunk["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "calculator"
        assert tool_call["id"].startswith("call_")

        # Verify finish_reason is tool_calls
        final_events = [
            e for e in events if e["choices"][0].get("finish_reason") == "tool_calls"
        ]
        assert len(final_events) == 1, (
            "Should have one chunk with finish_reason=tool_calls"
        )

    @pytest.mark.asyncio
    async def test_tool_call_with_complex_arguments(self, service, mock_gguf_model):
        """Test tool call with nested/complex arguments."""
        complex_tool = {
            "type": "function",
            "function": {
                "name": "create_event",
                "description": "Create a calendar event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "attendees": {"type": "array", "items": {"type": "string"}},
                        "metadata": {"type": "object"},
                    },
                },
            },
        }

        complex_args = {
            "title": "Team Meeting",
            "attendees": ["alice@example.com", "bob@example.com"],
            "metadata": {"priority": "high", "recurring": True},
        }

        model_response = f'<tool_call>{{"name": "create_event", "arguments": {json.dumps(complex_args)}}}</tool_call>'
        mock_gguf_model.generate = AsyncMock(return_value=model_response)

        messages = [{"role": "user", "content": "Schedule a team meeting"}]
        request = self.create_request(messages, tools=[complex_tool])

        with patch.object(service, "load_language", return_value=mock_gguf_model):
            response = await service.chat_completions(request)

        # Verify complex arguments are preserved
        tool_calls = response["choices"][0]["message"]["tool_calls"]
        parsed_args = json.loads(tool_calls[0]["function"]["arguments"])
        assert parsed_args == complex_args

    @pytest.mark.asyncio
    async def test_tool_call_with_empty_arguments(self, service, mock_gguf_model):
        """Test tool call with no arguments."""
        no_args_tool = {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current time",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        model_response = '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>'
        mock_gguf_model.generate = AsyncMock(return_value=model_response)

        messages = [{"role": "user", "content": "What time is it?"}]
        request = self.create_request(messages, tools=[no_args_tool])

        with patch.object(service, "load_language", return_value=mock_gguf_model):
            response = await service.chat_completions(request)

        tool_calls = response["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_time"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {}


class TestToolCallingOpenAICompatibility:
    """Test OpenAI SDK compatibility for tool calling responses."""

    @pytest.fixture
    def service(self):
        """Create a ChatCompletionsService instance."""
        return ChatCompletionsService()

    @pytest.fixture
    def mock_gguf_model(self):
        """Create a mock GGUF model."""
        from models.gguf_language_model import GGUFLanguageModel

        model = MagicMock(spec=GGUFLanguageModel)
        model.model_id = "test/model"
        model.context_manager = None
        return model

    @pytest.mark.asyncio
    async def test_response_can_be_parsed_by_openai_types(
        self, service, mock_gguf_model
    ):
        """Test that responses can be parsed by OpenAI SDK types."""
        from openai.types.chat import ChatCompletion

        model_response = (
            '<tool_call>{"name": "test", "arguments": {"key": "value"}}</tool_call>'
        )
        mock_gguf_model.generate = AsyncMock(return_value=model_response)

        request = ChatCompletionRequest(
            model="test/model",
            messages=[{"role": "user", "content": "Test"}],
            tools=[CALCULATOR_TOOL],
            max_tokens=100,
        )

        with patch.object(service, "load_language", return_value=mock_gguf_model):
            response = await service.chat_completions(request)

        # This should not raise - validates OpenAI SDK compatibility
        parsed = ChatCompletion.model_validate(response)
        assert parsed.choices[0].finish_reason == "tool_calls"
        assert len(parsed.choices[0].message.tool_calls) == 1
        assert parsed.choices[0].message.tool_calls[0].function.name == "test"

    @pytest.mark.asyncio
    async def test_tool_call_id_format(self, service, mock_gguf_model):
        """Test that tool call IDs match OpenAI's format."""
        model_response = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        mock_gguf_model.generate = AsyncMock(return_value=model_response)

        request = ChatCompletionRequest(
            model="test/model",
            messages=[{"role": "user", "content": "Test"}],
            tools=[CALCULATOR_TOOL],
            max_tokens=100,
        )

        with patch.object(service, "load_language", return_value=mock_gguf_model):
            response = await service.chat_completions(request)

        tool_call_id = response["choices"][0]["message"]["tool_calls"][0]["id"]
        # OpenAI format: call_<uuid>
        assert tool_call_id.startswith("call_")
        # UUID portion should be valid
        uuid_part = tool_call_id.replace("call_", "")
        assert len(uuid_part) > 0

    @pytest.mark.asyncio
    async def test_arguments_are_json_string(self, service, mock_gguf_model):
        """Test that arguments are returned as JSON string, not parsed object."""
        model_response = (
            '<tool_call>{"name": "test", "arguments": {"foo": "bar"}}</tool_call>'
        )
        mock_gguf_model.generate = AsyncMock(return_value=model_response)

        request = ChatCompletionRequest(
            model="test/model",
            messages=[{"role": "user", "content": "Test"}],
            tools=[CALCULATOR_TOOL],
            max_tokens=100,
        )

        with patch.object(service, "load_language", return_value=mock_gguf_model):
            response = await service.chat_completions(request)

        arguments = response["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
        # Arguments should be a JSON string, not a dict
        assert isinstance(arguments, str)
        # And parseable as JSON
        parsed = json.loads(arguments)
        assert parsed == {"foo": "bar"}
