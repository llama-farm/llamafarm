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
from fastapi import HTTPException

from models.gguf_language_model import GGUFLanguageModel
from routers.chat_completions.service import ChatCompletionsService
from routers.chat_completions.types import ChatCompletionRequest
from utils.context_manager import ContextBudget, ContextUsage, TruncationStrategy

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

        with patch("server.load_language", return_value=mock_gguf_model) as mock_loader:
            service.load_language = mock_loader
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

        with patch("server.load_language", return_value=mock_gguf_model) as mock_loader:
            service.load_language = mock_loader
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

        with patch("server.load_language", return_value=mock_gguf_model) as mock_loader:
            service.load_language = mock_loader
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

        with patch("server.load_language", return_value=mock_gguf_model) as mock_loader:
            service.load_language = mock_loader
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

        with patch("server.load_language", return_value=mock_gguf_model) as mock_loader:
            service.load_language = mock_loader
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

        with patch("server.load_language", return_value=mock_gguf_model) as mock_loader:
            service.load_language = mock_loader
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

        with patch("server.load_language", return_value=mock_gguf_model) as mock_loader:
            service.load_language = mock_loader
            response = await service.chat_completions(request)

        tool_calls = response["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_time"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {}

    @pytest.mark.asyncio
    async def test_http_exception_is_not_wrapped_as_500(self, service, mock_gguf_model):
        """HTTPException should propagate unchanged through the service layer."""
        context_error = HTTPException(
            status_code=400,
            detail={"error": "context_length_exceeded", "context_usage": {"total_context": 512}},
        )
        mock_gguf_model.generate = AsyncMock(side_effect=context_error)

        request = self.create_request(
            messages=[{"role": "user", "content": "Hello"}],
            tools=None,
            stream=False,
        )

        with (
            patch.object(service, "load_language", return_value=mock_gguf_model),
            pytest.raises(HTTPException) as exc_info,
        ):
            await service.chat_completions(request)

        assert exc_info.value.status_code == 400
        assert isinstance(exc_info.value.detail, dict)
        assert exc_info.value.detail["error"] == "context_length_exceeded"

    @pytest.mark.asyncio
    async def test_effective_max_tokens_uses_real_remaining_context(self, service):
        """Generation cap should use real remaining space, not reserved cap."""

        class FakeContextManager:
            def __init__(self):
                self.budget = ContextBudget(
                    total_context=2048,
                    max_prompt_tokens=1536,
                    reserved_completion=512,
                    safety_margin=102,
                )

            def validate_messages(self, _messages):
                # Simulate legacy reserved-completion-limited availability.
                return ContextUsage(
                    total_context=2048,
                    prompt_tokens=100,
                    available_for_completion=512,
                )

            def needs_truncation(self, _messages):
                return False

        model = GGUFLanguageModel("test/model", "cpu")
        model._context_manager = FakeContextManager()
        model._token_counter = None
        model.generate = AsyncMock(return_value="ok")

        request = self.create_request(
            messages=[{"role": "user", "content": "Hello"}],
            tools=None,
            stream=False,
        )
        request.max_tokens = 1200

        with patch.object(service, "load_language", return_value=model):
            await service.chat_completions(request)

        assert model.generate.await_count == 1
        assert model.generate.await_args.kwargs["max_tokens"] == 1200

    @pytest.mark.asyncio
    async def test_native_rendered_prompt_is_counted_for_context_budget(self, service):
        """Native-rendered tool prompts should be validated by rendered size."""

        class FakeContextManager:
            def __init__(self):
                self.budget = ContextBudget(
                    total_context=1000,
                    max_prompt_tokens=700,
                    reserved_completion=200,
                    safety_margin=50,
                )

            def validate_messages(self, _messages):
                raise AssertionError("native path should bypass message validation")

            def needs_truncation(self, _messages):
                return False

        model = GGUFLanguageModel("test/model", "cpu")
        model._context_manager = FakeContextManager()
        model._token_counter = MagicMock()
        model._token_counter.count_tokens.return_value = 900
        model.prepare_messages_for_context_validation = MagicMock(
            return_value=(
                [{"role": "user", "content": "short"}],
                False,
                "x" * 5000,
            )
        )
        model.generate = AsyncMock(return_value="should not run")

        request = self.create_request(
            messages=[{"role": "user", "content": "short"}],
            tools=[CALCULATOR_TOOL],
            stream=False,
        )

        with (
            patch.object(service, "load_language", return_value=model),
            pytest.raises(HTTPException) as exc_info,
        ):
            await service.chat_completions(request)

        assert exc_info.value.status_code == 400
        assert "Rendered prompt" in exc_info.value.detail["message"]
        assert model.generate.await_count == 0

    @pytest.mark.asyncio
    async def test_native_rendered_prompt_without_token_counter_returns_400(self, service):
        """Missing token counter on native path should return controlled HTTP error."""

        class FakeContextManager:
            def __init__(self):
                self.budget = ContextBudget(
                    total_context=1000,
                    max_prompt_tokens=700,
                    reserved_completion=200,
                    safety_margin=50,
                )

            def validate_messages(self, _messages):
                raise AssertionError("should fail before message validation fallback")

            def needs_truncation(self, _messages):
                return False

        model = GGUFLanguageModel("test/model", "cpu")
        model._context_manager = FakeContextManager()
        model._token_counter = None
        model.prepare_messages_for_context_validation = MagicMock(
            return_value=(
                [{"role": "user", "content": "short"}],
                False,
                "rendered native prompt",
            )
        )
        model.generate = AsyncMock(return_value="should not run")

        request = self.create_request(
            messages=[{"role": "user", "content": "short"}],
            tools=[CALCULATOR_TOOL],
            stream=False,
        )

        with (
            patch.object(service, "load_language", return_value=model),
            pytest.raises(HTTPException) as exc_info,
        ):
            await service.chat_completions(request)

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "context_validation_unavailable"
        assert "token counting is unavailable" in exc_info.value.detail["message"]
        assert model.generate.await_count == 0

    @pytest.mark.asyncio
    async def test_injected_tools_force_keep_system_over_sliding_window(self, service):
        """Injected tools should force system-preserving truncation strategy."""

        class FakeContextManager:
            def __init__(self):
                self.budget = ContextBudget(
                    total_context=1000,
                    max_prompt_tokens=700,
                    reserved_completion=200,
                    safety_margin=50,
                )
                self.last_strategy = None

            def validate_messages(self, _messages):
                return ContextUsage(
                    total_context=1000,
                    prompt_tokens=900,
                    available_for_completion=50,
                )

            def needs_truncation(self, _messages):
                return False

            def truncate_if_needed(self, messages, strategy):
                self.last_strategy = strategy
                return (
                    messages,
                    ContextUsage(
                        total_context=1000,
                        prompt_tokens=600,
                        available_for_completion=200,
                        truncated=True,
                        truncated_messages=1,
                        strategy_used=strategy.value,
                    ),
                )

        model = GGUFLanguageModel("test/model", "cpu")
        model._context_manager = FakeContextManager()
        model._token_counter = None
        model.prepare_messages_for_context_validation = MagicMock(
            return_value=(
                [
                    {"role": "system", "content": "<tools>calculator</tools>"},
                    {"role": "user", "content": "short"},
                ],
                True,
                None,
            )
        )
        model.generate = AsyncMock(return_value="ok")

        request = self.create_request(
            messages=[{"role": "user", "content": "short"}],
            tools=[CALCULATOR_TOOL],
            stream=False,
        )
        request.truncation_strategy = "sliding_window"

        with patch.object(service, "load_language", return_value=model):
            await service.chat_completions(request)

        assert model._context_manager.last_strategy == TruncationStrategy.KEEP_SYSTEM_SLIDING
        assert model.generate.await_args.kwargs["tools"] is None
        assert model.generate.await_args.kwargs["messages"][0]["role"] == "system"


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

        with patch("server.load_language", return_value=mock_gguf_model) as mock_loader:
            service.load_language = mock_loader
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
