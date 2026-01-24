"""
Tests for incremental streaming tool call support.

These tests verify that tool calls are streamed incrementally:
1. Tool name is emitted as soon as it's detected
2. Arguments are streamed character-by-character
3. OpenAI client libraries can consume the stream correctly
"""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from routers.chat_completions.service import ChatCompletionsService, ToolCallStreamState


class TestToolCallStreamState:
    """Tests for the ToolCallStreamState enum."""

    def test_state_values(self):
        """Verify all expected states exist."""
        assert ToolCallStreamState.NORMAL.value == "normal"
        assert ToolCallStreamState.BUFFERING_START.value == "buffering_start"
        assert ToolCallStreamState.STREAMING_ARGS.value == "streaming_args"


class TestIncrementalToolCallStreaming:
    """Integration tests for incremental tool call streaming."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that generates tool calls token by token."""
        model = MagicMock()
        model.model_id = "test-model"
        return model

    @pytest.fixture
    def service(self):
        """Create a chat completions service."""
        service = ChatCompletionsService()
        return service

    def create_token_stream(self, tokens: list[str]):
        """Create an async generator from a list of tokens."""

        async def token_generator():
            for token in tokens:
                yield token

        return token_generator()

    @pytest.mark.asyncio
    async def test_tool_name_emitted_before_arguments_complete(self):
        """Verify the tool name is emitted before the full arguments are received."""
        # Simulate tokens arriving one at a time for a tool call
        tokens = [
            "I'll",
            " help",
            " you",
            " with",
            " that",
            ".",
            " ",
            "<tool",
            "_call",
            ">",
            '{"name"',
            ":",
            ' "',
            "get",
            "_weather",
            '"',  # At this point, name should be emitted
            ",",
            ' "arguments"',
            ":",
            " {",
            '"location"',
            ":",
            ' "',
            "New",
            " York",
            '"}',
            "}",
            "</tool",
            "_call",
            ">",
        ]

        chunks_received = []

        # Parse the SSE chunks as they would be received
        # This simulates what the streaming logic produces
        from utils.tool_calling import (
            detect_probable_tool_call,
            extract_arguments_progress,
            extract_tool_name_from_partial,
            is_tool_call_complete,
        )

        accumulated = ""
        tool_state = ToolCallStreamState.NORMAL
        tool_name_found = None
        args_chunks = []

        for token in tokens:
            accumulated += token

            if tool_state == ToolCallStreamState.NORMAL:
                if detect_probable_tool_call(accumulated):
                    tool_state = ToolCallStreamState.BUFFERING_START

            elif tool_state == ToolCallStreamState.BUFFERING_START:
                name = extract_tool_name_from_partial(accumulated)
                if name:
                    tool_name_found = name
                    tool_state = ToolCallStreamState.STREAMING_ARGS
                    chunks_received.append({"type": "name", "value": name})

            elif tool_state == ToolCallStreamState.STREAMING_ARGS:
                progress = extract_arguments_progress(accumulated)
                if progress:
                    _, args = progress
                    if args and args not in args_chunks:
                        args_chunks.append(args)
                        chunks_received.append({"type": "args", "value": args})

                if is_tool_call_complete(accumulated):
                    chunks_received.append({"type": "complete"})
                    break

        # Verify: name was received before arguments were complete
        name_chunk_idx = next(
            (i for i, c in enumerate(chunks_received) if c["type"] == "name"), None
        )
        complete_chunk_idx = next(
            (i for i, c in enumerate(chunks_received) if c["type"] == "complete"), None
        )

        assert name_chunk_idx is not None, "Tool name should have been emitted"
        assert complete_chunk_idx is not None, "Tool call should have completed"
        assert name_chunk_idx < complete_chunk_idx, (
            "Name should be emitted before completion"
        )
        assert tool_name_found == "get_weather"

    @pytest.mark.asyncio
    async def test_arguments_streamed_incrementally(self):
        """Verify arguments are streamed in multiple chunks, not all at once."""
        from utils.tool_calling import (
            detect_probable_tool_call,
            extract_arguments_progress,
        )

        # Build a tool call with substantial arguments
        tool_call_content = '<tool_call>{"name": "search", "arguments": {"query": "artificial intelligence", "limit": 10, "sort": "relevance"}}</tool_call>'

        # Simulate receiving one character at a time
        accumulated = ""
        args_snapshots = []

        for char in tool_call_content:
            accumulated += char

            if detect_probable_tool_call(accumulated):
                progress = extract_arguments_progress(accumulated)
                if progress:
                    _, args = progress
                    # Only record when args changes
                    if not args_snapshots or args != args_snapshots[-1]:
                        args_snapshots.append(args)

        # Verify: arguments were captured incrementally (multiple snapshots)
        assert len(args_snapshots) > 1, (
            f"Arguments should be captured incrementally, got {len(args_snapshots)} snapshots"
        )

        # Verify the snapshots grow over time
        for i in range(1, len(args_snapshots)):
            assert len(args_snapshots[i]) >= len(args_snapshots[i - 1]), (
                "Arguments should grow monotonically"
            )

    @pytest.mark.asyncio
    async def test_incomplete_tool_call_handled_gracefully(self):
        """Verify incomplete tool calls are handled gracefully at stream end."""
        from utils.tool_calling import (
            detect_probable_tool_call,
            is_tool_call_complete,
        )

        # Incomplete tool call (stream ended before </tool_call>)
        incomplete_content = '<tool_call>{"name": "test", "arguments": {"partial": "da'

        assert detect_probable_tool_call(incomplete_content) is True
        assert is_tool_call_complete(incomplete_content) is False

        # In this case, the buffered content should be emitted as regular content

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_detection(self):
        """Verify multiple tool calls can be detected in sequence."""
        from utils.tool_calling import detect_tool_call_in_content

        content = """
        <tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>
        <tool_call>{"name": "get_time", "arguments": {"timezone": "EST"}}</tool_call>
        """

        result = detect_tool_call_in_content(content)
        assert result is not None
        assert len(result) == 2
        assert result[0][0] == "get_weather"
        assert result[1][0] == "get_time"

    @pytest.mark.asyncio
    async def test_openai_compatible_chunk_structure(self):
        """Verify the streaming chunks match OpenAI's expected format."""
        from openai.types.chat.chat_completion_chunk import (
            ChatCompletionChunk,
            ChoiceDelta,
            ChoiceDeltaToolCall,
            ChoiceDeltaToolCallFunction,
        )
        from openai.types.chat.chat_completion_chunk import (
            Choice as ChoiceChunk,
        )

        # Initial chunk with name
        initial_chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=int(datetime.now().timestamp()),
            model="test-model",
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_test123",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="get_weather",
                                    arguments="",
                                ),
                            )
                        ]
                    ),
                    finish_reason=None,
                )
            ],
        )

        # Verify it serializes correctly
        json_str = initial_chunk.model_dump_json(exclude_none=True)
        parsed = json.loads(json_str)

        assert parsed["object"] == "chat.completion.chunk"
        assert len(parsed["choices"]) == 1
        assert (
            parsed["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            == "get_weather"
        )
        assert (
            parsed["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
            == ""
        )

        # Incremental args chunk (no id, no type, just index and function.arguments)
        args_chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=int(datetime.now().timestamp()),
            model="test-model",
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(
                                    arguments='{"loc',
                                ),
                            )
                        ]
                    ),
                    finish_reason=None,
                )
            ],
        )

        args_json = args_chunk.model_dump_json(exclude_none=True)
        args_parsed = json.loads(args_json)

        assert (
            args_parsed["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
            == '{"loc'
        )

    @pytest.mark.asyncio
    async def test_state_machine_transitions(self):
        """Test that state machine transitions correctly."""
        # NORMAL -> BUFFERING_START when <tool_call> detected
        # BUFFERING_START -> STREAMING_ARGS when name found
        # STREAMING_ARGS -> (done) when </tool_call> found

        from utils.tool_calling import (
            detect_probable_tool_call,
            extract_tool_name_from_partial,
            is_tool_call_complete,
        )

        state = ToolCallStreamState.NORMAL
        accumulated = ""

        # Phase 1: Normal content
        for token in ["Hello", ", ", "I'll"]:
            accumulated += token
            assert state == ToolCallStreamState.NORMAL
            if detect_probable_tool_call(accumulated):
                state = ToolCallStreamState.BUFFERING_START

        assert state == ToolCallStreamState.NORMAL

        # Phase 2: Tool call starts
        accumulated += " <tool_call>"
        if detect_probable_tool_call(accumulated):
            state = ToolCallStreamState.BUFFERING_START

        assert state == ToolCallStreamState.BUFFERING_START

        # Phase 3: Name becomes available
        accumulated += '{"name": "test"'
        if state == ToolCallStreamState.BUFFERING_START:
            name = extract_tool_name_from_partial(accumulated)
            if name:
                state = ToolCallStreamState.STREAMING_ARGS

        assert state == ToolCallStreamState.STREAMING_ARGS

        # Phase 4: Tool call completes
        accumulated += ', "arguments": {}}</tool_call>'
        assert is_tool_call_complete(accumulated) is True

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_streaming_state_machine(self):
        """Test that state machine correctly handles multiple sequential tool calls."""
        from utils.tool_calling import (
            detect_probable_tool_call,
            detect_tool_call_in_content,
            extract_tool_name_from_partial,
            is_tool_call_complete,
            strip_tool_call_from_content,
        )

        # Simulate streaming two tool calls
        full_content = (
            "I will make two calls. "
            '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call> '
            '<tool_call>{"name": "get_time", "arguments": {"timezone": "EST"}}</tool_call>'
        )

        # Tokenize by character to simulate streaming
        state = ToolCallStreamState.NORMAL
        accumulated = ""
        tool_names_found = []
        tool_call_index = 0

        for char in full_content:
            accumulated += char

            if state == ToolCallStreamState.NORMAL:
                if detect_probable_tool_call(accumulated):
                    state = ToolCallStreamState.BUFFERING_START

            elif state == ToolCallStreamState.BUFFERING_START:
                name = extract_tool_name_from_partial(accumulated)
                if name:
                    tool_names_found.append(name)
                    state = ToolCallStreamState.STREAMING_ARGS

            elif state == ToolCallStreamState.STREAMING_ARGS and is_tool_call_complete(
                accumulated
            ):
                # Process completed tool call
                tool_calls = detect_tool_call_in_content(accumulated)
                assert tool_calls is not None
                assert len(tool_calls) >= 1

                # Reset state machine for next tool call
                accumulated = strip_tool_call_from_content(accumulated)
                state = ToolCallStreamState.NORMAL
                tool_call_index += 1

                # Check if there's already another tool call starting
                if detect_probable_tool_call(accumulated):
                    state = ToolCallStreamState.BUFFERING_START

        # Verify both tool calls were detected
        assert len(tool_names_found) == 2, (
            f"Expected 2 tool calls, got {tool_names_found}"
        )
        assert tool_names_found[0] == "get_weather"
        assert tool_names_found[1] == "get_time"
        assert tool_call_index == 2
