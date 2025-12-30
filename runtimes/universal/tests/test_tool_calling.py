"""
Tests for tool calling utilities.
"""

import json

from utils.tool_calling import (
    detect_probable_tool_call,
    detect_tool_call_in_content,
    inject_tools_into_messages,
    strip_tool_call_from_content,
)


class TestInjectToolsIntoMessages:
    """Tests for inject_tools_into_messages function."""

    def test_inject_tools_with_existing_system_message(self):
        """Test that tools are appended to existing system message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform arithmetic",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    },
                },
            }
        ]

        result = inject_tools_into_messages(messages, tools)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "You are a helpful assistant." in result[0]["content"]
        assert "<tools>" in result[0]["content"]
        assert "calculator" in result[0]["content"]
        assert "<tool_call>" in result[0]["content"]

    def test_inject_tools_creates_system_message_if_missing(self):
        """Test that system message is created if not present."""
        messages = [{"role": "user", "content": "What is the weather?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = inject_tools_into_messages(messages, tools)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "get_weather" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_inject_tools_empty_tools_returns_original(self):
        """Test that empty tools list returns original messages."""
        messages = [{"role": "user", "content": "Hello"}]
        result = inject_tools_into_messages(messages, [])
        assert result == messages

    def test_inject_tools_none_returns_original(self):
        """Test that None tools returns original messages."""
        messages = [{"role": "user", "content": "Hello"}]
        result = inject_tools_into_messages(messages, None)
        assert result == messages

    def test_inject_tools_does_not_modify_original(self):
        """Test that original messages list is not modified."""
        messages = [
            {"role": "system", "content": "Original content"},
            {"role": "user", "content": "Hello"},
        ]
        tools = [{"type": "function", "function": {"name": "test", "description": "Test"}}]

        original_content = messages[0]["content"]
        _ = inject_tools_into_messages(messages, tools)

        assert messages[0]["content"] == original_content


class TestDetectToolCallInContent:
    """Tests for detect_tool_call_in_content function."""

    def test_detect_single_tool_call(self):
        """Test detection of a single tool call."""
        content = 'I will call the calculator. <tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'

        result = detect_tool_call_in_content(content)

        assert result is not None
        assert len(result) == 1
        assert result[0][0] == "calculator"
        assert json.loads(result[0][1]) == {"expression": "2+2"}

    def test_detect_multiple_tool_calls(self):
        """Test detection of multiple tool calls."""
        content = """
        I need to make two calls.
        <tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>
        <tool_call>{"name": "get_time", "arguments": {"timezone": "EST"}}</tool_call>
        """

        result = detect_tool_call_in_content(content)

        assert result is not None
        assert len(result) == 2
        assert result[0][0] == "get_weather"
        assert result[1][0] == "get_time"

    def test_no_tool_call_returns_none(self):
        """Test that content without tool calls returns None."""
        content = "Just a regular response without any tool calls."
        result = detect_tool_call_in_content(content)
        assert result is None

    def test_empty_content_returns_none(self):
        """Test that empty content returns None."""
        assert detect_tool_call_in_content("") is None
        assert detect_tool_call_in_content(None) is None

    def test_malformed_json_is_skipped(self):
        """Test that malformed JSON tool calls are skipped."""
        content = '<tool_call>{"name": "test", invalid json}</tool_call>'
        result = detect_tool_call_in_content(content)
        assert result is None

    def test_tool_call_without_name_is_skipped(self):
        """Test that tool calls without name are skipped."""
        content = '<tool_call>{"arguments": {"foo": "bar"}}</tool_call>'
        result = detect_tool_call_in_content(content)
        assert result is None

    def test_tool_call_with_empty_arguments(self):
        """Test tool call with no arguments."""
        content = '<tool_call>{"name": "no_args_tool", "arguments": {}}</tool_call>'
        result = detect_tool_call_in_content(content)
        assert result is not None
        assert result[0][0] == "no_args_tool"
        assert result[0][1] == "{}"


class TestDetectProbableToolCall:
    """Tests for detect_probable_tool_call function."""

    def test_detects_opening_tag(self):
        """Test detection of opening tool_call tag."""
        assert detect_probable_tool_call("<tool_call>") is True
        assert detect_probable_tool_call("Some text <tool_call>") is True
        assert detect_probable_tool_call('<tool_call>{"name": "te') is True

    def test_no_tag_returns_false(self):
        """Test that content without tag returns False."""
        assert detect_probable_tool_call("Regular text") is False
        assert detect_probable_tool_call("") is False

    def test_complete_tag_returns_true(self):
        """Test that complete tool call still returns True."""
        content = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        assert detect_probable_tool_call(content) is True


class TestStripToolCallFromContent:
    """Tests for strip_tool_call_from_content function."""

    def test_strips_tool_call(self):
        """Test stripping tool call from content."""
        content = 'Here is the result: <tool_call>{"name": "test", "arguments": {}}</tool_call> Done.'
        result = strip_tool_call_from_content(content)
        assert result == "Here is the result:  Done."

    def test_strips_multiple_tool_calls(self):
        """Test stripping multiple tool calls."""
        content = '<tool_call>{"name": "a"}</tool_call> and <tool_call>{"name": "b"}</tool_call>'
        result = strip_tool_call_from_content(content)
        assert result == "and"

    def test_content_without_tool_call_unchanged(self):
        """Test that content without tool calls is unchanged."""
        content = "Regular content here"
        result = strip_tool_call_from_content(content)
        assert result == content
