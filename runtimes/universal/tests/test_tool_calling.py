"""
Tests for tool calling utilities.
"""

import json

from utils.tool_calling import (
    detect_probable_tool_call,
    detect_tool_call_in_content,
    extract_arguments_progress,
    extract_tool_name_from_partial,
    get_tool_call_content_after_tag,
    inject_tools_into_messages,
    is_tool_call_complete,
    parse_tool_choice,
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
        tools = [
            {"type": "function", "function": {"name": "test", "description": "Test"}}
        ]

        original_content = messages[0]["content"]
        _ = inject_tools_into_messages(messages, tools)

        assert messages[0]["content"] == original_content

    def test_inject_tools_with_tool_choice_none(self):
        """Test that tool_choice='none' skips tool injection."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        tools = [
            {
                "type": "function",
                "function": {"name": "calculator", "description": "Calculate"},
            }
        ]

        result = inject_tools_into_messages(messages, tools, tool_choice="none")

        # Messages should be unchanged (no tools injected)
        assert len(result) == 2
        assert result[0]["content"] == "You are a helpful assistant."
        assert "<tools>" not in result[0]["content"]

    def test_inject_tools_with_tool_choice_auto(self):
        """Test that tool_choice='auto' uses 'may call' wording."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "test", "description": "Test"},
            }
        ]

        result = inject_tools_into_messages(messages, tools, tool_choice="auto")

        assert "You may call" in result[0]["content"]
        assert "<tools>" in result[0]["content"]

    def test_inject_tools_with_tool_choice_required(self):
        """Test that tool_choice='required' uses 'MUST call' wording."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "test", "description": "Test"},
            }
        ]

        result = inject_tools_into_messages(messages, tools, tool_choice="required")

        assert "You MUST call" in result[0]["content"]
        assert "<tools>" in result[0]["content"]

    def test_inject_tools_with_specific_function(self):
        """Test that specific function tool_choice filters tools and uses specific wording."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather"},
            },
            {
                "type": "function",
                "function": {"name": "get_time", "description": "Get time"},
            },
        ]
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        result = inject_tools_into_messages(messages, tools, tool_choice=tool_choice)

        # Should only include get_weather tool
        assert "get_weather" in result[0]["content"]
        assert "get_time" not in result[0]["content"]
        # Should use specific function wording
        assert 'MUST call the function "get_weather"' in result[0]["content"]


class TestParseToolChoice:
    """Tests for parse_tool_choice function."""

    def test_parse_none_returns_auto(self):
        """Test that None returns 'auto' mode."""
        mode, func = parse_tool_choice(None)
        assert mode == "auto"
        assert func is None

    def test_parse_auto_string(self):
        """Test that 'auto' string returns 'auto' mode."""
        mode, func = parse_tool_choice("auto")
        assert mode == "auto"
        assert func is None

    def test_parse_none_string(self):
        """Test that 'none' string returns 'none' mode."""
        mode, func = parse_tool_choice("none")
        assert mode == "none"
        assert func is None

    def test_parse_required_string(self):
        """Test that 'required' string returns 'required' mode."""
        mode, func = parse_tool_choice("required")
        assert mode == "required"
        assert func is None

    def test_parse_specific_function_dict(self):
        """Test that specific function dict returns 'specific' mode with function name."""
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}
        mode, func = parse_tool_choice(tool_choice)
        assert mode == "specific"
        assert func == "get_weather"

    def test_parse_malformed_dict_returns_auto(self):
        """Test that malformed dict falls back to 'auto' mode."""
        tool_choice = {"type": "other", "something": "else"}
        mode, func = parse_tool_choice(tool_choice)
        assert mode == "auto"
        assert func is None

    def test_parse_unknown_string_returns_auto(self):
        """Test that unknown string falls back to 'auto' mode."""
        mode, func = parse_tool_choice("unknown_value")
        assert mode == "auto"
        assert func is None


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


# =============================================================================
# Tests for incremental streaming utilities
# =============================================================================


class TestExtractToolNameFromPartial:
    """Tests for extract_tool_name_from_partial function."""

    def test_extracts_name_from_incomplete_json(self):
        """Test extracting tool name from incomplete tool call JSON."""
        content = '<tool_call>{"name": "get_weather"'
        assert extract_tool_name_from_partial(content) == "get_weather"

    def test_extracts_name_with_arguments_starting(self):
        """Test extracting name when arguments key is starting."""
        content = '<tool_call>{"name": "calculator", "arguments": {'
        assert extract_tool_name_from_partial(content) == "calculator"

    def test_extracts_name_no_spaces(self):
        """Test extracting name with no spaces in JSON."""
        content = '<tool_call>{"name":"get_time"'
        assert extract_tool_name_from_partial(content) == "get_time"

    def test_returns_none_when_name_incomplete(self):
        """Test returns None when name value is incomplete."""
        content = '<tool_call>{"name": "get_wea'
        assert extract_tool_name_from_partial(content) is None

    def test_returns_none_when_no_tool_call_tag(self):
        """Test returns None when no tool_call tag present."""
        content = '{"name": "test"}'
        assert extract_tool_name_from_partial(content) is None

    def test_returns_none_for_empty_content(self):
        """Test returns None for empty content."""
        assert extract_tool_name_from_partial("") is None
        assert extract_tool_name_from_partial(None) is None

    def test_extracts_name_with_prefix_content(self):
        """Test extracting name when there's content before tool call."""
        content = 'I will call this function: <tool_call>{"name": "search"'
        assert extract_tool_name_from_partial(content) == "search"

    def test_extracts_from_complete_tool_call(self):
        """Test extracting name from complete tool call."""
        content = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        assert extract_tool_name_from_partial(content) == "test"


class TestExtractArgumentsProgress:
    """Tests for extract_arguments_progress function."""

    def test_extracts_partial_arguments(self):
        """Test extracting partial arguments during streaming."""
        content = '<tool_call>{"name": "test", "arguments": {"loc'
        result = extract_arguments_progress(content)
        assert result is not None
        _, args = result
        assert '{"loc' in args

    def test_extracts_more_complete_arguments(self):
        """Test extracting more complete arguments."""
        content = '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"'
        result = extract_arguments_progress(content)
        assert result is not None
        _, args = result
        assert "NYC" in args

    def test_returns_none_before_arguments_key(self):
        """Test returns None when arguments key not yet present."""
        content = '<tool_call>{"name": "test"'
        result = extract_arguments_progress(content)
        assert result is None

    def test_returns_none_for_empty_content(self):
        """Test returns None for empty content."""
        assert extract_arguments_progress("") is None
        assert extract_arguments_progress(None) is None

    def test_returns_none_without_tool_call_tag(self):
        """Test returns None when no tool_call tag."""
        content = '{"name": "test", "arguments": {}'
        assert extract_arguments_progress(content) is None

    def test_handles_no_space_after_colon(self):
        """Test handles arguments: with no space."""
        content = '<tool_call>{"name":"test","arguments":{"key'
        result = extract_arguments_progress(content)
        assert result is not None
        _, args = result
        assert '{"key' in args


class TestIsToolCallComplete:
    """Tests for is_tool_call_complete function."""

    def test_returns_true_for_complete_call(self):
        """Test returns True when closing tag is present."""
        content = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        assert is_tool_call_complete(content) is True

    def test_returns_true_with_prefix_content(self):
        """Test returns True when there's content before tool call."""
        content = 'Here it is: <tool_call>{"name": "x"}</tool_call>'
        assert is_tool_call_complete(content) is True

    def test_returns_false_for_incomplete_call(self):
        """Test returns False when closing tag is missing."""
        content = '<tool_call>{"name": "test", "arguments": {'
        assert is_tool_call_complete(content) is False

    def test_returns_false_for_empty_content(self):
        """Test returns False for empty content."""
        assert is_tool_call_complete("") is False
        assert is_tool_call_complete(None) is False

    def test_returns_false_for_opening_tag_only(self):
        """Test returns False when only opening tag is present."""
        content = "<tool_call>"
        assert is_tool_call_complete(content) is False


class TestGetToolCallContentAfterTag:
    """Tests for get_tool_call_content_after_tag function."""

    def test_extracts_content_from_incomplete_call(self):
        """Test extracting content from incomplete tool call."""
        content = '<tool_call>{"name": "test"'
        result = get_tool_call_content_after_tag(content)
        assert result == '{"name": "test"'

    def test_extracts_content_from_complete_call(self):
        """Test extracting content from complete tool call."""
        content = '<tool_call>{"name": "x", "arguments": {}}</tool_call>'
        result = get_tool_call_content_after_tag(content)
        assert result == '{"name": "x", "arguments": {}}'

    def test_returns_none_without_tag(self):
        """Test returns None when no tool_call tag."""
        content = '{"name": "test"}'
        assert get_tool_call_content_after_tag(content) is None

    def test_returns_none_for_empty_content(self):
        """Test returns None for empty content."""
        assert get_tool_call_content_after_tag("") is None
        assert get_tool_call_content_after_tag(None) is None

    def test_handles_content_before_tag(self):
        """Test handles content before tool_call tag."""
        content = 'Prefix text <tool_call>{"name": "test"}'
        result = get_tool_call_content_after_tag(content)
        assert result == '{"name": "test"}'
