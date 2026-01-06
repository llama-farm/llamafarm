"""
Prompt-based tool calling utilities.

This module provides functions for injecting tool definitions into prompts
and detecting tool calls in model outputs using XML tags.
"""

from __future__ import annotations

import copy
import json
import logging
import re

logger = logging.getLogger(__name__)


TOOLS_SYSTEM_MESSAGE_PREFIX = """

You may call one or more tools to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""

TOOLS_SYSTEM_MESSAGE_SUFFIX = """</tools>
For each tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call>.
If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.
"""


def format_tool_for_prompt(tool: dict) -> str:
    """Format a single tool definition for injection into the prompt.

    Args:
        tool: OpenAI-format tool definition with 'type' and 'function' keys.

    Returns:
        JSON string representation of the tool.
    """
    return json.dumps(tool, ensure_ascii=False)


def inject_tools_into_messages(
    messages: list[dict],
    tools: list[dict],
) -> list[dict]:
    """Inject tool definitions into the system message.

    If no system message exists, one is created. The tools are appended
    to the system message content using XML tags.

    Args:
        messages: List of chat messages (will not be modified).
        tools: List of tool definitions in OpenAI format.

    Returns:
        New list of messages with tools injected into system message.
    """
    if not tools:
        return messages

    # Deep copy to avoid modifying original
    messages = copy.deepcopy(messages)

    # Build tools section
    tools_section = TOOLS_SYSTEM_MESSAGE_PREFIX
    for tool in tools:
        tools_section += f"<tool>{format_tool_for_prompt(tool)}</tool>\n"
    tools_section += TOOLS_SYSTEM_MESSAGE_SUFFIX

    # Find system message and append tools
    system_found = False
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = content + tools_section
                system_found = True
                break
            # Non-string content (e.g., multimodal) - can't inject tools here
            # Continue searching for a string-content system message

    # If no system message, create one
    if not system_found:
        messages.insert(0, {"role": "system", "content": tools_section.strip()})

    return messages


def detect_tool_call_in_content(content: str) -> list[tuple[str, str]] | None:
    """Extract tool calls from content using XML tags.

    Looks for <tool_call>...</tool_call> patterns and extracts
    the tool name and arguments from each.

    Args:
        content: The model's response content.

    Returns:
        List of (tool_name, arguments_json) tuples, or None if no tool calls found.
    """
    if not content:
        return None

    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        return None

    results = []
    for match in matches:
        try:
            # Parse the JSON inside the tool_call tags
            tool_call_json = json.loads(match.strip())
            tool_name = tool_call_json.get("name")
            tool_args = tool_call_json.get("arguments", {})

            if tool_name:
                # Re-serialize arguments to ensure consistent JSON format
                args_json = json.dumps(tool_args, ensure_ascii=False)
                results.append((tool_name, args_json))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call JSON: {e}, content: {match[:100]}")
            continue

    return results if results else None


def detect_probable_tool_call(content: str) -> bool:
    """Check if content likely contains an incomplete tool call.

    Used during streaming to detect when we should start buffering
    instead of emitting tokens.

    Args:
        content: Accumulated content so far.

    Returns:
        True if content contains an opening <tool_call> tag.
    """
    return "<tool_call>" in content


def strip_tool_call_from_content(content: str) -> str:
    """Remove tool call XML tags from content.

    Args:
        content: The model's response content.

    Returns:
        Content with tool call tags removed.
    """
    pattern = r"<tool_call>.*?</tool_call>"
    return re.sub(pattern, "", content, flags=re.DOTALL).strip()


# =============================================================================
# Incremental streaming utilities
# =============================================================================


def extract_tool_name_from_partial(content: str) -> str | None:
    """Extract tool name from incomplete tool call JSON.

    Used during streaming to detect the tool name before the entire
    tool call JSON is complete. This enables emitting the initial
    tool call chunk early.

    Looks for patterns like:
    - <tool_call>{"name": "get_weather"
    - <tool_call>{"name":"get_weather",

    Args:
        content: Accumulated content that may contain a partial tool call.

    Returns:
        Tool name if found and complete, None otherwise.
    """
    if not content or "<tool_call>" not in content:
        return None

    # Find the start of the tool call JSON
    start_idx = content.find("<tool_call>")
    if start_idx == -1:
        return None

    # Extract everything after <tool_call>
    json_start = start_idx + len("<tool_call>")
    partial_json = content[json_start:]

    # Use regex to extract a complete "name" value
    # Matches: "name": "value" or "name":"value"
    # The name value must be complete (closing quote found)
    pattern = r'"name"\s*:\s*"([^"]+)"'
    match = re.search(pattern, partial_json)

    if match:
        return match.group(1)

    return None


def extract_arguments_progress(content: str) -> tuple[int, str] | None:
    """Extract the arguments JSON string progress from a partial tool call.

    Used during streaming to extract how much of the "arguments" value
    we have so far, enabling incremental streaming of arguments.

    Args:
        content: Accumulated content containing a partial tool call.

    Returns:
        Tuple of (start_position, arguments_so_far) where start_position
        is the character index where arguments value begins in the content,
        and arguments_so_far is the accumulated arguments string.
        Returns None if arguments section not yet started.
    """
    if not content or "<tool_call>" not in content:
        return None

    # Find the start of the tool call JSON
    tool_start = content.find("<tool_call>")
    if tool_start == -1:
        return None

    json_start = tool_start + len("<tool_call>")
    partial_json = content[json_start:]

    # Find "arguments": or "arguments" :
    args_pattern = r'"arguments"\s*:\s*'
    match = re.search(args_pattern, partial_json)

    if not match:
        return None

    # Position where the arguments value starts (after the colon and whitespace)
    args_value_start = json_start + match.end()

    # Extract everything from there
    remaining = content[args_value_start:]

    # Track brace depth to find the end of the arguments JSON value
    # Arguments is a JSON object, so we need to find where it closes
    args_content = _extract_json_value(remaining)

    if not args_content:
        return None

    return (args_value_start, args_content)


def _extract_json_value(content: str) -> str:
    """Extract a JSON value (object or array) from the start of content.

    Tracks brace/bracket depth to find where the JSON value ends.
    Handles incomplete JSON by returning what we have so far.

    Args:
        content: String starting with a JSON value.

    Returns:
        The JSON value string (possibly incomplete).
    """
    if not content:
        return ""

    content = content.strip()
    if not content:
        return ""

    # Determine the opening bracket type
    if content[0] == "{":
        open_char, close_char = "{", "}"
    elif content[0] == "[":
        open_char, close_char = "[", "]"
    else:
        # Not a JSON object/array, might be a primitive
        # For tool calls, arguments should always be an object
        return content

    depth = 0
    in_string = False
    escape_next = False
    end_pos = len(content)

    for i, char in enumerate(content):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                # Found the matching closing bracket
                end_pos = i + 1
                break

    # Return the JSON value (complete or partial)
    result = content[:end_pos]

    # Clean up any trailing content after the closing bracket
    # (like the closing brace of the outer object or </tool_call>)
    return result


def is_tool_call_complete(content: str) -> bool:
    """Check if content contains a complete tool call with closing tag.

    Args:
        content: Accumulated content that may contain a tool call.

    Returns:
        True if a complete <tool_call>...</tool_call> is found.
    """
    if not content:
        return False

    return "</tool_call>" in content


def get_tool_call_content_after_tag(content: str) -> str | None:
    """Extract the content inside <tool_call>...</tool_call> tags.

    Args:
        content: Content containing tool call tags.

    Returns:
        The content between the tags, or None if not found.
    """
    if not content or "<tool_call>" not in content:
        return None

    start_idx = content.find("<tool_call>")
    if start_idx == -1:
        return None

    json_start = start_idx + len("<tool_call>")
    end_idx = content.find("</tool_call>", json_start)

    if end_idx == -1:
        # No closing tag yet, return everything after opening tag
        return content[json_start:]

    return content[json_start:end_idx]
