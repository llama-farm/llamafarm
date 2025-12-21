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
