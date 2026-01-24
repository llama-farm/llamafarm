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

# Pre-compiled regex patterns for better performance
# Pattern to extract tool calls from <tool_call>...</tool_call> tags
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# Pattern to strip tool call tags from content
TOOL_CALL_STRIP_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)

# Pattern to extract tool name from partial JSON
TOOL_NAME_PATTERN = re.compile(r'"name"\s*:\s*"([^"]+)"')


# =============================================================================
# Prompt templates for different tool_choice modes
# =============================================================================

# tool_choice="auto" (default) - model may call tools if helpful
TOOLS_PREFIX_AUTO = """

You may call one or more tools to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""

TOOLS_SUFFIX_AUTO = """</tools>
For each tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call>.
If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.
"""

# tool_choice="required" - model MUST call at least one tool
TOOLS_PREFIX_REQUIRED = """

You MUST call one or more tools to respond to the user query. Do not respond with text alone.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""

TOOLS_SUFFIX_REQUIRED = """</tools>
You MUST use at least one of these tools. Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call>.
"""

# tool_choice={"type": "function", "function": {"name": "X"}} - model MUST call specific function
TOOLS_PREFIX_SPECIFIC = """

You MUST call the function "{function_name}" to respond to this query.
The function is defined within <tools></tools> XML tags:
<tools>
"""

TOOLS_SUFFIX_SPECIFIC = """</tools>
You MUST call the "{function_name}" function. Return a json object with the function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>{{"name": "{function_name}", "arguments": <args-json-object>}}</tool_call>.
"""

# Legacy aliases for backward compatibility
TOOLS_SYSTEM_MESSAGE_PREFIX = TOOLS_PREFIX_AUTO
TOOLS_SYSTEM_MESSAGE_SUFFIX = TOOLS_SUFFIX_AUTO


def format_tool_for_prompt(tool: dict) -> str:
    """Format a single tool definition for injection into the prompt.

    Args:
        tool: OpenAI-format tool definition with 'type' and 'function' keys.

    Returns:
        JSON string representation of the tool.
    """
    return json.dumps(tool, ensure_ascii=False)


def validate_tool_schema(tool: dict) -> list[str]:
    """Validate a tool definition schema.

    Args:
        tool: Tool definition in OpenAI format.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    if not isinstance(tool, dict):
        errors.append(f"Tool must be a dict, got {type(tool).__name__}")
        return errors

    # Check required top-level fields
    if "type" not in tool:
        errors.append("Tool missing required 'type' field")
    elif tool["type"] != "function":
        errors.append(f"Tool type must be 'function', got '{tool['type']}'")

    if "function" not in tool:
        errors.append("Tool missing required 'function' field")
        return errors

    func = tool["function"]
    if not isinstance(func, dict):
        errors.append(f"Tool 'function' must be a dict, got {type(func).__name__}")
        return errors

    # Check required function fields
    if "name" not in func:
        errors.append("Tool function missing required 'name' field")
    elif not isinstance(func["name"], str) or not func["name"]:
        errors.append("Tool function 'name' must be a non-empty string")

    # Check optional but commonly expected fields
    if "parameters" in func:
        params = func["parameters"]
        if not isinstance(params, dict):
            errors.append(
                f"Tool parameters must be a dict, got {type(params).__name__}"
            )

    return errors


def parse_tool_choice(tool_choice: str | dict | None) -> tuple[str, str | None]:
    """Parse tool_choice into a mode and optional function name.

    Args:
        tool_choice: Tool choice parameter from the API request.
            - None or "auto": Model decides whether to call tools
            - "none": Model should not call any tools
            - "required": Model must call at least one tool
            - {"type": "function", "function": {"name": "X"}}: Model must call function X

    Returns:
        Tuple of (mode, function_name) where mode is one of:
        "auto", "none", "required", "specific"
        and function_name is set only when mode is "specific".
    """
    if tool_choice is None or tool_choice == "auto":
        return ("auto", None)
    elif tool_choice == "none":
        return ("none", None)
    elif tool_choice == "required":
        return ("required", None)
    elif isinstance(tool_choice, dict):
        # Handle {"type": "function", "function": {"name": "X"}}
        if tool_choice.get("type") == "function":
            func_info = tool_choice.get("function", {})
            func_name = func_info.get("name")
            if func_name:
                return ("specific", func_name)
        # Fallback if dict format is unexpected
        logger.warning(
            f"Unexpected tool_choice dict format: {tool_choice}, using 'auto'"
        )
        return ("auto", None)
    else:
        logger.warning(f"Unknown tool_choice value: {tool_choice}, using 'auto'")
        return ("auto", None)


def inject_tools_into_messages(
    messages: list[dict],
    tools: list[dict],
    tool_choice: str | dict | None = None,
) -> list[dict]:
    """Inject tool definitions into the system message.

    If no system message exists, one is created. The tools are appended
    to the system message content using XML tags.

    Args:
        messages: List of chat messages (will not be modified).
        tools: List of tool definitions in OpenAI format.
        tool_choice: Tool choice strategy:
            - None or "auto": Model may call tools (default)
            - "none": Model should not call tools (returns messages unchanged)
            - "required": Model must call at least one tool
            - {"type": "function", "function": {"name": "X"}}: Must call specific function

    Returns:
        New list of messages with tools injected into system message.
    """
    if not tools:
        return messages

    # Validate tool schemas before injection
    valid_tools = []
    for i, tool in enumerate(tools):
        errors = validate_tool_schema(tool)
        if errors:
            tool_name = tool.get("function", {}).get("name", f"tool[{i}]")
            logger.warning(
                f"Skipping malformed tool '{tool_name}': {'; '.join(errors)}"
            )
        else:
            valid_tools.append(tool)

    if not valid_tools:
        logger.warning("No valid tools after validation, returning original messages")
        return messages

    tools = valid_tools

    # Parse tool_choice to determine mode
    mode, specific_func = parse_tool_choice(tool_choice)

    # "none" means don't inject tools at all
    if mode == "none":
        logger.debug("tool_choice='none', skipping tool injection")
        return messages

    # Deep copy to avoid modifying original
    messages = copy.deepcopy(messages)

    # Filter tools if a specific function is requested
    tools_to_inject = tools
    if mode == "specific" and specific_func:
        tools_to_inject = [
            t for t in tools if t.get("function", {}).get("name") == specific_func
        ]
        if not tools_to_inject:
            logger.warning(
                f"tool_choice specified function '{specific_func}' but it was not found "
                f"in provided tools. Available: {[t.get('function', {}).get('name') for t in tools]}"
            )
            # Fall back to auto mode with all tools
            mode = "auto"
            tools_to_inject = tools

    # Select prefix and suffix based on mode
    if mode == "required":
        prefix = TOOLS_PREFIX_REQUIRED
        suffix = TOOLS_SUFFIX_REQUIRED
    elif mode == "specific" and specific_func:
        prefix = TOOLS_PREFIX_SPECIFIC.format(function_name=specific_func)
        suffix = TOOLS_SUFFIX_SPECIFIC.format(function_name=specific_func)
    else:  # "auto" or fallback
        prefix = TOOLS_PREFIX_AUTO
        suffix = TOOLS_SUFFIX_AUTO

    # Build tools section
    tools_section = prefix
    for tool in tools_to_inject:
        tools_section += f"<tool>{format_tool_for_prompt(tool)}</tool>\n"
    tools_section += suffix

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

    matches = TOOL_CALL_PATTERN.findall(content)

    if not matches:
        return None

    results = []
    parse_errors = []
    for i, match in enumerate(matches):
        try:
            # Parse the JSON inside the tool_call tags
            tool_call_json = json.loads(match.strip())
            tool_name = tool_call_json.get("name")
            tool_args = tool_call_json.get("arguments", {})

            if tool_name:
                # Re-serialize arguments to ensure consistent JSON format
                args_json = json.dumps(tool_args, ensure_ascii=False)
                results.append((tool_name, args_json))
            else:
                parse_errors.append(f"Tool call {i + 1}: missing 'name' field")
        except json.JSONDecodeError as e:
            parse_errors.append(
                f"Tool call {i + 1}: JSON parse error - {e}, content: {match[:100]!r}"
            )

    # Log summary of parsing results
    if parse_errors:
        logger.error(
            f"Failed to parse {len(parse_errors)}/{len(matches)} tool call(s): "
            f"{'; '.join(parse_errors)}"
        )

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
    return TOOL_CALL_STRIP_PATTERN.sub("", content).strip()


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
    match = TOOL_NAME_PATTERN.search(partial_json)

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
