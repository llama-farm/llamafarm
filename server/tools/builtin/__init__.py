"""
Builtin Tools Infrastructure.

This module provides built-in tools that are available to all chat sessions.
Tools can be enabled/disabled via model configuration (builtin_tools.include).

Exports:
    BUILTIN_TOOL_NAMES: Set of all builtin tool names
    get_enabled_builtin_tool_names: Get enabled tool names based on model config
    BuiltinToolFactory: Factory for creating tool instances with context
    TasksTool: Task management tool implementation
"""

from tools.builtin.factory import BuiltinToolFactory
from tools.builtin.registry import BUILTIN_TOOL_NAMES, get_enabled_builtin_tool_names
from tools.builtin.tasks_tool import TasksTool

__all__ = [
    "BUILTIN_TOOL_NAMES",
    "get_enabled_builtin_tool_names",
    "BuiltinToolFactory",
    "TasksTool",
]
