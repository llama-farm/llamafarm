"""
Builtin Tools Infrastructure.

This module provides built-in tools that are available to all chat sessions.
Tools can be enabled/disabled via model configuration (builtin_tools.exclude).

Exports:
    BUILTIN_TOOLS: Registry of all builtin tool definitions
    get_enabled_builtin_tools: Filter tools based on model config
    BuiltinToolFactory: Factory for creating tool instances with context
    TasksTool: Task management tool implementation
"""

from tools.builtin.factory import BuiltinToolFactory
from tools.builtin.registry import BUILTIN_TOOLS, get_enabled_builtin_tools
from tools.builtin.tasks_tool import TasksTool

__all__ = [
    "BUILTIN_TOOLS",
    "get_enabled_builtin_tools",
    "BuiltinToolFactory",
    "TasksTool",
]
