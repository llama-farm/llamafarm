"""
Builtin Tools Registry.

This module defines the registry of all builtin tools and provides
filtering based on model configuration.
"""

from agents.base.types import ToolDefinition
from config.datamodel import Model


# Registry of all builtin tools keyed by name
BUILTIN_TOOLS: dict[str, ToolDefinition] = {
    "tasks": ToolDefinition(
        name="tasks",
        description=(
            "Manage tasks for the current session. Create, update, list, and get tasks "
            "to track work items. Tasks can have dependencies (blockedBy/blocks) and "
            "status (pending, in_progress, completed). Use status='deleted' to remove a task."
        ),
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create", "update", "list", "get"],
                    "description": "The operation to perform on tasks",
                },
                "taskId": {
                    "type": "string",
                    "description": "Task ID (required for get, update operations)",
                },
                "subject": {
                    "type": "string",
                    "description": "Task subject/title (for create, update)",
                },
                "description": {
                    "type": "string",
                    "description": "Task description (for create, update)",
                },
                "activeForm": {
                    "type": "string",
                    "description": (
                        "Present continuous form for display (e.g., 'Running tests')"
                    ),
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "deleted"],
                    "description": "Task status (for update). Use 'deleted' to remove.",
                },
                "blockedBy": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task IDs that must complete before this task (create)",
                },
                "addBlockedBy": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task IDs to add to blockedBy list (update)",
                },
                "addBlocks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task IDs that this task blocks (update)",
                },
            },
            "required": ["operation"],
        },
    ),
}


def get_enabled_builtin_tools(model_config: Model) -> list[ToolDefinition]:
    """Return builtin tools based on model config exclude list.

    Args:
        model_config: The model configuration containing builtin_tools settings

    Returns:
        List of ToolDefinition objects for enabled builtin tools
    """
    builtin_config = model_config.builtin_tools

    # If not specified, all enabled
    if builtin_config is None:
        return list(BUILTIN_TOOLS.values())

    # Master switch
    if not builtin_config.enabled:
        return []

    # Filter by exclude list
    exclude_set = set(builtin_config.exclude or [])
    return [
        tool for name, tool in BUILTIN_TOOLS.items() if name not in exclude_set
    ]
