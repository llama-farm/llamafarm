"""
Builtin Tools Registry.

This module defines the registry of all builtin tools and provides
filtering based on model configuration.

The registry only tracks tool names. Tool definitions are derived from
the Pydantic models in the tool classes themselves (single source of truth).
"""

from config.datamodel import Model

# Registry of all builtin tool names (public names users specify in config)
BUILTIN_TOOL_NAMES: set[str] = {"tasks"}

# Internal mapping from public tool names to actual tool names
_TOOL_EXPANSION: dict[str, list[str]] = {
    "tasks": ["task_create", "task_update", "task_list", "task_get"],
}


def get_enabled_builtin_tool_names(model_config: Model) -> set[str]:
    """Return names of enabled builtin tools based on model config include list.

    Args:
        model_config: The model configuration containing builtin_tools settings

    Returns:
        Set of tool names that are enabled for this model
    """
    if not model_config.builtin_tools or not model_config.builtin_tools.include:
        return set()

    result = set()
    for name in model_config.builtin_tools.include:
        if name in _TOOL_EXPANSION:
            result.update(_TOOL_EXPANSION[name])
    return result
