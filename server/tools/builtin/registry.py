"""
Builtin Tools Registry.

This module defines the registry of all builtin tools and provides
filtering based on model configuration.

The registry only tracks tool names. Tool definitions are derived from
the Pydantic models in the tool classes themselves (single source of truth).
"""

from config.datamodel import Model

# Registry of all builtin tool names
BUILTIN_TOOL_NAMES: set[str] = {"tasks"}


def get_enabled_builtin_tool_names(model_config: Model) -> set[str]:
    """Return names of enabled builtin tools based on model config include list.

    Args:
        model_config: The model configuration containing builtin_tools settings

    Returns:
        Set of tool names that are enabled for this model
    """
    builtin_config = model_config.builtin_tools

    # Default: no tools enabled
    if builtin_config is None:
        return set()

    # Only include explicitly listed tools that exist in registry
    include_list = builtin_config.include or []
    if not include_list:
        return set()

    return {name for name in include_list if name in BUILTIN_TOOL_NAMES}
