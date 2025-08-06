"""
Tool registry for managing and discovering tools.
"""

import logging
from typing import Any, Optional

from .base_tool import BaseTool
from .errors import ToolNotFoundError, ToolRegistryError

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing tool instances and discovery."""
    
    _tools: dict[str, BaseTool] = {}
    _tool_classes: dict[str, type[BaseTool]] = {}
    
    @classmethod
    def register(cls, name: str, tool_instance: BaseTool) -> None:
        """Register a tool instance."""
        if not isinstance(tool_instance, BaseTool):
            raise ToolRegistryError(
                f"Tool must be an instance of BaseTool, got {type(tool_instance)}")
        
        cls._tools[name] = tool_instance
        cls._tool_classes[name] = tool_instance.__class__
        logger.info(f"Tool registered: {name} ({tool_instance.__class__.__name__})")
    
    @classmethod
    def get_tool(cls, name: str) -> BaseTool:
        """Get a tool instance by name."""
        if name not in cls._tools:
            # Try to auto-discover and load the tool
            cls._try_load_tool(name)
        
        if name not in cls._tools:
            raise ToolNotFoundError(name)
        
        return cls._tools[name]
    
    @classmethod
    def list_tools(cls) -> list[str]:
        """List all registered tool names."""
        # Auto-discover tools if registry is empty
        if not cls._tools:
            cls._discover_tools()
        
        return list(cls._tools.keys())
    
    @classmethod
    def _try_load_tool(cls, name: str) -> None:
        """Try to load a tool by name."""
        try:
            # Try to load projects tool specifically
            if name == "projects":
                from tools.projects_tool.tool.projects_tool import ProjectsTool
                cls.register(name, ProjectsTool())
            # Add other tool discovery patterns here
        except Exception as e:
            logger.warning(f"Failed to auto-load tool '{name}': {e}")
    
    @classmethod
    def _discover_tools(cls) -> None:
        """Auto-discover available tools."""
        try:
            # Auto-load known tools
            cls._try_load_tool("projects")
        except Exception as e:
            logger.warning(f"Tool discovery failed: {e}")
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (mainly for testing)."""
        cls._tools.clear()
        cls._tool_classes.clear()


# Convenience functions
def register_tool(name: str, tool_instance: BaseTool) -> None:
    """Register a tool in the global registry."""
    ToolRegistry.register(name, tool_instance)


def get_tool(name: str) -> BaseTool:
    """Get a tool from the global registry."""
    return ToolRegistry.get_tool(name)


def list_tools() -> list[str]:
    """List all tools in the global registry."""
    return ToolRegistry.list_tools()


# Decorator for automatic tool registration
def tool(name: str | None = None):
    """Decorator to automatically register a tool class."""
    def decorator(tool_class: type[BaseTool]):
        tool_name = name or tool_class.__name__.lower().replace('tool', '')
        
        # Create instance and register
        try:
            tool_instance = tool_class()
            register_tool(tool_name, tool_instance)
        except Exception as e:
            logger.warning(f"Failed to auto-register tool '{tool_name}': {e}")
        
        return tool_class
    
    return decorator