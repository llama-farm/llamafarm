"""
Error classes for the tool system.
"""

class ToolError(Exception):
    """Base exception for tool-related errors"""
    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found in the registry"""
    
    def __init__(self, tool_name: str, message: str = None):
        self.tool_name = tool_name
        if message is None:
            message = f"Tool '{tool_name}' not found in registry"
        super().__init__(message)


class ToolExecutionError(ToolError):
    """Raised when a tool fails to execute properly"""
    
    def __init__(self, tool_name: str, message: str = None):
        self.tool_name = tool_name
        if message is None:
            message = f"Tool '{tool_name}' execution failed"
        super().__init__(message)


class ToolValidationError(ToolError):
    """Raised when tool input validation fails"""
    
    def __init__(self, tool_name: str, validation_message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' validation failed: {validation_message}")


class ToolRegistryError(ToolError):
    """Raised when there are issues with the tool registry"""
    pass


class ToolHealthCheckError(ToolError):
    """Raised when a tool health check fails"""
    
    def __init__(self, tool_name: str, message: str = None):
        self.tool_name = tool_name
        if message is None:
            message = f"Tool '{tool_name}' health check failed"
        super().__init__(message)