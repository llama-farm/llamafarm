"""
Base tool definitions and execution result classes.
"""

from dataclasses import dataclass
from typing import Any

from atomic_agents import BaseTool as AtomicBaseTool

# Re-export atomic_agents BaseTool for consistency
BaseTool = AtomicBaseTool


@dataclass
class ToolExecutionResult:
    """Result of tool execution."""
    success: bool
    data: Any | None = None
    error_message: str | None = None
    metadata: dict | None = None
    
    def __post_init__(self):
        """Ensure we have either data or error_message based on success."""
        if self.success and self.error_message is None:
            self.error_message = ""
        elif not self.success and self.error_message is None:
            self.error_message = "Unknown error occurred"