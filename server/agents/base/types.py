"""Types for LLamaFarm Agent framework."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ToolDefinition:
    """Provider-agnostic tool definition.

    This is a standardized format for tool definitions that can be
    converted to provider-specific formats (OpenAI, Ollama, etc.)
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters

    @classmethod
    def from_mcp_tool(cls, tool_class: type) -> "ToolDefinition":
        """Convert MCP tool class to ToolDefinition.

        Args:
            tool_class: The MCP tool class (from atomic-agents)

        Returns:
            ToolDefinition with extracted name, description, and parameters
        """
        tool_name = getattr(tool_class, "mcp_tool_name", tool_class.__name__)
        tool_description = tool_class.__doc__ or "No description"

        # Get input schema from tool
        input_schema_class = getattr(tool_class, "input_schema", None)
        if input_schema_class:
            schema = input_schema_class.model_json_schema()
            # Remove tool_name discriminator field from properties
            props = {
                k: v
                for k, v in schema.get("properties", {}).items()
                if k != "tool_name"
            }
            required = [r for r in schema.get("required", []) if r != "tool_name"]
            parameters = {"type": "object", "properties": props, "required": required}
        else:
            parameters = {"type": "object", "properties": {}}

        return cls(name=tool_name, description=tool_description, parameters=parameters)


@dataclass
class ToolCallRequest:
    """A tool call requested by the LLM.

    This represents the LLM's decision to call a tool with specific arguments.
    """

    id: str  # Unique ID for this tool call (for tracking in conversation)
    name: str
    arguments: dict[str, Any]


@dataclass
class StreamEvent:
    """Event from streaming chat with tool calling support.

    All LFAgentClient implementations must yield StreamEvent objects
    in a consistent format, regardless of how they implement tool calling
    (native API vs JSON prompting).
    """

    type: Literal["content", "tool_call"]

    # For type="content"
    content: str | None = None

    # For type="tool_call"
    tool_call: ToolCallRequest | None = None

    def is_content(self) -> bool:
        """Check if this event contains content."""
        return self.type == "content"

    def is_tool_call(self) -> bool:
        """Check if this event contains a tool call request."""
        return self.type == "tool_call"
