from atomic_agents import BaseTool
from atomic_agents.context import BaseDynamicContextProvider


class MCPToolsContextProvider(BaseDynamicContextProvider):
    """Context provider that adds MCP tool information to the agent's system prompt."""

    def __init__(self, title: str):
        super().__init__(title=title)
        self.tools: list[BaseTool] = []

    def set_tools(self, tools: list[BaseTool]):
        """Set the list of MCP tools (instances) to include in context."""
        self.tools = tools

    def get_info(self) -> str:
        """Generate context information about available MCP tools."""
        if not self.tools:
            return ""

        lines = [
            "STEPS TO PERFORM WHEN USING MCP TOOLS:",
            "1. Analyze the user's query to determine if one or more MCP tools could help.",
            "2. Choose the appropriate tool and set tool_parameters with the tool_name and required arguments.",
            "3. For complex queries, break them down into smaller tasks using sequential tool calls.",
            "4. When you have all the information needed, respond with FinalResponseSchema.",
            "5. Always provide clear reasoning for your tool selection.",
            "",
            "AVAILABLE MCP TOOLS:",
        ]

        for tool in self.tools:
            # Get tool name and description
            tool_name = getattr(tool, "mcp_tool_name", None) or getattr(
                tool, "tool_name", None
            )
            tool_description = getattr(tool, "__doc__", "No description available")

            if not tool_name:
                tool_name = tool.__class__.__name__

            # Try to get input schema information
            input_schema_class = getattr(tool, "input_schema", None)
            if input_schema_class:
                try:
                    # Get field names from the input schema
                    if hasattr(input_schema_class, "model_fields"):
                        fields = input_schema_class.model_fields
                        # Exclude tool_name from the argument list as it's the discriminator
                        arg_names = [
                            name for name in fields.keys() if name != "tool_name"
                        ]
                        arg_list = ", ".join(arg_names)
                    else:
                        arg_list = "unknown"
                except Exception:
                    arg_list = "unknown"
            else:
                arg_list = "unknown"

            lines.append(
                f"\n- **{tool_name}**: {tool_description}\n  Arguments: {arg_list}"
            )

        return "\n".join(lines)
