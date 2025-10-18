from atomic_agents import BaseTool  # type: ignore
from atomic_agents.context import BaseDynamicContextProvider  # type: ignore


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

        lines = """
            STEPS TO PERFORM WHEN USING MCP TOOLS:
            1. Analyze the user's query to determine if one or more
               MCP tools could help.
            2. Choose the appropriate tool and respond with the tool
               calling schema. If you are unsure about values for the
               tool parameters, ask the user for clarification.
            3. For complex queries, break them down into smaller tasks
               using sequential tool calls.
            4. When you have all the information needed, respond with
               FinalResponseSchema.
            5. Always provide clear reasoning for your tool selection.
            
            TOOL CALLING JSON SCHEMA WHEN REQUESTING A TOOL CALL:
            ```json
            {
              "type": "object",
              "properties": {
                  "tool_name": {
                      "type": "string",
                      "description": "The name of the tool to call.",
                  },
                  "tool_parameters": {
                      "type": "object",
                      "additionalProperties": true,
                      "description": "Parameters for the tool. See the
                                      tool's input schema below.",
                  }
              },
            },
            ```
            
            AVAILABLE MCP TOOLS:
        """

        import json

        for tool in self.tools:
            # Get tool name and description
            tool_name = getattr(tool, "mcp_tool_name", None) or getattr(
                tool, "tool_name", None
            )
            tool_description = getattr(tool, "__doc__", "No description available")

            if not tool_name:
                tool_name = tool.__class__.__name__

            # Try to get input schema as JSON schema
            input_schema_class = getattr(tool, "input_schema", None)
            if input_schema_class:
                try:
                    # Get the JSON schema from the Pydantic model
                    if hasattr(input_schema_class, "model_json_schema"):
                        schema = input_schema_class.model_json_schema()
                        # Remove tool_name from properties if it exists
                        # (it's the discriminator)
                        props = schema.get("properties", {})
                        if "tool_name" in props:
                            schema_copy = schema.copy()
                            schema_copy["properties"] = {
                                k: v for k, v in props.items() if k != "tool_name"
                            }
                            if "required" in schema_copy:
                                schema_copy["required"] = [
                                    r
                                    for r in schema_copy["required"]
                                    if r != "tool_name"
                                ]
                            schema = schema_copy
                        schema_str = json.dumps(schema, indent=2)
                    else:
                        schema_str = "Schema not available"
                except Exception as e:
                    schema_str = f"Error getting schema: {e}"
            else:
                schema_str = "No input schema defined"

            lines += f"\n\n- **{tool_name}**: {tool_description}\n"
            lines += f"  Input Schema (JSON):\n```json\n{schema_str}\n```"

        return lines
