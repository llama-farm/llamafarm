from typing import Literal, get_type_hints

from atomic_agents import BaseIOSchema, BasicChatOutputSchema  # type: ignore
from atomic_agents.connectors.mcp import create_mcp_orchestrator_schema  # type: ignore
from config.datamodel import LlamaFarmConfig
from pydantic import Field

from agents.agent import LFAgent, LFAgentConfig
from context_providers.mcp_tools_context_provider import MCPToolsContextProvider
from core.logging import FastAPIStructLogger
from services.mcp_service import MCPService
from tools.mcp_tool.tool.mcp_tool_factory import MCPToolFactory

logger = FastAPIStructLogger(__name__)


class FinalResponseSchema(BaseIOSchema):
    """Final response tool - use when you have enough info to answer.

    This is a special tool that signals you are done calling tools
    and ready to provide the final answer.

    Use this when:
    - You have gathered all necessary information from tools
    - You can directly answer the user's question
    - No additional tool calls are needed
    """

    chat_message: str = Field(
        ...,
        description="Your complete final answer to the user's question. Be clear, concise, and helpful.",
    )


class MCPOrchestrator(LFAgent):
    """Orchestrator agent that uses MCP tools with atomic-agents orchestrator pattern.

    Supports both structured (instructor) and unstructured (vanilla OpenAI) modes.
    """

    _mcp_service: MCPService | None = None
    _mcp_tools_loaded: bool = False
    _mcp_tools: list = []  # List of MCP tool classes (Type[BaseTool])
    _mcp_tool_factory: MCPToolFactory

    def __init__(
        self,
        *,
        config: LFAgentConfig,
        mcp_service: MCPService,
    ):
        """Initialize MCPOrchestrator with tools and configuration.

        Args:
            config: LFAgentConfig with client and model settings
            tools: List of MCP tool classes (not instances) from atomic-agents
        """
        super().__init__(config=config)
        self._mcp_service = mcp_service
        self._mcp_tool_factory = MCPToolFactory(mcp_service)

        mcp_context_provider = MCPToolsContextProvider(title="MCP Tools")
        self.register_context_provider("mcp", mcp_context_provider)

    async def load_mcp_tools(self) -> None:
        """Load MCP tools dynamically (call from async context to avoid deadlock)."""

        if self._mcp_tools_loaded:
            return  # Already loaded

        try:
            # Get tool classes (not instances) from factory
            # These are Type[BaseTool] classes created by atomic-agents
            tool_classes = await self._mcp_tool_factory.create_all_tools()
            if tool_classes:
                # Store tool classes for orchestrator pattern
                self._mcp_tools = tool_classes

                # MCPToolsContextProvider expects tool instances
                # Create temporary instances for context provider
                # (Orchestrator will create its own when executing)
                tool_instances_for_context = []
                for tool_class in tool_classes:
                    try:
                        tool_instances_for_context.append(tool_class())
                    except Exception as e:
                        logger.warning(
                            "Failed to create tool instance for context",
                            tool_class=tool_class.__name__,
                            error=str(e),
                        )

                if hasattr(self, "mcp_context_provider"):
                    self.mcp_context_provider.set_tools(tool_instances_for_context)

                tool_names = [
                    getattr(t, "mcp_tool_name", t.__name__) for t in self._mcp_tools
                ]
                logger.info(
                    "MCP tools loaded for orchestrator pattern",
                    tool_count=len(self._mcp_tools),
                    tool_names=tool_names,
                )
                self._mcp_tools_loaded = True
            else:
                logger.info("No MCP tools available from configured servers")
        except Exception:
            logger.warning("Failed to load MCP tools dynamically", exc_info=True)

        # Create orchestrator schema with all tool input schemas
        orchestrator_schema = create_mcp_orchestrator_schema(self._mcp_tools)
        if orchestrator_schema is None:
            raise ValueError("Could not create orchestrator schema for MCP tools")

        # Get the tool_parameters field type from the orchestrator schema
        type_hints = get_type_hints(orchestrator_schema)
        tool_params_type = type_hints.get("tool_parameters")

        # Add FinalResponseSchema to union so LLM can choose to stop
        # This is critical - without it, orchestrator loops forever
        extended_tool_params_type = (
            tool_params_type | FinalResponseSchema  # type: ignore
        )

        # Build output schema with tool_parameters or FinalResponseSchema
        class MCPOrchestratorOutputSchema(BaseIOSchema):
            """Output schema for orchestrator with tool or final response.

            Choose the appropriate tool_name based on what you need:
            - Set tool_name='final_response' when ready to provide the final answer
            - Set tool_name to a specific tool name when you need more information
            """

            tool_parameters: extended_tool_params_type = Field(  # type: ignore
                ...,
                description=(
                    "The tool to use, identified by tool_name:\n"
                    "- tool_name='final_response': Use when you have enough information "
                    "to answer. Include your answer in 'chat_message' field.\n"
                    "- tool_name='<specific_tool>': Use when you need to call a tool. "
                    "Include the tool's required parameters.\n\n"
                    "The tool_name field determines which schema to use."
                ),
            )

            model_config = {"arbitrary_types_allowed": True}

        # Store the orchestrator output schema
        self._orchestrator_output_schema = MCPOrchestratorOutputSchema

        # DEBUG: Log the schema to verify FinalResponseSchema is included
        try:
            schema_dict = MCPOrchestratorOutputSchema.model_json_schema()
            logger.info(
                "Orchestrator schema created",
                schema_keys=list(schema_dict.keys()),
                tool_parameters_def=schema_dict.get("properties", {}).get(
                    "tool_parameters"
                ),
            )
        except Exception as e:
            logger.warning("Failed to log schema", error=str(e))

    async def run_async(self, user_input):
        """Run with orchestrator pattern using MCP tools."""
        return await self._run_with_tools(user_input)

    async def run_async_stream(self, user_input):
        """Stream with orchestrator pattern.

        Note: Orchestrator pattern doesn't support true streaming as it
        needs to make structured decisions. Falls back to non-streaming
        execution and yields the final result.
        """
        final_response = await self._run_with_tools(user_input)
        yield final_response

    async def cleanup(self) -> None:
        """Clean up resources, including persistent MCP sessions.

        Should be called when the orchestrator is no longer needed.
        """
        if self._mcp_service:
            await self._mcp_service.close_all_persistent_sessions()
            logger.info("Cleaned up MCP orchestrator resources")

    async def _run_with_tools(self, user_input):
        """Execute orchestrator pattern: LLM decides tool, execute in loop.

        The orchestrator loop:
        1. LLM receives input and chooses a tool (or FinalResponseSchema)
        2. If tool chosen, execute it and feed result back to LLM
        3. Repeat until LLM returns FinalResponseSchema

        Handles both structured and unstructured modes.
        """
        # Store original schemas
        original_output_schema = self.output_schema

        try:
            # Switch to orchestrator output schema
            self.__class__.output_schema = property(
                lambda s: self._orchestrator_output_schema
            )

            max_iterations = 3  # Prevent infinite loops
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                logger.info(
                    "Orchestrator iteration",
                    iteration=iteration,
                    use_structured=self._use_structured_output,
                )

                # Get LLM decision
                # if self._use_structured_output:
                # DEBUG: Log what we're about to ask the LLM
                logger.info(
                    "Calling LLM with orchestrator schema",
                    iteration=iteration,
                    input_preview=str(user_input)[:200],
                    output_schema=self._orchestrator_output_schema.__name__,
                )
                # Structured mode: instructor will parse to orchestrator schema
                orchestrator_output = await super().run_async(user_input)
                # else:
                #     # Unstructured mode: manually parse response
                #     # For now, doesn't support tool orchestration
                #     # Fall back to direct response
                #     logger.warning(
                #         "Orchestrator pattern not fully supported in "
                #         "unstructured mode; returning direct response"
                #     )
                #     response = await super().run_async(user_input)
                #     return FinalResponseSchema(chat_message=response.chat_message)

                # Extract tool_parameters from orchestrator output
                if not hasattr(orchestrator_output, "tool_parameters"):
                    logger.warning("No tool_parameters in output")
                    return FinalResponseSchema(
                        chat_message=(
                            "I apologize, I couldn't process that request properly."
                        ),
                    )

                tool_params = orchestrator_output.tool_parameters

                # DEBUG: Log what the LLM chose
                logger.info(
                    "LLM response received",
                    tool_params_type=type(tool_params).__name__,
                    tool_params_dict=tool_params.model_dump()
                    if hasattr(tool_params, "model_dump")
                    else str(tool_params),
                )

                # Extract tool name to determine action
                tool_name = getattr(tool_params, "tool_name", None)

                # Check if it's the final response
                if isinstance(tool_params, FinalResponseSchema):
                    logger.info("Orchestrator selected final_response")
                    # Return the final response with chat_message
                    return FinalResponseSchema(
                        chat_message=tool_params.chat_message,
                    )
                if not tool_name:
                    logger.warning("No tool_name in tool_parameters")
                    return FinalResponseSchema(
                        chat_message="I couldn't determine which tool to use.",
                    )

                # Find the tool class
                tool_class = next(
                    (
                        t
                        for t in self._mcp_tools
                        if getattr(t, "mcp_tool_name", None) == tool_name
                    ),
                    None,
                )

                if not tool_class:
                    logger.warning("Tool not found", tool_name=tool_name)
                    return FinalResponseSchema(
                        chat_message=f"Tool '{tool_name}' not found.",
                    )

                # Execute the tool
                try:
                    logger.info("Executing MCP tool", tool_name=tool_name)
                    tool_instance = tool_class()
                    tool_result = await tool_instance.arun(tool_params)

                    # Format tool result as user message for next iteration
                    result_content = getattr(tool_result, "result", str(tool_result))
                    message = BasicChatOutputSchema(
                        chat_message=(
                            f"Tool '{tool_name}' returned: {result_content}\n\n"
                            "Now decide your next action:\n"
                            "1. If you have enough information to fully answer the user's question:\n"
                            "   Set tool_name='final_response' and provide your complete answer in chat_message\n"
                            "2. If you need more information:\n"
                            "   Set tool_name to the specific tool you need (e.g., 'get_weather') "
                            "with the appropriate parameters\n\n"
                            "Think carefully: Do you have all the information needed to answer now?"
                        )
                    )
                    self.history.add_message("assistant", message)

                    logger.info(
                        "Tool execution successful",
                        tool_name=tool_name,
                        result_preview=str(result_content)[:200],
                    )

                    user_input.chat_message = f"Answer this user's prompt based on the existing tool results. {user_input.chat_message}"

                except Exception as e:
                    logger.error(
                        "Tool execution failed",
                        tool_name=tool_name,
                        error=str(e),
                        exc_info=True,
                    )
                    return FinalResponseSchema(
                        chat_message=f"Error executing tool '{tool_name}': {str(e)}",
                    )

            # Max iterations reached
            logger.warning("Max orchestrator iterations reached")
            return FinalResponseSchema(
                chat_message=(
                    "I've reached the maximum number of tool calls. "
                    "Please try rephrasing your request."
                ),
            )

        finally:
            # Restore original output schema
            self.__class__.output_schema = original_output_schema


class MCPOrchestratorFactory:
    """Factory for creating MCPOrchestrator instances."""

    @staticmethod
    async def create_agent(
        config: LFAgentConfig, project_config: LlamaFarmConfig
    ) -> MCPOrchestrator:
        mcp_service = MCPService(project_config)
        agent = MCPOrchestrator(config=config, mcp_service=mcp_service)
        await agent.load_mcp_tools()
        return agent
