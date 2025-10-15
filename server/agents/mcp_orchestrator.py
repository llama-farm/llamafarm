from typing import get_type_hints

from atomic_agents import BaseIOSchema, BaseTool, BasicChatOutputSchema  # type: ignore
from atomic_agents.connectors.mcp import create_mcp_orchestrator_schema  # type: ignore
from context_providers.mcp_tools_context_provider import MCPToolsContextProvider
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import Field

from agents.agent import LFAgent, LFAgentConfig
from core.logging import FastAPIStructLogger
from services.mcp_service import MCPService
from tools.mcp_tool.tool.mcp_tool_factory import MCPToolFactory

from config.datamodel import LlamaFarmConfig

logger = FastAPIStructLogger(__name__)


class FinalResponseSchema(BasicChatOutputSchema):
    """Final response schema for when no more tools need to be called."""

    __doc__ = BasicChatOutputSchema.__doc__
    pass


class MCPOrchestrator(LFAgent):
    """Orchestrator agent that uses MCP tools with atomic-agents orchestrator pattern.

    Supports both structured (instructor) and unstructured (vanilla OpenAI) modes.
    """

    _mcp_service: MCPService | None = None
    _mcp_tools_loaded: bool = False
    _mcp_tools: list = []  # List of MCP tool classes (Type[BaseTool])
    _mcp_tool_factory: MCPToolFactory
    _mcp_session: ClientSession | None = None

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

        # Build the output schema with tool_parameters union
        class MCPOrchestratorOutputSchema(BaseIOSchema):
            """Output schema for orchestrator with tool parameters."""

            tool_parameters: tool_params_type = Field(  # type: ignore
                ...,
                description=(
                    "The parameters for the selected tool, "
                    "matching its specific schema (includes 'tool_name')."
                ),
            )

            model_config = {"arbitrary_types_allowed": True}

        # Store the orchestrator output schema
        self._orchestrator_output_schema = MCPOrchestratorOutputSchema

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

            max_iterations = 10  # Prevent infinite loops
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                logger.info(
                    "Orchestrator iteration",
                    iteration=iteration,
                    use_structured=self._use_structured_output,
                )

                # Get LLM decision
                if self._use_structured_output:
                    # Structured mode: instructor will parse to orchestrator schema
                    orchestrator_output = await super().run_async(user_input)
                else:
                    # Unstructured mode: manually parse response
                    # For now, doesn't support tool orchestration
                    # Fall back to direct response
                    logger.warning(
                        "Orchestrator pattern not fully supported in "
                        "unstructured mode; returning direct response"
                    )
                    response = await super().run_async(user_input)
                    return FinalResponseSchema(chat_message=response.chat_message)

                # Extract tool_parameters from orchestrator output
                if not hasattr(orchestrator_output, "tool_parameters"):
                    logger.warning("No tool_parameters in output")
                    return FinalResponseSchema(
                        chat_message=(
                            "I apologize, I couldn't process that request properly."
                        )
                    )

                tool_params = orchestrator_output.tool_parameters

                # Check if it's the final response
                if isinstance(tool_params, FinalResponseSchema):
                    logger.info("Orchestrator selected FinalResponseSchema")
                    return tool_params

                # Extract tool name and find matching tool class
                tool_name = getattr(tool_params, "tool_name", None)
                if not tool_name:
                    logger.warning("No tool_name in tool_parameters")
                    return FinalResponseSchema(
                        chat_message="I couldn't determine which tool to use."
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
                        chat_message=f"Tool '{tool_name}' not found."
                    )

                # Execute the tool
                try:
                    logger.info("Executing MCP tool", tool_name=tool_name)
                    tool_instance = tool_class()
                    tool_result = await tool_instance.arun(tool_params)

                    # Format tool result as user message for next iteration
                    result_content = getattr(tool_result, "result", str(tool_result))
                    user_input = self.input_schema(
                        chat_message=(
                            f"Tool '{tool_name}' returned: {result_content}"
                            "\n\nPlease interpret this result and provide "
                            "a response to the user."
                        )
                    )

                    logger.info(
                        "Tool execution successful",
                        tool_name=tool_name,
                        result_preview=str(result_content)[:200],
                    )

                except Exception as e:
                    logger.error(
                        "Tool execution failed",
                        tool_name=tool_name,
                        error=str(e),
                        exc_info=True,
                    )
                    return FinalResponseSchema(
                        chat_message=f"Error executing tool '{tool_name}': {str(e)}"
                    )

            # Max iterations reached
            logger.warning("Max orchestrator iterations reached")
            return FinalResponseSchema(
                chat_message=(
                    "I've reached the maximum number of tool calls. "
                    "Please try rephrasing your request."
                )
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
