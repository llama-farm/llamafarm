import json
import os
from collections.abc import AsyncGenerator
from pathlib import Path

from atomic_agents import BaseTool  # type: ignore
from config.datamodel import LlamaFarmConfig, Provider

from agents.base.agent import LFAgent, LFAgentConfig
from agents.base.clients.openai import LFAgentClientOpenAI
from agents.base.history import LFAgentChatMessage, LFAgentHistory
from agents.base.system_prompt_generator import LFAgentSystemPromptGenerator
from agents.base.types import ToolDefinition
from core.logging import FastAPIStructLogger
from core.mcp_registry import register_mcp_service
from services.mcp_service import MCPService
from services.model_service import ModelService
from services.runtime_service.runtime_service import RuntimeService
from tools.mcp_tool.tool.mcp_tool_factory import MCPToolFactory

logger = FastAPIStructLogger(__name__)

CLIENT_CLASSES = {
    Provider.openai: LFAgentClientOpenAI,
    Provider.ollama: LFAgentClientOpenAI,
    Provider.lemonade: LFAgentClientOpenAI,
}

# Constants for orchestration loop
MAX_TOOL_ITERATIONS = 10
MAX_ITERATIONS_MESSAGE = (
    "I've reached the maximum number of tool calls. Please try rephrasing your request."
)


class ChatOrchestratorAgent(LFAgent):
    _persist_enabled: bool
    _project_dir: str
    _project_config: LlamaFarmConfig
    model_name: str
    _session_id: str | None = None
    _mcp_enabled: bool = False
    _mcp_service: MCPService | None = None
    _mcp_tool_factory: MCPToolFactory | None = None
    _mcp_tools: list[BaseTool] = []

    def __init__(
        self,
        *,
        project_config: LlamaFarmConfig,
        project_dir: str,
        model_name: str | None = None,
    ):
        self._project_config = project_config
        self._project_dir = project_dir
        self._session_id = None
        self._persist_enabled = False

        # Get the model config - if model_name is None, get_model returns the default
        model_config = ModelService.get_model(project_config, model_name)
        # Store the model name (the config name), not the model string
        # This allows lookup by name in the config
        self.model_name = model_config.name
        # Store the actual model string for reference (e.g., "llama3.2:latest")
        self._model_string = model_config.model

        history = self._get_history(project_config)
        provider = RuntimeService.get_provider(model_config)
        client = provider.get_client()

        system_prompt_generator = LFAgentSystemPromptGenerator(
            prompts=self._get_prompts_for_model(model_config.name)
        )
        config = LFAgentConfig(
            history=history,
            system_prompt_generator=system_prompt_generator,
            client=client,
        )

        super().__init__(config=config)

    def _create_tool_result_guidance_message(
        self, tool_name: str, result_content: str
    ) -> str:
        """Create guidance message to send after tool execution."""
        return (
            f"Tool '{tool_name}' returned: {result_content}\n\n"
            "Based on this tool result, please provide your "
            "complete final answer to my original question. "
            "Do not call the same tool again unless you need "
            "additional different information. Answer:"
        )

    def _persist_history_safe(self) -> None:
        """Safely persist history with error handling."""
        try:
            self._persist_history()
        except Exception:
            logger.warning("History persistence failed", exc_info=True)

    async def run_async(self, user_input: LFAgentChatMessage | None = None) -> str:
        """Run the agent with MCP tool calling support.

        The agent will:
        1. Get response from LLM
        2. Check if response requests a tool call
        3. Execute the tool and feed result back to LLM
        4. Repeat until LLM provides final answer
        """
        iteration = 0

        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1

            try:
                # Get LLM response
                response = await super().run_async(user_input=user_input)

                # Try to parse as JSON to check for tool calls
                try:
                    response_data = json.loads(response)
                except (json.JSONDecodeError, TypeError):
                    # Not JSON, treat as final response
                    self._persist_history_safe()
                    return response

                # Check if this is a tool call request
                tool_name = response_data.get("tool_name")
                tool_parameters = response_data.get("tool_parameters", {})

                if not tool_name:
                    # No tool requested, this is the final response
                    # Extract the actual message if present
                    final_message = response_data.get("message") or response
                    self._persist_history_safe()
                    return final_message

                # Execute the tool
                logger.info(
                    "Executing MCP tool",
                    tool_name=tool_name,
                    iteration=iteration,
                )

                # Find the tool in our loaded tools
                tool_class = next(
                    (
                        t
                        for t in self._mcp_tools
                        if getattr(t, "mcp_tool_name", None) == tool_name
                    ),
                    None,
                )

                if not tool_class:
                    error_msg = f"Tool '{tool_name}' not found"
                    logger.warning(error_msg)
                    # Feed error back to LLM
                    user_input = LFAgentChatMessage(
                        role="user",
                        content=(
                            f"Error: {error_msg}. Please try again or "
                            "provide a direct answer."
                        ),
                    )
                    continue

                # Call the tool
                try:
                    tool_instance = tool_class()
                    # Create input schema instance with parameters
                    input_schema_class = tool_class.input_schema
                    tool_input = input_schema_class(
                        tool_name=tool_name, **tool_parameters
                    )
                    tool_result = await tool_instance.arun(tool_input)

                    # Extract result content
                    result_content = getattr(tool_result, "result", str(tool_result))

                    logger.info(
                        "Tool execution successful",
                        tool_name=tool_name,
                        result_preview=str(result_content)[:200],
                    )

                    # Feed result back to LLM with clear instruction
                    user_input = LFAgentChatMessage(
                        role="user",
                        content=self._create_tool_result_guidance_message(
                            tool_name, result_content
                        ),
                    )

                except Exception as e:
                    error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    # Feed error back to LLM
                    user_input = LFAgentChatMessage(
                        role="user",
                        content=(
                            f"{error_msg}. Please try again or provide a direct answer."
                        ),
                    )
                    continue

            except Exception as e:
                logger.error("Error in orchestrator loop", exc_info=True)
                raise e

        # Max iterations reached
        logger.warning("Max iterations reached in orchestrator")
        self._persist_history_safe()
        return MAX_ITERATIONS_MESSAGE

    async def run_async_stream(
        self, user_input: LFAgentChatMessage | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat with MCP tool execution support."""

        if not self._mcp_enabled or not self._mcp_tools:
            # No MCP tools, use standard streaming
            accumulated_content = ""
            async for chunk in super().run_async_stream(user_input=user_input):
                yield chunk
                accumulated_content += chunk

            # Add complete assistant response to history
            if accumulated_content:
                self.history.add_message(
                    LFAgentChatMessage(
                        role="assistant",
                        content=accumulated_content,
                    )
                )

            self._persist_history_safe()
            return

        # Convert MCP tools to ToolDefinition format
        tools = [ToolDefinition.from_mcp_tool(t) for t in self._mcp_tools]

        iteration = 0
        current_input = user_input

        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1

            logger.info("Starting tool calling iteration", iteration=iteration)

            # Stream chat with tools
            tool_call_made = False
            accumulated_content = ""  # Accumulate chunks for history

            async for event in self.stream_chat_with_tools(
                user_input=current_input, tools=tools
            ):
                if event.is_content():
                    # Stream content to user
                    if event.content:
                        yield event.content
                        accumulated_content += event.content

                elif event.is_tool_call():
                    # Add accumulated content to history before tool call
                    if accumulated_content:
                        self.history.add_message(
                            LFAgentChatMessage(
                                role="assistant",
                                content=accumulated_content,
                            )
                        )

                    # Execute the tool
                    tool_call_made = True
                    tool_call = event.tool_call

                    if not tool_call:
                        continue

                    logger.info(
                        "Executing MCP tool",
                        tool_name=tool_call.name,
                        iteration=iteration,
                    )

                    yield f"\n\n🔧 Calling {tool_call.name}...\nParameters: {tool_call.arguments}\n"

                    # Execute the MCP tool
                    result = await self._execute_mcp_tool(
                        tool_call.name, tool_call.arguments
                    )

                    # Add tool call and result to history with guidance
                    self.history.add_message(
                        LFAgentChatMessage(
                            role="tool",
                            content=result,
                        )
                    )
                    self.history.add_message(
                        LFAgentChatMessage(
                            role="assistant",
                            content=(
                                "Based on the tool result above, I should now provide "
                                "my complete final answer to the user's original question. "
                                "I should not call the same tool again unless I need "
                                "additional different information. Answer:"
                            ),
                        )
                    )

                    # Prepare for next iteration
                    current_input = None  # History already updated
                    break  # Exit event loop, continue while loop

            # If no tool was called, we're done
            if not tool_call_made:
                # Add final accumulated content to history
                if accumulated_content:
                    self.history.add_message(
                        LFAgentChatMessage(
                            role="assistant",
                            content=accumulated_content,
                        )
                    )
                logger.info("No tool call made, conversation complete")
                break

        # Save history
        if iteration >= MAX_TOOL_ITERATIONS:
            logger.warning("Max iterations reached", max_iterations=MAX_TOOL_ITERATIONS)
            yield f"\n\n{MAX_ITERATIONS_MESSAGE}\n"

        self._persist_history_safe()

    def enable_persistence(
        self,
        *,
        session_id: str,
    ) -> None:
        """Enable disk persistence for this agent and restore history.

        Use this when the agent was constructed without context (e.g., via a factory
        mocked in tests) but we still want to persist session history.
        """
        try:
            self._persist_enabled = True
            self._session_id = session_id
            self._restore_persisted_history()

        except Exception:
            logger.warning("Failed to enable persistence", exc_info=True)

    async def enable_mcp(self):
        """Enable MCP tool calling support."""
        if self._mcp_enabled:
            return
        self._mcp_service = MCPService(self._project_config)
        self._mcp_tool_factory = MCPToolFactory(self._mcp_service)
        # Register for cleanup on shutdown
        register_mcp_service(self._mcp_service)
        self._mcp_enabled = True
        await self._load_mcp_tools()

    async def _load_mcp_tools(self):
        """Load MCP tools from configured servers."""
        if not self._mcp_enabled:
            await self.enable_mcp()
        self._mcp_tools = await self._mcp_tool_factory.create_all_tools()
        logger.info(
            "MCP tools loaded",
            tool_count=len(self._mcp_tools),
            tool_names=[
                getattr(t, "mcp_tool_name", t.__name__) for t in self._mcp_tools
            ],
        )

    async def _execute_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute an MCP tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool parameters

        Returns:
            Tool result as string
        """
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
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(
                error_msg,
                available_tools=[
                    getattr(t, "mcp_tool_name", t.__name__) for t in self._mcp_tools
                ],
            )
            return f"Error: {error_msg}"

        try:
            # Instantiate and execute tool
            tool_instance = tool_class()
            input_schema_class = tool_class.input_schema

            # Create input with tool_name discriminator
            tool_input = input_schema_class(tool_name=tool_name, **arguments)

            # Execute tool
            result = await tool_instance.arun(tool_input)

            # Extract result content
            result_content = getattr(result, "result", str(result))

            logger.info(
                "Tool execution successful",
                tool_name=tool_name,
                result_length=len(str(result_content)),
            )

            return str(result_content)

        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def reset_history(self):
        super().reset_history()
        # Clear persisted history by removing the file
        path = self._history_file_path
        if path:
            path.unlink(missing_ok=True)

    def _populate_history_with_non_system_prompts(
        self, history: LFAgentHistory, project_config: LlamaFarmConfig
    ):
        prompts = self._get_prompts_for_model(self.model_name)
        for prompt in prompts:
            # Only add non-system prompts to the history
            if prompt.role != "system":
                history.add_message(prompt)

    def _get_history(self, project_config: LlamaFarmConfig) -> LFAgentHistory:
        history = LFAgentHistory()
        self._populate_history_with_non_system_prompts(history, project_config)
        return history

    def _get_prompts_for_model(self, model_name: str) -> list[LFAgentChatMessage]:
        model_config = ModelService.get_model(self._project_config, model_name)
        provider = RuntimeService.get_provider(model_config)
        Client = provider.get_client().__class__

        if model_config.prompts:
            prompts = [
                prompt
                for prompt in (self._project_config.prompts or [])
                if getattr(prompt, "name", None) in (model_config.prompts or [])
            ]
        else:
            prompts = self._project_config.prompts or []

        return [
            message
            for prompt in prompts
            for message in Client.prompt_to_message(prompt)
        ]

    @property
    def _history_file_path(self) -> Path | None:
        if not self._persist_enabled or not self._session_id:
            return None
        base_dir = Path(self._project_dir)
        sessions_dir = base_dir / "sessions" / self._session_id
        try:
            sessions_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort; if mkdir fails, disable persistence
            logger.warning(
                "Failed to create sessions directory",
                path=str(sessions_dir),
                exc_info=True,
            )
            return None
        return sessions_dir / "history.json"

    def _restore_persisted_history(self) -> None:
        path = self._history_file_path
        if not path:
            return
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning(
                "Failed to read/parse history file", path=str(path), exc_info=True
            )
            return

        # Add messages into history in order
        items = data if isinstance(data, list) else []
        for item in items:
            try:
                role = item.get("role")
                content = item.get("content", "")
                if not role or not isinstance(content, str):
                    continue
                self.history.add_message(
                    message=LFAgentChatMessage(
                        role=role,
                        content=content,
                    )
                )
            except Exception:
                # Skip malformed entries defensively
                continue

    def _persist_history(self) -> None:
        path = self._history_file_path
        if not path:
            return
        try:
            history = self.history.get_history()
            tmp_path = Path(str(path) + ".tmp")
            tmp_path.write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        except Exception:
            logger.warning(
                "Failed to persist history",
                path=str(path) if path else None,
                exc_info=True,
            )


class ChatOrchestratorAgentFactory:
    @staticmethod
    async def create_agent(
        *,
        project_config: LlamaFarmConfig,
        project_dir: str,
        model_name: str | None = None,
        session_id: str | None = None,
    ) -> LFAgent:
        agent = ChatOrchestratorAgent(
            project_config=project_config,
            project_dir=project_dir,
            model_name=model_name,
        )
        if session_id:
            agent.enable_persistence(session_id=session_id)

        if project_config.mcp and project_config.mcp.servers:
            await agent.enable_mcp()

        return agent
