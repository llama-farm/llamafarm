import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

from atomic_agents import BaseTool  # type: ignore
from config.datamodel import LlamaFarmConfig
from openai.types.chat import ChatCompletionMessageFunctionToolCallParam
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function,
)

from agents.base.agent import LFAgent, LFAgentConfig
from agents.base.clients.client import LFChatCompletion, LFChatCompletionChunk
from agents.base.history import (
    LFAgentHistory,
    LFChatCompletionAssistantMessageParam,
    LFChatCompletionMessageParam,
    LFChatCompletionToolMessageParam,
)
from agents.base.system_prompt_generator import LFAgentSystemPromptGenerator
from agents.base.types import ToolDefinition
from core.logging import FastAPIStructLogger
from core.mcp_registry import register_mcp_service
from services.mcp_service import MCPService
from services.model_service import ModelService
from services.prompt_service import PromptService  # type: ignore
from services.runtime_service.runtime_service import RuntimeService
from tools.mcp_tool.tool.mcp_tool_factory import MCPToolFactory

logger = FastAPIStructLogger(__name__)

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
    _mcp_tools: list[type[BaseTool]] = []

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
            prompts=self._get_prompt_messages_for_model(model_config.name)
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

    async def run_async(
        self,
        user_input: LFChatCompletionMessageParam | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> LFChatCompletion:
        """Run the agent with MCP tool calling support.

        The agent will:
        1. Get response from LLM
        2. Check if response requests a tool call
        3. Execute the tool and feed result back to LLM
        4. Repeat until LLM provides final answer
        """
        iteration = 0
        tools = [ToolDefinition.from_mcp_tool(t) for t in self._mcp_tools]

        final_response: LFChatCompletion | None = None

        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1

            try:
                # Get LLM response
                response = await super().run_async(user_input=user_input, tools=tools)

                assistant_message = response.choices[0].message
                tool_calls = assistant_message.tool_calls

                # FIXME: only handle one tool call for now
                tool_call = tool_calls[0] if tool_calls else None
                if not tool_call:
                    final_response = response
                    break

                if tool_call.type != "function":
                    logger.warning(
                        "Model returned a tool call of type %s, but we only support function tool calls",
                        tool_call.type,
                    )
                    final_response = response
                    break

                # Save the assistant tool call message to history
                self.history.add_message(
                    LFChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=assistant_message.content,
                        tool_calls=[
                            ChatCompletionMessageFunctionToolCallParam(
                                type="function",
                                id=tool_call.id,
                                function=Function(
                                    name=tool_call.function.name,
                                    arguments=tool_call.function.arguments,
                                ),
                            )
                            for tool_call in tool_calls or []
                            if hasattr(tool_call, "function")
                        ],
                    )
                )

                result = await self._execute_mcp_tool(
                    tool_call.function.name, tool_call.function.arguments
                )

                self.history.add_message(
                    LFChatCompletionToolMessageParam(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                    )
                )

                user_input = None

            except Exception as e:
                logger.error("Error in orchestrator loop", exc_info=True)
                raise e

        if final_response:
            self.history.add_message(
                LFChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=final_response.choices[0].message.content,
                    # reasoning=final_response.choices[0].message.reasoning,
                )
            )
            self._persist_history_safe()
            return final_response

        # Max iterations reached
        logger.warning("Max iterations reached in orchestrator")

        return LFChatCompletion(
            id=f"chat-{uuid.uuid4()}",
            created=int(time.time()),
            model=self.model_name,
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=MAX_ITERATIONS_MESSAGE,
                    ),
                    finish_reason="stop",
                )
            ],
        )

    async def run_async_stream(
        self,
        user_input: LFChatCompletionMessageParam | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncGenerator[LFChatCompletionChunk]:
        """Stream chat with MCP tool execution support."""

        # Convert MCP tools to ToolDefinition format
        tools = [ToolDefinition.from_mcp_tool(t) for t in self._mcp_tools]

        iteration = 0
        current_input = user_input
        done = False

        while not done and iteration < MAX_TOOL_ITERATIONS:
            iteration += 1

            # Stream chat with tools
            accumulated_content = ""  # Accumulate chunks for history
            accumulated_reasoning = ""  # Accumulate chunks for reasoning

            async for chunk in super().run_async_stream(
                user_input=current_input, tools=tools
            ):
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    accumulated_content += delta.content

                if hasattr(delta, "reasoning"):
                    accumulated_reasoning += delta.reasoning

                tool_call = delta.tool_calls[0] if delta.tool_calls else None
                if not tool_call:
                    if choice.finish_reason == "stop":
                        done = True
                        yield chunk
                        continue
                    else:
                        yield chunk
                        continue

                if tool_call.type != "function":
                    logger.warning(
                        "Model returned a tool call of type %s, but we only support function tool calls",
                        tool_call.type,
                    )
                    yield chunk
                    continue

                if not (
                    tool_call.function
                    and tool_call.function.name
                    and tool_call.function.arguments
                    and tool_call.id
                ):
                    logger.warning("Tool call function missing required fields")
                    yield chunk
                    continue

                logger.info(
                    "Executing MCP tool",
                    tool_name=tool_call.function.name,
                    iteration=iteration,
                )

                tool_call_message = LFChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=choice.delta.content,
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCallParam(
                            type="function",
                            id=tool_call.id or f"call_{uuid.uuid4()}",
                            function=Function(
                                name=tool_call.function.name,  # type: ignore
                                arguments=tool_call.function.arguments,  # type: ignore
                            ),
                        )
                        for tool_call in choice.delta.tool_calls or []
                        if hasattr(tool_call, "function")
                    ],
                )
                self.history.add_message(tool_call_message)
                # Yield the tool call chunk - CLI will handle rendering
                yield chunk

                # Execute the MCP tool
                result = await self._execute_mcp_tool(
                    tool_call.function.name, tool_call.function.arguments
                )

                # Add tool call and result to history
                self.history.add_message(
                    LFChatCompletionToolMessageParam(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                    )
                )

                # Prepare for next iteration
                current_input = None  # No need to pass input to next iteration
                break  # Exit event loop, continue while loop

        # Add final accumulated content to history
        if accumulated_content:
            self.history.add_message(
                LFChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=accumulated_content,
                    # reasoning=accumulated_reasoning, # TODO: we'll need to adjust types to support saving this
                )
            )

        # Save history
        if iteration >= MAX_TOOL_ITERATIONS:
            logger.warning("Max iterations reached", max_iterations=MAX_TOOL_ITERATIONS)

            yield LFChatCompletionChunk(
                id=f"chat-{uuid.uuid4()}",
                created=int(time.time()),
                model=self.model_name,
                object="chat.completion.chunk",
                choices=[
                    ChoiceChunk(
                        index=0,
                        delta=ChoiceDelta(
                            role="assistant",
                            content=MAX_ITERATIONS_MESSAGE,
                        ),
                        finish_reason="stop",
                    )
                ],
            )
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

    async def setup_tools(self):
        """
        Setup tools that the agent can use.

        For now, this only pertains to tools associated with MCP servers.
        In the future, we may support custom tool definitions through the
        project config.
        """
        await self.enable_mcp()

    async def enable_mcp(self):
        """Enable MCP tool calling support."""
        if self._mcp_enabled:
            return
        self._mcp_service = MCPService(self._project_config, self.model_name)
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

    async def _execute_mcp_tool(self, tool_name: str, arguments: str | None) -> str:
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
            tool_input = input_schema_class(
                tool_name=tool_name, **json.loads(arguments or "{}")
            )

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
        prompts = self._get_prompt_messages_for_model(self.model_name)
        for prompt in prompts:
            # Only add non-system prompts to the history
            if prompt.get("role") != "system":
                history.add_message(prompt)

    def _get_history(self, project_config: LlamaFarmConfig) -> LFAgentHistory:
        history = LFAgentHistory()
        self._populate_history_with_non_system_prompts(history, project_config)
        return history

    def _get_prompt_messages_for_model(
        self, model_name: str
    ) -> list[LFChatCompletionMessageParam]:
        model_config = ModelService.get_model(self._project_config, model_name)
        provider = RuntimeService.get_provider(model_config)
        ClientClass = provider.get_client().__class__

        messages = PromptService.resolve_prompts_for_model(
            self._project_config, model_config
        )

        return [
            ClientClass.prompt_message_to_chat_completion_message(message)
            for message in messages
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
        for item in data:
            self.history.add_message(LFAgentHistory.message_from_dict(item))

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

        await agent.setup_tools()

        return agent
