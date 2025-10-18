import json
import os
from collections.abc import AsyncGenerator
from pathlib import Path

from config.datamodel import LlamaFarmConfig, Provider

from agents.llamagent.agent import LFAgent, LFAgentConfig
from agents.llamagent.clients.openai import LFAgentClientOpenAI
from agents.llamagent.history import LFAgentChatMessage, LFAgentHistory
from agents.llamagent.system_prompt_generator import LFAgentSystemPromptGenerator
from context_providers.mcp_tools_context_provider import MCPToolsContextProvider
from core.logging import FastAPIStructLogger
from services.mcp_service import MCPService
from services.model_service import ModelService
from services.runtime_service.runtime_service import RuntimeService
from tools.mcp_tool.tool.mcp_tool_factory import BaseTool, MCPToolFactory

logger = FastAPIStructLogger(__name__)

CLIENT_CLASSES = {
    Provider.openai: LFAgentClientOpenAI,
    Provider.ollama: LFAgentClientOpenAI,
    Provider.lemonade: LFAgentClientOpenAI,
}


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
        self.model_name = (
            model_name or ModelService.get_model(project_config, model_name).name
        )

        history = self._get_history(project_config)

        model_config = ModelService.get_model(project_config, self.model_name)
        provider = RuntimeService.get_provider(model_config)
        client = provider.get_client()

        system_prompt_generator = LFAgentSystemPromptGenerator(
            prompts=self._get_prompts_for_model(self.model_name)
        )
        config = LFAgentConfig(
            history=history,
            system_prompt_generator=system_prompt_generator,
            client=client,
        )

        super().__init__(config=config)

    async def run_async(self, user_input: LFAgentChatMessage | None = None) -> str:
        """Run the agent with MCP tool calling support.

        The agent will:
        1. Get response from LLM
        2. Check if response requests a tool call
        3. Execute the tool and feed result back to LLM
        4. Repeat until LLM provides final answer
        """
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            try:
                # Get LLM response
                response = await super().run_async(user_input=user_input)

                # Try to parse as JSON to check for tool calls
                try:
                    response_data = json.loads(response)
                except (json.JSONDecodeError, TypeError):
                    # Not JSON, treat as final response
                    try:
                        self._persist_history()
                    except Exception:
                        logger.warning("History persistence failed", exc_info=True)
                    return response

                # Check if this is a tool call request
                tool_name = response_data.get("tool_name")
                tool_parameters = response_data.get("tool_parameters", {})

                if not tool_name:
                    # No tool requested, this is the final response
                    # Extract the actual message if present
                    final_message = response_data.get("message") or response
                    try:
                        self._persist_history()
                    except Exception:
                        logger.warning("History persistence failed", exc_info=True)
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

                    # Feed result back to LLM for next iteration
                    user_input = LFAgentChatMessage(
                        role="user",
                        content=(
                            f"Tool '{tool_name}' returned: {result_content}\n\n"
                            "Based on this result, provide your final answer "
                            "or call another tool if needed."
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
        final_response = (
            "I've reached the maximum number of tool calls. "
            "Please try rephrasing your request."
        )
        try:
            self._persist_history()
        except Exception:
            logger.warning("History persistence failed", exc_info=True)
        return final_response

    async def run_async_stream(
        self, user_input: LFAgentChatMessage | None = None
    ) -> AsyncGenerator[str, None]:
        # If MCP is enabled, we can't stream the response
        if self._mcp_enabled:
            response = await self.run_async(user_input=user_input)
            yield response
            return

        async for chunk in super().run_async_stream(user_input=user_input):
            yield chunk
        try:
            self._persist_history()
        except Exception:
            logger.warning(
                "History persistence failed after run_async_stream", exc_info=True
            )

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
        if self._mcp_enabled:
            return
        self._mcp_service = MCPService(self._project_config)
        self._mcp_tool_factory = MCPToolFactory(self._mcp_service)
        mcp_context_provider = MCPToolsContextProvider(title="MCP Tools")
        self.register_context_provider("mcp", mcp_context_provider)
        self._mcp_enabled = True
        await self._load_mcp_tools()

    async def _load_mcp_tools(self):
        if not self._mcp_enabled:
            await self.enable_mcp()
        self._mcp_tools = await self._mcp_tool_factory.create_all_tools()
        self.get_context_provider("mcp").set_tools(self._mcp_tools)

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

        return [
            Client.prompt_to_message(prompt)
            for prompt in self._project_config.prompts or []
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
