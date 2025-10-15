import json
import os
from pathlib import Path
from typing import Any

import instructor
from atomic_agents import BaseIOSchema  # type: ignore
from atomic_agents.agents.atomic_agent import (  # type: ignore
    ChatHistory,
    SystemPromptGenerator,
)
from config.datamodel import (
    LlamaFarmConfig,
    Model,
    Prompt,
    PromptFormat,
)
from openai import AsyncOpenAI
from pydantic import Field

from agents.agent import LFAgent, LFAgentConfig
from agents.mcp_orchestrator import (
    FinalResponseSchema,
    MCPOrchestrator,
    MCPOrchestratorFactory,
)
from context_providers.docs_context_provider import DocsContextProvider
from context_providers.mcp_tools_context_provider import MCPToolsContextProvider
from core.logging import FastAPIStructLogger
from services import runtime_service

logger = FastAPIStructLogger(__name__)


class ProjectChatOrchestratorAgentInputSchema(BaseIOSchema):
    """
    Input schema for the project chat orchestrator agent.
    """

    chat_message: str


class ProjectChatOrchestratorAgentOutputSchema(BaseIOSchema):
    """Output schema for the project chat orchestrator agent.

    This schema is intentionally simple to ensure compatibility
    with Ollama's JSON parsing.
    """

    chat_message: str


class ToolCallDecision(BaseIOSchema):
    """Decision about whether to call a tool and which one."""

    should_call_tool: bool = Field(
        default=False,
        description="Whether a tool should be called to answer the user's query",
    )
    tool_name: str | None = Field(
        default=None,
        description="Name of the tool to call (if should_call_tool is True)",
    )
    tool_arguments: dict[str, Any] | None = Field(
        default=None,
        description="Arguments to pass to the tool (if should_call_tool is True)",
    )
    reasoning: str | None = Field(
        default=None,
        description="Brief explanation of why this tool was chosen",
    )


class OrchestratorOutputSchema(BaseIOSchema):
    """Output schema for orchestrator mode with MCP tools.

    The agent decides whether to call a tool or respond directly.
    """

    tool_call: ToolCallDecision | None = Field(
        default=None,
        description="Tool call decision (if agent wants to use a tool)",
    )
    chat_message: str = Field(
        default="",
        description=(
            "Direct response to user (if not calling a tool), or "
            "explanation of tool results (after tool execution)"
        ),
    )


class ProjectChatOrchestratorAgent(LFAgent):
    """Project chat orchestrator agent that uses MCP tools with atomic-agents orchestrator pattern."""

    def __init__(
        self,
        project_config: LlamaFarmConfig,
        project_dir: str,
        model_name: str | None = None,
    ):
        # Resolve model configuration
        from services.model_service import ModelService

        logger.info(
            "Creating ProjectChatOrchestratorAgent",
            model_name=model_name,
            project=project_config.name,
        )
        model_config = ModelService.get_model(project_config, model_name)
        logger.info(
            "Resolved model configuration",
            model_name=model_config.name,
            provider=model_config.provider.value,
            model=model_config.model,
            base_url=model_config.base_url,
        )

        # Build base history from config
        history = _get_history(project_config)
        client = _get_client_for_model(model_config)

        lf_config = LFAgentConfig(
            client=client,
            model=model_config.model,
            history=history,
            system_prompt_generator=LFSystemPromptGenerator(
                project_config=project_config
            ),
            model_api_parameters=model_config.model_api_parameters,
        )

        super().__init__(config=lf_config)

        # Session-scoped persistence context
        self._namespace = project_config.namespace
        self._project_id = project_config.name
        self._project_dir = project_dir
        self._persist_enabled = False
        self._project_config = project_config

        self.model_name = model_config.model  # Store model name for API responses

        # Register docs context provider
        self.docs_context_provider = DocsContextProvider(title="Relevant Documentation")
        self.register_context_provider("docs", self.docs_context_provider)

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

    def reset_history(self):
        super().reset_history()
        # Clear persisted history by removing the file
        path = self._history_file_path()
        if path:
            path.unlink(missing_ok=True)

    # -------------------- Persistence helpers --------------------
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
        path = self._history_file_path()
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
                if role == "user":
                    content_instance = self.input_schema.model_validate_json(content)
                elif role == "assistant":
                    content_instance = self.output_schema.model_validate_json(content)
                else:
                    # Skip system or unknown roles;
                    # system prompts are handled separately
                    continue
                self.history.add_message(role, content_instance)
            except Exception:
                # Skip malformed entries defensively
                continue

    def _persist_history(self) -> None:
        path = self._history_file_path()
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

    # -------------------- Execution overrides --------------------
    async def run_async(self, user_input):
        """Run with orchestrator pattern if MCP tools are loaded."""

        if self._project_config.mcp and self._project_config.mcp.servers:
            # Create orchestrator with same config as this agent
            orchestrator_config = LFAgentConfig(
                client=self.client,
                model=self.model,
                history=self.history,
                system_prompt_generator=self.system_prompt_generator,
                model_api_parameters=self.model_api_parameters,
            )

            mcp_orchestrator = await MCPOrchestratorFactory.create_agent(
                config=orchestrator_config,
                project_config=(self._project_config),
            )

            # Run orchestrator and convert result to our output schema
            orchestrator_result = await mcp_orchestrator.run_async(user_input)

            # Convert FinalResponseSchema to our output schema
            if isinstance(orchestrator_result, FinalResponseSchema):
                response = ProjectChatOrchestratorAgentOutputSchema(
                    chat_message=orchestrator_result.chat_message
                )
            else:
                # Fallback: extract chat_message if available
                chat_msg = getattr(
                    orchestrator_result,
                    "chat_message",
                    str(orchestrator_result),
                )
                response = ProjectChatOrchestratorAgentOutputSchema(
                    chat_message=chat_msg
                )

            try:
                self._persist_history()
            except Exception:
                logger.warning(
                    "History persistence failed after orchestrator run",
                    exc_info=True,
                )

            return response

        # No MCP tools - use normal flow
        response = await super().run_async(user_input)
        try:
            self._persist_history()
        except Exception:
            logger.warning("History persistence failed after run_async", exc_info=True)
        return response

    async def run_async_stream(self, user_input):
        """Stream with orchestrator pattern if MCP tools are loaded."""
        # For now, MCPOrchestrator agent doesn't support streaming
        # Fall back to non-streaming execution and yield the final result
        if self._project_config.mcp and self._project_config.mcp.servers:
            # Create orchestrator with same config as this agent
            orchestrator_config = LFAgentConfig(
                client=self.client,
                model=self.model,
                history=self.history,
                system_prompt_generator=self.system_prompt_generator,
                model_api_parameters=self.model_api_parameters,
            )

            mcp_orchestrator = await MCPOrchestratorFactory.create_agent(
                config=orchestrator_config,
                project_config=(self._project_config),
            )

            # Run orchestrator and convert result
            orchestrator_result = await mcp_orchestrator.run_async(user_input)

            # Convert to our output schema
            if isinstance(orchestrator_result, FinalResponseSchema):
                response = ProjectChatOrchestratorAgentOutputSchema(
                    chat_message=orchestrator_result.chat_message
                )
            else:
                chat_msg = getattr(
                    orchestrator_result,
                    "chat_message",
                    str(orchestrator_result),
                )
                response = ProjectChatOrchestratorAgentOutputSchema(
                    chat_message=chat_msg
                )

            try:
                self._persist_history()
            except Exception:
                logger.warning(
                    "History persistence failed after orchestrator stream",
                    exc_info=True,
                )

            yield response
            return

        # No MCP tools - use normal streaming
        async for chunk in super().run_async_stream(user_input):
            yield chunk
        try:
            self._persist_history()
        except Exception:
            logger.warning(
                "History persistence failed after run_async_stream", exc_info=True
            )


class LFSystemPromptGenerator(SystemPromptGenerator):
    def __init__(self, project_config: LlamaFarmConfig):
        logger.info(f"Project config: {project_config}")
        self.system_prompts = [
            prompt
            for prompt in (project_config.prompts or [])
            if prompt.role == "system"
        ]
        super().__init__()

    def generate_prompt(self) -> str:
        # return "\nYou are a helpful assistant that can answer "
        # "questions and help with tasks."
        prompt_parts = []
        for prompt in self.system_prompts:
            prompt_parts.append(prompt.content)
            prompt_parts.append("")

        if self.context_providers:
            prompt_parts.append("# EXTRA INFORMATION AND CONTEXT")
            for provider in self.context_providers.values():
                info = provider.get_info()
                if info:
                    prompt_parts.append(f"## {provider.title}")
                    prompt_parts.append(info)
                    prompt_parts.append("")

        return "\n".join(prompt_parts)


def _prompt_to_content_schema(prompt: Prompt) -> BaseIOSchema:
    if prompt.role == "assistant":
        return ProjectChatOrchestratorAgentOutputSchema(
            chat_message=prompt.content,
        )
    elif prompt.role == "user":
        return ProjectChatOrchestratorAgentInputSchema(
            chat_message=prompt.content,
        )
    else:
        raise ValueError(f"Unsupported role: {prompt.role}")


def _populate_history_with_non_system_prompts(
    history: ChatHistory, project_config: LlamaFarmConfig
):
    for prompt in project_config.prompts or []:
        # Only add non-system prompts to the history
        if prompt.role != "system":
            history.add_message(
                role=prompt.role,
                content=_prompt_to_content_schema(prompt),
            )


def _get_history(project_config: LlamaFarmConfig) -> ChatHistory:
    history = ChatHistory()
    _populate_history_with_non_system_prompts(history, project_config)
    return history


def _get_client_for_model(
    model_config: Model,
) -> AsyncOpenAI | instructor.client.AsyncInstructor:
    """Get client for a specific model configuration using provider registry.

    Args:
        model_config: ModelConfig instance from ModelService

    Returns:
        AsyncOpenAI client (possibly instructor-wrapped)
    """
    provider = runtime_service.get_provider(model_config.provider, model_config)
    return provider.get_client()


class ProjectChatOrchestratorAgentFactory:
    @staticmethod
    async def create_agent(
        project_config: LlamaFarmConfig,
        project_dir: str,
        model_name: str | None = None,
        session_id: str | None = None,
    ) -> ProjectChatOrchestratorAgent:
        from services.model_service import ModelService

        # Get model config for logging
        model_config = ModelService.get_model(project_config, model_name)
        pf = model_config.prompt_format or PromptFormat.unstructured
        selected_name = model_config.name

        logger.info(
            "Creating chat agent",
            prompt_format=pf.value if pf else "unstructured",
            model=model_config.model,
            model_name=selected_name,
            provider=model_config.provider.value,
        )

        agent = ProjectChatOrchestratorAgent(
            project_config, project_dir=project_dir, model_name=model_name
        )

        # Enable session persistence if session_id provided
        if session_id:
            agent.enable_persistence(session_id=session_id)

        return agent
