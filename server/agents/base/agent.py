from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict, Field

from agents.base.clients.client import (
    LFAgentClient,
    LFChatCompletion,
    LFChatCompletionChunk,
)
from agents.base.types import ToolDefinition
from core.logging import FastAPIStructLogger

from .context_provider import LFAgentContextProvider
from .history import (
    LFAgentHistory,
    LFChatCompletionMessageParam,
    LFChatCompletionSystemMessageParam,
)
from .system_prompt_generator import LFAgentSystemPromptGenerator

logger = FastAPIStructLogger(__name__)


class LFAgentConfig(BaseModel):
    client: LFAgentClient = Field(..., description="The client for the agent")
    history: LFAgentHistory = Field(..., description="The history of the agent")
    system_prompt_generator: LFAgentSystemPromptGenerator = Field(
        ..., description="The system prompt generator for the agent"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LFAgent:
    history: LFAgentHistory
    _system_prompt_generator: LFAgentSystemPromptGenerator
    _client: LFAgentClient
    _request_system_prompt: str | None

    def __init__(self, config: LFAgentConfig):
        self.history = config.history
        self._system_prompt_generator = config.system_prompt_generator
        self._client = config.client
        self._request_system_prompt = None

    @property
    def config_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition.from_datamodel_tool(t)
            for t in self._client._model_config.tools or []
        ]

    async def run_async(
        self,
        *,
        messages: list[LFChatCompletionMessageParam] | None = None,
        tools: list[ToolDefinition] | None = None,
        extra_body: dict | None = None,
    ) -> LFChatCompletion:
        if messages is not None:
            self._request_system_prompt = None
            for message in messages:
                if message.get("role") == "system":
                    content = message.get("content")
                    self._request_system_prompt = (
                        content if isinstance(content, str) else ""
                    )
                else:
                    self.history.add_message(message)

        messages = self._prepare_messages()

        # Combine config tools with extra tools
        tools = self.config_tools + (tools or [])

        return await self._client.chat(
            messages=messages, tools=tools, extra_body=extra_body
        )

    async def run_async_stream(
        self,
        *,
        messages: list[LFChatCompletionMessageParam] | None = None,
        tools: list[ToolDefinition] | None = None,
        extra_body: dict | None = None,
    ) -> AsyncGenerator[LFChatCompletionChunk]:
        if messages is not None:
            self._request_system_prompt = None
            for message in messages:
                if message.get("role") == "system":
                    content = message.get("content")
                    self._request_system_prompt = (
                        content if isinstance(content, str) else ""
                    )
                else:
                    self.history.add_message(message)
        messages = self._prepare_messages()

        # Combine config tools with extra tools
        tools = self.config_tools + (tools or [])

        async for chunk in self._client.stream_chat(
            messages=messages, tools=tools, extra_body=extra_body
        ):
            yield chunk

    def register_context_provider(
        self, title: str, context_provider: LFAgentContextProvider
    ):
        if self._system_prompt_generator.context_providers.get(title):
            raise ValueError(f"Context provider already registered: {title}")
        self._system_prompt_generator.context_providers[title] = context_provider

    def get_context_provider(self, title: str) -> LFAgentContextProvider | None:
        return self._system_prompt_generator.context_providers.get(title, None)

    def remove_context_provider(self, title: str):
        self._system_prompt_generator.context_providers.pop(title, None)

    def reset_history(self):
        """Reset the agent's conversation history."""
        self.history.history.clear()

    def _get_context_provider_text(self) -> str:
        """Get context provider text (RAG, etc.) without config system prompts."""
        parts: list[str] = []
        providers = self._system_prompt_generator.context_providers
        if providers:
            parts.append("# EXTRA INFORMATION AND CONTEXT")
            for provider in providers.values():
                info = provider.get_info()
                if info:
                    parts.append(f"## {provider.title}")
                    parts.append(info)
                    parts.append("")
        return "\n".join(parts)

    def _prepare_messages(self) -> list[LFChatCompletionMessageParam]:
        messages: list[LFChatCompletionMessageParam] = []

        if self._request_system_prompt is not None:
            # API system prompt overrides config system prompts
            system_content = self._request_system_prompt
            context_suffix = self._get_context_provider_text()
            if context_suffix:
                system_content += "\n" + context_suffix
            messages.append(
                LFChatCompletionSystemMessageParam(
                    role="system", content=system_content
                )
            )
        else:
            # No API override — use config system prompt
            system_prompt = self._system_prompt_generator.generate_prompt()
            if system_prompt:
                messages.append(
                    LFChatCompletionSystemMessageParam(
                        role="system", content=system_prompt
                    )
                )

        for message in self.history.history:
            # Serialize messages to ensure proper JSON-compatible format
            # This handles OpenAI SDK types (Pydantic models) that may not
            # serialize correctly when passed through the API client
            serialized = LFAgentHistory._serialize_message(message)
            messages.append(serialized)

        return messages
