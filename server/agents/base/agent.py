from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict, Field

from agents.base.clients.client import LFAgentClient
from agents.base.types import StreamEvent, ToolDefinition
from core.logging import FastAPIStructLogger

from .context_provider import LFAgentContextProvider
from .history import LFAgentChatMessage, LFAgentHistory
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

    def __init__(self, config: LFAgentConfig):
        self.history = config.history
        self._system_prompt_generator = config.system_prompt_generator
        self._client = config.client

    async def run_async(
        self,
        *,
        user_input: LFAgentChatMessage | None = None,
    ) -> str:
        if user_input:
            self.history.add_message(user_input)

        messages = self._prepare_messages()
        return await self._client.chat(messages=messages)  # type: ignore[attr-defined]

    async def run_async_stream(
        self,
        *,
        user_input: LFAgentChatMessage | None = None,
    ) -> AsyncGenerator[str, None]:
        if user_input:
            self.history.add_message(user_input)
        messages = self._prepare_messages()

        async for chunk in self._client.stream_chat(  # type: ignore[attr-defined]
            messages=messages
        ):
            yield chunk

    async def stream_chat_with_tools(
        self,
        *,
        user_input: LFAgentChatMessage | None = None,
        tools: list[ToolDefinition],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream chat with tool calling support.

        This method is provider-agnostic. The client handles:
        - How tools are injected (API param vs system prompt)
        - How tool calls are detected (native vs JSON)

        Args:
            user_input: Optional user message to add to history
            tools: Available tools for the LLM to use

        Yields:
            StreamEvent: Content chunks or tool call requests
        """
        if user_input:
            self.history.add_message(user_input)

        # Prepare messages (system prompt + history)
        messages = self._prepare_messages()

        # Delegate to client - it handles provider-specific logic
        async for event in self._client.stream_chat_with_tools(
            messages=messages, tools=tools
        ):
            yield event

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

    def _prepare_messages(self) -> list[LFAgentChatMessage]:
        messages: list[LFAgentChatMessage] = []
        system_prompt = self._system_prompt_generator.generate_prompt()
        if system_prompt:
            messages.append(LFAgentChatMessage(role="system", content=system_prompt))

        for message in self.history.history:
            messages.append(
                LFAgentChatMessage(role=message.role, content=message.content)
            )

        return messages
