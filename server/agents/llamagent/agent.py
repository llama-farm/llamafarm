from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict, Field

from agents.llamagent.clients.client import LlamAgentClient
from core.logging import FastAPIStructLogger

from .context_provider import LlamAgentContextProvider
from .history import LlamAgentChatMessage, LlamAgentHistory
from .system_prompt_generator import LlamAgentSystemPromptGenerator

logger = FastAPIStructLogger(__name__)


class LlamAgentConfig(BaseModel):
    client: LlamAgentClient = Field(..., description="The client for the agent")
    history: LlamAgentHistory = Field(..., description="The history of the agent")
    system_prompt_generator: LlamAgentSystemPromptGenerator = Field(
        ..., description="The system prompt generator for the agent"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LlamAgent:
    history: LlamAgentHistory
    _system_prompt_generator: LlamAgentSystemPromptGenerator
    _client: LlamAgentClient

    def __init__(self, config: LlamAgentConfig):
        self.history = config.history
        self._system_prompt_generator = config.system_prompt_generator
        self._client = config.client

    async def run_async(
        self,
        *,
        user_input: LlamAgentChatMessage | None = None,
    ) -> str:
        if user_input:
            self.history.add_message(user_input)

        messages = self._prepare_messages()
        return await self._client.chat(messages=messages)

    async def run_async_stream(
        self,
        *,
        user_input: LlamAgentChatMessage | None = None,
    ) -> AsyncGenerator[str, None]:
        if user_input:
            self.history.add_message(user_input)
        messages = self._prepare_messages()

        async for chunk in self._client.stream_chat(messages=messages):
            yield chunk

    def register_context_provider(
        self, title: str, context_provider: LlamAgentContextProvider
    ):
        if self._system_prompt_generator.context_providers.get(title):
            raise ValueError(f"Context provider already registered: {title}")
        self._system_prompt_generator.context_providers[title] = context_provider

    def get_context_provider(self, title: str) -> LlamAgentContextProvider | None:
        return self._system_prompt_generator.context_providers.get(title, None)

    def remove_context_provider(self, title: str):
        self._system_prompt_generator.context_providers.pop(title, None)

    def _prepare_messages(self) -> list[LlamAgentChatMessage]:
        messages: list[LlamAgentChatMessage] = []
        system_prompt = self._system_prompt_generator.generate_prompt()
        if system_prompt:
            messages.append(LlamAgentChatMessage(role="system", content=system_prompt))

        for message in self.history.get_history():
            messages.append(
                LlamAgentChatMessage(role=message.role, content=message.content)
            )

        return messages
