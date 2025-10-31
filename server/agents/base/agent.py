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

TOOLS_SYSTEM_MESSAGE_PREFIX = """

You may call one or more tools to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""

TOOLS_SYSTEM_MESSAGE_SUFFIX = """
</tools>
For each tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>.
If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.
"""


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
        user_input: LFChatCompletionMessageParam | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> LFChatCompletion:
        if user_input:
            self.history.add_message(user_input)

        messages = self._prepare_messages()
        return await self._client.chat(messages=messages, tools=tools)

    async def run_async_stream(
        self,
        *,
        user_input: LFChatCompletionMessageParam | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncGenerator[LFChatCompletionChunk]:
        if user_input:
            self.history.add_message(user_input)
        messages = self._prepare_messages()

        async for chunk in self._client.stream_chat(messages=messages, tools=tools):
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

    def _prepare_messages(self) -> list[LFChatCompletionMessageParam]:
        messages: list[LFChatCompletionMessageParam] = []
        system_prompt = self._system_prompt_generator.generate_prompt()
        if system_prompt:
            messages.append(
                LFChatCompletionSystemMessageParam(role="system", content=system_prompt)
            )

        for message in self.history.history:
            messages.append(message)

        return messages
