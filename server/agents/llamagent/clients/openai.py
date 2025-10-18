from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from config.datamodel import Prompt
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from ..history import LlamAgentChatMessage
from ..tool import LlamAgentToolDefinition
from .client import LlamAgentClient


class ABCToolsProvider(ABC):
    @abstractmethod
    def get_tool_definitions(self) -> list[LlamAgentToolDefinition]:
        pass


class LlamAgentClientOpenAI(LlamAgentClient):
    """Some clients need to specify tools when models support native
    tool calling."""

    async def chat(self, *, messages: list[LlamAgentChatMessage]) -> str:
        # Call the existing stream_chat and return the accumulated content
        content = ""
        async for chunk in self.stream_chat(messages=messages):
            content += chunk
        return content

    async def stream_chat(  # type: ignore[override]
        self, *, messages: list[LlamAgentChatMessage]
    ) -> AsyncGenerator[str, None]:
        client = AsyncOpenAI(
            api_key=self._model_config.api_key or "",
            base_url=self._model_config.base_url or "",
        )
        response_stream = await client.chat.completions.create(
            model=self._model_config.model,
            messages=[self._message_to_openai_message(message) for message in messages],
            stream=True,
            **(self._model_config.model_api_parameters or {}),
        )

        content = ""
        async for partial_response in response_stream:
            # Try to extract content from the streamed message
            if hasattr(partial_response, "choices") and partial_response.choices:
                choice = partial_response.choices[0]
                if (
                    hasattr(choice, "delta")
                    and choice.delta
                    and hasattr(choice.delta, "content")
                ):
                    delta_content = choice.delta.content or ""
                    content += delta_content
                    # Yield just the delta content string
                    if delta_content:
                        yield delta_content
                elif (
                    hasattr(choice, "message")
                    and choice.message
                    and hasattr(choice.message, "content")
                ):
                    # Some APIs may use message.content in streaming
                    message_content = choice.message.content or ""
                    content += message_content
                    if message_content:
                        yield message_content

    @staticmethod
    def prompt_to_message(prompt: Prompt) -> LlamAgentChatMessage:
        return LlamAgentChatMessage(role="system", content=prompt.content)

    def _message_to_openai_message(
        self, message: LlamAgentChatMessage
    ) -> ChatCompletionMessageParam:
        match message.role:
            case "system":
                return ChatCompletionSystemMessageParam(
                    role="system", content=message.content
                )
            case "user":
                return ChatCompletionUserMessageParam(
                    role="user", content=message.content
                )
            case "assistant":
                return ChatCompletionAssistantMessageParam(
                    role="assistant", content=message.content
                )
            case "tool":
                return ChatCompletionToolMessageParam(
                    role="tool", tool_call_id=message.content, content=message.content
                )
            case "developer":
                return ChatCompletionDeveloperMessageParam(
                    role="developer", content=message.content
                )
            case _:
                raise ValueError(f"Unknown message role: {message.role}")
