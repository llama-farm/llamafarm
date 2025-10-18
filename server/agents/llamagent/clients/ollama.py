from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from config.datamodel import Prompt
from ollama import AsyncClient, Message

from ..history import LFAgentChatMessage
from ..tool import LFAgentToolDefinition
from .client import LFAgentClient


class ABCToolsProvider(ABC):
    @abstractmethod
    def get_tool_definitions(self) -> list[LFAgentToolDefinition]:
        pass


class LFAgentClientOllama(LFAgentClient):
    async def chat(self, *, messages: list[LFAgentChatMessage]) -> str:
        # Call the existing stream_chat and return the accumulated content
        content = ""
        async for chunk in self.stream_chat(messages=messages):
            content += chunk
        return content

    async def stream_chat(  # type: ignore[override]
        self, *, messages: list[LFAgentChatMessage]
    ) -> AsyncGenerator[str, None]:
        client = AsyncClient(
            host=(
                self._model_config.base_url.rstrip("/v1")
                if self._model_config.base_url
                else ""
            ),
        )
        response_stream = await client.chat(
            model=self._model_config.model,
            messages=[self._message_to_openai_message(message) for message in messages],
            **(self._model_config.model_api_parameters or {}),
            stream=True,
        )

        async for partial_response in response_stream:
            # Try to extract content from the streamed message
            message_content = partial_response.message.content
            if message_content:
                yield message_content

    @staticmethod
    def prompt_to_message(prompt: Prompt) -> LFAgentChatMessage:
        return LFAgentChatMessage(role="system", content=prompt.content)

    def _message_to_openai_message(self, message: LFAgentChatMessage) -> Message:
        match message.role:
            case "system":
                return Message(role="system", content=message.content)
            case "user":
                return Message(role="user", content=message.content)
            case "assistant":
                return Message(role="assistant", content=message.content)
            case "developer":
                return Message(role="developer", content=message.content)
            case "function":
                return Message(role="function", content=message.content)
            case _:
                raise ValueError(f"Unknown message role: {message.role}")
