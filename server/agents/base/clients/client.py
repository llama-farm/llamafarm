from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from config.datamodel import Model, Prompt

from agents.base.history import LFAgentChatMessage
from agents.base.types import StreamEvent, ToolDefinition


class LFAgentClient(ABC):
    """Abstract base class for LLM clients.

    Each implementation handles tool calling in their own way:
    - OpenAI: Uses native `tools` parameter
    - Ollama: Injects tools into system prompt, detects JSON
    - Others: Whatever works for that provider

    All implementations must return the same StreamEvent format.
    """

    _model_name: str
    _model_config: Model

    def __init__(self, *, model_config: Model):
        self._model_name = model_config.name
        self._model_config = model_config

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def prompt_to_message(prompt: Prompt) -> list[LFAgentChatMessage]:
        """
        Converts a llamafarm Prompt set into a list of LFAgentChatMessages.
        """
        return [
            LFAgentChatMessage(role=message.role, content=message.content)  # type: ignore
            for message in prompt.messages
        ]

    @abstractmethod
    async def chat(self, *, messages: list[LFAgentChatMessage]) -> str:
        """Simple chat without tool calling support (for backwards compatibility)."""
        pass

    @abstractmethod
    async def stream_chat(
        self, *, messages: list[LFAgentChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Stream chat without tool calling support (for backwards compatibility)."""
        ...
        # Async generator - implementations should use async def with yield
        yield  # type: ignore

    @abstractmethod
    async def stream_chat_with_tools(
        self,
        *,
        messages: list[LFAgentChatMessage],
        tools: list[ToolDefinition],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream chat with tool calling support.

        The implementation is responsible for:
        1. Injecting tools (via API param, system prompt, etc.)
        2. Detecting tool call requests in response
        3. Yielding StreamEvent objects in consistent format

        Args:
            messages: Conversation history
            tools: Available tools (provider-agnostic format)

        Yields:
            StreamEvent: Either content chunks or tool call requests
        """
        ...
        # Async generator - implementations should use async def with yield
        # Type checkers: return type is AsyncGenerator when async def uses yield
        yield  # type: ignore
