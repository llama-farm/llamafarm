from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Literal, TypeAlias

from config.datamodel import Model, PromptMessage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from agents.base.history import (
    LFChatCompletionAssistantMessageParam,
    LFChatCompletionMessageParam,
    LFChatCompletionSystemMessageParam,
    LFChatCompletionToolMessageParam,
    LFChatCompletionUserMessageParam,
)
from agents.base.types import ToolDefinition


class LFToolCall(BaseModel):
    type: Literal["function"] = "function"
    function: ToolDefinition


class LFChatResponse(BaseModel):
    message: ChatCompletion


LFChatCompletion: TypeAlias = ChatCompletion
LFChatCompletionChunk: TypeAlias = ChatCompletionChunk


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
    def prompt_message_to_chat_completion_message(
        message: PromptMessage,
    ) -> LFChatCompletionMessageParam:
        match message.role:
            case "system":
                return LFChatCompletionSystemMessageParam(
                    role="system", content=message.content
                )
            case "user":
                return LFChatCompletionUserMessageParam(
                    role="user", content=message.content
                )
            case "assistant":
                return LFChatCompletionAssistantMessageParam(
                    role="assistant", content=message.content
                )
            case "tool":
                return LFChatCompletionToolMessageParam(
                    role="tool",
                    content=message.content,
                    tool_call_id=message.tool_call_id or "",
                )
            case _:
                return LFChatCompletionUserMessageParam(
                    role="user", content=message.content
                )

    @abstractmethod
    async def chat(
        self,
        *,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
        extra_body: dict | None = None,
    ) -> LFChatCompletion:
        """Simple chat without tool calling support (for backwards compatibility)."""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        *,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
        extra_body: dict | None = None,
    ) -> AsyncGenerator[LFChatCompletionChunk]:
        """Stream chat with tool calling support.

        The implementation is responsible for:
        1. Injecting tools (via API param, system prompt, etc.)
        2. Detecting tool call requests in response
        3. Yielding StreamEvent objects in consistent format

        Args:
            messages: Conversation history
            tools: Available tools (provider-agnostic format)
            extra_body: Additional parameters to pass to the API

        Yields:
            StreamEvent: Either content chunks or tool call requests
        """
        ...
        # Async generator - implementations should use async def with yield
        # Type checkers: return type is AsyncGenerator when async def uses yield
        yield  # type: ignore
