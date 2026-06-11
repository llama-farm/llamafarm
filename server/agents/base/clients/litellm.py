"""LiteLLM client - unified gateway for 100+ LLM providers."""

from collections.abc import AsyncGenerator

from openai.types.chat import ChatCompletionToolParam

from agents.base.history import LFChatCompletionMessageParam
from agents.base.types import ToolDefinition
from core.logging import FastAPIStructLogger

from .client import (
    LFAgentClient,
    LFChatCompletion,
    LFChatCompletionChunk,
)

logger = FastAPIStructLogger(__name__)


class LFAgentClientLiteLLM(LFAgentClient):
    """LiteLLM client for 100+ LLM providers via unified SDK.

    Uses litellm.acompletion() which returns OpenAI-format responses,
    so tool calling, streaming, and response parsing work identically
    to the OpenAI client.
    """

    async def chat(
        self,
        *,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
        extra_body: dict | None = None,
    ) -> LFChatCompletion:
        import litellm

        openai_tools = (
            [self._tool_to_openai_format(t) for t in tools] if tools else None
        )

        api_params = (self._model_config.model_api_parameters or {}).copy()
        extra_body_copy = dict(extra_body or {})
        if "max_tokens" in extra_body_copy and "max_tokens" not in api_params:
            api_params["max_tokens"] = extra_body_copy.pop("max_tokens")

        kwargs = {
            "model": self._model_config.model,
            "messages": messages,
            "drop_params": True,
            **api_params,
        }
        if self._model_config.api_key:
            kwargs["api_key"] = self._model_config.api_key
        if self._model_config.base_url:
            kwargs["api_base"] = self._model_config.base_url
        if openai_tools:
            kwargs["tools"] = openai_tools

        return await litellm.acompletion(**kwargs)

    async def stream_chat(
        self,
        *,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
        extra_body: dict | None = None,
    ) -> AsyncGenerator[LFChatCompletionChunk]:
        import litellm

        openai_tools = (
            [self._tool_to_openai_format(t) for t in tools] if tools else None
        )

        api_params = (self._model_config.model_api_parameters or {}).copy()
        extra_body_copy = dict(extra_body or {})
        if "max_tokens" in extra_body_copy and "max_tokens" not in api_params:
            api_params["max_tokens"] = extra_body_copy.pop("max_tokens")

        kwargs = {
            "model": self._model_config.model,
            "messages": messages,
            "stream": True,
            "drop_params": True,
            **api_params,
        }
        if self._model_config.api_key:
            kwargs["api_key"] = self._model_config.api_key
        if self._model_config.base_url:
            kwargs["api_base"] = self._model_config.base_url
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = await litellm.acompletion(**kwargs)
        async for chunk in response:
            yield chunk

    def _tool_to_openai_format(self, tool: ToolDefinition) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(
            type="function",
            function={
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        )
