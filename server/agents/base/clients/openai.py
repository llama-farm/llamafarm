from collections.abc import AsyncGenerator
from typing import Literal

from openai import NOT_GIVEN, AsyncOpenAI
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


class LFAgentClientOpenAI(LFAgentClient):
    """OpenAI client using native function calling.

    This client:
    1. Passes tools via the `tools` API parameter
    2. Detects tool calls from native `tool_calls` in response
    3. Streams both content and tool calls as StreamEvents
    """

    async def chat(
        self,
        *,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
    ) -> LFChatCompletion:
        """Chat with tool calling support."""
        client = AsyncOpenAI(
            api_key=self._model_config.api_key or "",
            base_url=self._model_config.base_url or "",
        )

        # Convert tools to OpenAI format
        openai_tools = (
            [self._tool_to_openai_format(t) for t in tools] if tools else NOT_GIVEN
        )

        # Create non-streaming request
        stream_param: Literal[False] = False
        completion = await client.chat.completions.create(
            messages=messages,
            model=self._model_config.model,
            tools=openai_tools,
            tool_choice="auto" if tools else NOT_GIVEN,
            **(self._model_config.model_api_parameters or {}),
            stream=stream_param,
        )

        return completion

    async def stream_chat(
        self,
        *,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncGenerator[LFChatCompletionChunk]:
        """Stream chat with native OpenAI function calling."""

        client = AsyncOpenAI(
            api_key=self._model_config.api_key or "",
            base_url=self._model_config.base_url or "",
        )

        # Convert tools to OpenAI format
        openai_tools = (
            [self._tool_to_openai_format(t) for t in tools] if tools else NOT_GIVEN
        )

        # Create streaming request with proper overload typing
        # Use NOT_GIVEN for optional parameters to match OpenAI overload signatures
        stream_param: Literal[True] = True
        response_stream = await client.chat.completions.create(
            messages=messages,
            model=self._model_config.model,
            tools=openai_tools,
            tool_choice="auto" if tools else NOT_GIVEN,
            **(self._model_config.model_api_parameters or {}),
            stream=stream_param,
        )

        async for chunk in response_stream:
            yield chunk

    def _tool_to_openai_format(self, tool: ToolDefinition) -> ChatCompletionToolParam:
        """Convert ToolDefinition to OpenAI function calling format."""
        return ChatCompletionToolParam(
            type="function",
            function={
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        )
