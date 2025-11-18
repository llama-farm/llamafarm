import datetime
import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

from ollama import AsyncClient, Message
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChoiceChunk,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

from agents.base.history import LFChatCompletionMessageParam
from agents.base.types import ToolDefinition
from core.logging import FastAPIStructLogger

from .client import LFAgentClient, LFChatCompletion, LFChatCompletionChunk

logger = FastAPIStructLogger(__name__)


class LFAgentClientOllama(LFAgentClient):
    """Ollama client using JSON-based tool calling.

    This client:
    1. Injects tools into the system prompt as instructions + schemas
    2. Detects JSON responses that match tool call format
    3. Yields StreamEvents in the same format as OpenAI client
    """

    async def chat(
        self,
        *,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
        **_,
    ) -> LFChatCompletion:
        """Chat with tool calling support."""
        client = AsyncClient(
            host=(
                self._model_config.base_url.rstrip("/v1")
                if self._model_config.base_url
                else ""
            ),
        )
        # Convert tools to Ollama format
        ollama_tools = (
            [self._tool_to_ollama_format(t) for t in tools] if tools else None
        )

        # Convert messages to Ollama format
        ollama_messages = [self._message_to_ollama_message(m) for m in messages]

        # Create non-streaming request
        stream_param: Literal[False] = False
        response = await client.chat(
            model=self._model_config.model,
            messages=ollama_messages,
            stream=stream_param,
            tools=ollama_tools,
        )

        # Convert Ollama response to OpenAI ChatCompletion format

        finish = (
            response.done_reason
            if response.done_reason
            in ("stop", "length", "tool_calls", "content_filter")
            else "stop"
        )

        # Build message with reasoning if available
        message = ChatCompletionMessage(
            role="assistant",
            content=response.message.content or "",
        )
        if response.message.thinking:
            message.reasoning = response.message.thinking  # type: ignore

        if response.message.tool_calls:
            message.tool_calls = [
                ChatCompletionMessageFunctionToolCall(
                    type="function",
                    id=f"call_{uuid.uuid4()}",
                    function=Function(
                        name=tool_call.function.name,
                        arguments=json.dumps(tool_call.function.arguments),
                    ),
                )
                for tool_call in response.message.tool_calls
            ]

        created_timestamp = self._ollama_created_at_datetime(response.created_at)

        return ChatCompletion(
            id=f"chatcmpl-{created_timestamp}",
            model=response.model or "unknown",
            object="chat.completion",
            created=created_timestamp,
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason=finish,  # type: ignore
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=response.prompt_eval_count or 0,
                completion_tokens=response.eval_count or 0,
                total_tokens=(
                    (response.prompt_eval_count or 0) + (response.eval_count or 0)
                ),
            ),
        )

    async def stream_chat(
        self,
        *,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
        **_,
    ) -> AsyncGenerator[LFChatCompletionChunk]:
        """Stream chat, converting Ollama chunks to OpenAI format."""

        client = AsyncClient(
            host=(
                self._model_config.base_url.rstrip("/v1")
                if self._model_config.base_url
                else ""
            ),
        )

        ollama_tools = (
            [self._tool_to_ollama_format(t) for t in tools] if tools else None
        )

        stream_param: Literal[True] = True
        response_stream = await client.chat(
            model=self._model_config.model,
            messages=[self._message_to_ollama_message(m) for m in messages],
            stream=stream_param,
            tools=ollama_tools,
        )

        finish_reason_set_to_tool_calls_once = False

        async for chunk in response_stream:
            # Convert Ollama chunk to OpenAI format
            created_timestamp = self._ollama_created_at_datetime(chunk.created_at)

            # Build delta with content and reasoning (if available)
            delta = ChoiceDelta(
                role="assistant" if chunk.message.role else None,
                content=chunk.message.content or None,
            )

            finish = (
                chunk.done_reason
                if chunk.done_reason
                in ("stop", "length", "tool_calls", "content_filter")
                else None
            )

            # HACK: ollama is setting done and done_reason to "stop" when a tool call is present
            # We are going to immitate openai behavior and set the finish_reason to "tool_calls"
            if (
                not chunk.message.content
                and not chunk.message.thinking
                and chunk.message.tool_calls
                and not finish_reason_set_to_tool_calls_once
            ):
                finish_reason_set_to_tool_calls_once = True
                finish = "tool_calls"

            # Add reasoning from Ollama's thinking field
            if chunk.message.thinking:
                delta.reasoning = chunk.message.thinking  # type: ignore

            if chunk.message.tool_calls:
                # HACK: ollama is setting done and done_reason to "stop" when a tool call is present
                # We are going to immitate openai behavior and set the finish_reason to "tool_calls"
                if (
                    not chunk.message.content
                    and not chunk.message.thinking
                    and chunk.message.tool_calls
                    and not finish_reason_set_to_tool_calls_once
                ):
                    finish_reason_set_to_tool_calls_once = True
                    finish = "tool_calls"

                delta.tool_calls = [
                    ChoiceDeltaToolCall(
                        index=idx,
                        type="function",
                        id=f"call_{uuid.uuid4()}",
                        function=ChoiceDeltaToolCallFunction(
                            name=tool_call.function.name,
                            arguments=json.dumps(tool_call.function.arguments),
                        ),
                    )
                    for idx, tool_call in enumerate(chunk.message.tool_calls)
                ]

            # Build the chunk
            completion_chunk = LFChatCompletionChunk(
                id=f"chatcmpl-{created_timestamp}",
                model=chunk.model or "unknown",
                object="chat.completion.chunk",
                created=created_timestamp,
                choices=[
                    ChoiceChunk(
                        index=0,
                        delta=delta,
                        finish_reason=finish,  # type: ignore
                    ),
                ],
            )

            # Add usage info on final chunk (when done=True)
            if chunk.done and (chunk.prompt_eval_count or chunk.eval_count):
                from openai.types.completion_usage import CompletionUsage

                completion_chunk.usage = CompletionUsage(  # type: ignore
                    prompt_tokens=chunk.prompt_eval_count or 0,
                    completion_tokens=chunk.eval_count or 0,
                    total_tokens=(
                        (chunk.prompt_eval_count or 0) + (chunk.eval_count or 0)
                    ),
                )

            yield completion_chunk

    def _ollama_created_at_datetime(self, created_at: str | None) -> int:
        """Convert Ollama created_at string to datetime."""

        if created_at:
            # Parse RFC3339/ISO8601 with/without subsecond precision and Zulu
            try:
                created_dt = datetime.datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                )
                created_timestamp = int(created_dt.timestamp())
            except Exception:
                created_timestamp = 0
        else:
            created_timestamp = 0
        return created_timestamp

    def _tool_to_ollama_format(self, tool: ToolDefinition) -> dict:
        """Convert ToolDefinition to Ollama format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _message_to_ollama_message(
        self, message: LFChatCompletionMessageParam
    ) -> Message:
        """Convert LFAgentChatMessage to Ollama Message format."""
        role = message.get("role", "")
        content = message.get("content", "")

        content_str = str(content)
        # Ensure content is always a string for the Ollama Message
        if not isinstance(content, str):
            if content is None:
                content_str = ""
            elif isinstance(content, list | tuple):
                # Join parts if they're dicts with "text", else str()
                parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        parts.append(part["text"])
                    elif isinstance(part, str):
                        parts.append(part)
                    else:
                        parts.append(str(part))
                content_str = "".join(parts)
            else:
                content_str = str(content)

        match role:
            case "system":
                return Message(role="system", content=content_str)
            case "user":
                return Message(role="user", content=content_str)
            case "assistant":
                return Message(role="assistant", content=content_str)
            case "tool":
                tool_call_id: str = message.get("tool_call_id", "")  # type: ignore
                return Message(role="tool", tool_name=tool_call_id, content=content_str)
            case _:
                err = f"Unknown message role: {role}"
                raise ValueError(err)
