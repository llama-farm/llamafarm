import json
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from agents.base.history import LFAgentChatMessage
from agents.base.types import StreamEvent, ToolCallRequest, ToolDefinition
from core.logging import FastAPIStructLogger

from .client import LFAgentClient

logger = FastAPIStructLogger(__name__)


class LFAgentClientOpenAI(LFAgentClient):
    """OpenAI client using native function calling.

    This client:
    1. Passes tools via the `tools` API parameter
    2. Detects tool calls from native `tool_calls` in response
    3. Streams both content and tool calls as StreamEvents
    """

    async def chat(self, *, messages: list[LFAgentChatMessage]) -> str:
        """Simple chat without tool calling support."""
        content = ""
        async for event in self.stream_chat_with_tools(messages=messages, tools=[]):
            if event.is_content():
                content += event.content or ""
        return content

    async def stream_chat(
        self, *, messages: list[LFAgentChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Stream chat without tool calling support."""
        async for event in self.stream_chat_with_tools(messages=messages, tools=[]):
            if event.is_content() and event.content:
                yield event.content

    async def stream_chat_with_tools(
        self,
        *,
        messages: list[LFAgentChatMessage],
        tools: list[ToolDefinition],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream chat with native OpenAI function calling."""

        client = AsyncOpenAI(
            api_key=self._model_config.api_key or "",
            base_url=self._model_config.base_url or "",
        )

        # Convert tools to OpenAI format
        openai_tools = (
            [self._tool_to_openai_format(t) for t in tools] if tools else None
        )

        # Create streaming request
        params = {
            "model": self._model_config.model,
            "messages": [self._message_to_openai_message(m) for m in messages],
            "stream": True,
            **(self._model_config.model_api_parameters or {}),
        }
        if openai_tools:
            params["tools"] = openai_tools
            params["tool_choice"] = "auto"

        response_stream = await client.chat.completions.create(**params)  # type: ignore

        # Track partial tool calls as they stream in
        current_tool_calls: dict[int, dict] = {}

        async for chunk in response_stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Yield content chunks
            if delta.content:
                yield StreamEvent(type="content", content=delta.content)

            # Handle tool call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index

                    # Initialize tool call if new
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tc_delta.id or f"call_{idx}",
                            "name": "",
                            "arguments": "",
                        }

                    # Accumulate name
                    if tc_delta.function and tc_delta.function.name:
                        current_tool_calls[idx]["name"] = tc_delta.function.name

                    # Accumulate arguments
                    if tc_delta.function and tc_delta.function.arguments:
                        current_tool_calls[idx]["arguments"] += (
                            tc_delta.function.arguments
                        )

            # When stream finishes with tool calls, yield them
            if choice.finish_reason == "tool_calls":
                for tc_data in current_tool_calls.values():
                    try:
                        args = json.loads(tc_data["arguments"])
                        yield StreamEvent(
                            type="tool_call",
                            tool_call=ToolCallRequest(
                                id=tc_data["id"], name=tc_data["name"], arguments=args
                            ),
                        )
                    except json.JSONDecodeError as e:
                        logger.error(
                            "Failed to parse tool call arguments",
                            arguments=tc_data["arguments"],
                            error=str(e),
                        )

    def _tool_to_openai_format(self, tool: ToolDefinition) -> dict:
        """Convert ToolDefinition to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _message_to_openai_message(
        self, message: LFAgentChatMessage
    ) -> ChatCompletionMessageParam:
        """Convert LFAgentChatMessage to OpenAI format."""
        match message.role:
            case "system":
                return {"role": "system", "content": message.content}
            case "user":
                return {"role": "user", "content": message.content}
            case "assistant":
                return {"role": "assistant", "content": message.content}
            case "tool":
                # For tool results, format as user message with result
                return {"role": "user", "content": message.content}
            case _:
                raise ValueError(f"Unknown message role: {message.role}")
