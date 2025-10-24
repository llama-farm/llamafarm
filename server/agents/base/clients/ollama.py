import json
from collections.abc import AsyncGenerator

from config.datamodel import Prompt
from ollama import AsyncClient, Message

from agents.base.history import LFAgentChatMessage
from agents.base.types import StreamEvent, ToolCallRequest, ToolDefinition
from core.logging import FastAPIStructLogger

from .client import LFAgentClient

logger = FastAPIStructLogger(__name__)


class LFAgentClientOllama(LFAgentClient):
    """Ollama client using JSON-based tool calling.

    This client:
    1. Injects tools into the system prompt as instructions + schemas
    2. Detects JSON responses that match tool call format
    3. Yields StreamEvents in the same format as OpenAI client
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
        """Stream chat with JSON-based tool calling."""

        # Inject tools into system message
        if tools:
            tool_instruction = self._create_tool_instruction(tools)

            # Prepend tool instruction to first system message or create new one
            modified_messages = []
            system_injected = False
            for msg in messages:
                if msg.role == "system" and not system_injected:
                    # Prepend to existing system message
                    modified_msg = LFAgentChatMessage(
                        role="system", content=f"{tool_instruction}\n\n{msg.content}"
                    )
                    modified_messages.append(modified_msg)
                    system_injected = True
                else:
                    modified_messages.append(msg)

            # If no system message, create one
            if not system_injected:
                modified_messages.insert(
                    0, LFAgentChatMessage(role="system", content=tool_instruction)
                )

            messages = modified_messages

        # Stream response
        client = AsyncClient(
            host=(
                self._model_config.base_url.rstrip("/v1")
                if self._model_config.base_url
                else ""
            ),
        )

        response_stream = await client.chat(
            model=self._model_config.model,
            messages=[self._message_to_ollama_message(m) for m in messages],
            stream=True,
            think=False,
            # **(self._model_config.model_api_parameters or {}),
        )

        # Buffer for detecting JSON tool calls
        buffer = ""
        looks_like_json = False

        async for chunk in response_stream:
            content = chunk.message.content
            if not content:
                continue

            buffer += content

            # Detect if this looks like a JSON tool call
            stripped = buffer.strip()
            if not looks_like_json and stripped.startswith("{"):
                looks_like_json = True

            # If not JSON, stream content normally
            if not looks_like_json:
                yield StreamEvent(type="content", content=content)

        # After stream ends, check if buffer contains tool call
        if looks_like_json:
            try:
                data = json.loads(buffer.strip())

                # Check if it matches tool call format
                if "tool_name" in data and "tool_parameters" in data:
                    yield StreamEvent(
                        type="tool_call",
                        tool_call=ToolCallRequest(
                            id=f"call_{data['tool_name']}",
                            name=data["tool_name"],
                            arguments=data.get("tool_parameters", {}),
                        ),
                    )
                else:
                    # JSON but not a tool call, treat as content
                    yield StreamEvent(type="content", content=buffer)
            except json.JSONDecodeError:
                # Not valid JSON, treat as content
                yield StreamEvent(type="content", content=buffer)

    def _create_tool_instruction(self, tools: list[ToolDefinition]) -> str:
        """Create system prompt instructions for JSON-based tool calling."""
        tool_schemas = []
        for tool in tools:
            tool_schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            )

        instruction = (
            "You have access to the following tools. When you want to use a tool, "
            "respond with ONLY a JSON object in this exact format:\n\n"
            "{\n"
            '  "tool_name": "<tool_name>",\n'
            '  "tool_parameters": {<parameters>}\n'
            "}\n\n"
            "Do not include any other text or explanation when calling a tool.\n\n"
            f"Available tools:\n{json.dumps(tool_schemas, indent=2)}"
        )
        return instruction

    def _message_to_ollama_message(self, message: LFAgentChatMessage) -> Message:
        """Convert LFAgentChatMessage to Ollama Message format."""
        match message.role:
            case "system":
                return Message(role="system", content=message.content)
            case "user":
                return Message(role="user", content=message.content)
            case "assistant":
                return Message(role="assistant", content=message.content)
            case "tool":
                # Tool results as user messages
                return Message(role="user", content=message.content)
            case _:
                raise ValueError(f"Unknown message role: {message.role}")
