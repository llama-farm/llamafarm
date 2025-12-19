import json
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Literal

from config.datamodel import ToolCallStrategy
from openai import NOT_GIVEN, AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChoiceChunk,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

from agents.base.history import LFChatCompletionMessageParam
from agents.base.types import ToolDefinition
from core.logging import FastAPIStructLogger

from .client import (
    LFAgentClient,
    LFChatCompletion,
    LFChatCompletionChunk,
)

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
        extra_body: dict | None = None,
    ) -> LFChatCompletion:
        """Chat with tool calling support.

        Args:
            messages: Chat messages
            tools: Tool definitions
            extra_body: Additional parameters to pass to the API (e.g., n_ctx for GGUF models)
        """
        client = AsyncOpenAI(
            api_key=self._model_config.api_key or "",
            base_url=self._model_config.base_url or "",
        )

        # Convert tools to OpenAI format
        if self._model_config.tool_call_strategy == ToolCallStrategy.native_api:
            openai_tools = (
                [self._tool_to_openai_format(t) for t in tools] if tools else NOT_GIVEN
            )
        else:
            openai_tools = NOT_GIVEN
            self._update_system_message_with_tools(messages, tools)

        # Prepare API parameters
        # model_api_parameters go as direct kwargs, extra_body goes in extra_body
        api_params = (self._model_config.model_api_parameters or {}).copy()

        # Extract standard OpenAI parameters from extra_body into api_params
        # These are first-class OpenAI API parameters, not provider-specific extensions
        # Only use per-request value if project config doesn't already define it
        extra_body_copy = dict(extra_body or {})
        if "max_tokens" in extra_body_copy:
            if "max_tokens" not in api_params:
                api_params["max_tokens"] = extra_body_copy.pop("max_tokens")
            else:
                # Project config takes precedence, discard per-request value
                extra_body_copy.pop("max_tokens")
        # Note: think and thinking_budget stay in extra_body - they're not standard OpenAI params
        # The universal runtime extracts them from extra_body

        # Convert extra_body from Pydantic model to dict if needed
        config_extra_body = {}
        if self._model_config.extra_body:
            config_extra_body = (
                self._model_config.extra_body.model_dump(exclude_none=True)
                if hasattr(self._model_config.extra_body, "model_dump")
                else dict(self._model_config.extra_body)
            )

        # Project-level config takes precedence over per-request params
        # to ensure enforced limits (n_ctx, etc.) can't be bypassed
        extra_body_params = {
            **extra_body_copy,
            **config_extra_body,
        }

        # Create non-streaming request
        stream_param: Literal[False] = False

        completion = await client.chat.completions.create(
            messages=messages,
            model=self._model_config.model,
            tools=openai_tools,
            **api_params,
            extra_body=extra_body_params,
            stream=stream_param,
        )

        if (
            self._model_config.tool_call_strategy == ToolCallStrategy.prompt_based
            and self._contains_tool_call(completion)
        ):
            return self._create_synthetic_tool_call(completion)

        return completion

    async def stream_chat(
        self,
        *,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
        extra_body: dict | None = None,
    ) -> AsyncGenerator[LFChatCompletionChunk]:
        """Stream chat with native OpenAI function calling.

        Args:
            messages: Chat messages
            tools: Tool definitions
            extra_body: Additional parameters to pass to the API (e.g., n_ctx for GGUF models)
        """

        client = AsyncOpenAI(
            api_key=self._model_config.api_key or "",
            base_url=self._model_config.base_url or "",
        )

        # Convert tools to OpenAI format
        if self._model_config.tool_call_strategy == ToolCallStrategy.native_api:
            openai_tools = (
                [self._tool_to_openai_format(t) for t in tools] if tools else NOT_GIVEN
            )
        else:
            openai_tools = NOT_GIVEN
            self._update_system_message_with_tools(messages, tools)

        # Prepare API parameters
        # model_api_parameters go as direct kwargs, extra_body goes in extra_body
        api_params = (self._model_config.model_api_parameters or {}).copy()

        # Extract standard OpenAI parameters from extra_body into api_params
        # These are first-class OpenAI API parameters, not provider-specific extensions
        # Only use per-request value if project config doesn't already define it
        extra_body_copy = dict(extra_body or {})
        if "max_tokens" in extra_body_copy:
            if "max_tokens" not in api_params:
                api_params["max_tokens"] = extra_body_copy.pop("max_tokens")
            else:
                # Project config takes precedence, discard per-request value
                extra_body_copy.pop("max_tokens")
        # Note: think and thinking_budget stay in extra_body - they're not standard OpenAI params
        # The universal runtime extracts them from extra_body

        # Convert extra_body from Pydantic model to dict if needed
        config_extra_body = {}
        if self._model_config.extra_body:
            config_extra_body = (
                self._model_config.extra_body.model_dump(exclude_none=True)
                if hasattr(self._model_config.extra_body, "model_dump")
                else dict(self._model_config.extra_body)
            )

        # Project-level config takes precedence over per-request params
        # to ensure enforced limits (n_ctx, etc.) can't be bypassed
        extra_body_params = {
            **extra_body_copy,
            **config_extra_body,
        }

        stream_param: Literal[True] = True

        response_stream = await client.chat.completions.create(
            messages=messages,
            model=self._model_config.model,
            tools=openai_tools,
            **api_params,
            extra_body=extra_body_params,
            stream=stream_param,
        )

        if self._model_config.tool_call_strategy == ToolCallStrategy.native_api:
            async for chunk in response_stream:
                yield chunk
            return

        # For prompt-based strategy, we need to buffer content to detect tool calls
        accumulated_content = ""
        is_in_tool_call = False

        async for chunk in response_stream:
            # For native tool calls, pass through immediately
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                yield chunk
                continue

            # Accumulate content
            delta_content = chunk.choices[0].delta.content if chunk.choices else None
            if delta_content:
                accumulated_content += delta_content

            # No complete tool call yet, yield the chunk normally, unless we are
            # probably in a tool call and are just accumulating the tool call JSON.
            if not self._detect_probable_tool_call_in_content(accumulated_content):
                yield chunk

            (tool_name, tool_args_json) = self._detect_tool_call_in_content(
                accumulated_content
            ) or (None, None)

            if not tool_name:
                # Continue to accumulate tool call JSON content
                continue

            if is_in_tool_call:
                # We have already yielded the first tool call chunk. now just yield the
                # arguments delta
                yield self._create_synthetic_tool_call_chunk(
                    base_chunk=chunk,
                    tool_arguments=tool_args_json,
                )
                yield self._create_synthetic_tool_call_chunk(
                    base_chunk=chunk,
                    is_finished=True,
                )
                break
            else:
                # First tool call chunk should just have the ID and name
                tool_call_id = f"call_{uuid.uuid4()}"
                yield self._create_synthetic_tool_call_chunk(
                    base_chunk=chunk,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_arguments="",
                )
                is_in_tool_call = True

    def _detect_probable_tool_call_in_content(self, content: str) -> bool:
        """Detect if the content is probably a tool call.

        Returns:
            True if probable tool call indicated by the presence of an opening
            <tool_call> tag in the content, False otherwise.
        """
        return bool(re.search(r"<tool_call>", content, re.DOTALL))

    def _detect_tool_call_in_content(self, content: str) -> tuple[str, str] | None:
        """Detect and extract tool call from accumulated content.

        Returns:
            Tuple of (tool_name, tool_arguments_json) if found, None if not found.
        """
        tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if not tool_call_match:
            return None

        try:
            tool_call_json = json.loads(tool_call_match.group(1))
            tool_call_name = tool_call_json["name"]
            tool_call_arguments = json.dumps(tool_call_json["arguments"])
            return (tool_call_name, tool_call_arguments)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(
                "Failed to parse tool call from content",
                error=str(e),
                content=content[:200],
            )
            return None

    def _create_synthetic_tool_call_chunk(
        self,
        *,
        base_chunk: ChatCompletionChunk,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        tool_arguments: str | None = None,
        is_finished: bool = False,
    ) -> ChatCompletionChunk:
        """Create a synthetic tool call chunk from a content chunk.

        Args:
            base_chunk: The chunk to use for metadata (id, timestamp, etc.)
            tool_call_id: ID of the tool call
            tool_name: Name of the tool being called
            tool_arguments: JSON string of tool arguments

        Returns:
            A new ChatCompletionChunk with tool call information
        """
        delta = (
            ChoiceDelta()
            if is_finished
            else (
                ChoiceDelta(
                    role="assistant",
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id=tool_call_id,
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                name=tool_name,
                                arguments=tool_arguments,
                            ),
                        )
                    ],
                )
            )
        )

        return ChatCompletionChunk(
            id=base_chunk.id,
            object="chat.completion.chunk",
            created=base_chunk.created,
            model=base_chunk.model,
            system_fingerprint=base_chunk.system_fingerprint,
            service_tier=base_chunk.service_tier,
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=delta,
                    finish_reason="tool_calls" if is_finished else None,
                ),
            ],
            usage=base_chunk.usage,
        )

    def _update_system_message_with_tools(
        self,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
    ):
        """Update system message to add a special TOOLS section."""
        if not tools:
            return

        for msg in messages:
            msg_content = msg.get("content")
            if msg.get("role") == "system" and isinstance(msg_content, str):
                new_content = msg_content + TOOLS_SYSTEM_MESSAGE_PREFIX
                for tool in tools:
                    openai_tool = self._tool_to_openai_format(tool)
                    new_content += f"<tool>{json.dumps(openai_tool)}</tool>\n"
                new_content += TOOLS_SYSTEM_MESSAGE_SUFFIX
                msg.update({"content": new_content})
                break

    def _contains_tool_call(self, completion: ChatCompletion) -> bool:
        """Check if the completion contains a tool call."""
        if completion.choices[0].message.tool_calls:
            return True

        content = completion.choices[0].message.content
        return (
            re.search(r"<tool_call>.*?</tool_call>", str(content), re.DOTALL)
            is not None
        )

    def _create_synthetic_tool_call(self, completion: ChatCompletion) -> ChatCompletion:
        if completion.choices[0].message.tool_calls:
            return completion

        """Create a completion with a tool call."""
        tool_call = re.search(
            r"<tool_call>(.*?)</tool_call>",
            str(completion.choices[0].message.content),
            re.DOTALL,
        )
        if not tool_call:
            return completion

        tool_call_json = json.loads(tool_call.group(1))
        tool_call_name = tool_call_json["name"]
        tool_call_arguments = json.dumps(tool_call_json["arguments"])

        return ChatCompletion(
            id=completion.id,
            object="chat.completion",
            created=completion.created,
            model=completion.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageFunctionToolCall(
                                type="function",
                                id=f"call_{uuid.uuid4()}",
                                function=Function(
                                    name=tool_call_name,
                                    arguments=tool_call_arguments,
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                ),
            ],
            usage=completion.usage,
        )

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
