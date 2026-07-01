"""LiteLLM client - unified gateway for 100+ LLM providers."""

import json
import re
import uuid
from collections.abc import AsyncGenerator

from config.datamodel import ToolCallStrategy
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
from .openai import TOOLS_SYSTEM_MESSAGE_PREFIX, TOOLS_SYSTEM_MESSAGE_SUFFIX

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

        # Check for native_api strategy (handle both enum and string values)
        strategy = self._model_config.tool_call_strategy
        use_native_api = strategy in (
            ToolCallStrategy.native_api,
            "native_api",
            None,  # Default to native_api if not set
        )

        if use_native_api:
            openai_tools = (
                [self._tool_to_openai_format(t) for t in tools] if tools else None
            )
        else:
            openai_tools = None
            self._update_system_message_with_tools(messages, tools)

        api_params = (self._model_config.model_api_parameters or {}).copy()

        # Extract standard OpenAI parameters from extra_body into api_params
        extra_body_copy = dict(extra_body or {})
        if "max_tokens" in extra_body_copy:
            if "max_tokens" not in api_params:
                api_params["max_tokens"] = extra_body_copy.pop("max_tokens")
            else:
                extra_body_copy.pop("max_tokens")
        # Note: think, thinking_budget, n_ctx stay in extra_body

        # Convert extra_body from Pydantic model to dict if needed
        config_extra_body = {}
        if self._model_config.extra_body:
            config_extra_body = (
                self._model_config.extra_body.model_dump(exclude_none=True)
                if hasattr(self._model_config.extra_body, "model_dump")
                else dict(self._model_config.extra_body)
            )

        # Project-level config takes precedence over per-request params
        extra_body_params = {
            **extra_body_copy,
            **config_extra_body,
        }

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
        if extra_body_params:
            kwargs["extra_body"] = extra_body_params

        completion = await litellm.acompletion(**kwargs)

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
        import litellm

        # Check for native_api strategy (handle both enum and string values)
        strategy = self._model_config.tool_call_strategy
        use_native_api = strategy in (
            ToolCallStrategy.native_api,
            "native_api",
            None,  # Default to native_api if not set
        )

        if use_native_api:
            openai_tools = (
                [self._tool_to_openai_format(t) for t in tools] if tools else None
            )
        else:
            openai_tools = None
            self._update_system_message_with_tools(messages, tools)

        api_params = (self._model_config.model_api_parameters or {}).copy()

        # Extract standard OpenAI parameters from extra_body into api_params
        extra_body_copy = dict(extra_body or {})
        if "max_tokens" in extra_body_copy:
            if "max_tokens" not in api_params:
                api_params["max_tokens"] = extra_body_copy.pop("max_tokens")
            else:
                extra_body_copy.pop("max_tokens")
        # Note: think, thinking_budget, n_ctx stay in extra_body

        # Convert extra_body from Pydantic model to dict if needed
        config_extra_body = {}
        if self._model_config.extra_body:
            config_extra_body = (
                self._model_config.extra_body.model_dump(exclude_none=True)
                if hasattr(self._model_config.extra_body, "model_dump")
                else dict(self._model_config.extra_body)
            )

        # Project-level config takes precedence over per-request params
        extra_body_params = {
            **extra_body_copy,
            **config_extra_body,
        }

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
        if extra_body_params:
            kwargs["extra_body"] = extra_body_params

        response = await litellm.acompletion(**kwargs)

        if use_native_api:
            async for chunk in response:
                yield chunk
            return

        # For prompt-based strategy, buffer content to detect tool calls
        accumulated_content = ""
        is_in_tool_call = False

        async for chunk in response:
            # For native tool calls, pass through immediately
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                yield chunk
                continue

            # Accumulate content
            delta_content = chunk.choices[0].delta.content if chunk.choices else None
            if delta_content:
                accumulated_content += delta_content

            if not self._detect_probable_tool_call_in_content(accumulated_content):
                yield chunk

            (tool_name, tool_args_json) = self._detect_tool_call_in_content(
                accumulated_content
            ) or (None, None)

            if not tool_name:
                continue

            if is_in_tool_call:
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
                tool_call_id = f"call_{uuid.uuid4()}"
                yield self._create_synthetic_tool_call_chunk(
                    base_chunk=chunk,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_arguments="",
                )
                is_in_tool_call = True

    def _detect_probable_tool_call_in_content(self, content: str) -> bool:
        """Detect if the content probably contains a tool call."""
        return bool(re.search(r"<tool_call>", content, re.DOTALL))

    def _detect_tool_call_in_content(self, content: str) -> tuple[str, str] | None:
        """Detect and extract tool call from accumulated content."""
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
        """Create a synthetic tool call chunk from a content chunk."""
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
        """Create a completion with a synthetic tool call from prompt-based response."""
        if completion.choices[0].message.tool_calls:
            return completion

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

    def _tool_to_openai_format(self, tool: ToolDefinition) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(
            type="function",
            function={
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        )
