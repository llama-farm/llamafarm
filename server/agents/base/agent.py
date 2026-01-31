from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict, Field

from agents.base.clients.client import (
    LFAgentClient,
    LFChatCompletion,
    LFChatCompletionChunk,
)
from agents.base.types import ToolDefinition
from config.datamodel import PromptsMode, ToolsMode
from core.logging import FastAPIStructLogger

from .context_provider import LFAgentContextProvider
from .history import (
    LFAgentHistory,
    LFChatCompletionMessageParam,
    LFChatCompletionSystemMessageParam,
)
from .system_prompt_generator import LFAgentSystemPromptGenerator

logger = FastAPIStructLogger(__name__)


class LFAgentConfig(BaseModel):
    client: LFAgentClient = Field(..., description="The client for the agent")
    history: LFAgentHistory = Field(..., description="The history of the agent")
    system_prompt_generator: LFAgentSystemPromptGenerator = Field(
        ..., description="The system prompt generator for the agent"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LFAgent:
    history: LFAgentHistory
    _system_prompt_generator: LFAgentSystemPromptGenerator
    _client: LFAgentClient

    def __init__(self, config: LFAgentConfig):
        self.history = config.history
        self._system_prompt_generator = config.system_prompt_generator
        self._client = config.client

    @property
    def config_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition.from_datamodel_tool(t)
            for t in self._client._model_config.tools or []
        ]

    @property
    def tools_mode(self) -> ToolsMode:
        """Get the tools_mode from model config, defaulting to api_supplement."""
        return self._client._model_config.tools_mode or ToolsMode.api_supplement

    @property
    def prompts_mode(self) -> PromptsMode:
        """Get the prompts_mode from model config, defaulting to api_supplement."""
        return self._client._model_config.prompts_mode or PromptsMode.api_supplement

    def _combine_tools(
        self, api_tools: list[ToolDefinition] | None
    ) -> list[ToolDefinition]:
        """Combine config tools with API tools based on tools_mode.

        Args:
            api_tools: Tools provided via the API request (may include MCP tools)

        Returns:
            Combined tool list based on the model's tools_mode setting:
            - config_only: Only config tools; API tools are ignored
            - api_replace: API tools replace config tools entirely
            - api_supplement: API tools are added to config tools (default)
        """
        mode = self.tools_mode
        config_tools = self.config_tools

        if mode == ToolsMode.config_only:
            # Secure mode: ignore all API-provided tools
            if api_tools:
                logger.debug(
                    "tools_mode=config_only: ignoring %d API-provided tools",
                    len(api_tools),
                )
            return config_tools

        if mode == ToolsMode.api_replace:
            # API tools completely replace config tools
            if api_tools:
                logger.debug(
                    "tools_mode=api_replace: using %d API tools, ignoring %d config tools",
                    len(api_tools),
                    len(config_tools),
                )
                return api_tools
            # No API tools provided, fall back to config tools
            return config_tools

        # Default: api_supplement - combine both
        return config_tools + (api_tools or [])

    def _extract_system_messages(
        self, messages: list[LFChatCompletionMessageParam] | None
    ) -> tuple[list[LFChatCompletionMessageParam], list[LFChatCompletionMessageParam]]:
        """Separate system messages from other messages.

        System messages are applied per-request but not stored in history.
        This follows OpenAI-compatible behavior where the client sends
        the system prompt with each request.

        Returns:
            Tuple of (system_messages, non_system_messages)
        """
        if not messages:
            return [], []

        system_msgs = []
        other_msgs = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                other_msgs.append(msg)
        return system_msgs, other_msgs

    async def run_async(
        self,
        *,
        messages: list[LFChatCompletionMessageParam] | None = None,
        tools: list[ToolDefinition] | None = None,
        extra_body: dict | None = None,
    ) -> LFChatCompletion:
        # Separate system messages (per-request) from other messages (stored in history)
        request_system_msgs, other_msgs = self._extract_system_messages(messages)

        # Only add non-system messages to history
        for message in other_msgs:
            self.history.add_message(message)

        # Prepare messages including request system prompts for this request only
        messages = self._prepare_messages(request_system_messages=request_system_msgs)

        # Combine tools based on tools_mode setting
        tools = self._combine_tools(tools)

        return await self._client.chat(
            messages=messages, tools=tools, extra_body=extra_body
        )

    async def run_async_stream(
        self,
        *,
        messages: list[LFChatCompletionMessageParam] | None = None,
        tools: list[ToolDefinition] | None = None,
        extra_body: dict | None = None,
    ) -> AsyncGenerator[LFChatCompletionChunk]:
        # Separate system messages (per-request) from other messages (stored in history)
        request_system_msgs, other_msgs = self._extract_system_messages(messages)

        # Only add non-system messages to history
        for message in other_msgs:
            self.history.add_message(message)

        # Prepare messages including request system prompts for this request only
        messages = self._prepare_messages(request_system_messages=request_system_msgs)

        # Combine tools based on tools_mode setting
        tools = self._combine_tools(tools)

        async for chunk in self._client.stream_chat(
            messages=messages, tools=tools, extra_body=extra_body
        ):
            yield chunk

    def register_context_provider(
        self, title: str, context_provider: LFAgentContextProvider
    ):
        if self._system_prompt_generator.context_providers.get(title):
            raise ValueError(f"Context provider already registered: {title}")
        self._system_prompt_generator.context_providers[title] = context_provider

    def get_context_provider(self, title: str) -> LFAgentContextProvider | None:
        return self._system_prompt_generator.context_providers.get(title, None)

    def remove_context_provider(self, title: str):
        self._system_prompt_generator.context_providers.pop(title, None)

    def reset_history(self):
        """Reset the agent's conversation history."""
        self.history.history.clear()

    def _prepare_messages(
        self,
        request_system_messages: list[LFChatCompletionMessageParam] | None = None,
    ) -> list[LFChatCompletionMessageParam]:
        """Prepare the full message list for the LLM.

        Message order depends on prompts_mode:
        - config_only: Only config prompts (API system prompts ignored)
        - api_replace: Only API system prompts (config prompts ignored if API prompts provided)
        - api_supplement: Config prompts + API system prompts (default)

        Then conversation history is appended.

        Args:
            request_system_messages: System messages from the current API request.
                These are included for this request only, not stored in history.
        """
        messages: list[LFChatCompletionMessageParam] = []
        mode = self.prompts_mode

        # Get config system prompt
        config_system_prompt = self._system_prompt_generator.generate_prompt()

        if mode == PromptsMode.config_only:
            # Secure mode: only use config prompts, ignore API system prompts
            if request_system_messages:
                logger.debug(
                    "prompts_mode=config_only: ignoring %d API-provided system messages",
                    len(request_system_messages),
                )
            if config_system_prompt:
                messages.append(
                    LFChatCompletionSystemMessageParam(
                        role="system", content=config_system_prompt
                    )
                )

        elif mode == PromptsMode.api_replace:
            # API system prompts replace config prompts (if provided)
            if request_system_messages:
                logger.debug(
                    "prompts_mode=api_replace: using %d API system messages, ignoring config prompt",
                    len(request_system_messages),
                )
                for sys_msg in request_system_messages:
                    serialized = LFAgentHistory._serialize_message(sys_msg)
                    messages.append(serialized)
            elif config_system_prompt:
                # No API prompts provided, fall back to config
                messages.append(
                    LFChatCompletionSystemMessageParam(
                        role="system", content=config_system_prompt
                    )
                )

        else:
            # Default: api_supplement - config prompts + API system prompts
            if config_system_prompt:
                messages.append(
                    LFChatCompletionSystemMessageParam(
                        role="system", content=config_system_prompt
                    )
                )
            if request_system_messages:
                for sys_msg in request_system_messages:
                    serialized = LFAgentHistory._serialize_message(sys_msg)
                    messages.append(serialized)

        # Append conversation history
        for message in self.history.history:
            # Serialize messages to ensure proper JSON-compatible format
            # This handles OpenAI SDK types (Pydantic models) that may not
            # serialize correctly when passed through the API client
            serialized = LFAgentHistory._serialize_message(message)
            messages.append(serialized)

        return messages
