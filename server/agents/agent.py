from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import instructor
from atomic_agents import (  # type: ignore
    AgentConfig,
    AtomicAgent,
    BasicChatInputSchema,
    BasicChatOutputSchema,
)
from atomic_agents.context.chat_history import ChatHistory  # type: ignore
from atomic_agents.context.system_prompt_generator import (  # type: ignore
    SystemPromptGenerator,  # type: ignore
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger(__name__)


class LFAgentConfig(BaseModel):
    client: AsyncOpenAI | instructor.client.AsyncInstructor = Field(
        ..., description="OpenAI or instructor client instance."
    )
    model: str = Field(
        default="gpt-4o-mini", description="The model to use for generating responses."
    )
    history: ChatHistory | None = Field(
        default=None, description="History component for storing chat history."
    )
    system_prompt_generator: SystemPromptGenerator | None = Field(
        default=None, description="Component for generating system prompts."
    )
    system_role: str | None = Field(
        default="system",
        description=(
            "The role of the system in the conversation. None means no system prompt."
        ),
    )
    model_api_parameters: dict[str, Any] | None = Field(
        None,
        description="Additional parameters passed to the API provider.",
    )

    # Allow non-pydantic types like AsyncOpenAI
    model_config = {"arbitrary_types_allowed": True}


class LFAgent[InputSchema: BasicChatInputSchema, OutputSchema: BasicChatOutputSchema](
    AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema]
):
    def __init__(self, config: LFAgentConfig):
        # For structured mode: AtomicAgent needs instructor client
        # For unstructured mode: we'll bypass AtomicAgent's message preparation entirely
        if isinstance(config.client, AsyncOpenAI):
            client_for_atomic_agent = instructor.from_openai(config.client)
            self._use_structured_output = False
            self._unstructured_client = config.client
        else:
            client_for_atomic_agent = config.client
            self._use_structured_output = True

        atomic_agent_config = AgentConfig(
            client=client_for_atomic_agent,
            model=config.model,
            history=config.history,
            system_prompt_generator=config.system_prompt_generator,
            system_role=config.system_role,
            model_api_parameters=config.model_api_parameters,
        )
        super().__init__(config=atomic_agent_config)

    def _prepare_messages(self):
        """Prepare messages for the model.

        For structured mode: use AtomicAgent's built-in preparation
        For unstructured mode: build plain text messages directly from history
        """
        if self._use_structured_output:
            # Structured mode: use parent's schema-based message preparation
            return super()._prepare_messages()

        # Unstructured mode: build plain messages manually, no schema wrapping
        self.messages = []

        # Add system prompt if configured
        if self.system_prompt_generator and self.system_role:
            system_prompt = self.system_prompt_generator.generate_prompt()
            if system_prompt:
                self.messages.append(
                    {"role": self.system_role, "content": system_prompt}
                )

        # Add history messages as plain text
        for msg in self.history.get_history():
            role = getattr(msg, "role", None) or (
                msg.get("role") if isinstance(msg, dict) else None
            )
            content_str = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None
            )

            # Extract plain text from content
            if role == "user":
                content_instance = self.input_schema.model_validate_json(content_str)
            elif role == "assistant":
                content_instance = self.output_schema.model_validate_json(content_str)

            if role and content_instance:
                self.messages.append(
                    {"role": role, "content": content_instance.chat_message}
                )

    async def run_async(self, user_input: InputSchema | None = None) -> OutputSchema:
        if self._use_structured_output:
            # Structured mode: use instructor client (defer to parent)
            return await super().run_async(user_input)

        return await self._run_async_unstructured(user_input)

    async def run_async_stream(
        self, user_input: InputSchema | None = None
    ) -> AsyncGenerator[OutputSchema, None]:
        if self._use_structured_output:
            async for chunk in super().run_async_stream(user_input):
                yield chunk
            return

        async for chunk in self._run_async_stream_unstructured(user_input):
            yield chunk

    async def _run_async_unstructured(
        self, user_input: InputSchema | None = None
    ) -> OutputSchema:
        if user_input:
            self.history.initialize_turn()
            self.current_user_input = user_input
            self.history.add_message("user", user_input)

        self._prepare_messages()

        # Unstructured mode: use plain OpenAI client
        completion = await self._unstructured_client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **(self.model_api_parameters or {}),
        )

        content = ""
        try:
            if completion.choices and completion.choices[0].message:
                content = completion.choices[0].message.content or ""
        except Exception:
            content = ""

        response = self.output_schema(chat_message=content)
        self.history.add_message("assistant", response)
        return response

    async def _run_async_stream_unstructured(
        self, user_input: InputSchema | None = None
    ) -> AsyncGenerator[OutputSchema, None]:
        if user_input:
            self.history.initialize_turn()
            self.current_user_input = user_input
            self.history.add_message("user", user_input)

        self._prepare_messages()

        # Unstructured mode: use plain OpenAI client, plain text messages
        response_stream = await self._unstructured_client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **(self.model_api_parameters or {}),
            stream=True,
        )

        content = ""
        async for partial_response in response_stream:
            # Try to extract content from the streamed message
            if hasattr(partial_response, "choices") and partial_response.choices:
                choice = partial_response.choices[0]
                if (
                    hasattr(choice, "delta")
                    and choice.delta
                    and hasattr(choice.delta, "content")
                ):
                    delta_content = choice.delta.content or ""
                    content += delta_content
                elif (
                    hasattr(choice, "message")
                    and choice.message
                    and hasattr(choice.message, "content")
                ):
                    # Some APIs may use message.content in streaming
                    content += choice.message.content or ""

            # Yield the current accumulated content as output schema
            output = self.output_schema(chat_message=content)
            yield output

        if content:
            full_response_content = self.output_schema(chat_message=content)
            self.history.add_message("assistant", full_response_content)
