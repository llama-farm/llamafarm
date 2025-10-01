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
from atomic_agents.base.base_io_schema import BaseIOSchema  # type: ignore
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
        description="The role of the system in the conversation. None means no system prompt.",
    )
    model_api_parameters: dict[str, Any] | None = Field(
        None,
        description="Additional parameters passed to the API provider.",
    )

    # Allow non-pydantic types like AsyncOpenAI
    model_config = {"arbitrary_types_allowed": True}


class LFAgent[InputSchema: BasicChatInputSchema, OutputSchema: BasicChatOutputSchema](
    AtomicAgent[BasicChatInputSchema, OutputSchema]
):
    def __init__(self, config: LFAgentConfig):
        # Check if client is already wrapped with instructor (structured mode)
        is_instructor_client = isinstance(config.client, instructor.client.AsyncInstructor)

        # AtomicAgent requires instructor client, so wrap plain AsyncOpenAI
        if not is_instructor_client:
            client_for_parent = instructor.from_openai(config.client, mode=instructor.Mode.MD_JSON)
        else:
            client_for_parent = config.client

        atomic_agent_config = AgentConfig(
            client=client_for_parent,
            model=config.model,
            history=config.history,
            system_prompt_generator=config.system_prompt_generator,
            system_role=config.system_role,
            model_api_parameters=config.model_api_parameters,
        )
        super().__init__(config=atomic_agent_config)

        # Store original client and mode
        self.client = config.client  # Original unwrapped client for API calls
        self._use_structured_output = is_instructor_client

    def _prepare_messages(self):
        """Prepare messages for unstructured mode - no JSON wrapping."""
        super()._prepare_messages()

    async def run_async_stream(
        self, user_input: InputSchema | None = None
    ) -> AsyncGenerator[OutputSchema, None]:
        if user_input:
            self.history.initialize_turn()
            self.current_user_input = user_input

            self.history.add_message("user", user_input)

        self._prepare_messages()

        if isinstance(self.client, instructor.client.AsyncInstructor):
            logger.debug(f"Using STRUCTURED output with instructor client, response_model={self.output_schema}")
            response_stream = self.client.chat.completions.create_partial(
                model=self.model,
                messages=self.messages,
                response_model=self.output_schema,
                **self.model_api_parameters,
                stream=True,
            )

            last_response = None
            async for partial_response in response_stream:
                last_response = partial_response
                yield partial_response

            if last_response:
                full_response_content = self.output_schema(**last_response.model_dump())
                self.history.add_message("assistant", full_response_content)
        else:
            response_stream = await self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                **(self.model_api_parameters or {}),
                stream=True,
            )

            content = ""
            last_response = None
            async for partial_response in response_stream:
                last_response = partial_response
                try:
                    # Try to extract content from the streamed message
                    if (
                        hasattr(partial_response, "choices")
                        and partial_response.choices
                    ):
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
                except Exception:
                    pass  # If structure is unexpected, just skip

                # Yield the current accumulated content as output schema
                output = self.output_schema(chat_message=content)
                yield output

            if content:
                full_response_content = self.output_schema(chat_message=content)
                self.history.add_message("assistant", full_response_content)

    async def run_async(self, user_input: BaseIOSchema | None = None) -> BaseIOSchema:
        # If using AsyncInstructor, defer to the base implementation which expects Instructor
        if isinstance(self.client, instructor.client.AsyncInstructor):
            return await super().run_async(user_input)

        # Raw AsyncOpenAI implementation
        if user_input:
            self.history.initialize_turn()
            self.current_user_input = user_input

            self.history.add_message("user", user_input)

        self._prepare_messages()

        completion = await self.client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            messages=self.messages,  # type: ignore[attr-defined]
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
