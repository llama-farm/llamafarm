import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import instructor
from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema  # type: ignore
from atomic_agents.agents.atomic_agent import (  # type: ignore
    ChatHistory,
    SystemPromptGenerator,
)
from openai import AsyncOpenAI

from core.settings import settings  # type: ignore

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
from config.datamodel import (  # noqa: E402
    AgentHandler,
    LlamaFarmConfig,
    Prompt,
    Provider,
    Runtime,
)

from core.logging import FastAPIStructLogger  # noqa: E402

logger = FastAPIStructLogger(__name__)


class ProjectChatOrchestratorAgentInputSchema(BaseIOSchema):
    """
    Input schema for the project chat orchestrator agent.
    """

    chat_message: str


class ProjectChatOrchestratorAgentOutputSchema(BaseIOSchema):
    """
    Output schema for the project chat orchestrator agent.
    This schema is intentionally simple to ensure compatibility with Ollama's JSON parsing.
    """

    chat_message: str


class ProjectChatOrchestratorAgent(
    AtomicAgent[
        ProjectChatOrchestratorAgentInputSchema,
        ProjectChatOrchestratorAgentOutputSchema,
    ]
):
    def __init__(self, project_config: LlamaFarmConfig):
        history = _get_history(project_config)
        client = _get_client(project_config)

        agent_config = AgentConfig(
            client=client,
            model=project_config.runtime.model,
            history=history,
            system_prompt_generator=LFSystemPromptGenerator(
                project_config=project_config
            ),
            model_api_parameters=project_config.runtime.model_api_parameters,
        )
        super().__init__(config=agent_config)


class LFSystemPromptGenerator(SystemPromptGenerator):
    def __init__(self, project_config: LlamaFarmConfig):
        logger.info(f"Project config: {project_config}")
        self.system_prompts = [
            prompt
            for prompt in (project_config.prompts or [])
            if prompt.role == "system"
        ]
        super().__init__()

    def generate_prompt(self) -> str:
        # return "\nYou are a helpful assistant that can answer questions and help with tasks."
        prompt_parts = []
        for prompt in self.system_prompts:
            prompt_parts.append(prompt.content)
            prompt_parts.append("")

        if self.context_providers:
            prompt_parts.append("# EXTRA INFORMATION AND CONTEXT")
            for provider in self.context_providers.values():
                info = provider.get_info()
                if info:
                    prompt_parts.append(f"## {provider.title}")
                    prompt_parts.append(info)
                    prompt_parts.append("")

        return "\n".join(prompt_parts)


def _prompt_to_content_schema(prompt: Prompt) -> BaseIOSchema:
    if prompt.role == "assistant":
        return ProjectChatOrchestratorAgentOutputSchema(
            chat_message=prompt.content,
        )
    elif prompt.role == "user":
        return ProjectChatOrchestratorAgentInputSchema(
            chat_message=prompt.content,
        )
    else:
        raise ValueError(f"Unsupported role: {prompt.role}")


def _populate_history_with_non_system_prompts(
    history: ChatHistory, project_config: LlamaFarmConfig
):
    for prompt in project_config.prompts or []:
        # Only add non-system prompts to the history
        if prompt.role != "system":
            history.add_message(
                role=prompt.role,
                content=_prompt_to_content_schema(prompt),
            )


def _get_history(project_config: LlamaFarmConfig) -> ChatHistory:
    history = ChatHistory()
    _populate_history_with_non_system_prompts(history, project_config)
    return history


def _get_client(project_config: LlamaFarmConfig) -> instructor.client.Instructor:
    # Use the configured instructor mode or default based on provider
    if project_config.runtime.instructor_mode is not None:
        # It's a string value
        mode_str = project_config.runtime.instructor_mode
        
        # Map the configured mode string to instructor.Mode
        try:
            mode = instructor.mode.Mode[mode_str.upper()]
            logger.info(f"Using configured instructor mode: {mode}")
        except KeyError:
            # Invalid mode specified
            raise ValueError(
                f"Invalid instructor_mode '{mode_str}'. "
                f"Common modes include: tools, json, md_json, anthropic_tools, gemini_json. "
                f"See instructor documentation for full list of supported modes."
            )
    elif project_config.runtime.provider == Provider.ollama:
        # Default to MD_JSON for Ollama as it's most compatible
        mode = instructor.Mode.MD_JSON
        logger.info(f"Using MD_JSON mode for Ollama provider (default)")
    else:
        mode = instructor.Mode.TOOLS
        logger.info(f"Using TOOLS mode (default for non-Ollama)")
    
    logger.info(f"Instructor mode: {mode}")

    if project_config.runtime.provider == Provider.openai:
        return instructor.from_openai(
            AsyncOpenAI(
                api_key=project_config.runtime.api_key,
                base_url=project_config.runtime.base_url,
            ),
            mode=mode,
        )
    if project_config.runtime.provider == Provider.ollama:
        client = instructor.from_openai(
            AsyncOpenAI(
                api_key=project_config.runtime.api_key or settings.ollama_api_key,
                base_url=project_config.runtime.base_url
                or f"{settings.ollama_host}/v1",
            ),
            mode=mode,
        )
        # Set max_retries to handle Ollama's occasional parsing issues
        client.max_retries = 2
        return client
    else:
        raise ValueError(f"Unsupported provider: {project_config.runtime.provider}")


class ProjectChatOrchestratorAgentFactory:
    @staticmethod
    def create_agent(project_config: LlamaFarmConfig) -> ProjectChatOrchestratorAgent:
        runtime = project_config.runtime
        _validate_runtime(runtime)

        handler = runtime.agent_handler
        logger.info("Creating chat agent", handler=handler.value, model=runtime.model)

        if handler == AgentHandler.structured_rag:
            if not _model_supports_structured_output(runtime.model):
                logger.warning(
                    "Model lacks structured-output support; falling back to simple_rag",
                    model=runtime.model,
                )
                handler = AgentHandler.simple_rag
            else:
                return ProjectChatOrchestratorAgent(project_config)

        if handler == AgentHandler.simple_rag:
            return SimpleChatAgent(project_config)

        if handler == AgentHandler.simple_chat:
            return SimpleChatAgent(project_config)

        if handler == AgentHandler.classifier:
            # Placeholder: classifier behaviour to be implemented.
            return SimpleChatAgent(project_config)

        raise ValueError(f"Unsupported agent handler: {handler}")


class SimpleChatAgent:
    def __init__(self, project_config: LlamaFarmConfig):
        self.project_config = project_config
        self.client = _build_async_client(project_config)
        self.history: list[dict[str, str]] = []
        self.context_providers: dict[str, Any] = {}

    def register_context_provider(self, name: str, provider: Any) -> None:
        self.context_providers[name] = provider

    def reset_history(self) -> None:
        self.history = []

    async def run_async(self, input_schema: ProjectChatOrchestratorAgentInputSchema):
        user_message = input_schema.chat_message
        messages = self._build_messages(user_message)
        response = await self.client.chat.completions.create(
            model=self.project_config.runtime.model,
            messages=messages,
            stream=False,
            **(self.project_config.runtime.model_api_parameters or {}),
        )
        content = response.choices[0].message.content or ""
        self._append_history("user", user_message)
        self._append_history("assistant", content)
        return SimpleNamespace(chat_message=content)

    async def run_async_stream(self, input_schema: ProjectChatOrchestratorAgentInputSchema):
        user_message = input_schema.chat_message
        messages = self._build_messages(user_message)
        stream = await self.client.chat.completions.create(
            model=self.project_config.runtime.model,
            messages=messages,
            stream=True,
            **(self.project_config.runtime.model_api_parameters or {}),
        )

        assembled = ""
        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            content = getattr(delta, "content", None)
            if not content:
                continue
            assembled += content
            yield SimpleNamespace(chat_message=assembled)

        self._append_history("user", user_message)
        self._append_history("assistant", assembled)

    def _build_messages(self, user_message: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for prompt in self.project_config.prompts or []:
            if prompt.role == "system" and prompt.content:
                messages.append({"role": "system", "content": prompt.content})

        if self.context_providers:
            context_parts: list[str] = []
            for provider in self.context_providers.values():
                info = provider.get_info()
                if info:
                    context_parts.append(f"## {provider.title}\n{info}")
            if context_parts:
                context_block = "# EXTRA INFORMATION AND CONTEXT\n" + "\n\n".join(
                    context_parts
                )
                messages.append({"role": "system", "content": context_block})

        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _append_history(self, role: str, content: str) -> None:
        if not content:
            return
        if role == "assistant" and self.history and self.history[-1]["role"] == "assistant":
            self.history[-1]["content"] = content
        else:
            self.history.append({"role": role, "content": content})


def _build_async_client(project_config: LlamaFarmConfig) -> AsyncOpenAI:
    if project_config.runtime.provider == Provider.openai:
        return AsyncOpenAI(
            api_key=project_config.runtime.api_key,
            base_url=project_config.runtime.base_url,
        )
    if project_config.runtime.provider == Provider.ollama:
        return AsyncOpenAI(
            api_key=project_config.runtime.api_key or settings.ollama_api_key,
            base_url=project_config.runtime.base_url or f"{settings.ollama_host}/v1",
        )
    raise ValueError(f"Unsupported provider: {project_config.runtime.provider}")


def _model_supports_structured_output(model: str) -> bool:
    lowered = model.lower()
    unsupported_tokens = (
        "tinyllama",
        "tiny-stories",
        "tinystories",
    )
    return not any(token in lowered for token in unsupported_tokens)


def _validate_runtime(runtime: Runtime) -> None:
    if runtime.provider == Provider.openai and runtime.base_url:
        lowered = runtime.base_url.lower()
        if "11434" in lowered or "ollama" in lowered:
            raise ValueError(
                "runtime.provider is set to 'openai' but base_url points to an Ollama host. Set provider to 'ollama'."
            )
    if runtime.provider == Provider.ollama and runtime.base_url:
        lowered = runtime.base_url.lower()
        if "openai" in lowered:
            logger.warning(
                "runtime.provider is 'ollama' but base_url references OpenAI; continuing",
                base_url=runtime.base_url,
            )
