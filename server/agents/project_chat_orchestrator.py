import sys
from pathlib import Path

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
from config.datamodel import LlamaFarmConfig, Prompt, Provider  # noqa: E402

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
    """

    chat_message: str


class ProjectChatOrchestratorAgent(
    AtomicAgent[
        ProjectChatOrchestratorAgentInputSchema,
        ProjectChatOrchestratorAgentOutputSchema,
    ]
):
    def __init__(self, project_config: LlamaFarmConfig, model_name: str | None = None):
        history = _get_history(project_config)
        runtime_config = _get_runtime_config(project_config, model_name)
        client = _get_client_from_runtime(runtime_config)

        agent_config = AgentConfig(
            client=client,
            model=runtime_config.get("model"),
            history=history,
            system_prompt_generator=LFSystemPromptGenerator(
                project_config=project_config
            ),
            model_api_parameters=runtime_config.get("parameters", {}),
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


def _get_runtime_config(project_config: LlamaFarmConfig, model_name: str | None = None) -> dict:
    """Get runtime configuration from the new multi-model format."""
    
    # Require runtime_models to be present
    if not hasattr(project_config, 'runtime_models') or not project_config.runtime_models:
        raise ValueError(
            "No runtime_models configured. Please update your llamafarm.yaml to use the new multi-model format:\n"
            "  default_model: primary\n"
            "  runtime_models:\n"
            "    - name: primary\n"
            "      provider: ollama\n"
            "      model: llama3.1:8b\n"
            "      ...\n"
        )
    
    # If model_name specified, find that model
    if model_name:
        for model in project_config.runtime_models:
            if model.name == model_name:
                return model.model_dump()
        
        # Model not found - provide helpful error with available models
        available = ", ".join([m.name for m in project_config.runtime_models])
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available}"
        )
    
    # Use default_model if specified
    if hasattr(project_config, 'default_model') and project_config.default_model:
        for model in project_config.runtime_models:
            if model.name == project_config.default_model:
                return model.model_dump()
        
        # Default model not found - this is a config error
        raise ValueError(
            f"Default model '{project_config.default_model}' not found in runtime_models"
        )
    
    # No default specified, use first model with a warning
    logger.warning(
        "No default_model specified, using first model in runtime_models list. "
        "Consider setting default_model in your configuration."
    )
    return project_config.runtime_models[0].model_dump()


def _get_client_from_runtime(runtime_config: dict) -> instructor.client.Instructor:
    """Create client from runtime configuration dictionary."""
    instructor_mode = runtime_config.get("instructor_mode", "TOOLS")
    
    # Handle both string and enum values
    if hasattr(instructor_mode, 'value'):
        # It's an enum, get its value
        instructor_mode = instructor_mode.value
    elif instructor_mode is None:
        instructor_mode = "TOOLS"
    
    # Convert to uppercase for Mode lookup
    mode = instructor.mode.Mode[instructor_mode.upper()]

    provider = runtime_config.get("provider", "")
    
    # Handle both string and enum values for provider
    if hasattr(provider, 'value'):
        # It's an enum, get its value
        provider = provider.value
    
    provider = str(provider).lower()
    
    # Get base_url and convert from Pydantic URL if needed
    base_url = runtime_config.get("base_url")
    if base_url and hasattr(base_url, '__str__'):
        base_url = str(base_url)
    
    if provider == "openai":
        return instructor.from_openai(
            AsyncOpenAI(
                api_key=runtime_config.get("api_key"),
                base_url=base_url,
            ),
            mode=mode,
        )
    if provider == "ollama":
        return instructor.from_openai(
            AsyncOpenAI(
                api_key=runtime_config.get("api_key") or settings.ollama_api_key,
                base_url=base_url or f"{settings.ollama_host}/v1",
            ),
            mode=mode,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _get_client(project_config: LlamaFarmConfig) -> instructor.client.Instructor:
    """Create client from project configuration (uses default model)."""
    runtime_config = _get_runtime_config(project_config, None)
    return _get_client_from_runtime(runtime_config)


class ProjectChatOrchestratorAgentFactory:
    @staticmethod
    def create_agent(project_config: LlamaFarmConfig, model_name: str | None = None) -> ProjectChatOrchestratorAgent:
        return ProjectChatOrchestratorAgent(project_config, model_name)
