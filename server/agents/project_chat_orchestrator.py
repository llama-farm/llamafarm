import sys
from pathlib import Path
from typing import override

import instructor
from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema  # type: ignore
from atomic_agents.agents.atomic_agent import ChatHistory, SystemPromptGenerator  # type: ignore
from openai import OpenAI

from core.settings import settings  # type: ignore

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
from config.datamodel import LlamaFarmConfig, Provider  # noqa: E402


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
    def __init__(self, project_config: LlamaFarmConfig):
        history = ChatHistory()
        history.add_message(
            "assistant",
            ProjectChatOrchestratorAgentOutputSchema(
                chat_message="Hello! How can I assist you today?"
            ),
        )
        client = _get_client(project_config)
        super().__init__(
            config=AgentConfig(
                client=client,
                model=project_config.runtime.model,
                history=history,
                system_prompt_generator=LFSystemPromptGenerator(
                    project_config=project_config
                ),
                system_role="system",
                model_api_parameters=project_config.runtime.model_api_parameters,
            )
        )


class LFSystemPromptGenerator(SystemPromptGenerator):
    def __init__(self, project_config: LlamaFarmConfig):
        self.prompt = project_config.prompts[0]
        super().__init__()

    def _generate_prompt_from_sections(self) -> str:
        sections = [
            (section.title, section.content) for section in self.prompt.sections or []
        ]

        prompt_parts = []

        for title, content in sections:
            if content:
                prompt_parts.append(f"# {title}")
                prompt_parts.extend(f"- {item}" for item in content)
                prompt_parts.append("")

        if self.context_providers:
            prompt_parts.append("# EXTRA INFORMATION AND CONTEXT")
            for provider in self.context_providers.values():
                if info := provider.get_info():
                    prompt_parts.extend((f"## {provider.title}", info, ""))
        return "\n".join(prompt_parts).strip()

    def generate_prompt(self) -> str:
        if self.prompt.raw_text:
            return self.prompt.raw_text
        elif self.prompt.sections:
            return self._generate_prompt_from_sections()
        return ""


def _get_client(project_config: LlamaFarmConfig) -> instructor.client.Instructor:
    if project_config.runtime.provider == Provider.openai:
        return instructor.from_openai(
            OpenAI(
                api_key=project_config.runtime.api_key,
                base_url=project_config.runtime.base_url,
            )
        )
    else:
        raise ValueError(f"Unsupported provider: {project_config.runtime.provider}")


class ProjectChatOrchestratorAgentFactory:
    @staticmethod
    def create_agent(project_config: LlamaFarmConfig) -> ProjectChatOrchestratorAgent:
        return ProjectChatOrchestratorAgent(project_config)
