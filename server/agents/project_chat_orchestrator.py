import sys
from pathlib import Path

import instructor
from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema  # type: ignore
from atomic_agents.agents.atomic_agent import ChatHistory  # type: ignore
from openai import OpenAI

from core.settings import settings  # type: ignore

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
from config.datamodel import LlamaFarmConfig  # noqa: E402


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
                model=settings.ollama_model,
                history=history,
                # TODO: add a SystemPromptGenerator using the project's prompt config
            )
        )


def _get_client(project_config: LlamaFarmConfig) -> instructor.client.Instructor:
    # TODO: use the values from the project's model strategy config
    # api_key=project_config.rag.strategies[0].api_key
    # base_url=project_config.rag.strategies[0].base_url

    api_key = settings.ollama_api_key
    base_url = settings.ollama_host

    return instructor.from_openai(OpenAI(api_key=api_key, base_url=base_url))


class ProjectChatOrchestratorAgentFactory:
    @staticmethod
    def create_agent(project_config: LlamaFarmConfig) -> ProjectChatOrchestratorAgent:
        return ProjectChatOrchestratorAgent(project_config)
