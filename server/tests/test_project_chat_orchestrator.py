import tempfile

import pytest
from config.datamodel import (
    PromptFormat,
    LlamaFarmConfig,
    Prompt,
    Message,
    Provider,
    Runtime,
    Version,
    Model,
)

from agents.project_chat_orchestrator import (
    ProjectChatOrchestratorAgent,
    ProjectChatOrchestratorAgentFactory,
    ProjectChatOrchestratorAgentInputSchema,
)
from context_providers.project_chat_context_provider import (
    ChunkItem,
    ProjectChatContextProvider,
)


@pytest.fixture()
def dummy_client():
    class DummyCompletions:
        async def create(
            self, *args, **kwargs
        ):  # pragma: no cover - unused in factory test
            return None

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        chat = DummyChat()

    return DummyClient()


def make_config(
    prompt_format: PromptFormat, model: str = "tinyllama:latest"
) -> LlamaFarmConfig:
    return LlamaFarmConfig(
        version=Version.v1,
        name="demo",
        namespace="default",
        runtime=Runtime(
            models=[
                Model(
                    name="default",
                    provider=Provider.ollama,
                    model=model,
                    base_url="http://localhost:11434/v1",
                    prompt_format=prompt_format,
                    api_key="ollama",
                    instructor_mode="tools",
                    model_api_parameters={},
                )
            ]
        ),
        prompts=[Prompt(name="default", messages=[Message(role="system", content="You are a helpful assistant.")])],
        rag=None,  # Don't set RAG if not needed, avoids validation errors
        datasets=[],
    )


def test_factory_returns_unstructured_agent(monkeypatch, dummy_client):
    config = make_config(PromptFormat.unstructured)

    with tempfile.TemporaryDirectory() as project_dir:
        agent = ProjectChatOrchestratorAgentFactory.create_agent(config, project_dir)
    assert isinstance(agent, ProjectChatOrchestratorAgent)
    assert hasattr(agent, "client")


def test_factory_returns_structured_agent(monkeypatch, dummy_client):
    config = make_config(PromptFormat.structured, model="qwen3:8b")

    class DummyAgent(ProjectChatOrchestratorAgent):
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        "agents.project_chat_orchestrator.ProjectChatOrchestratorAgent",
        DummyAgent,
    )

    with tempfile.TemporaryDirectory() as project_dir:
        agent = ProjectChatOrchestratorAgentFactory.create_agent(config, project_dir)
    assert isinstance(agent, DummyAgent)


def test_structured_agent_falls_back_for_unsupported_model(monkeypatch, dummy_client):
    config = make_config(PromptFormat.structured, model="tinyllama:latest")

    with tempfile.TemporaryDirectory() as project_dir:
        agent = ProjectChatOrchestratorAgentFactory.create_agent(config, project_dir)
    assert isinstance(agent, ProjectChatOrchestratorAgent)


@pytest.mark.asyncio
async def test_simple_rag_agent_injects_context(monkeypatch):
    import json

    captured = {}

    config = make_config(PromptFormat.unstructured, model="tinyllama:latest")

    # Intercept LFAgent.run_async to capture messages (no network calls)
    from agents.agent import LFAgent

    async def fake_run_async(self, user_input=None):
        if user_input:
            self.history.initialize_turn()
            self.current_user_input = user_input
            self.history.add_message("user", user_input)
        self._prepare_messages()
        captured["messages"] = self.messages
        return self.output_schema(chat_message="ok")

    monkeypatch.setattr(LFAgent, "run_async", fake_run_async)

    with tempfile.TemporaryDirectory() as project_dir:
        agent = ProjectChatOrchestratorAgentFactory.create_agent(config, project_dir)
    assert isinstance(agent, ProjectChatOrchestratorAgent)

    provider = ProjectChatContextProvider(title="Context")
    provider.chunks.append(
        ChunkItem(content="Important note", metadata={"source": "doc"})
    )
    agent.register_context_provider("project_chat_context", provider)

    await agent.run_async(
        ProjectChatOrchestratorAgentInputSchema(chat_message="Hello there")
    )

    messages = captured.get("messages", [])
    assert messages

    def norm_content(val):
        if hasattr(val, "chat_message"):
            return val.chat_message
        if isinstance(val, dict) and "chat_message" in val:
            return val.get("chat_message")
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict) and "chat_message" in parsed:
                    return parsed.get("chat_message")
            except Exception:
                pass
        return val

    assert messages[0]["role"] == "system"
    assert "You are a helpful assistant." in norm_content(messages[0]["content"])
    assert any(
        "Important note" in norm_content(msg["content"])
        for msg in messages
        if msg["role"] == "system"
    )
    assert (
        messages[-1]["role"] == "user"
        and norm_content(messages[-1]["content"]) == "Hello there"
    )
