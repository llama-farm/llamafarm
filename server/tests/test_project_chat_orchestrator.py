import asyncio
from types import SimpleNamespace

import pytest

from agents.project_chat_orchestrator import (
    ProjectChatOrchestratorAgent,
    ProjectChatOrchestratorAgentInputSchema,
    ProjectChatOrchestratorAgentFactory,
    SimpleChatAgent,
)
from config.datamodel import AgentHandler, LlamaFarmConfig, Prompt, Provider, Runtime
from context_providers.project_chat_context_provider import (
    ChunkItem,
    ProjectChatContextProvider,
)


@pytest.fixture()
def dummy_client():
    class DummyCompletions:
        async def create(self, *args, **kwargs):  # pragma: no cover - unused in factory test
            return None

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        chat = DummyChat()

    return DummyClient()


def make_config(handler: AgentHandler, model: str = "tinyllama:latest") -> LlamaFarmConfig:
    return LlamaFarmConfig(
        version="v1",
        name="demo",
        namespace="default",
        runtime=Runtime(
            provider=Provider.ollama,
            model=model,
            base_url="http://localhost:11434/v1",
            agent_handler=handler,
        ),
        prompts=[Prompt(role="system", content="You are a helpful assistant.")],
    )


def test_factory_returns_simple_chat_agent(monkeypatch, dummy_client):
    config = make_config(AgentHandler.simple_chat)

    monkeypatch.setattr(
        "agents.project_chat_orchestrator._build_async_client",
        lambda _cfg: dummy_client,
    )

    agent = ProjectChatOrchestratorAgentFactory.create_agent(config)

    assert isinstance(agent, SimpleChatAgent)
    assert agent.client is dummy_client


def test_factory_returns_structured_rag_agent(monkeypatch, dummy_client):
    config = make_config(AgentHandler.structured_rag, model="qwen3:8b")

    class DummyAgent(ProjectChatOrchestratorAgent):
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        "agents.project_chat_orchestrator.ProjectChatOrchestratorAgent",
        DummyAgent,
    )

    agent = ProjectChatOrchestratorAgentFactory.create_agent(config)

    assert isinstance(agent, DummyAgent)


def test_rag_agent_falls_back_for_unsupported_model(monkeypatch, dummy_client):
    config = make_config(AgentHandler.structured_rag, model="tinyllama:latest")

    monkeypatch.setattr(
        "agents.project_chat_orchestrator._build_async_client",
        lambda _cfg: dummy_client,
    )

    agent = ProjectChatOrchestratorAgentFactory.create_agent(config)

    assert isinstance(agent, SimpleChatAgent)


@pytest.mark.asyncio
async def test_simple_rag_agent_injects_context(monkeypatch):
    captured = {}

    class DummyCompletions:
        async def create(self, *_, **kwargs):  # pragma: no cover - networking stub
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        chat = DummyChat()

    config = make_config(AgentHandler.simple_rag, model="tinyllama:latest")

    monkeypatch.setattr(
        "agents.project_chat_orchestrator._build_async_client",
        lambda _cfg: DummyClient(),
    )

    agent = ProjectChatOrchestratorAgentFactory.create_agent(config)
    assert isinstance(agent, SimpleChatAgent)

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
    assert messages[0]["role"] == "system"
    assert "You are a helpful assistant." in messages[0]["content"]
    assert any("Important note" in msg["content"] for msg in messages if msg["role"] == "system")
    assert messages[-1]["role"] == "user" and messages[-1]["content"] == "Hello there"
