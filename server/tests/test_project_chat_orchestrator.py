import tempfile
from types import SimpleNamespace

from agents.base.history import LFChatCompletionUserMessageParam
import pytest

from config.datamodel import (
    LlamaFarmConfig,
    PromptMessage,
    Model,
    PromptSet,
    Provider,
    Runtime,
    Version,
)

from agents.chat_orchestrator import (
    ChatOrchestratorAgent,
    ChatOrchestratorAgentFactory,
)
from context_providers.rag_context_provider import (
    ChunkItem,
    RAGContextProvider,
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


def make_config(model: str = "tinyllama:latest") -> LlamaFarmConfig:
    return LlamaFarmConfig(
        version=Version.v1,
        name="demo",
        namespace="default",
        runtime=Runtime(
            default_model="default",
            models=[
                Model(
                    name="default",
                    description=None,
                    provider_config=None,
                    prompts=None,
                    provider=Provider.ollama,
                    model=model,
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                    instructor_mode="tools",
                    model_api_parameters={},
                )
            ],
        ),
        prompts=[
            PromptSet(
                name="default",
                messages=[
                    PromptMessage(role="system", content="You are a helpful assistant.")
                ],
            )
        ],
        rag=None,  # Don't set RAG if not needed, avoids validation errors
        datasets=[],
        mcp=None,
    )


@pytest.mark.asyncio
async def test_factory_returns_unstructured_agent(monkeypatch, dummy_client):
    config = make_config()

    with tempfile.TemporaryDirectory() as project_dir:
        agent = await ChatOrchestratorAgentFactory.create_agent(
            project_config=config, project_dir=project_dir
        )
    assert isinstance(agent, ChatOrchestratorAgent)
    # Client is now private (_client)
    assert hasattr(agent, "_client")
    assert agent.model_name == "default"


@pytest.mark.asyncio
async def test_simple_rag_agent_injects_context(monkeypatch):
    captured = {}

    config = make_config(model="tinyllama:latest")

    # Intercept LFAgent.run_async to capture messages (no network calls)
    from agents.base.agent import LFAgent

    async def fake_run_async(self, *, user_input=None, tools=None):
        # LFAgent.run_async adds user_input to history if provided
        if user_input:
            self.history.add_message(user_input)
        # Capture messages after preparation
        messages = self._prepare_messages()
        captured["messages"] = messages
        # Return a simple string response (not a schema object)
        message = SimpleNamespace(role="assistant", content="ok", tool_calls=None)
        choice = SimpleNamespace(index=0, finish_reason="stop", message=message)
        return SimpleNamespace(choices=[choice], model=self.model_name)

    monkeypatch.setattr(LFAgent, "run_async", fake_run_async)

    with tempfile.TemporaryDirectory() as project_dir:
        agent = await ChatOrchestratorAgentFactory.create_agent(
            project_config=config,
            project_dir=project_dir,
            model_name="default",
        )
    assert isinstance(agent, ChatOrchestratorAgent)

    provider = RAGContextProvider(title="Context")
    provider.chunks.append(
        ChunkItem(content="Important note", metadata={"source": "doc"})
    )
    agent.register_context_provider("project_chat_context", provider)

    await agent.run_async(
        user_input=LFChatCompletionUserMessageParam(role="user", content="Hello there")
    )

    messages = captured.get("messages", [])
    assert messages

    # Messages are now LFAgentChatMessage objects
    assert len(messages) >= 2
    # First message should be system prompt
    assert messages[0]["role"] == "system"
    assert "You are a helpful assistant." in messages[0]["content"]
    # Check that RAG context was injected
    assert any(
        "Important note" in msg["content"]
        for msg in messages
        if msg["role"] == "system"
    )
    # Last message should be the user input
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Hello there"
