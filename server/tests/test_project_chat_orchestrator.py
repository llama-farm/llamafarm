import tempfile

from agents.base.history import LFAgentChatMessage
import pytest

from config.datamodel import (
    LlamaFarmConfig,
    Message,
    Model,
    Prompt,
    PromptFormat,
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


def make_config(
    prompt_format: PromptFormat, model: str = "tinyllama:latest"
) -> LlamaFarmConfig:
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
                    prompt_format=prompt_format,
                    api_key="ollama",
                    instructor_mode="tools",
                    model_api_parameters={},
                )
            ],
        ),
        prompts=[
            Prompt(
                name="default",
                messages=[
                    Message(role="system", content="You are a helpful assistant.")
                ],
            )
        ],
        rag=None,  # Don't set RAG if not needed, avoids validation errors
        datasets=[],
        mcp=None,
    )


@pytest.mark.asyncio
async def test_factory_returns_unstructured_agent(monkeypatch, dummy_client):
    config = make_config(PromptFormat.unstructured)

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

    config = make_config(PromptFormat.unstructured, model="tinyllama:latest")

    # Intercept LFAgent.run_async to capture messages (no network calls)
    from agents.base.agent import LFAgent

    async def fake_run_async(self, *, user_input=None):
        # LFAgent.run_async adds user_input to history if provided
        if user_input:
            self.history.add_message(user_input)
        # Capture messages after preparation
        messages = self._prepare_messages()
        captured["messages"] = messages
        # Return a simple string response (not a schema object)
        return "ok"

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
        user_input=LFAgentChatMessage(role="user", content="Hello there")
    )

    messages = captured.get("messages", [])
    assert messages

    # Messages are now LFAgentChatMessage objects
    assert len(messages) >= 2
    # First message should be system prompt
    assert messages[0].role == "system"
    assert "You are a helpful assistant." in messages[0].content
    # Check that RAG context was injected
    assert any(
        "Important note" in msg.content for msg in messages if msg.role == "system"
    )
    # Last message should be the user input
    assert messages[-1].role == "user"
    assert messages[-1].content == "Hello there"
