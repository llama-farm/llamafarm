import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from config.datamodel import (
    LlamaFarmConfig,
    Model,
    PromptMessage,
    PromptSet,
    Provider,
    Runtime,
    Version,
)
from pydantic import BaseModel

from agents.base.history import LFChatCompletionUserMessageParam
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

    # Mock ProjectService.get_project to avoid file system dependency
    from services.project_service import Project, ProjectService

    def mock_get_project(namespace: str, project_id: str):
        return Project(
            namespace=namespace,
            name=project_id,
            config=config,
            validation_error=None,
            last_modified=None,
        )

    monkeypatch.setattr(ProjectService, "get_project", mock_get_project)

    # Intercept LFAgent.run_async to capture messages (no network calls)
    from agents.base.agent import LFAgent

    async def fake_run_async(self, *, messages=None, tools=None, extra_body=None):
        # LFAgent.run_async adds messages to history if provided
        if messages:
            for message in messages:
                self.history.add_message(message)
        # Capture messages after preparation
        prepared_messages = self._prepare_messages()
        captured["messages"] = prepared_messages
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
        messages=[LFChatCompletionUserMessageParam(role="user", content="Hello there")]
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


def test_schema_field_is_preserved_and_applied(monkeypatch):
    cfg = make_config()
    config_payload = cfg.model_dump(mode="python", by_alias=True)
    config_payload["schema"] = "schemas/person.py::Person"
    config_with_schema = LlamaFarmConfig.model_validate(config_payload)
    assert config_with_schema.schema_ == "schemas/person.py::Person"

    class Person(BaseModel):
        name: str

    class DummyClient:
        def __init__(self):
            self.response_model = None

        def set_response_model(self, model):
            self.response_model = model

    agent = ChatOrchestratorAgent.__new__(ChatOrchestratorAgent)
    agent._project_config = config_with_schema
    monkeypatch.setattr(agent, "_load_response_model", lambda _: Person)
    client = DummyClient()
    agent._apply_response_model(client)
    assert client.response_model is Person


def test_apply_response_model_falls_back_to_response_model_attr(monkeypatch):
    cfg = make_config()
    config_payload = cfg.model_dump(mode="python", by_alias=True)
    config_payload["schema"] = "schemas/person.py::Person"
    config_with_schema = LlamaFarmConfig.model_validate(config_payload)

    class Person(BaseModel):
        name: str

    class LegacyClient:
        def __init__(self):
            self.response_model = None

    agent = ChatOrchestratorAgent.__new__(ChatOrchestratorAgent)
    agent._project_config = config_with_schema
    monkeypatch.setattr(agent, "_load_response_model", lambda _: Person)
    client = LegacyClient()
    agent._apply_response_model(client)
    assert client.response_model is Person


def test_apply_response_model_raises_for_unsupported_client(monkeypatch):
    cfg = make_config()
    config_payload = cfg.model_dump(mode="python", by_alias=True)
    config_payload["schema"] = "schemas/person.py::Person"
    config_with_schema = LlamaFarmConfig.model_validate(config_payload)

    class Person(BaseModel):
        name: str

    class UnsupportedClient:
        pass

    agent = ChatOrchestratorAgent.__new__(ChatOrchestratorAgent)
    agent._project_config = config_with_schema
    monkeypatch.setattr(agent, "_load_response_model", lambda _: Person)

    with pytest.raises(TypeError, match="does not support structured output"):
        agent._apply_response_model(UnsupportedClient())


def test_load_response_model_missing_class_has_clear_error(tmp_path: Path):
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    schema_file = schemas_dir / "person.py"
    schema_file.write_text(
        "from pydantic import BaseModel\n\nclass Person(BaseModel):\n    name: str\n",
        encoding="utf-8",
    )

    agent = ChatOrchestratorAgent.__new__(ChatOrchestratorAgent)
    agent._project_dir = str(tmp_path)

    with pytest.raises(ValueError, match="is not a valid class"):
        agent._load_response_model("schemas/person.py::MissingClass")


def test_load_response_model_wraps_exec_errors(tmp_path: Path):
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    schema_file = schemas_dir / "broken.py"
    schema_file.write_text(
        "import definitely_missing_module\n\nclass Person:\n    pass\n",
        encoding="utf-8",
    )

    agent = ChatOrchestratorAgent.__new__(ChatOrchestratorAgent)
    agent._project_dir = str(tmp_path)

    with pytest.raises(ValueError, match="Failed to load schema module"):
        agent._load_response_model("schemas/broken.py::Person")
