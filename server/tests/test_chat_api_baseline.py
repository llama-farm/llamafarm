"""Baseline chat API behaviour highlighting current session issues."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import llama_farm_api
from config.datamodel import LlamaFarmConfig, Prompt, Message, Provider, Runtime, Model
from services.project_chat_service import FALLBACK_ECHO_RESPONSE


@pytest.fixture(autouse=True)
def reset_agent_sessions():
    from api.routers.projects import projects

    with projects._agent_sessions_lock:  # noqa: SLF001
        projects.agent_sessions.clear()
    yield
    with projects._agent_sessions_lock:  # noqa: SLF001
        projects.agent_sessions.clear()


@pytest.fixture()
def app_client(mocker):
    default_config = LlamaFarmConfig(
        version="v1",
        name="llamafarm-1",
        namespace="default",
        prompts=[
            Prompt(name="default", messages=[Message(role="system", content="You are the default project assistant.")])
        ],
        runtime=Runtime(
            models=[
                Model(
                    name="default",
                    provider=Provider.ollama,
                    model="dummy-model",
                )
            ]
        ),
    )
    seed_config = LlamaFarmConfig(
        version="v1",
        name="project_seed",
        namespace="llamafarm",
        prompts=[Prompt(name="default", messages=[Message(role="system", content="You are the seed project assistant.")])],
        runtime=Runtime(
            models=[
                Model(
                    name="default",
                    provider=Provider.ollama,
                    model="dummy-model",
                )
            ]
        ),
    )

    def load_config(namespace: str, project_id: str):
        if namespace == "llamafarm" and project_id == "project_seed":
            return seed_config
        if namespace == "default" and project_id == "llamafarm-1":
            return default_config
        raise AssertionError(f"Unexpected project request: {namespace}/{project_id}")

    mocker.patch(
        "api.routers.projects.projects.ProjectService.load_config",
        side_effect=load_config,
    )
    mocker.patch(
        "api.routers.projects.projects.ProjectService.get_project_dir",
        return_value="/tmp",
    )

    class StubAgent:
        def __init__(self, tag: str):
            self.tag = tag
            self.model_name = "stub-model"
            self.context_providers = {}
            self.history = []
            self._persist_enabled = False
            self._session_id = None

        def register_context_provider(self, name: str, provider):
            self.context_providers[name] = provider

        def enable_persistence(self, *, session_id: str):
            self._persist_enabled = True
            self._session_id = session_id

        async def run_async(self, input_schema):
            self.history.append(input_schema.chat_message)
            return SimpleNamespace(
                chat_message=f"{self.tag}:{input_schema.chat_message}"
            )

        async def run_async_stream(self, input_schema):
            self.history.append(input_schema.chat_message)
            if input_schema.chat_message == "no-content":
                # Simulate providers that stream no usable content
                yield SimpleNamespace(chat_message="")
            else:
                # Mirror current bug: stream yields exactly the user input
                yield SimpleNamespace(chat_message=input_schema.chat_message)

    def make_agent(
        project_config: LlamaFarmConfig,
        project_dir: str,
        model_name: str | None = None,
        session_id: str | None = None,
    ):
        tag = f"{project_config.namespace}/{project_config.name}"
        agent = StubAgent(tag)
        # Add model_name attribute for fallback testing
        agent.model_name = model_name or "test-model"
        return agent

    mocker.patch(
        "api.routers.projects.projects.ProjectChatOrchestratorAgentFactory.create_agent",
        side_effect=make_agent,
    )

    return TestClient(llama_farm_api())


def _post_chat(
    client: TestClient,
    namespace: str,
    project: str,
    payload: dict,
    session: str | None = None,
):
    headers = {"Content-Type": "application/json"}
    if session:
        headers["X-Session-ID"] = session
    return client.post(
        f"/v1/projects/{namespace}/{project}/chat/completions",
        json=payload,
        headers=headers,
    )


def _delete_session(client: TestClient, namespace: str, project: str, session: str):
    return client.delete(f"/v1/projects/{namespace}/{project}/chat/sessions/{session}")


def _delete_all_sessions(client: TestClient, namespace: str, project: str):
    return client.delete(f"/v1/projects/{namespace}/{project}/chat/sessions")


def _stream_chat(
    client: TestClient,
    namespace: str,
    project: str,
    payload: dict,
    session: str | None = None,
) -> str:
    headers = {"Content-Type": "application/json"}
    if session:
        headers["X-Session-ID"] = session
    chunks: list[str] = []
    with client.stream(
        "POST",
        f"/v1/projects/{namespace}/{project}/chat/completions",
        json=payload,
        headers=headers,
    ) as resp:
        assert resp.status_code == 200, resp.text
        for line in resp.iter_lines():
            if isinstance(line, bytes):
                if not line or not line.startswith(b"data:"):
                    continue
                data_bytes = line[len(b"data:") :].strip()
            else:
                if not line or not line.startswith("data:"):
                    continue
                data_bytes = line[len("data:") :].strip().encode()
            if data_bytes == b"[DONE]":
                break
            try:
                import json

                payload_json = json.loads(data_bytes.decode())
            except ValueError:
                continue
            delta = payload_json.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                chunks.append(content)
    return "".join(chunks)


def test_default_project_chat_should_not_use_seed_session(app_client):
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    shared_session = "sess-123"

    seed_resp = _post_chat(
        app_client, "llamafarm", "project_seed", payload, session=shared_session
    )
    assert seed_resp.status_code == 200
    assert seed_resp.json()["choices"][0]["message"]["content"].startswith(
        "llamafarm/project_seed:"
    )

    default_resp = _post_chat(
        app_client, "default", "llamafarm-1", payload, session=shared_session
    )
    assert default_resp.status_code == 200
    content = default_resp.json()["choices"][0]["message"]["content"]
    assert content.startswith("default/llamafarm-1:"), (
        "Default project reuses seed agent session"
    )


def test_seed_project_chat_creates_session(app_client):
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    resp = _post_chat(app_client, "llamafarm", "project_seed", payload)
    assert resp.status_code == 200
    assert resp.headers.get("X-Session-ID")
    assert resp.json()["choices"][0]["message"]["content"].startswith(
        "llamafarm/project_seed:"
    )


def test_delete_specific_session(app_client):
    payload = {"messages": [{"role": "user", "content": "reset"}]}
    resp = _post_chat(app_client, "default", "llamafarm-1", payload)
    assert resp.status_code == 200
    session = resp.headers.get("X-Session-ID")
    assert session

    delete_resp = _delete_session(app_client, "default", "llamafarm-1", session)
    assert delete_resp.status_code == 200
    assert delete_resp.json()["message"].startswith("Session")

    resp_again = _post_chat(
        app_client, "default", "llamafarm-1", payload, session=session
    )
    assert resp_again.status_code == 200
    assert resp_again.headers.get("X-Session-ID") == session


def test_delete_all_project_sessions(app_client):
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    first = _post_chat(app_client, "llamafarm", "project_seed", payload)
    second = _post_chat(app_client, "llamafarm", "project_seed", payload)
    assert first.headers.get("X-Session-ID")
    assert second.headers.get("X-Session-ID")

    delete_resp = _delete_all_sessions(app_client, "llamafarm", "project_seed")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["count"] >= 1

    third = _post_chat(app_client, "llamafarm", "project_seed", payload)
    third_session = third.headers.get("X-Session-ID")
    assert third_session
    assert third_session != first.headers.get("X-Session-ID")


def test_stream_default_project_returns_non_echo_content(app_client):
    payload = {"messages": [{"role": "user", "content": "hello"}], "stream": True}
    streamed = _stream_chat(app_client, "default", "llamafarm-1", payload)
    assert streamed == FALLBACK_ECHO_RESPONSE


def test_stream_dev_project_returns_non_echo_content(app_client):
    payload = {"messages": [{"role": "user", "content": "hello"}], "stream": True}
    streamed = _stream_chat(app_client, "llamafarm", "project_seed", payload)
    assert streamed == FALLBACK_ECHO_RESPONSE


def test_stream_empty_output_uses_fallback(app_client):
    payload = {"messages": [{"role": "user", "content": "no-content"}], "stream": True}
    streamed = _stream_chat(app_client, "default", "llamafarm-1", payload)
    assert streamed == FALLBACK_ECHO_RESPONSE
