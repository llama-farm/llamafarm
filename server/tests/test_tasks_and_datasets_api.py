from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import llama_farm_api


def _client() -> TestClient:
    app = llama_farm_api()
    return TestClient(app)


def test_dataset_actions_ingest_triggers_task_and_returns_task_uri(mocker):
    # Patch the task object on the SUT import path
    mocked_task = mocker.Mock()
    mocked_task.delay.return_value = SimpleNamespace(id="task-123")
    mocker.patch(
        "api.routers.datasets.datasets.process_dataset_task",
        mocked_task,
    )

    client = _client()
    resp = client.post(
        "/v1/projects/ns1/proj1/datasets/ds1/actions",
        json={"action_type": "ingest"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Accepted"
    assert data["task_uri"].endswith("/v1/projects/ns1/proj1/tasks/task-123")
    mocked_task.delay.assert_called_once_with("ns1", "proj1", "ds1")


def test_dataset_actions_invalid_type_returns_400():
    client = _client()
    resp = client.post(
        "/v1/projects/ns1/proj1/datasets/ds1/actions",
        json={"action_type": "unknown"},
    )
    assert resp.status_code == 400


class _FakeAsyncResult:
    def __init__(
        self, state: str, info=None, result=None, traceback: str | None = None
    ):
        self.state = state
        self.info = info
        self.result = result
        self.traceback = traceback


def test_get_task_pending_state(mocker):
    fake = _FakeAsyncResult(state="PENDING")
    mocked_app = mocker.patch("api.routers.projects.projects.app")
    mocked_app.AsyncResult.return_value = fake

    client = _client()
    resp = client.get("/v1/projects/ns1/proj1/tasks/tk-1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "tk-1"
    assert body["state"] == "PENDING"
    assert body["meta"] is None
    assert body["result"] is None
    assert body["error"] is None
    assert body["traceback"] is None
    mocked_app.AsyncResult.assert_called_once_with("tk-1")


def test_get_task_success_state(mocker):
    fake = _FakeAsyncResult(state="SUCCESS", result={"ok": True})
    mocked_app = mocker.patch("api.routers.projects.projects.app")
    mocked_app.AsyncResult.return_value = fake

    client = _client()
    resp = client.get("/v1/projects/ns1/proj1/tasks/tk-2")

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "SUCCESS"
    assert body["result"] == {"ok": True}
    assert body["error"] is None
    assert body["traceback"] is None


def test_get_task_failure_state(mocker):
    fake = _FakeAsyncResult(
        state="FAILURE",
        result=RuntimeError("boom"),
        traceback="traceback text",
    )
    mocked_app = mocker.patch("api.routers.projects.projects.app")
    mocked_app.AsyncResult.return_value = fake

    client = _client()
    resp = client.get("/v1/projects/ns1/proj1/tasks/tk-3")

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "FAILURE"
    assert body["error"] == "boom"
    assert body["traceback"] == "traceback text"
