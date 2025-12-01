from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import llama_farm_api


def _client() -> TestClient:
    app = llama_farm_api()
    return TestClient(app)


def test_dataset_actions_ingest_triggers_task_and_returns_task_uri(mocker):
    launch = SimpleNamespace(task_id="task-123", message="Dataset ingestion started")
    start_ingest = mocker.patch(
        "api.routers.datasets.datasets.DatasetService.start_dataset_ingestion",
        return_value=launch,
    )

    client = _client()
    resp = client.post(
        "/v1/projects/ns1/proj1/datasets/ds1/actions",
        json={"action_type": "process"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Dataset ingestion started"
    assert data["task_uri"].endswith("/v1/projects/ns1/proj1/tasks/task-123")
    assert data["task_id"] == "task-123"
    start_ingest.assert_called_once_with("ns1", "proj1", "ds1")


def test_dataset_actions_invalid_type_returns_400():
    client = _client()
    resp = client.post(
        "/v1/projects/ns1/proj1/datasets/ds1/actions",
        json={"action_type": "unknown"},
    )
    # Pydantic validation returns 422 for invalid enum values
    assert resp.status_code == 422


class _FakeAsyncResult:
    def __init__(
        self, state: str, info=None, result=None, traceback: str | None = None
    ):
        self.state = state
        self.info = info
        self.result = result
        self.traceback = traceback

    def ready(self):
        """Return True if task has finished (SUCCESS or FAILURE)"""
        return self.state in ("SUCCESS", "FAILURE")

    def successful(self):
        """Return True if task completed successfully"""
        return self.state == "SUCCESS"

    def failed(self):
        """Return True if task failed"""
        return self.state == "FAILURE"


def test_get_task_pending_state(mocker):
    fake = _FakeAsyncResult(state="PENDING")
    mocked_app = mocker.patch("api.routers.projects.projects.app")
    mocked_app.AsyncResult.return_value = fake

    # Mock GroupResult.restore to return None (not a group task)
    mocker.patch("celery.result.GroupResult.restore", return_value=None)

    client = _client()
    resp = client.get("/v1/projects/ns1/proj1/tasks/tk-1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "tk-1"
    assert body["state"] == "PENDING"
    assert not hasattr(body, "meta")
    assert not hasattr(body, "result")
    assert not hasattr(body, "error")
    assert not hasattr(body, "traceback")
    mocked_app.AsyncResult.assert_called_once_with("tk-1")


def test_get_task_success_state(mocker):
    fake = _FakeAsyncResult(state="SUCCESS", result={"ok": True})
    mocked_app = mocker.patch("api.routers.projects.projects.app")
    mocked_app.AsyncResult.return_value = fake

    # Mock GroupResult.restore to return None (not a group task)
    mocker.patch("celery.result.GroupResult.restore", return_value=None)

    client = _client()
    resp = client.get("/v1/projects/ns1/proj1/tasks/tk-2")

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "SUCCESS"
    assert body["result"] == {"ok": True}
    assert not hasattr(body, "error")
    assert not hasattr(body, "traceback")


def test_get_task_failure_state(mocker):
    fake = _FakeAsyncResult(
        state="FAILURE",
        result=RuntimeError("boom"),
        traceback="traceback text",
    )
    mocked_app = mocker.patch("api.routers.projects.projects.app")
    mocked_app.AsyncResult.return_value = fake

    # Mock GroupResult.restore to return None (not a group task)
    mocker.patch("celery.result.GroupResult.restore", return_value=None)

    client = _client()
    resp = client.get("/v1/projects/ns1/proj1/tasks/tk-3")

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "FAILURE"
    assert body["error"] == "boom"
    assert body["traceback"] == "traceback text"
