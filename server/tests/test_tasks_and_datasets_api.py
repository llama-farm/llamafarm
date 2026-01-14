import json
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


def test_dataset_upload_auto_process_defaults_true(mocker):
    launch = SimpleNamespace(task_id="task-xyz", message="Dataset ingestion started")
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.get_dataset_config",
        return_value=SimpleNamespace(auto_process=True),
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.add_file_to_dataset",
        return_value=(True, SimpleNamespace(hash="abc123", original_file_name="doc.pdf")),
    )
    start_ingest = mocker.patch(
        "api.routers.datasets.datasets.DatasetService.start_ingestion_for_hashes",
        return_value=launch,
    )

    client = _client()
    resp = client.post(
        "/v1/projects/ns1/proj1/datasets/ds1/data",
        files={"file": ("doc.pdf", b"hello")},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] is True
    assert data["task_id"] == "task-xyz"
    assert data["status"] == "processing"
    start_ingest.assert_called_once()


def test_dataset_upload_skipped_duplicate_no_processing(mocker):
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.get_dataset_config",
        return_value=SimpleNamespace(auto_process=True),
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.add_file_to_dataset",
        return_value=(False, SimpleNamespace(hash="abc123", original_file_name="doc.pdf")),
    )
    start_ingest = mocker.patch(
        "api.routers.datasets.datasets.DatasetService.start_ingestion_for_hashes",
    )

    client = _client()
    resp = client.post(
        "/v1/projects/ns1/proj1/datasets/ds1/data",
        files={"file": ("doc.pdf", b"hello")},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] is False
    assert data["skipped"] is True
    assert data["status"] == "skipped"
    start_ingest.assert_not_called()


def test_dataset_upload_rejects_chunk_overlap_exceeding_default(mocker):
    dataset_cfg = SimpleNamespace(
        auto_process=False, data_processing_strategy="default"
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.get_dataset_config",
        return_value=dataset_cfg,
    )
    add_file = mocker.patch(
        "api.routers.datasets.datasets.DatasetService.add_file_to_dataset",
        return_value=(True, SimpleNamespace(hash="abc123", original_file_name="doc.pdf")),
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.start_ingestion_for_hashes",
    )

    processing_strategy = SimpleNamespace(
        name="default",
        parsers=[
            SimpleNamespace(
                type="PDFParser_LlamaIndex",
                config={"chunk_size": 1000},
            )
        ],
    )
    project_config = SimpleNamespace(
        rag=SimpleNamespace(data_processing_strategies=[processing_strategy])
    )
    mocker.patch(
        "api.routers.datasets.datasets.ProjectService.load_config",
        return_value=project_config,
    )

    client = _client()
    resp = client.post(
        "/v1/projects/ns1/proj1/datasets/ds1/data",
        files={"file": ("doc.pdf", b"hello")},
        data={
            "parser_overrides": json.dumps(
                {"PDFParser_LlamaIndex": {"chunk_overlap": 5000}}
            )
        },
    )

    assert resp.status_code == 400
    assert "chunk_overlap" in resp.json()["detail"]
    add_file.assert_not_called()


def test_dataset_upload_rejects_zero_chunk_size(mocker):
    dataset_cfg = SimpleNamespace(
        auto_process=False, data_processing_strategy="default"
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.get_dataset_config",
        return_value=dataset_cfg,
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.add_file_to_dataset",
        return_value=(True, SimpleNamespace(hash="abc123", original_file_name="doc.pdf")),
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.start_ingestion_for_hashes",
    )

    processing_strategy = SimpleNamespace(
        name="default",
        parsers=[
            SimpleNamespace(
                type="PDFParser_LlamaIndex",
                config={"chunk_size": 1000},
            )
        ],
    )
    project_config = SimpleNamespace(
        rag=SimpleNamespace(data_processing_strategies=[processing_strategy])
    )
    mocker.patch(
        "api.routers.datasets.datasets.ProjectService.load_config",
        return_value=project_config,
    )

    client = _client()
    resp = client.post(
        "/v1/projects/ns1/proj1/datasets/ds1/data",
        files={"file": ("doc.pdf", b"hello")},
        data={
            "parser_overrides": json.dumps(
                {"PDFParser_LlamaIndex": {"chunk_size": 0, "chunk_overlap": 0}}
            )
        },
    )

    assert resp.status_code == 400
    assert "chunk_size" in resp.json()["detail"]


def test_dataset_upload_rejects_negative_overlap(mocker):
    dataset_cfg = SimpleNamespace(
        auto_process=False, data_processing_strategy="default"
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.get_dataset_config",
        return_value=dataset_cfg,
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.add_file_to_dataset",
        return_value=(True, SimpleNamespace(hash="abc123", original_file_name="doc.pdf")),
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.start_ingestion_for_hashes",
    )

    processing_strategy = SimpleNamespace(
        name="default",
        parsers=[
            SimpleNamespace(
                type="PDFParser_LlamaIndex",
                config={"chunk_size": 1000},
            )
        ],
    )
    project_config = SimpleNamespace(
        rag=SimpleNamespace(data_processing_strategies=[processing_strategy])
    )
    mocker.patch(
        "api.routers.datasets.datasets.ProjectService.load_config",
        return_value=project_config,
    )

    client = _client()
    resp = client.post(
        "/v1/projects/ns1/proj1/datasets/ds1/data",
        files={"file": ("doc.pdf", b"hello")},
        data={
            "parser_overrides": json.dumps(
                {"PDFParser_LlamaIndex": {"chunk_overlap": -1}}
            )
        },
    )

    assert resp.status_code == 400
    assert "chunk_overlap" in resp.json()["detail"]


def test_dataset_upload_rejects_non_dict_override(mocker):
    dataset_cfg = SimpleNamespace(
        auto_process=False, data_processing_strategy="default"
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.get_dataset_config",
        return_value=dataset_cfg,
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.add_file_to_dataset",
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.start_ingestion_for_hashes",
    )

    processing_strategy = SimpleNamespace(
        name="default",
        parsers=[
            SimpleNamespace(
                type="PDFParser_LlamaIndex",
                config={"chunk_size": 1000},
            )
        ],
    )
    project_config = SimpleNamespace(
        rag=SimpleNamespace(data_processing_strategies=[processing_strategy])
    )
    mocker.patch(
        "api.routers.datasets.datasets.ProjectService.load_config",
        return_value=project_config,
    )

    client = _client()
    resp = client.post(
        "/v1/projects/ns1/proj1/datasets/ds1/data",
        files={"file": ("doc.pdf", b"hello")},
        data={"parser_overrides": json.dumps({"*": "invalid"})},
    )

    assert resp.status_code == 400
    assert "override" in resp.json()["detail"].lower()


def test_dataset_bulk_upload_defaults_no_processing(mocker):
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.get_dataset_config",
        return_value=SimpleNamespace(auto_process=True),
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.add_file_to_dataset",
        return_value=(True, SimpleNamespace(hash="abc123", original_file_name="doc.pdf")),
    )
    start_ingest = mocker.patch(
        "api.routers.datasets.datasets.DatasetService.start_ingestion_for_hashes",
    )

    client = _client()
    resp = client.post(
        "/v1/projects/ns1/proj1/datasets/ds1/data/bulk",
        files=[
            ("files", ("doc1.pdf", b"hello")),
            ("files", ("doc2.pdf", b"world")),
        ],
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["uploaded"] == 2
    assert data["status"] == "uploaded"
    assert data.get("task_id") is None
    start_ingest.assert_not_called()


def test_dataset_bulk_upload_with_auto_process_true(mocker):
    launch = SimpleNamespace(task_id="task-xyz", message="Dataset ingestion started")
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.get_dataset_config",
        return_value=SimpleNamespace(auto_process=False),
    )
    mocker.patch(
        "api.routers.datasets.datasets.DatasetService.add_file_to_dataset",
        return_value=(True, SimpleNamespace(hash="abc123", original_file_name="doc.pdf")),
    )
    start_ingest = mocker.patch(
        "api.routers.datasets.datasets.DatasetService.start_ingestion_for_hashes",
        return_value=launch,
    )

    client = _client()
    resp = client.post(
        "/v1/projects/ns1/proj1/datasets/ds1/data/bulk?auto_process=true",
        files=[
            ("files", ("doc1.pdf", b"hello")),
            ("files", ("doc2.pdf", b"world")),
        ],
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["uploaded"] == 2
    assert data["status"] == "processing"
    assert data["task_id"] == "task-xyz"
    start_ingest.assert_called_once()


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
