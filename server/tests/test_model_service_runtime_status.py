"""Tests for project model runtime status collection."""

from types import SimpleNamespace

from server.services.model_service import ModelService
from server.services.runtime_service.providers.base import RuntimeModelStatus


def test_list_model_runtime_statuses_collects_provider_data(mocker):
    model = SimpleNamespace(name="fast", provider=SimpleNamespace(value="ollama"))
    project_config = SimpleNamespace(
        runtime=SimpleNamespace(models=[model], default_model="fast")
    )

    provider = mocker.Mock()
    provider.get_model_runtime_status.return_value = RuntimeModelStatus(
        status="running",
        host="http://localhost:11434",
        loaded=True,
        running=True,
        memory_usage_human="4.2 GB",
        gpu_allocation="3.8 GB",
    )
    mocker.patch(
        "server.services.model_service.runtime_service.get_provider",
        return_value=provider,
    )

    statuses = ModelService.list_model_runtime_statuses(project_config)

    assert statuses == {
        "fast": {
            "runtime_status": "running",
            "runtime_loaded": True,
            "runtime_running": True,
            "runtime_host": "http://localhost:11434",
            "memory_usage_human": "4.2 GB",
            "gpu_allocation": "3.8 GB",
        }
    }


def test_list_model_runtime_statuses_returns_unknown_on_provider_failure(mocker):
    model = SimpleNamespace(
        name="powerful", provider=SimpleNamespace(value="universal")
    )
    project_config = SimpleNamespace(
        runtime=SimpleNamespace(models=[model], default_model="powerful")
    )

    mocker.patch(
        "server.services.model_service.runtime_service.get_provider",
        side_effect=RuntimeError("boom"),
    )

    statuses = ModelService.list_model_runtime_statuses(project_config)

    assert statuses["powerful"]["runtime_status"] == "unknown"
    assert statuses["powerful"]["runtime_loaded"] is False
    assert statuses["powerful"]["runtime_running"] is False
    assert "boom" in statuses["powerful"]["runtime_message"]
