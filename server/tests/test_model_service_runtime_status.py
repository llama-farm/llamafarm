"""Tests for project model runtime status collection."""

from types import SimpleNamespace

from server.services.model_service import ModelService, _reset_uptime_tracker
from server.services.runtime_service.providers.base import RuntimeModelStatus


def _project_config_with(*models):
    return SimpleNamespace(
        runtime=SimpleNamespace(
            models=list(models),
            default_model=models[0].name if models else None,
        )
    )


def test_list_model_runtime_statuses_collects_provider_data(mocker):
    _reset_uptime_tracker()
    model = SimpleNamespace(name="fast", provider=SimpleNamespace(value="ollama"))
    project_config = _project_config_with(model)

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

    status = statuses["fast"]
    assert status["runtime_status"] == "running"
    assert status["runtime_loaded"] is True
    assert status["runtime_running"] is True
    assert status["runtime_host"] == "http://localhost:11434"
    assert status["memory_usage_human"] == "4.2 GB"
    assert status["gpu_allocation"] == "3.8 GB"
    # The service enriches the first sighting with an uptime of zero.
    assert status["uptime_seconds"] == 0
    assert status["uptime_human"] == "0s"


def test_list_model_runtime_statuses_returns_unknown_on_provider_failure(mocker):
    _reset_uptime_tracker()
    model = SimpleNamespace(
        name="powerful", provider=SimpleNamespace(value="universal")
    )
    project_config = _project_config_with(model)

    mocker.patch(
        "server.services.model_service.runtime_service.get_provider",
        side_effect=RuntimeError("boom"),
    )

    statuses = ModelService.list_model_runtime_statuses(project_config)

    assert statuses["powerful"]["runtime_status"] == "unknown"
    assert statuses["powerful"]["runtime_loaded"] is False
    assert statuses["powerful"]["runtime_running"] is False
    assert "boom" in statuses["powerful"]["runtime_message"]
    assert "uptime_seconds" not in statuses["powerful"]


def test_list_model_runtime_statuses_accumulates_uptime(mocker):
    _reset_uptime_tracker()
    model = SimpleNamespace(name="fast", provider=SimpleNamespace(value="ollama"))
    project_config = _project_config_with(model)

    def _fresh_status(*_args, **_kwargs):
        return RuntimeModelStatus(
            status="running",
            host="http://localhost:11434",
            loaded=True,
            running=True,
        )

    provider = mocker.Mock()
    provider.get_model_runtime_status.side_effect = _fresh_status
    mocker.patch(
        "server.services.model_service.runtime_service.get_provider",
        return_value=provider,
    )

    monotonic_values = iter([100.0, 175.0])
    mocker.patch(
        "server.services.model_service.time.monotonic",
        side_effect=lambda: next(monotonic_values),
    )

    first = ModelService.list_model_runtime_statuses(project_config)
    second = ModelService.list_model_runtime_statuses(project_config)

    assert first["fast"]["uptime_seconds"] == 0
    assert second["fast"]["uptime_seconds"] == 75
    assert second["fast"]["uptime_human"] == "1m"


def test_list_model_runtime_statuses_resets_uptime_when_model_unloads(mocker):
    _reset_uptime_tracker()
    model = SimpleNamespace(name="fast", provider=SimpleNamespace(value="ollama"))
    project_config = _project_config_with(model)

    loaded = RuntimeModelStatus(
        status="running",
        host="http://localhost:11434",
        loaded=True,
        running=True,
    )
    idle = RuntimeModelStatus(
        status="idle",
        host="http://localhost:11434",
        loaded=False,
        running=False,
    )
    reloaded = RuntimeModelStatus(
        status="running",
        host="http://localhost:11434",
        loaded=True,
        running=True,
    )
    provider = mocker.Mock()
    provider.get_model_runtime_status.side_effect = [loaded, idle, reloaded]
    mocker.patch(
        "server.services.model_service.runtime_service.get_provider",
        return_value=provider,
    )

    monotonic_values = iter([10.0, 70.0, 200.0, 220.0])
    mocker.patch(
        "server.services.model_service.time.monotonic",
        side_effect=lambda: next(monotonic_values),
    )

    first = ModelService.list_model_runtime_statuses(project_config)
    second = ModelService.list_model_runtime_statuses(project_config)
    third = ModelService.list_model_runtime_statuses(project_config)

    assert first["fast"]["uptime_seconds"] == 0
    assert "uptime_seconds" not in second["fast"]
    # After a reload the counter starts over from the new first-seen timestamp.
    assert third["fast"]["uptime_seconds"] == 0


def test_list_model_runtime_statuses_preserves_provider_uptime(mocker):
    _reset_uptime_tracker()
    model = SimpleNamespace(name="fast", provider=SimpleNamespace(value="ollama"))
    project_config = _project_config_with(model)

    provider = mocker.Mock()
    provider.get_model_runtime_status.return_value = RuntimeModelStatus(
        status="running",
        host="http://localhost:11434",
        loaded=True,
        running=True,
        uptime_seconds=3723,
        uptime_human="1h 2m",
    )
    mocker.patch(
        "server.services.model_service.runtime_service.get_provider",
        return_value=provider,
    )

    statuses = ModelService.list_model_runtime_statuses(project_config)

    assert statuses["fast"]["uptime_seconds"] == 3723
    assert statuses["fast"]["uptime_human"] == "1h 2m"
