from datetime import datetime, timezone

import httpx
from fastapi.testclient import TestClient

from api.main import app
from api.routers.system import upgrades as upgrades_router
from services.upgrade_service import CLIRelease


client = TestClient(app)


def test_cli_upgrade_success(monkeypatch):
    release = CLIRelease(
        tag_name="v1.2.3",
        html_url="https://example.com/release",
        published_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        name="LlamaFarm v1.2.3",
        body="- Improvements",
        from_cache=False,
    )

    async def fake_get_latest_cli_release(force_refresh: bool = False):
        return release

    monkeypatch.setattr(
        upgrades_router, "get_latest_cli_release", fake_get_latest_cli_release
    )

    response = client.get("/v1/system/version-check")
    assert response.status_code == 200
    payload = response.json()
    assert payload["latest_version"] == "v1.2.3"
    assert payload["release_url"] == "https://example.com/release"
    assert payload["published_at"].startswith("2025-01-01T12:00:00+00:00")
    assert payload["install"]["mac_linux"].startswith("curl -fsSL")


def test_cli_upgrade_failure(monkeypatch):
    async def fake_get_latest_cli_release(force_refresh: bool = False):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(
        upgrades_router, "get_latest_cli_release", fake_get_latest_cli_release
    )

    response = client.get("/v1/system/version-check")
    assert response.status_code == 503
    assert response.json()["detail"] == "Unable to reach GitHub releases"
