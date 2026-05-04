"""Tests for /v1/completions offline-mode error mapping.

When LLAMAFARM_OFFLINE=1 and the requested model is not cached locally,
``load_language()`` raises ``OfflineModelNotCachedError`` from
llamafarm_common. The endpoint must translate that into a 404
``model_not_cached`` response matching /v1/chat/completions, not a
generic 500.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llamafarm_common.model_format import OfflineModelNotCachedError

from routers import completions as completions_router


@pytest.fixture
def client(monkeypatch):
    async def _raise_offline(model, n_ctx=None, n_gpu_layers=None):
        raise OfflineModelNotCachedError(
            f"detect_model_format({model!r}) refused in offline mode "
            "(LLAMAFARM_OFFLINE=1). Place the model under $LLAMAFARM_MODEL_DIR "
            "or pre-populate the HuggingFace cache."
        )

    async def _raise_other(model, n_ctx=None, n_gpu_layers=None):
        raise FileNotFoundError("some other missing file")

    import sys
    import types

    fake_server = types.ModuleType("server")
    fake_server.load_language = _raise_offline
    monkeypatch.setitem(sys.modules, "server", fake_server)

    app = FastAPI()
    app.include_router(completions_router.router)
    return TestClient(app, raise_server_exceptions=False), fake_server, _raise_other


def test_offline_returns_404_model_not_cached(client):
    test_client, _, _ = client
    resp = test_client.post(
        "/v1/completions",
        json={"model": "some/missing-model", "prompt": "hi"},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["detail"]["error"] == "model_not_cached"
    assert "some/missing-model" in body["detail"]["message"]


def test_non_offline_filenotfounderror_propagates(client):
    test_client, fake_server, raise_other = client
    fake_server.load_language = raise_other
    resp = test_client.post(
        "/v1/completions",
        json={"model": "some/model", "prompt": "hi"},
    )
    # FastAPI surfaces unhandled exceptions as 500.
    assert resp.status_code == 500
