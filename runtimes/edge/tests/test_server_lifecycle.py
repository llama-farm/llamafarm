"""Tests for server.py backend-state wiring.

Closes the unit-coverage gap left by the original PR:

1. `_init_llama_backend()` outcome recording — success marks the backend
   as initialized and leaves readiness at INITIALIZING (the lifespan
   finalize step decides READY/DEGRADED later); ImportError and generic
   Exception both flip readiness to UNAVAILABLE with a reason string.

2. `_finalize_backend_readiness()` — projects preload outcomes onto
   readiness without ever overwriting an UNAVAILABLE set by backend init,
   so the original failure reason survives.

The finalize logic is split out of `lifespan()` so it can be tested
without spinning up the full FastAPI app, KV cache manager, Zenoh
session, etc.
"""

from __future__ import annotations

import sys
import types

import pytest

import server
from core.backend_state import BackendState, Readiness


@pytest.fixture
def fresh_state(monkeypatch):
    """Replace server.BACKEND_STATE with a clean instance for the test.

    The module-level `_init_llama_backend()` call at import time has
    already mutated the real singleton; swapping in a fresh instance
    here gives each test a known starting point and prevents cross-test
    pollution.
    """
    state = BackendState()
    monkeypatch.setattr(server, "BACKEND_STATE", state)
    return state


def _install_fake_bindings(monkeypatch, ensure_backend):
    """Install a fake `llamafarm_llama._bindings` module that exposes
    `ensure_backend`. Both the parent and submodule must be in sys.modules
    for `from llamafarm_llama._bindings import ensure_backend` to resolve."""
    parent = types.ModuleType("llamafarm_llama")
    submodule = types.ModuleType("llamafarm_llama._bindings")
    submodule.ensure_backend = ensure_backend
    parent._bindings = submodule
    monkeypatch.setitem(sys.modules, "llamafarm_llama", parent)
    monkeypatch.setitem(sys.modules, "llamafarm_llama._bindings", submodule)


class TestInitLlamaBackend:
    def test_success_marks_backend_initialized(self, fresh_state, monkeypatch):
        calls = {"n": 0}

        def ensure_backend():
            calls["n"] += 1

        _install_fake_bindings(monkeypatch, ensure_backend)

        server._init_llama_backend()

        assert calls["n"] == 1
        snap = fresh_state.snapshot()
        assert snap["backend_initialized"] is True
        # Readiness stays INITIALIZING — finalize step decides READY/DEGRADED.
        assert snap["readiness"] == "initializing"
        assert snap["reason"] == ""

    def test_import_error_sets_unavailable(self, fresh_state, monkeypatch):
        # sys.modules[name] = None forces `import name` to raise ImportError.
        monkeypatch.setitem(sys.modules, "llamafarm_llama", None)
        monkeypatch.setitem(sys.modules, "llamafarm_llama._bindings", None)

        server._init_llama_backend()

        snap = fresh_state.snapshot()
        assert snap["readiness"] == "unavailable"
        assert "llamafarm_llama not installed" in snap["reason"]
        assert snap["backend_initialized"] is False

    def test_runtime_failure_sets_unavailable_with_message(
        self, fresh_state, monkeypatch
    ):
        def ensure_backend():
            raise RuntimeError("CUDA driver mismatch")

        _install_fake_bindings(monkeypatch, ensure_backend)

        server._init_llama_backend()

        snap = fresh_state.snapshot()
        assert snap["readiness"] == "unavailable"
        assert "backend init failed" in snap["reason"]
        # Underlying error message must propagate so operators can diagnose.
        assert "CUDA driver mismatch" in snap["reason"]
        assert snap["backend_initialized"] is False


class TestFinalizeBackendReadiness:
    def test_uninitialized_backend_is_not_overwritten(self, fresh_state):
        """If backend init failed (UNAVAILABLE + initialized=False), the
        finalize step must preserve the original reason — not mask it
        with a preload-derived state."""
        fresh_state.set(Readiness.UNAVAILABLE, "backend init failed: missing lib")

        server._finalize_backend_readiness("model-a", ["model-a"], [])

        snap = fresh_state.snapshot()
        assert snap["readiness"] == "unavailable"
        assert snap["reason"] == "backend init failed: missing lib"

    def test_no_preload_configured_is_ready(self, fresh_state):
        fresh_state.mark_backend_initialized()

        server._finalize_backend_readiness("", [], [])

        snap = fresh_state.snapshot()
        assert snap["readiness"] == "ready"
        assert snap["reason"] == ""

    def test_all_preloads_succeeded_is_ready(self, fresh_state):
        fresh_state.mark_backend_initialized()

        server._finalize_backend_readiness("a,b", ["a", "b"], [])

        snap = fresh_state.snapshot()
        assert snap["readiness"] == "ready"
        assert snap["reason"] == ""

    def test_partial_preload_failure_is_degraded(self, fresh_state):
        fresh_state.mark_backend_initialized()

        server._finalize_backend_readiness("a,b", ["a"], ["b"])

        snap = fresh_state.snapshot()
        assert snap["readiness"] == "degraded"
        assert "preload failed" in snap["reason"]
        assert "b" in snap["reason"]
        # Successful model name must not appear in the degraded reason —
        # operators should see only what failed.
        assert "a" not in snap["reason"].split(":")[-1]

    def test_all_preloads_failed_is_unavailable(self, fresh_state):
        fresh_state.mark_backend_initialized()

        server._finalize_backend_readiness("a,b", [], ["a", "b"])

        snap = fresh_state.snapshot()
        assert snap["readiness"] == "unavailable"
        assert "all preloads failed" in snap["reason"]
        assert "a" in snap["reason"]
        assert "b" in snap["reason"]
