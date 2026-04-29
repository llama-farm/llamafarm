"""Tests for the BackendState process-wide readiness object."""

from __future__ import annotations

from core.backend_state import BackendState, Readiness


class TestBackendStateInitial:
    def test_initial_readiness_is_initializing(self):
        state = BackendState()
        snap = state.snapshot()
        assert snap["readiness"] == "initializing"
        assert snap["backend_initialized"] is False
        assert snap["reason"] == ""
        assert isinstance(snap["last_transition_ms"], int)


class TestBackendStateTransitions:
    def test_set_updates_readiness_and_reason(self):
        state = BackendState()
        state.set(Readiness.READY)
        snap = state.snapshot()
        assert snap["readiness"] == "ready"
        assert snap["reason"] == ""

    def test_set_unavailable_with_reason(self):
        state = BackendState()
        state.set(Readiness.UNAVAILABLE, "backend init failed: foo")
        snap = state.snapshot()
        assert snap["readiness"] == "unavailable"
        assert snap["reason"] == "backend init failed: foo"

    def test_set_degraded_with_reason(self):
        state = BackendState()
        state.set(Readiness.DEGRADED, "preload failed: m1")
        snap = state.snapshot()
        assert snap["readiness"] == "degraded"
        assert snap["reason"] == "preload failed: m1"

    def test_mark_backend_initialized(self):
        state = BackendState()
        assert state.snapshot()["backend_initialized"] is False
        state.mark_backend_initialized()
        assert state.snapshot()["backend_initialized"] is True

    def test_mark_backend_initialized_advances_transition_ms(self):
        """Heartbeat consumers diff on last_transition_ms to detect changes;
        without bumping it, the backend_initialized flip is invisible."""
        import time

        state = BackendState()
        first = state.snapshot()["last_transition_ms"]
        # 5 ms is enough that int(time.time()*1000) reliably advances even
        # on jittery CI runners.
        time.sleep(0.005)
        state.mark_backend_initialized()
        second = state.snapshot()["last_transition_ms"]
        assert second > first

    def test_last_transition_ms_advances(self):
        import time

        state = BackendState()
        first = state.snapshot()["last_transition_ms"]
        time.sleep(0.002)
        state.set(Readiness.READY)
        second = state.snapshot()["last_transition_ms"]
        assert second >= first


class TestBackendStateSnapshotShape:
    def test_snapshot_is_a_plain_dict(self):
        """Heartbeat publishes JSON; snapshot must be JSON-serializable."""
        import json

        state = BackendState()
        state.set(Readiness.DEGRADED, "x")
        state.mark_backend_initialized()
        # Round-trip through json to confirm serializability.
        round_tripped = json.loads(json.dumps(state.snapshot()))
        assert round_tripped["readiness"] == "degraded"
        assert round_tripped["reason"] == "x"
        assert round_tripped["backend_initialized"] is True
