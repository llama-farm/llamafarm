"""Tests for ZenohIPC admission control and heartbeat state propagation.

These tests stub out the Zenoh session and event loop wiring — they
validate the *logic* in _handle_request and _heartbeat_loop, not the
zenoh transport itself.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core.backend_state import BackendState, Readiness
from services.zenoh_ipc import ZenohIPC


class FakeSession:
    def __init__(self):
        self.puts: list[tuple[str, dict]] = []

    def put(self, topic, payload):
        self.puts.append((topic, json.loads(bytes(payload))))


@pytest.fixture
def fake_session():
    return FakeSession()


@pytest.fixture
def state():
    return BackendState()


@pytest.fixture
def inference_calls():
    return []


@pytest.fixture
def ipc(fake_session, state, inference_calls):
    async def fake_inference(request):
        inference_calls.append(request)
        return "ok"

    ipc = ZenohIPC(
        inference_fn=fake_inference,
        state_provider=state.snapshot,
        state_setter=state.set,
    )
    ipc._session = fake_session
    return ipc


class TestRequestAdmissionControl:
    @pytest.mark.asyncio
    async def test_refuses_when_initializing(self, ipc, fake_session, inference_calls):
        await ipc._handle_request({"request_id": "r1", "model": "m"})
        assert inference_calls == []
        assert len(fake_session.puts) == 1
        topic, payload = fake_session.puts[0]
        assert topic == "local/llm/response"
        assert payload["error"] == "backend_unavailable"
        assert payload["readiness"] == "initializing"
        assert payload["request_id"] == "r1"

    @pytest.mark.asyncio
    async def test_refuses_when_unavailable(self, ipc, state, fake_session, inference_calls):
        state.set(Readiness.UNAVAILABLE, "backend init failed: missing lib")
        await ipc._handle_request({"request_id": "r2", "model": "m"})
        assert inference_calls == []
        topic, payload = fake_session.puts[0]
        assert payload["error"] == "backend_unavailable"
        assert payload["reason"] == "backend init failed: missing lib"
        assert payload["readiness"] == "unavailable"

    @pytest.mark.asyncio
    async def test_refuses_when_degraded(self, ipc, state, fake_session, inference_calls):
        state.set(Readiness.DEGRADED, "preload failed: m1")
        await ipc._handle_request({"request_id": "r3", "model": "m"})
        assert inference_calls == []
        topic, payload = fake_session.puts[0]
        assert payload["error"] == "backend_unavailable"
        assert payload["readiness"] == "degraded"

    @pytest.mark.asyncio
    async def test_passes_through_when_ready(self, ipc, state, fake_session, inference_calls):
        state.mark_backend_initialized()
        state.set(Readiness.READY)
        await ipc._handle_request({"request_id": "r4", "model": "m"})
        assert len(inference_calls) == 1
        topic, payload = fake_session.puts[0]
        assert payload["content"] == "ok"
        assert "error" not in payload


class TestRuntimeFailureMarksDegraded:
    @pytest.mark.asyncio
    async def test_offline_mode_error_flips_to_degraded(self, fake_session, state):
        # Skip if huggingface_hub isn't installed in this environment.
        pytest.importorskip("huggingface_hub")
        from huggingface_hub.errors import OfflineModeIsEnabled

        async def boom(request):
            raise OfflineModeIsEnabled("offline")

        ipc = ZenohIPC(
            inference_fn=boom,
            state_provider=state.snapshot,
            state_setter=state.set,
        )
        ipc._session = fake_session
        state.mark_backend_initialized()
        state.set(Readiness.READY)

        await ipc._handle_request({"request_id": "r5", "model": "m1"})

        assert state.snapshot()["readiness"] == "degraded"
        assert "m1" in state.snapshot()["reason"]
        # And the response was published as a generic inference failure.
        topic, payload = fake_session.puts[0]
        assert payload["error"] == "inference failed"

    @pytest.mark.asyncio
    async def test_generic_exception_does_not_flip_state(self, fake_session, state):
        async def boom(request):
            raise RuntimeError("decode error")

        ipc = ZenohIPC(
            inference_fn=boom,
            state_provider=state.snapshot,
            state_setter=state.set,
        )
        ipc._session = fake_session
        state.mark_backend_initialized()
        state.set(Readiness.READY)

        await ipc._handle_request({"request_id": "r6", "model": "m1"})

        # Generic inference errors do NOT poison the backend's status.
        assert state.snapshot()["readiness"] == "ready"


class TestHeartbeatPublishesSnapshot:
    @pytest.mark.asyncio
    async def test_heartbeat_publishes_current_state(self, fake_session, state):
        ipc = ZenohIPC(
            inference_fn=None,
            state_provider=state.snapshot,
            state_setter=state.set,
        )
        ipc._session = fake_session
        state.set(Readiness.UNAVAILABLE, "backend init failed: foo")

        # Run one heartbeat iteration by patching sleep to immediately cancel.
        async def fake_sleep(_):
            raise asyncio.CancelledError

        import services.zenoh_ipc as mod

        original_sleep = mod.asyncio.sleep
        mod.asyncio.sleep = fake_sleep
        try:
            with pytest.raises(asyncio.CancelledError):
                await ipc._heartbeat_loop()
        finally:
            mod.asyncio.sleep = original_sleep

        # Exactly one heartbeat published before the cancel.
        assert len(fake_session.puts) == 1
        topic, payload = fake_session.puts[0]
        assert topic == "local/llm/status"
        assert payload["status"] == "unavailable"
        assert payload["readiness"] == "unavailable"
        assert payload["reason"] == "backend init failed: foo"
        assert payload["service"] == "edge-runtime"

    @pytest.mark.asyncio
    async def test_heartbeat_legacy_mode_without_state_provider(self, fake_session):
        """Without a state_provider, fall back to the old hardcoded `ready`
        so existing deployments that haven't wired in BACKEND_STATE still
        work."""
        ipc = ZenohIPC(inference_fn=None)
        ipc._session = fake_session

        async def fake_sleep(_):
            raise asyncio.CancelledError

        import services.zenoh_ipc as mod

        original_sleep = mod.asyncio.sleep
        mod.asyncio.sleep = fake_sleep
        try:
            with pytest.raises(asyncio.CancelledError):
                await ipc._heartbeat_loop()
        finally:
            mod.asyncio.sleep = original_sleep

        topic, payload = fake_session.puts[0]
        assert payload["status"] == "ready"
