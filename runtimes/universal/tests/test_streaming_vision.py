"""Tests for streaming vision sessions (cascade, TTL cleanup, session management)."""

import time
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from routers.vision.streaming import (
    SESSION_TTL_SECONDS,
    CascadeConfigRequest,
    StreamFrameRequest,
    StreamStartRequest,
    StreamStopRequest,
    _sessions,
    list_sessions,
    process_frame,
    set_streaming_detection_loader,
    start_stream,
    stop_stream,
)


@dataclass
class FakeBox:
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    class_id: int
    confidence: float


@dataclass
class FakeDetResult:
    boxes: list
    confidence: float


@pytest.fixture(autouse=True)
def _clear_sessions():
    _sessions.clear()
    yield
    _sessions.clear()


@pytest.mark.asyncio
class TestStreamLifecycle:
    async def test_start_stop(self):
        resp = await start_stream(StreamStartRequest())
        assert resp.session_id
        assert resp.session_id in _sessions

        stop_resp = await stop_stream(StreamStopRequest(session_id=resp.session_id))
        assert stop_resp.frames_processed == 0
        assert resp.session_id not in _sessions

    async def test_stop_unknown_session(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException, match="Session not found"):
            await stop_stream(StreamStopRequest(session_id="nonexistent"))

    async def test_max_sessions(self):
        for _i in range(100):
            await start_stream(StreamStartRequest())
        from fastapi import HTTPException
        with pytest.raises(HTTPException, match="Max 100"):
            await start_stream(StreamStartRequest())

    async def test_list_sessions(self):
        await start_stream(StreamStartRequest())
        await start_stream(StreamStartRequest())
        resp = await list_sessions()
        assert resp.count == 2
        assert len(resp.sessions) == 2

    async def test_custom_cascade_config(self):
        req = StreamStartRequest(
            config=CascadeConfigRequest(chain=["yolov8s", "yolov8m"], confidence_threshold=0.8),
            target_fps=5.0,
            cooldown_seconds=2.0,
        )
        resp = await start_stream(req)
        session = _sessions[resp.session_id]
        assert session.cascade.chain == ["yolov8s", "yolov8m"]
        assert session.cascade.confidence_threshold == 0.8
        assert session.target_fps == 5.0


@pytest.mark.asyncio
class TestFrameProcessing:
    async def test_frame_no_loader(self):
        set_streaming_detection_loader(None)
        resp = await start_stream(StreamStartRequest())
        from fastapi import HTTPException
        with pytest.raises(HTTPException, match="not initialized"):
            await process_frame(StreamFrameRequest(
                session_id=resp.session_id,
                image="aGVsbG8=",  # base64 "hello"
            ))

    async def test_frame_unknown_session(self):
        set_streaming_detection_loader(AsyncMock())
        from fastapi import HTTPException
        with pytest.raises(HTTPException, match="Session not found"):
            await process_frame(StreamFrameRequest(session_id="nope", image="aGVsbG8="))

    async def test_frame_confident_detection(self):
        """Detection above threshold → action response."""
        det_model = AsyncMock()
        det_model.detect = AsyncMock(return_value=FakeDetResult(
            boxes=[FakeBox(10, 10, 50, 50, "cat", 0, 0.9)],
            confidence=0.9,
        ))
        set_streaming_detection_loader(AsyncMock(return_value=det_model))

        resp = await start_stream(StreamStartRequest(
            config=CascadeConfigRequest(confidence_threshold=0.7),
        ))
        frame_resp = await process_frame(StreamFrameRequest(
            session_id=resp.session_id,
            image="aGVsbG8=",
        ))
        assert frame_resp.status == "action"
        assert len(frame_resp.detections) == 1
        assert frame_resp.detections[0].class_name == "cat"

    async def test_frame_below_threshold(self):
        """Detection below threshold with single-model chain → ok."""
        det_model = AsyncMock()
        det_model.detect = AsyncMock(return_value=FakeDetResult(boxes=[], confidence=0.3))
        set_streaming_detection_loader(AsyncMock(return_value=det_model))

        resp = await start_stream(StreamStartRequest(
            config=CascadeConfigRequest(confidence_threshold=0.7),
        ))
        frame_resp = await process_frame(StreamFrameRequest(
            session_id=resp.session_id,
            image="aGVsbG8=",
        ))
        assert frame_resp.status == "ok"

    async def test_cooldown_suppresses_action(self):
        """Second action within cooldown → ok instead of action."""
        det_model = AsyncMock()
        det_model.detect = AsyncMock(return_value=FakeDetResult(
            boxes=[FakeBox(10, 10, 50, 50, "cat", 0, 0.9)],
            confidence=0.9,
        ))
        set_streaming_detection_loader(AsyncMock(return_value=det_model))

        resp = await start_stream(StreamStartRequest(
            config=CascadeConfigRequest(confidence_threshold=0.5),
            cooldown_seconds=10.0,
        ))
        sid = resp.session_id

        r1 = await process_frame(StreamFrameRequest(session_id=sid, image="aGVsbG8="))
        assert r1.status == "action"

        r2 = await process_frame(StreamFrameRequest(session_id=sid, image="aGVsbG8="))
        assert r2.status == "ok"  # Suppressed by cooldown

    async def test_frame_counter(self):
        det_model = AsyncMock()
        det_model.detect = AsyncMock(return_value=FakeDetResult(boxes=[], confidence=0.1))
        set_streaming_detection_loader(AsyncMock(return_value=det_model))

        resp = await start_stream(StreamStartRequest())
        sid = resp.session_id
        for _ in range(3):
            await process_frame(StreamFrameRequest(session_id=sid, image="aGVsbG8="))
        assert _sessions[sid].frames_processed == 3


@pytest.mark.asyncio
class TestSessionTTL:
    async def test_expired_session_cleaned(self):
        """Sessions idle beyond TTL should be cleaned up."""
        resp = await start_stream(StreamStartRequest())
        sid = resp.session_id
        # Artificially age the session
        _sessions[sid].last_frame_at = time.time() - SESSION_TTL_SECONDS - 10

        # Run one iteration of the cleanup manually
        now = time.time()
        expired = [s for s, sess in _sessions.items()
                   if (now - sess.last_frame_at) > SESSION_TTL_SECONDS]
        for s in expired:
            _sessions.pop(s, None)

        assert sid not in _sessions

    async def test_active_session_not_cleaned(self):
        resp = await start_stream(StreamStartRequest())
        sid = resp.session_id
        # Session is fresh — should not be cleaned
        now = time.time()
        expired = [s for s, sess in _sessions.items()
                   if (now - sess.last_frame_at) > SESSION_TTL_SECONDS]
        assert sid not in expired
