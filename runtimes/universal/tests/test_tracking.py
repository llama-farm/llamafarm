"""Unit tests for vision tracking router."""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from routers.vision.tracking import (
    MAX_SESSIONS,
    SESSION_TTL_SECONDS,
    TrackSession,
    TrackStartRequest,
    TracksSummary,
    TrackedDetection,
    VALID_TRACKERS,
    _resolve_model_path,
    _run_track,
    _sessions,
    set_tracking_models_dir,
)


# ── Model resolution tests ──────────────────────────────────────────────────


def test_resolve_rejects_traversal():
    """Path traversal in model name rejected."""
    from fastapi import HTTPException
    set_tracking_models_dir(Path("/tmp/models"))
    with pytest.raises(HTTPException) as exc:
        _resolve_model_path("../etc/passwd")
    assert exc.value.status_code == 400


def test_resolve_rejects_backslash():
    from fastapi import HTTPException
    set_tracking_models_dir(Path("/tmp/models"))
    with pytest.raises(HTTPException):
        _resolve_model_path("model\\evil")


def test_resolve_rejects_colon():
    from fastapi import HTTPException
    set_tracking_models_dir(Path("/tmp/models"))
    with pytest.raises(HTTPException):
        _resolve_model_path("C:model")


def test_resolve_model_not_found():
    from fastapi import HTTPException
    set_tracking_models_dir(Path("/tmp/nonexistent_models_dir"))
    with pytest.raises(HTTPException) as exc:
        _resolve_model_path("no-such-model")
    assert exc.value.status_code == 404


def test_resolve_model_from_dir(tmp_path):
    """Resolves current.pt from models dir."""
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    pt = model_dir / "current.pt"
    pt.write_bytes(b"fake")
    set_tracking_models_dir(tmp_path)
    result = _resolve_model_path("my-model")
    assert result == str(pt)


def test_resolve_versioned_fallback(tmp_path):
    """Falls back to latest versioned .pt."""
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    (model_dir / "v1.pt").write_bytes(b"old")
    (model_dir / "v2.pt").write_bytes(b"new")
    set_tracking_models_dir(tmp_path)
    result = _resolve_model_path("my-model")
    assert "v2.pt" in result


# ── Tracker validation ──────────────────────────────────────────────────────


def test_valid_trackers():
    assert "bytetrack" in VALID_TRACKERS
    assert "botsort" in VALID_TRACKERS
    assert "ocsort" in VALID_TRACKERS
    assert "invalid" not in VALID_TRACKERS


# ── Request model validation ────────────────────────────────────────────────


def test_start_request_defaults():
    req = TrackStartRequest(model="yolov8n")
    assert req.tracker == "bytetrack"
    assert req.confidence_threshold == 0.25
    assert req.target_fps == 10.0
    assert req.image is None


def test_start_request_custom():
    req = TrackStartRequest(
        model="custom-model",
        tracker="botsort",
        confidence_threshold=0.5,
        target_fps=30.0,
    )
    assert req.tracker == "botsort"
    assert req.confidence_threshold == 0.5


# ── Session management ──────────────────────────────────────────────────────


def test_session_dataclass():
    mock_model = MagicMock()
    session = TrackSession(
        session_id="test123",
        model_id="yolov8n",
        tracker="bytetrack",
        yolo_model=mock_model,
    )
    assert session.frames_processed == 0
    assert session.total_tracks_created == 0
    assert session.confidence_threshold == 0.25


def test_session_ttl_constant():
    assert SESSION_TTL_SECONDS == 120.0


def test_max_sessions_constant():
    assert MAX_SESSIONS == 50


# ── TrackedDetection model ──────────────────────────────────────────────────


def test_tracked_detection():
    det = TrackedDetection(
        x1=10, y1=20, x2=100, y2=200,
        class_name="person", class_id=0,
        confidence=0.95, track_id=1,
    )
    assert det.track_state == "tracked"
    assert det.track_id == 1


def test_tracks_summary():
    summary = TracksSummary(active=5, total_created=12)
    assert summary.active == 5
    assert summary.total_created == 12
