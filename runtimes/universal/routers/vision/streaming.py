"""Streaming vision router — simplified cascade detection.

Cascade: if confidence < threshold, try next model in chain.
Chain can include "remote:{url}" entries for Atmosphere readiness.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.error_handler import handle_endpoint_errors

from .utils import decode_base64_image

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-streaming"])

# Dependency injection
_load_detection_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None

# SSRF protection: allowlist of remote hosts for cascade
_ALLOWED_REMOTE_HOSTS: set[str] = set()


def set_streaming_detection_loader(fn: Callable[..., Coroutine[Any, Any, Any]] | None) -> None:
    global _load_detection_fn
    _load_detection_fn = fn


def set_allowed_remote_hosts(hosts: set[str]) -> None:
    """Set allowlist of remote hosts for cascade (SSRF mitigation)."""
    global _ALLOWED_REMOTE_HOSTS
    _ALLOWED_REMOTE_HOSTS = hosts


# =============================================================================
# Session management
# =============================================================================

@dataclass
class CascadeConfig:
    """Cascade chain config. Models tried in order."""
    chain: list[str] = field(default_factory=lambda: ["yolov8n"])
    confidence_threshold: float = 0.7

@dataclass
class StreamSession:
    session_id: str
    cascade: CascadeConfig
    target_fps: float = 1.0
    action_classes: list[str] | None = None
    cooldown_seconds: float = 5.0
    frames_processed: int = 0
    actions_triggered: int = 0
    escalations: int = 0
    created_at: float = field(default_factory=time.time)
    last_action_at: float = 0.0
    last_frame_at: float = field(default_factory=time.time)

_sessions: dict[str, StreamSession] = {}
_http_client: httpx.AsyncClient | None = None
_cleanup_task: asyncio.Task | None = None
SESSION_TTL_SECONDS: float = 60.0  # Auto-expire after no frames for this long


async def _session_cleanup_loop() -> None:
    """Background task that expires orphaned streaming sessions."""
    while True:
        await asyncio.sleep(15)  # Check every 15 seconds
        now = time.time()
        expired = [
            sid for sid, s in _sessions.items()
            if (now - s.last_frame_at) > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            session = _sessions.pop(sid, None)
            if session:
                logger.info(
                    f"Expired orphaned stream session {sid} "
                    f"(idle {now - session.last_frame_at:.0f}s, "
                    f"{session.frames_processed} frames processed)"
                )


def start_session_cleanup() -> None:
    """Start the background session cleanup task. Call once at server startup."""
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_session_cleanup_loop())


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=10.0)
    return _http_client


# =============================================================================
# Request/Response models
# =============================================================================

class CascadeConfigRequest(BaseModel):
    chain: list[str] = Field(default=["yolov8n"], description="Model chain, can include 'remote:http://...'")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class StreamStartRequest(BaseModel):
    config: CascadeConfigRequest = Field(default_factory=CascadeConfigRequest)
    target_fps: float = 1.0
    action_classes: list[str] | None = None
    cooldown_seconds: float = 5.0

class StreamStartResponse(BaseModel):
    session_id: str

class StreamFrameRequest(BaseModel):
    session_id: str
    image: str = Field(..., description="Base64-encoded image")

class DetectionItem(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    class_id: int
    confidence: float

class StreamFrameResponse(BaseModel):
    status: str  # "ok", "action", "escalated"
    detections: list[DetectionItem] | None = None
    confidence: float | None = None
    resolved_by: str | None = None

class StreamStopRequest(BaseModel):
    session_id: str

class StreamStopResponse(BaseModel):
    session_id: str
    frames_processed: int
    actions_triggered: int
    escalations: int
    duration_seconds: float


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/v1/vision/stream/start", response_model=StreamStartResponse)
@handle_endpoint_errors("vision_stream_start")
async def start_stream(request: StreamStartRequest) -> StreamStartResponse:
    """Start a streaming detection session with cascade config."""
    # Limit concurrent sessions to prevent memory growth
    MAX_SESSIONS = 100
    if len(_sessions) >= MAX_SESSIONS:
        raise HTTPException(status_code=429, detail=f"Max {MAX_SESSIONS} concurrent sessions")
    sid = str(uuid.uuid4())[:8]
    _sessions[sid] = StreamSession(
        session_id=sid,
        cascade=CascadeConfig(
            chain=request.config.chain,
            confidence_threshold=request.config.confidence_threshold,
        ),
        target_fps=request.target_fps,
        action_classes=request.action_classes,
        cooldown_seconds=request.cooldown_seconds,
    )
    return StreamStartResponse(session_id=sid)


@router.post("/v1/vision/stream/frame", response_model=StreamFrameResponse)
@handle_endpoint_errors("vision_stream_frame")
async def process_frame(request: StreamFrameRequest) -> StreamFrameResponse:
    """Process a frame through the cascade chain."""
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if _load_detection_fn is None:
        raise HTTPException(status_code=500, detail="Detection loader not initialized")

    session.frames_processed += 1
    session.last_frame_at = time.time()
    image_bytes = decode_base64_image(request.image)

    # Try each model in the cascade chain
    for i, model_ref in enumerate(session.cascade.chain):
        if model_ref.startswith("remote:"):
            # Remote model — HTTP POST
            url = model_ref[7:]  # strip "remote:"
            result = await _call_remote(url, image_bytes, session)
            if result:
                if i > 0:
                    session.escalations += 1
                return _build_response(result, model_ref, i > 0, session)
        else:
            # Local model
            model = await _load_detection_fn(model_ref)
            det_result = await model.detect(
                image=image_bytes,
                confidence_threshold=0.1,  # Low threshold, we check ourselves
                classes=session.action_classes,
            )
            if det_result.confidence >= session.cascade.confidence_threshold:
                if i > 0:
                    session.escalations += 1
                return _build_response(det_result, model_ref, i > 0, session)

    # No model in chain was confident enough
    return StreamFrameResponse(status="ok")


@router.post("/v1/vision/stream/stop", response_model=StreamStopResponse)
@handle_endpoint_errors("vision_stream_stop")
async def stop_stream(request: StreamStopRequest) -> StreamStopResponse:
    """Stop a streaming session."""
    session = _sessions.pop(request.session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return StreamStopResponse(
        session_id=session.session_id,
        frames_processed=session.frames_processed,
        actions_triggered=session.actions_triggered,
        escalations=session.escalations,
        duration_seconds=time.time() - session.created_at,
    )


class SessionInfo(BaseModel):
    session_id: str
    frames_processed: int
    actions_triggered: int
    escalations: int
    chain: list[str]
    idle_seconds: float
    duration_seconds: float


class SessionsListResponse(BaseModel):
    sessions: list[SessionInfo]
    count: int


@router.get("/v1/vision/stream/sessions", response_model=SessionsListResponse)
@handle_endpoint_errors("vision_stream_sessions")
async def list_sessions() -> SessionsListResponse:
    """List active streaming sessions."""
    now = time.time()
    sessions = [
        SessionInfo(
            session_id=s.session_id,
            frames_processed=s.frames_processed,
            actions_triggered=s.actions_triggered,
            escalations=s.escalations,
            chain=s.cascade.chain,
            idle_seconds=round(now - s.last_frame_at, 1),
            duration_seconds=round(now - s.created_at, 1),
        )
        for s in _sessions.values()
    ]
    return SessionsListResponse(sessions=sessions, count=len(sessions))


# =============================================================================
# Helpers
# =============================================================================

def _build_response(det_result: Any, model_ref: str, escalated: bool,
                    session: StreamSession) -> StreamFrameResponse:
    """Build response from detection result."""
    # Check cooldown
    now = time.time()
    if now - session.last_action_at < session.cooldown_seconds:
        return StreamFrameResponse(status="ok")

    session.actions_triggered += 1
    session.last_action_at = now

    detections = []
    if hasattr(det_result, "boxes"):
        detections = [
            DetectionItem(
                x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2,
                class_name=b.class_name, class_id=b.class_id, confidence=b.confidence,
            ) for b in det_result.boxes
        ]

    return StreamFrameResponse(
        status="escalated" if escalated else "action",
        detections=detections,
        confidence=det_result.confidence if hasattr(det_result, "confidence") else None,
        resolved_by=model_ref,
    )


async def _call_remote(url: str, image_bytes: bytes, session: StreamSession) -> Any | None:
    """Call a remote vision detection endpoint.
    
    SSRF Protection: Only calls URLs with hosts in the allowlist.
    If allowlist is empty, all remote calls are rejected.
    """
    import base64
    
    # Validate URL against allowlist
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.warning(f"Invalid scheme in remote URL: {url}")
            return None
        if not _ALLOWED_REMOTE_HOSTS:
            logger.warning("Remote cascade disabled: no allowed hosts configured")
            raise HTTPException(status_code=403, detail="Remote cascade not allowed")
        if parsed.hostname not in _ALLOWED_REMOTE_HOSTS:
            logger.warning(f"Remote host {parsed.hostname} not in allowlist")
            raise HTTPException(status_code=403, detail=f"Remote host not allowed: {parsed.hostname}")
    except ValueError as e:
        logger.warning(f"Malformed remote URL: {url} - {e}")
        return None
    
    try:
        client = _get_http_client()
        resp = await client.post(url, json={
            "image": base64.b64encode(image_bytes).decode(),
            "confidence_threshold": session.cascade.confidence_threshold,
            "classes": session.action_classes,
        })
        if resp.status_code == 200:
            data = resp.json()
            return _RemoteResult(data)
    except Exception as e:
        logger.warning(f"Remote cascade call to {url} failed: {e}")
    return None


@dataclass
class _RemoteBox:
    """Bounding box from a remote detection result."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    class_id: int
    confidence: float


class _RemoteResult:
    """Simple wrapper for remote detection results."""
    def __init__(self, data: dict):
        dets = data.get("detections", [])
        self.confidence = max((d.get("confidence", 0) for d in dets), default=0.0)
        self.boxes = []
        for d in dets:
            box = d.get("box", {})
            try:
                self.boxes.append(_RemoteBox(
                    x1=box.get("x1", 0), y1=box.get("y1", 0),
                    x2=box.get("x2", 0), y2=box.get("y2", 0),
                    class_name=d.get("class_name", "unknown"),
                    class_id=d.get("class_id", 0),
                    confidence=d.get("confidence", 0),
                ))
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping malformed remote detection: {e}")
