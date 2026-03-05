"""Object tracking router — persistent track IDs across frames.

Uses ultralytics model.track(persist=True) for ByteTrack/BoT-SORT/OC-SORT.
Sessions hold a loaded YOLO model with tracker state. TTL auto-expires idle sessions.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.error_handler import handle_endpoint_errors

from .utils import decode_base64_image

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-tracking"])

# ── Dependency injection ────────────────────────────────────────────────────

_VISION_MODELS_DIR: Path | None = None


def set_tracking_models_dir(d: Path) -> None:
    global _VISION_MODELS_DIR
    _VISION_MODELS_DIR = d


# ── Session state ───────────────────────────────────────────────────────────

VALID_TRACKERS = {"bytetrack", "botsort", "ocsort"}
MAX_SESSIONS = 50
SESSION_TTL_SECONDS = 120.0  # 2 min idle → expire


@dataclass
class TrackSession:
    session_id: str
    model_id: str
    tracker: str
    yolo_model: Any  # Loaded YOLO model instance
    confidence_threshold: float = 0.25
    target_fps: float = 10.0
    frames_processed: int = 0
    total_tracks_created: int = 0
    created_at: float = field(default_factory=time.time)
    last_frame_at: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock
    )  # Prevent concurrent frame races


_sessions: dict[str, TrackSession] = {}
_cleanup_task: asyncio.Task | None = None  # Held to prevent GC of the background task


async def _session_cleanup_loop() -> None:
    """Expire idle tracking sessions."""
    while True:
        await asyncio.sleep(15)
        now = time.time()
        expired = [
            sid
            for sid, s in _sessions.items()
            if (now - s.last_frame_at) > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            session = _sessions.pop(sid, None)
            if session:
                logger.info(
                    f"Expired tracking session {sid} "
                    f"(idle {now - session.last_frame_at:.0f}s, "
                    f"{session.frames_processed} frames)"
                )


def start_tracking_cleanup() -> None:
    """Start background cleanup task. Call once at server startup."""
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_session_cleanup_loop())


async def stop_tracking_cleanup() -> None:
    """Cancel the background tracking cleanup task (call during shutdown)."""
    global _cleanup_task
    if _cleanup_task is not None and not _cleanup_task.done():
        _cleanup_task.cancel()
        import contextlib
        with contextlib.suppress(asyncio.CancelledError):
            await _cleanup_task
        logger.info("Vision tracking cleanup task stopped")
    _cleanup_task = None


# ── Model resolution ───────────────────────────────────────────────────────


def _resolve_model_path(model: str) -> str:
    """Resolve model name to .pt path."""
    p = Path(model)
    if p.exists() and p.suffix == ".pt":
        # Containment check
        resolved = p.resolve()
        home_lf = (Path.home() / ".llamafarm").resolve()
        cwd = Path.cwd().resolve()
        if not (resolved.is_relative_to(home_lf) or resolved.is_relative_to(cwd)):
            raise HTTPException(400, "Model path outside allowed directories")
        return str(resolved)

    if _VISION_MODELS_DIR:
        safe_name = Path(model).name
        if safe_name != model or ".." in model or ":" in model or "\\" in model:
            raise HTTPException(400, "Invalid model name")

        current = _VISION_MODELS_DIR / safe_name / "current.pt"
        if current.exists():
            return str(current)

        model_dir = _VISION_MODELS_DIR / safe_name
        if model_dir.is_dir():
            versions = sorted(model_dir.glob("v*.pt"), reverse=True)
            if versions:
                return str(versions[0])

    raise HTTPException(404, f"Model not found: {model}")


# ── Request/Response models ─────────────────────────────────────────────────


class TrackStartRequest(BaseModel):
    """Start a tracking session. First frame can be included."""

    model: str = Field(..., description="Model name or path to .pt file")
    tracker: str = Field(
        "bytetrack", description="Tracker algorithm: bytetrack, botsort, ocsort"
    )
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    target_fps: float = Field(10.0, ge=0.1, le=60.0)
    image: str | None = Field(None, description="Optional base64 image for first frame")


class TrackedDetection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    class_id: int
    confidence: float
    track_id: int
    track_state: str = "tracked"  # new, tracked, lost


class TracksSummary(BaseModel):
    active: int = 0
    total_created: int = 0


class TrackStartResponse(BaseModel):
    session_id: str
    tracker: str
    model: str
    # If first frame was included, return detections
    detections: list[TrackedDetection] | None = None
    tracks_summary: TracksSummary | None = None
    inference_time_ms: float | None = None
    tracking_time_ms: float | None = None


class TrackFrameRequest(BaseModel):
    session_id: str
    image: str = Field(..., description="Base64-encoded image")


class TrackFrameResponse(BaseModel):
    detections: list[TrackedDetection]
    tracks_summary: TracksSummary
    inference_time_ms: float
    tracking_time_ms: float
    frame_number: int


class TrackStatusResponse(BaseModel):
    session_id: str
    model: str
    tracker: str
    frames_processed: int
    total_tracks_created: int
    idle_seconds: float
    duration_seconds: float


class TrackStopResponse(BaseModel):
    session_id: str
    frames_processed: int
    total_tracks_created: int
    duration_seconds: float


# ── Core tracking logic ─────────────────────────────────────────────────────


def _run_track(
    session: TrackSession, image_bytes: bytes
) -> tuple[list[TrackedDetection], TracksSummary, float, float]:
    """Run model.track() synchronously. Called via asyncio.to_thread()."""
    import io

    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img)

    t0 = time.time()
    results = session.yolo_model.track(
        img_array,
        persist=True,
        tracker=f"{session.tracker}.yaml",
        conf=session.confidence_threshold,
        verbose=False,
    )
    t1 = time.time()

    detections: list[TrackedDetection] = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        names = results[0].names or {}

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            cls_name = names.get(cls_id, f"class_{cls_id}")

            track_id = -1
            if boxes.id is not None:
                track_id = int(boxes.id[i])

            detections.append(
                TrackedDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    class_name=cls_name,
                    class_id=cls_id,
                    confidence=conf,
                    track_id=track_id,
                )
            )

    active_tracks = len({d.track_id for d in detections if d.track_id >= 0})
    # Track total unique IDs seen
    max_id = max((d.track_id for d in detections if d.track_id >= 0), default=0)
    if max_id > session.total_tracks_created:
        session.total_tracks_created = max_id

    summary = TracksSummary(
        active=active_tracks,
        total_created=session.total_tracks_created,
    )

    total_ms = (t1 - t0) * 1000
    # Approximate split: tracking is ~10% of total for ByteTrack
    tracking_ms = total_ms * 0.1
    inference_ms = total_ms * 0.9

    return detections, summary, inference_ms, tracking_ms


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/v1/vision/track/start", response_model=TrackStartResponse)
@handle_endpoint_errors("vision_track_start")
async def start_tracking(request: TrackStartRequest) -> TrackStartResponse:
    """Start a tracking session. Optionally include first frame."""
    if request.tracker not in VALID_TRACKERS:
        raise HTTPException(
            400,
            f"Invalid tracker: {request.tracker}. Valid: {', '.join(sorted(VALID_TRACKERS))}",
        )
    if len(_sessions) >= MAX_SESSIONS:
        raise HTTPException(429, f"Max {MAX_SESSIONS} concurrent tracking sessions")

    model_path = _resolve_model_path(request.model)

    # Load a fresh YOLO model in thread pool to avoid blocking the event loop
    from ultralytics import YOLO

    yolo_model = await asyncio.to_thread(YOLO, model_path)

    # Collision-safe short ID (MAX_SESSIONS=50, so collisions are rare but possible)
    sid = str(uuid.uuid4())[:8]
    while sid in _sessions:
        sid = str(uuid.uuid4())[:8]
    session = TrackSession(
        session_id=sid,
        model_id=request.model,
        tracker=request.tracker,
        yolo_model=yolo_model,
        confidence_threshold=request.confidence_threshold,
        target_fps=request.target_fps,
    )
    _sessions[sid] = session

    response = TrackStartResponse(
        session_id=sid,
        tracker=request.tracker,
        model=request.model,
    )

    # If first frame included, process it
    if request.image:
        image_bytes = decode_base64_image(request.image)
        session.frames_processed += 1
        session.last_frame_at = time.time()
        detections, summary, inf_ms, trk_ms = await asyncio.to_thread(
            _run_track, session, image_bytes
        )
        response.detections = detections
        response.tracks_summary = summary
        response.inference_time_ms = round(inf_ms, 2)
        response.tracking_time_ms = round(trk_ms, 2)

    logger.info(
        f"Started tracking session {sid}: model={request.model} tracker={request.tracker}"
    )
    return response


@router.post("/v1/vision/track/frame", response_model=TrackFrameResponse)
@handle_endpoint_errors("vision_track_frame")
async def track_frame(request: TrackFrameRequest) -> TrackFrameResponse:
    """Process a frame through the tracker. Returns detections with persistent track IDs."""
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, "Tracking session not found")

    image_bytes = decode_base64_image(request.image)

    # Lock per-session to prevent concurrent frames racing on tracker state
    async with session.lock:
        session.frames_processed += 1
        session.last_frame_at = time.time()

        detections, summary, inf_ms, trk_ms = await asyncio.to_thread(
            _run_track, session, image_bytes
        )

    return TrackFrameResponse(
        detections=detections,
        tracks_summary=summary,
        inference_time_ms=round(inf_ms, 2),
        tracking_time_ms=round(trk_ms, 2),
        frame_number=session.frames_processed,
    )


@router.get("/v1/vision/track/{session_id}", response_model=TrackStatusResponse)
@handle_endpoint_errors("vision_track_status")
async def track_status(session_id: str) -> TrackStatusResponse:
    """Get tracking session status."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Tracking session not found")
    now = time.time()
    return TrackStatusResponse(
        session_id=session.session_id,
        model=session.model_id,
        tracker=session.tracker,
        frames_processed=session.frames_processed,
        total_tracks_created=session.total_tracks_created,
        idle_seconds=round(now - session.last_frame_at, 1),
        duration_seconds=round(now - session.created_at, 1),
    )


class TrackStopRequest(BaseModel):
    session_id: str


@router.post("/v1/vision/track/stop", response_model=TrackStopResponse)
@handle_endpoint_errors("vision_track_stop")
async def stop_tracking(request: TrackStopRequest) -> TrackStopResponse:
    """Stop a tracking session and release resources."""
    session = _sessions.pop(request.session_id, None)
    if not session:
        raise HTTPException(404, "Tracking session not found")
    logger.info(
        f"Stopped tracking session {session.session_id}: "
        f"{session.frames_processed} frames, {session.total_tracks_created} tracks"
    )
    return TrackStopResponse(
        session_id=session.session_id,
        frames_processed=session.frames_processed,
        total_tracks_created=session.total_tracks_created,
        duration_seconds=round(time.time() - session.created_at, 1),
    )
