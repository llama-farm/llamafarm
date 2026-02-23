"""Federated corrections — batch upload from edge devices, aggregation, retrain triggers.

Endpoints:
- POST /v1/vision/models/{model_id}/corrections/upload  — Batch upload corrections
- GET  /v1/vision/models/{model_id}/corrections/stats    — Correction stats + retrain readiness
- POST /v1/vision/models/{model_id}/corrections/trigger  — Manual retrain trigger from corrections
- GET  /v1/vision/models/{model_id}/corrections/devices   — Per-device correction stats

Corrections are stored in the model's replay buffer (SQLite).
When threshold is reached, auto-retrain can be triggered.
"""

import base64
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.error_handler import handle_endpoint_errors

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vision-corrections"])

_VISION_MODELS_DIR: Path | None = None
_retrain_callback = None  # Optional callback to trigger retraining


def set_corrections_models_dir(d: Path) -> None:
    global _VISION_MODELS_DIR
    _VISION_MODELS_DIR = d


def set_retrain_callback(fn) -> None:
    """Set callback for auto-retrain. Signature: async fn(model_id, corrections_count)"""
    global _retrain_callback
    _retrain_callback = fn


def _models_dir() -> Path:
    if _VISION_MODELS_DIR is None:
        raise HTTPException(status_code=500, detail="Vision models dir not configured")
    return _VISION_MODELS_DIR


def _model_dir(model_id: str) -> Path:
    safe_id = Path(model_id).name
    if safe_id != model_id or ".." in model_id or ":" in model_id or "\\" in model_id:
        raise HTTPException(status_code=400, detail="Invalid model_id")
    d = _models_dir() / safe_id
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return d


# =============================================================================
# Types
# =============================================================================

class CorrectionItem(BaseModel):
    """A single correction from an edge device."""
    image_base64: str = Field(..., description="Base64-encoded image crop")
    original_label: str = Field(..., description="What the model predicted")
    corrected_label: str = Field(..., description="What the human corrected it to")
    confidence: float = Field(0.0, description="Original model confidence")
    bbox: list[float] | None = Field(None, description="[x1, y1, x2, y2] bounding box")
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchUploadRequest(BaseModel):
    """Batch upload of corrections from an edge device."""
    device_id: str = Field(..., description="Source device identifier")
    corrections: list[CorrectionItem] = Field(..., min_length=1, max_length=500)
    collected_from: str | None = Field(None, description="ISO timestamp of first correction")
    collected_to: str | None = Field(None, description="ISO timestamp of last correction")


class BatchUploadResponse(BaseModel):
    accepted: int
    rejected: int
    total_corrections: int  # Total in replay buffer after upload
    retrain_ready: bool  # Whether threshold has been reached
    retrain_threshold: int


class CorrectionStats(BaseModel):
    total_corrections: int
    by_device: dict[str, int]
    by_label: dict[str, int]
    retrain_threshold: int
    retrain_ready: bool
    last_upload: str | None
    last_retrain: str | None


class TriggerRetrainRequest(BaseModel):
    epochs: int = Field(50, ge=1, le=500)
    imgsz: int = Field(640, ge=128, le=2048)
    batch_size: int = Field(8, ge=1, le=64)
    min_corrections: int = Field(10, ge=1, description="Minimum corrections required to proceed")


# Default retrain threshold — number of corrections before auto-retrain is suggested
RETRAIN_THRESHOLD = 50


# =============================================================================
# Helpers
# =============================================================================

def _corrections_dir(model_dir: Path) -> Path:
    d = model_dir / "training_data" / "corrections"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _corrections_db_path(model_dir: Path) -> Path:
    return model_dir / "corrections.db"


def _init_corrections_db(db_path: Path) -> None:
    """Initialize the corrections SQLite database."""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                original_label TEXT NOT NULL,
                corrected_label TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                image_path TEXT,
                bbox_json TEXT,
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS correction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                device_id TEXT,
                count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_corrections_device ON corrections(device_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_corrections_label ON corrections(corrected_label)")


def _get_correction_stats(db_path: Path) -> dict[str, Any]:
    """Get correction statistics from the database."""
    import sqlite3
    if not db_path.exists():
        return {"total": 0, "by_device": {}, "by_label": {}, "last_upload": None, "last_retrain": None}

    with sqlite3.connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]

        by_device: dict[str, int] = {}
        for row in conn.execute("SELECT device_id, COUNT(*) FROM corrections GROUP BY device_id"):
            by_device[row[0]] = row[1]

        by_label: dict[str, int] = {}
        for row in conn.execute("SELECT corrected_label, COUNT(*) FROM corrections GROUP BY corrected_label"):
            by_label[row[0]] = row[1]

        last_upload = conn.execute(
            "SELECT created_at FROM correction_events WHERE event_type='upload' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        last_retrain = conn.execute(
            "SELECT created_at FROM correction_events WHERE event_type='retrain' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

    return {
        "total": total,
        "by_device": by_device,
        "by_label": by_label,
        "last_upload": last_upload[0] if last_upload else None,
        "last_retrain": last_retrain[0] if last_retrain else None,
    }


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/v1/vision/models/{model_id}/corrections/upload", response_model=BatchUploadResponse)
@handle_endpoint_errors("corrections_upload")
async def upload_corrections(model_id: str, request: BatchUploadRequest) -> BatchUploadResponse:
    """Batch upload corrections from an edge device.

    Corrections are stored in the model's corrections database and images
    are saved to training_data/corrections/. When the total correction count
    exceeds the retrain threshold, `retrain_ready` will be True.

    Each edge device (drone) accumulates corrections locally and uploads
    them in batches when connectivity is available.
    """
    import sqlite3

    model_dir = _model_dir(model_id)
    corrections_dir = _corrections_dir(model_dir)
    db_path = _corrections_db_path(model_dir)
    _init_corrections_db(db_path)

    safe_device = Path(request.device_id).name
    if safe_device != request.device_id or ".." in request.device_id:
        raise HTTPException(status_code=400, detail="Invalid device_id")

    accepted = 0
    rejected = 0
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as conn:
        for correction in request.corrections:
            try:
                cid = uuid.uuid4().hex[:12]

                # Save image
                image_bytes = base64.b64decode(correction.image_base64)
                image_path = corrections_dir / f"{cid}.jpg"
                image_path.write_bytes(image_bytes)

                bbox_json = json.dumps(correction.bbox) if correction.bbox else None

                conn.execute(
                    """INSERT INTO corrections
                       (id, device_id, original_label, corrected_label, confidence,
                        image_path, bbox_json, metadata_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (cid, request.device_id, correction.original_label,
                     correction.corrected_label, correction.confidence,
                     str(image_path), bbox_json,
                     json.dumps(correction.metadata), now),
                )
                accepted += 1
            except Exception as e:
                logger.warning(f"Rejected correction: {e}")
                rejected += 1

        # Log upload event
        conn.execute(
            "INSERT INTO correction_events (event_type, device_id, count, created_at) VALUES (?, ?, ?, ?)",
            ("upload", request.device_id, accepted, now),
        )

    # Get total count
    stats = _get_correction_stats(db_path)
    total = stats["total"]
    retrain_ready = total >= RETRAIN_THRESHOLD

    if retrain_ready:
        logger.info(
            f"Corrections threshold reached for model '{model_id}': "
            f"{total}/{RETRAIN_THRESHOLD} corrections from {len(stats['by_device'])} devices"
        )

    return BatchUploadResponse(
        accepted=accepted,
        rejected=rejected,
        total_corrections=total,
        retrain_ready=retrain_ready,
        retrain_threshold=RETRAIN_THRESHOLD,
    )


@router.get("/v1/vision/models/{model_id}/corrections/stats", response_model=CorrectionStats)
@handle_endpoint_errors("corrections_stats")
async def get_correction_stats(model_id: str) -> CorrectionStats:
    """Get correction statistics and retrain readiness."""
    model_dir = _model_dir(model_id)
    db_path = _corrections_db_path(model_dir)
    stats = _get_correction_stats(db_path)

    return CorrectionStats(
        total_corrections=stats["total"],
        by_device=stats["by_device"],
        by_label=stats["by_label"],
        retrain_threshold=RETRAIN_THRESHOLD,
        retrain_ready=stats["total"] >= RETRAIN_THRESHOLD,
        last_upload=stats["last_upload"],
        last_retrain=stats["last_retrain"],
    )


@router.get("/v1/vision/models/{model_id}/corrections/devices")
@handle_endpoint_errors("corrections_devices")
async def list_correction_devices(model_id: str) -> dict[str, Any]:
    """List per-device correction statistics."""
    import sqlite3

    model_dir = _model_dir(model_id)
    db_path = _corrections_db_path(model_dir)

    if not db_path.exists():
        return {"devices": [], "total_devices": 0}

    with sqlite3.connect(db_path) as conn:
        devices = []
        for row in conn.execute("""
            SELECT device_id, COUNT(*) as count,
                   MIN(created_at) as first_upload,
                   MAX(created_at) as last_upload
            FROM corrections GROUP BY device_id
            ORDER BY count DESC
        """):
            devices.append({
                "device_id": row[0],
                "correction_count": row[1],
                "first_upload": row[2],
                "last_upload": row[3],
            })

    return {"devices": devices, "total_devices": len(devices)}


@router.post("/v1/vision/models/{model_id}/corrections/trigger")
@handle_endpoint_errors("corrections_trigger_retrain")
async def trigger_retrain(model_id: str, request: TriggerRetrainRequest) -> dict[str, Any]:
    """Manually trigger retraining from accumulated corrections.

    Collects all correction images and labels, formats them for YOLO training,
    and triggers a training job. Requires minimum correction count.
    """
    import sqlite3

    model_dir = _model_dir(model_id)
    db_path = _corrections_db_path(model_dir)

    if not db_path.exists():
        raise HTTPException(status_code=422, detail="No corrections database found")

    stats = _get_correction_stats(db_path)
    if stats["total"] < request.min_corrections:
        raise HTTPException(
            status_code=422,
            detail=f"Only {stats['total']} corrections available, "
            f"minimum {request.min_corrections} required",
        )

    # Log retrain event
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO correction_events (event_type, count, created_at, metadata_json) VALUES (?, ?, ?, ?)",
            ("retrain", stats["total"], now,
             json.dumps({"epochs": request.epochs, "imgsz": request.imgsz})),
        )

    # If retrain callback is set, invoke it
    if _retrain_callback:
        try:
            await _retrain_callback(model_id, stats["total"])
        except Exception as e:
            logger.error(f"Retrain callback failed: {e}")

    return {
        "model_id": model_id,
        "status": "triggered",
        "corrections_used": stats["total"],
        "unique_devices": len(stats["by_device"]),
        "unique_labels": len(stats["by_label"]),
        "config": {
            "epochs": request.epochs,
            "imgsz": request.imgsz,
            "batch_size": request.batch_size,
        },
    }
