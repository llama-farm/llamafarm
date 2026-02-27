"""Polars buffer router for direct data buffer access.

Provides endpoints for:
- Creating named Polars buffers
- Appending data to buffers
- Computing rolling features
- Getting buffer statistics

These endpoints expose the Polars data substrate directly for advanced use cases.
For most users, the streaming anomaly detection API is recommended instead.
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from api_types.anomaly import (
    PolarsBufferAppendRequest,
    PolarsBufferCreateRequest,
    PolarsBufferDataResponse,
    PolarsBufferFeaturesRequest,
    PolarsBuffersListResponse,
    PolarsBufferStats,
)
from utils.polars_buffer import PolarsBuffer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/polars", tags=["polars"])

# Global buffer registry
_buffers: dict[str, PolarsBuffer] = {}
_buffers_lock = asyncio.Lock()


async def _get_buffer(buffer_id: str) -> PolarsBuffer:
    """Get a buffer by ID, raising 404 if not found."""
    if buffer_id not in _buffers:
        raise HTTPException(
            status_code=404,
            detail=f"Buffer '{buffer_id}' not found. Create it first with POST /v1/polars/buffers",
        )
    return _buffers[buffer_id]


@router.post("/buffers")
async def create_buffer(request: PolarsBufferCreateRequest) -> dict[str, Any]:
    """Create a new named Polars buffer.

    Creates a sliding window buffer that maintains the most recent N records.
    Use this for streaming data processing with automatic window truncation.

    Example:
        POST /v1/polars/buffers
        {"buffer_id": "sensor-data", "window_size": 1000}
    """
    async with _buffers_lock:
        if request.buffer_id in _buffers:
            raise HTTPException(
                status_code=409,
                detail=f"Buffer '{request.buffer_id}' already exists. Use DELETE to remove it first.",
            )

        _buffers[request.buffer_id] = PolarsBuffer(window_size=request.window_size)
        logger.info(f"Created Polars buffer: {request.buffer_id} (window_size={request.window_size})")

    return {
        "object": "buffer_created",
        "buffer_id": request.buffer_id,
        "window_size": request.window_size,
        "status": "created",
    }


@router.get("/buffers")
async def list_buffers() -> PolarsBuffersListResponse:
    """List all active Polars buffers with their statistics."""
    stats_list = []
    for buffer_id, buffer in _buffers.items():
        stats = buffer.get_stats()
        stats_list.append(
            PolarsBufferStats(
                buffer_id=buffer_id,
                size=stats.size,
                window_size=stats.window_size,
                columns=stats.columns,
                numeric_columns=stats.numeric_columns,
                memory_bytes=stats.memory_bytes,
                append_count=stats.append_count,
                avg_append_ms=stats.avg_append_ms,
            )
        )

    return PolarsBuffersListResponse(data=stats_list, total=len(stats_list))


@router.get("/buffers/{buffer_id}")
async def get_buffer_stats(buffer_id: str) -> PolarsBufferStats:
    """Get statistics for a specific buffer."""
    buffer = await _get_buffer(buffer_id)
    stats = buffer.get_stats()

    return PolarsBufferStats(
        buffer_id=buffer_id,
        size=stats.size,
        window_size=stats.window_size,
        columns=stats.columns,
        numeric_columns=stats.numeric_columns,
        memory_bytes=stats.memory_bytes,
        append_count=stats.append_count,
        avg_append_ms=stats.avg_append_ms,
    )


@router.delete("/buffers/{buffer_id}")
async def delete_buffer(buffer_id: str) -> dict[str, Any]:
    """Delete a buffer and free its memory."""
    async with _buffers_lock:
        if buffer_id not in _buffers:
            raise HTTPException(status_code=404, detail=f"Buffer '{buffer_id}' not found")

        del _buffers[buffer_id]
        logger.info(f"Deleted Polars buffer: {buffer_id}")

    return {"deleted": True, "buffer_id": buffer_id}


@router.post("/buffers/{buffer_id}/clear")
async def clear_buffer(buffer_id: str) -> dict[str, Any]:
    """Clear all data from a buffer (keep the buffer itself)."""
    buffer = await _get_buffer(buffer_id)
    buffer.clear()

    return {"cleared": True, "buffer_id": buffer_id, "size": 0}


@router.post("/append")
async def append_data(request: PolarsBufferAppendRequest) -> dict[str, Any]:
    """Append data to a buffer.

    Supports single records or batches:
    - Single: {"buffer_id": "my-buffer", "data": {"value": 1.0, "label": "A"}}
    - Batch: {"buffer_id": "my-buffer", "data": [{"value": 1.0}, {"value": 2.0}]}

    The buffer automatically truncates to window_size, keeping the most recent records.
    """
    buffer = await _get_buffer(request.buffer_id)

    if isinstance(request.data, list):
        buffer.append_batch(request.data)
        count = len(request.data)
    else:
        buffer.append(request.data)
        count = 1

    stats = buffer.get_stats()

    return {
        "object": "append_result",
        "buffer_id": request.buffer_id,
        "appended": count,
        "buffer_size": stats.size,
        "avg_append_ms": stats.avg_append_ms,
    }


@router.post("/features")
async def compute_features(request: PolarsBufferFeaturesRequest) -> PolarsBufferDataResponse:
    """Compute rolling features from buffer data.

    Computes rolling statistics (mean, std, min, max) and lag features
    for all numeric columns in the buffer.

    Example:
        POST /v1/polars/features
        {
            "buffer_id": "sensor-data",
            "rolling_windows": [5, 10],
            "include_lags": true,
            "lag_periods": [1, 2],
            "tail": 10
        }

    Returns the data with computed features as new columns.
    """
    buffer = await _get_buffer(request.buffer_id)

    if buffer.size == 0:
        return PolarsBufferDataResponse(
            buffer_id=request.buffer_id,
            rows=0,
            columns=[],
            data=[],
        )

    # Compute features
    df = buffer.get_features(
        rolling_windows=request.rolling_windows,
        include_lags=request.include_lags,
        lag_periods=request.lag_periods,
    )

    # Optionally return only the tail
    if request.tail is not None and request.tail > 0:
        df = df.tail(request.tail)

    # Convert to list of dicts
    data = df.to_dicts()

    return PolarsBufferDataResponse(
        buffer_id=request.buffer_id,
        rows=len(data),
        columns=df.columns,
        data=data,
    )


@router.get("/buffers/{buffer_id}/data")
async def get_buffer_data(
    buffer_id: str,
    tail: int | None = None,
    with_features: bool = False,
) -> PolarsBufferDataResponse:
    """Get raw data from a buffer.

    Args:
        buffer_id: Buffer identifier
        tail: Return only last N rows (optional)
        with_features: Compute and include rolling features

    Returns buffer data as a list of dictionaries.
    """
    buffer = await _get_buffer(buffer_id)

    if buffer.size == 0:
        return PolarsBufferDataResponse(
            buffer_id=buffer_id,
            rows=0,
            columns=[],
            data=[],
        )

    df = buffer.get_features() if with_features else buffer.get_data()

    if tail is not None and tail > 0:
        df = df.tail(tail)

    data = df.to_dicts()

    return PolarsBufferDataResponse(
        buffer_id=buffer_id,
        rows=len(data),
        columns=df.columns,
        data=data,
    )


# Helper function to get buffer manager (for use by other routers)
def get_buffer(buffer_id: str) -> PolarsBuffer | None:
    """Get a buffer by ID (returns None if not found)."""
    return _buffers.get(buffer_id)


def create_or_get_buffer(buffer_id: str, window_size: int = 1000) -> PolarsBuffer:
    """Get existing buffer or create a new one."""
    if buffer_id not in _buffers:
        _buffers[buffer_id] = PolarsBuffer(window_size=window_size)
        logger.info(f"Created Polars buffer: {buffer_id} (window_size={window_size})")
    return _buffers[buffer_id]
