"""Streaming Data Router - Real-time data ingestion for datasets.

Phase 19: Streaming Data Endpoint

Provides:
- POST /stream/ingest - Batch ingest stream records
- POST /stream/record - Single record ingestion
- GET /stream/status - Stream ingestion status
- SSE endpoint for real-time data push
"""

from .router import router
from .types import (
    StreamBatchRequest,
    StreamBatchResponse,
    StreamRecord,
    StreamRecordRequest,
    StreamRecordResponse,
    StreamStatusResponse,
)

__all__ = [
    "router",
    "StreamBatchRequest",
    "StreamBatchResponse",
    "StreamRecord",
    "StreamRecordRequest",
    "StreamRecordResponse",
    "StreamStatusResponse",
]
