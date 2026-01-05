"""Streaming Data Router - Real-time data ingestion for typed datasets.

Phase 19: Streaming Data Endpoint

Provides endpoints for streaming real-time data into datasets:
- Single record ingestion
- Batch ingestion
- Status and statistics
- SSE for real-time updates
"""

import logging
from collections.abc import AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from .types import (
    DatasetStreamStats,
    StreamBatchRequest,
    StreamBatchResponse,
    StreamEvent,
    StreamRecordRequest,
    StreamRecordResponse,
    StreamRecordResult,
    StreamStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/projects/{namespace}/{project}/datasets/{dataset}/stream",
    tags=["streaming"],
)


# =============================================================================
# Service Integration (lazy import to avoid circular dependencies)
# =============================================================================


def _get_dataset_store(namespace: str, project: str, dataset: str):
    """Get or create a UnifiedDatasetStore for the dataset.

    This lazily imports from RAG to avoid circular dependencies.
    """
    import importlib.util
    import sys
    from pathlib import Path

    # Resolve RAG directory path
    current_file = Path(__file__).resolve()
    server_dir = current_file.parent.parent.parent.parent  # server/
    project_root = server_dir.parent  # llamafarm/
    rag_dir = project_root / "rag"

    # Add RAG to path if not already there
    rag_path = str(rag_dir)
    if rag_path not in sys.path:
        sys.path.insert(0, rag_path)

    # Import UnifiedDatasetStore
    module_path = rag_dir / "core" / "unified_store.py"
    if not module_path.exists():
        raise ImportError(f"UnifiedDatasetStore module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("unified_store", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    UnifiedDatasetStore = module.UnifiedDatasetStore

    # Get project data directory
    from services.project_service import ProjectService

    project_dir = ProjectService.get_project_path(namespace, project)
    if not project_dir:
        raise HTTPException(
            status_code=404,
            detail=f"Project not found: {namespace}/{project}",
        )

    # Load dataset config
    from services.dataset_service import DatasetService

    datasets = DatasetService.list_datasets(namespace, project)
    dataset_config = None
    for ds in datasets:
        if ds.name == dataset:
            dataset_config = ds.model_dump()
            break

    if dataset_config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {dataset}",
        )

    # Create store with dataset config
    store = UnifiedDatasetStore(
        config={"name": dataset, "type": dataset_config.get("type", "realtime")},
        project_path=str(project_dir),
    )

    return store


# =============================================================================
# Single Record Ingestion
# =============================================================================


@router.post(
    "/record",
    response_model=StreamRecordResponse,
    operation_id="stream_record_ingest",
    summary="Ingest a single stream record",
    description="""Ingest a single real-time data record into the dataset.

The record is automatically routed to the appropriate stores based on
the dataset type configuration:
- **timeseries**: Stores time-indexed data in DuckDB
- **spatial**: Stores geo-located data with R-tree indexing
- **working_memory**: TTL-based buffer for recent data

Example request:
```json
{
    "dataset": "telemetry",
    "record": {
        "data": {"temperature": 72.5, "humidity": 45.2},
        "data_type": "sensor_reading",
        "latitude": 35.78,
        "longitude": -78.64,
        "source_id": "sensor-001"
    }
}
```
""",
)
async def ingest_record(
    namespace: str,
    project: str,
    dataset: str,
    request: StreamRecordRequest,
) -> StreamRecordResponse:
    """Ingest a single stream record."""
    logger.info(f"Ingesting record to {namespace}/{project}/{dataset}")

    try:
        store = _get_dataset_store(namespace, project, dataset)

        result = store.add_stream_record(
            data=request.record.data,
            data_type=request.record.data_type,
            timestamp=request.record.timestamp,
            latitude=request.record.latitude,
            longitude=request.record.longitude,
            altitude=request.record.altitude,
            metadata={
                **(request.record.metadata or {}),
                "source_id": request.record.source_id,
            }
            if request.record.source_id
            else request.record.metadata,
        )

        store.close()

        return StreamRecordResponse(
            success=True,
            record_id=result.get("record_id"),
            dataset=dataset,
            stores=result.get("stores", []),
            timestamp=result.get("timestamp"),
            message="Record ingested successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest record: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Batch Ingestion
# =============================================================================


@router.post(
    "/batch",
    response_model=StreamBatchResponse,
    operation_id="stream_batch_ingest",
    summary="Batch ingest multiple stream records",
    description="""Ingest multiple records in a single request for high-throughput scenarios.

Supports up to 1000 records per batch. Records are processed sequentially
and results are aggregated.

Example request:
```json
{
    "dataset": "telemetry",
    "records": [
        {"data": {"temp": 72}, "source_id": "sensor-001"},
        {"data": {"temp": 68}, "source_id": "sensor-002"}
    ],
    "fail_fast": false
}
```
""",
)
async def ingest_batch(
    namespace: str,
    project: str,
    dataset: str,
    request: StreamBatchRequest,
) -> StreamBatchResponse:
    """Batch ingest multiple stream records."""
    logger.info(
        f"Batch ingesting {len(request.records)} records to {namespace}/{project}/{dataset}"
    )

    try:
        store = _get_dataset_store(namespace, project, dataset)

        results = []
        successful = 0
        failed = 0

        for record in request.records:
            try:
                result = store.add_stream_record(
                    data=record.data,
                    data_type=record.data_type,
                    timestamp=record.timestamp,
                    latitude=record.latitude,
                    longitude=record.longitude,
                    altitude=record.altitude,
                    metadata={
                        **(record.metadata or {}),
                        "source_id": record.source_id,
                    }
                    if record.source_id
                    else record.metadata,
                )

                successful += 1
                results.append(
                    StreamRecordResult(
                        success=True,
                        record_id=result.get("record_id"),
                        stores=result.get("stores", []),
                        timestamp=result.get("timestamp"),
                    )
                )

            except Exception as e:
                failed += 1
                results.append(
                    StreamRecordResult(
                        success=False,
                        error=str(e),
                    )
                )

                if request.fail_fast:
                    break

        store.close()

        return StreamBatchResponse(
            success=failed == 0,
            dataset=dataset,
            total_records=len(request.records),
            successful=successful,
            failed=failed,
            results=[r for r in results if not r.success],  # Only return failures
            message=f"Processed {successful}/{len(request.records)} records",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Stream Status
# =============================================================================


@router.get(
    "/status",
    response_model=StreamStatusResponse,
    operation_id="stream_status",
    summary="Get stream ingestion status",
    description="Get the current status and statistics for stream ingestion on this dataset.",
)
async def get_stream_status(
    namespace: str,
    project: str,
    dataset: str,
) -> StreamStatusResponse:
    """Get stream ingestion status and statistics."""
    try:
        store = _get_dataset_store(namespace, project, dataset)

        stats = store.get_stats()
        enabled_stores = store.get_enabled_stores()

        # Build statistics
        timeseries_stats = stats.get("timeseries", {})
        spatial_stats = stats.get("spatial", {})

        stream_stats = DatasetStreamStats(
            dataset_name=stats.get("dataset_name", dataset),
            dataset_type=stats.get("dataset_type", "unknown"),
            total_records=timeseries_stats.get("record_count", 0),
            stores_enabled=enabled_stores,
            oldest_record=timeseries_stats.get("oldest_record"),
            newest_record=timeseries_stats.get("newest_record"),
            spatial_bounds=spatial_stats.get("bounding_box"),
        )

        store.close()

        return StreamStatusResponse(
            active=True,  # Always active if dataset exists and supports streaming
            dataset=dataset,
            stats=stream_stats,
            message="Stream ingestion is active",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stream status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Server-Sent Events (SSE) Endpoint
# =============================================================================


async def _generate_sse_events(
    namespace: str,
    project: str,
    dataset: str,
    heartbeat_interval: int = 30,
) -> AsyncGenerator[str, None]:
    """Generate SSE events for stream updates.

    This is a placeholder implementation. In production, this would
    subscribe to a message queue or use polling with cursors.
    """
    import asyncio
    import json

    event_count = 0

    while True:
        # Send heartbeat
        event = StreamEvent(
            event_type="heartbeat",
            dataset=dataset,
            data={"event_count": event_count},
            timestamp=datetime.utcnow(),
        )

        yield f"event: {event.event_type}\n"
        yield f"data: {json.dumps(event.model_dump(mode='json'))}\n\n"

        event_count += 1

        # Wait for next heartbeat
        await asyncio.sleep(heartbeat_interval)


@router.get(
    "/events",
    operation_id="stream_events_sse",
    summary="Subscribe to stream events (SSE)",
    description="""Subscribe to real-time stream events using Server-Sent Events.

Events include:
- `heartbeat`: Periodic keepalive (default every 30s)
- `record_added`: When a new record is ingested
- `batch_complete`: When a batch ingestion completes
- `error`: When an error occurs

Example usage with curl:
```bash
curl -N http://localhost:8000/v1/projects/ns/proj/datasets/ds/stream/events
```
""",
)
async def stream_events(
    namespace: str,
    project: str,
    dataset: str,
    heartbeat_interval: int = Query(
        default=30,
        ge=5,
        le=300,
        description="Heartbeat interval in seconds",
    ),
) -> StreamingResponse:
    """Subscribe to stream events via SSE."""
    # Verify dataset exists
    _get_dataset_store(namespace, project, dataset)

    return StreamingResponse(
        _generate_sse_events(namespace, project, dataset, heartbeat_interval),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
