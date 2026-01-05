"""Pydantic models for Streaming Data API endpoints.

Phase 19: Streaming Data Endpoint

These models define the request/response schemas for real-time
data ingestion into typed datasets.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Request Types
# =============================================================================


class StreamRecord(BaseModel):
    """A single stream record for ingestion."""

    data: dict[str, Any] = Field(..., description="The data payload to ingest")
    data_type: str = Field(
        default="telemetry",
        description="Type of data: telemetry, event, observation, measurement",
    )
    timestamp: datetime | None = Field(
        default=None, description="Optional timestamp (defaults to now)"
    )
    latitude: float | None = Field(
        default=None, description="Latitude for spatial data"
    )
    longitude: float | None = Field(
        default=None, description="Longitude for spatial data"
    )
    altitude: float | None = Field(
        default=None, description="Altitude in meters for 3D spatial data"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata to attach to the record"
    )
    source_id: str | None = Field(
        default=None, description="Source identifier (e.g., sensor ID, device ID)"
    )


class StreamRecordRequest(BaseModel):
    """Request to ingest a single stream record."""

    dataset: str = Field(..., description="Target dataset name")
    record: StreamRecord = Field(..., description="The stream record to ingest")


class StreamBatchRequest(BaseModel):
    """Request to batch ingest multiple stream records."""

    dataset: str = Field(..., description="Target dataset name")
    records: list[StreamRecord] = Field(
        ...,
        description="List of stream records to ingest",
        min_length=1,
        max_length=1000,  # Limit batch size to prevent memory issues
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop on first error (default: continue and report failures)",
    )


# =============================================================================
# Response Types
# =============================================================================


class StreamRecordResult(BaseModel):
    """Result of ingesting a single stream record."""

    success: bool
    record_id: str | None = None
    stores: list[str] = Field(
        default_factory=list,
        description="Stores the record was written to",
    )
    timestamp: datetime | None = None
    error: str | None = None


class StreamRecordResponse(BaseModel):
    """Response from ingesting a single stream record."""

    success: bool
    record_id: str | None = None
    dataset: str
    stores: list[str] = Field(
        default_factory=list,
        description="Stores the record was written to (timeseries, spatial, working_memory)",
    )
    timestamp: datetime | None = None
    message: str | None = None


class StreamBatchResponse(BaseModel):
    """Response from batch ingestion."""

    success: bool
    dataset: str
    total_records: int = Field(..., description="Total records in batch")
    successful: int = Field(..., description="Number of records successfully ingested")
    failed: int = Field(..., description="Number of records that failed")
    results: list[StreamRecordResult] = Field(
        default_factory=list,
        description="Individual results (only for failures or if requested)",
    )
    message: str | None = None


class DatasetStreamStats(BaseModel):
    """Stream statistics for a dataset."""

    dataset_name: str
    dataset_type: str
    total_records: int = 0
    records_today: int = 0
    records_last_hour: int = 0
    stores_enabled: list[str] = Field(default_factory=list)
    oldest_record: datetime | None = None
    newest_record: datetime | None = None
    spatial_bounds: dict[str, float] | None = Field(
        default=None,
        description="Bounding box: min_lat, max_lat, min_lon, max_lon",
    )


class StreamStatusResponse(BaseModel):
    """Response from stream status endpoint."""

    active: bool = Field(
        ..., description="Whether streaming is active for this dataset"
    )
    dataset: str
    stats: DatasetStreamStats | None = None
    message: str | None = None


# =============================================================================
# Event Types (for SSE)
# =============================================================================


class StreamEvent(BaseModel):
    """Server-Sent Event for stream updates."""

    event_type: str = Field(
        ..., description="Event type: record_added, batch_complete, error, heartbeat"
    )
    dataset: str
    data: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
