"""
Pydantic models for event logs API.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EventSummary(BaseModel):
    """Summary of an event for list responses."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of event (inference, rag_processing, etc.)")
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(..., description="Event start timestamp")
    namespace: str = Field(..., description="Project namespace")
    project: str = Field(..., description="Project name")
    status: str = Field(..., description="Event status (completed, failed)")
    duration_ms: float | None = Field(None, description="Total event duration in milliseconds")
    config_hash: str = Field(..., description="Config hash at time of event")


class SubEvent(BaseModel):
    """Individual sub-event within an event."""

    timestamp: datetime = Field(..., description="Sub-event timestamp")
    event_name: str = Field(..., description="Sub-event name")
    duration_ms: float = Field(..., description="Duration from event start in milliseconds")
    data: dict[str, Any] = Field(..., description="Sub-event data")


class EventDetail(BaseModel):
    """Full event details including all sub-events."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of event")
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(..., description="Event start timestamp")
    namespace: str = Field(..., description="Project namespace")
    project: str = Field(..., description="Project name")
    config_hash: str = Field(..., description="Config hash at time of event")
    events: list[SubEvent] = Field(..., description="List of sub-events")
    status: str = Field(..., description="Event status")
    error: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    total_elapsed_time_ms: float | None = Field(None, description="Total event duration in milliseconds")
    time_to_first_token_ms: float | None = Field(None, description="Time to first token in milliseconds (for streaming inference)")


class ListEventsResponse(BaseModel):
    """Response for list events endpoint."""

    total: int = Field(..., description="Total number of matching events")
    events: list[EventSummary] = Field(..., description="List of event summaries")
    limit: int = Field(..., description="Limit applied")
    offset: int = Field(..., description="Offset applied")
