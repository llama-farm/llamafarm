"""
Pydantic models for Per-Project Memory API endpoints.

Phase 12: Per-Project Memory API Router

These models define the request/response schemas for the
per-project memory API at /v1/projects/{namespace}/{project}/memory/*.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Request Types
# =============================================================================


class ProjectMemoryAddRequest(BaseModel):
    """Request to add data to a project's memory store."""

    data: Any = Field(..., description="The data to store (text, dict, or structured)")
    data_type: str = Field(
        default="text",
        description="Type of data: text, telemetry, chat, audio, node, edge",
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata to attach to the record"
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


class ProjectMemoryConsolidateRequest(BaseModel):
    """Request to trigger memory consolidation for a project."""

    use_llm: bool = Field(
        default=False,
        description="Whether to use LLM for synthesis (falls back to rule-based)",
    )


# =============================================================================
# Response Types
# =============================================================================


class ProjectMemoryAddResponse(BaseModel):
    """Response from adding data to project memory."""

    success: bool
    uuid: str | None = None
    store: str | None = Field(
        default=None,
        description="Which store component the data was added to",
    )
    component_id: str | None = None
    message: str | None = None


class ProjectMemoryRecord(BaseModel):
    """A single record from the project memory system."""

    uuid: str | None = None
    content: Any | None = None
    data_type: str | None = None
    store: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] | None = None
    distance_m: float | None = Field(
        default=None, description="Distance in meters (for spatial queries)"
    )


class ProjectMemoryQueryResponse(BaseModel):
    """Response from querying project memory."""

    success: bool = True
    results: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    error: str | None = None


class ProjectMemoryContextResponse(BaseModel):
    """Response containing aggregated context from project memory."""

    success: bool = True
    working_memory: list[dict[str, Any]] = Field(default_factory=list)
    graph: list[dict[str, Any]] = Field(default_factory=list)
    timeseries: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class ProjectMemoryDeleteResponse(BaseModel):
    """Response from deleting a memory record."""

    success: bool
    uuid: str
    deleted_from: list[str] | None = Field(
        default=None, description="Stores the record was deleted from"
    )
    message: str | None = None


class ProjectMemoryClearResponse(BaseModel):
    """Response from clearing a memory table."""

    success: bool
    table: str
    cleared: dict[str, Any] | None = Field(
        default=None, description="Details of what was cleared"
    )
    message: str | None = None


class ProjectMemoryConsolidateResponse(BaseModel):
    """Response from memory consolidation."""

    success: bool
    records_processed: int = 0
    facts_extracted: int = 0
    nodes_created: int = 0
    pruned: int = 0
    skipped: bool = False
    synthesis_method: str | None = None
    message: str | None = None


class ProjectMemoryPruneResponse(BaseModel):
    """Response from pruning expired records."""

    success: bool
    pruned_count: int = 0
    remaining_count: int | None = None
    message: str | None = None


class ProjectMemoryStatsResponse(BaseModel):
    """Response containing storage statistics for project memory."""

    success: bool = True
    working_memory: dict[str, Any] = Field(default_factory=dict)
    graph: dict[str, Any] = Field(default_factory=dict)
    timeseries: dict[str, Any] = Field(default_factory=dict)
    linkage: dict[str, Any] = Field(default_factory=dict)
    store_path: str | None = None
    total_size_bytes: int | None = None
    error: str | None = None
