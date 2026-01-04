"""
Pydantic models for Memory API endpoints.

These models define the request/response schemas for the
Embedded Trinity Memory System API.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Request Types
# =============================================================================


class MemoryAddRequest(BaseModel):
    """Request to add data to the memory system."""

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


class MemoryConsolidateRequest(BaseModel):
    """Request to trigger memory consolidation."""

    use_llm: bool = Field(
        default=False,
        description="Whether to use LLM for synthesis (falls back to rule-based)",
    )
    force: bool = Field(
        default=False,
        description="Force consolidation even if below threshold",
    )


class MemoryPruneRequest(BaseModel):
    """Request to prune expired records."""

    older_than_hours: int | None = Field(
        default=None,
        description="Prune records older than this many hours (overrides TTL)",
    )


# =============================================================================
# Response Types
# =============================================================================


class MemoryAddResponse(BaseModel):
    """Response from adding data to memory."""

    success: bool
    uuid: str | None = None
    store: str | None = Field(
        default=None,
        description="Which store the data was added to: vector, timeseries, graph, working_memory",
    )
    message: str | None = None
    nodes_created: int | None = None
    edges_created: int | None = None
    expires_at: str | None = None


class MemoryRecord(BaseModel):
    """A single record from the memory system."""

    uuid: str | None = None
    content: Any | None = None
    data_type: str | None = None
    store: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] | None = None
    distance_m: float | None = Field(
        default=None, description="Distance in meters (for spatial queries)"
    )


class MemoryQueryResponse(BaseModel):
    """Response from querying memory."""

    results: list[MemoryRecord]
    total_count: int


class MemoryDeleteResponse(BaseModel):
    """Response from deleting a memory record."""

    success: bool
    uuid: str
    deleted_from: list[str] | None = Field(
        default=None, description="Stores the record was deleted from"
    )
    message: str | None = None


class MemoryConsolidateResponse(BaseModel):
    """Response from memory consolidation."""

    success: bool
    records_processed: int = 0
    facts_extracted: int = 0
    nodes_created: int = 0
    pruned: int = 0
    skipped: bool = False
    synthesis_method: str | None = None
    message: str | None = None


class WorkingMemoryStats(BaseModel):
    """Statistics for working memory."""

    total_records: int = 0
    by_type: dict[str, int] | None = None


class GraphStats(BaseModel):
    """Statistics for graph store."""

    total_nodes: int = 0
    total_edges: int = 0


class TimeseriesStats(BaseModel):
    """Statistics for time-series store."""

    total_records: int = 0
    oldest_record: str | None = None
    newest_record: str | None = None


class LinkageStats(BaseModel):
    """Statistics for linkage table."""

    total_links: int = 0


class MemoryStatsResponse(BaseModel):
    """Response containing storage statistics."""

    working_memory: dict[str, Any] = Field(default_factory=dict)
    graph: dict[str, Any] = Field(default_factory=dict)
    timeseries: dict[str, Any] = Field(default_factory=dict)
    linkage: dict[str, Any] = Field(default_factory=dict)
    vector: dict[str, Any] | None = None


class MemoryContextResponse(BaseModel):
    """Response containing aggregated context."""

    working_memory: list[dict[str, Any]] = Field(default_factory=list)
    graph: list[dict[str, Any]] = Field(default_factory=list)
    timeseries: list[dict[str, Any]] = Field(default_factory=list)
    summary: str | None = None


class MemoryPruneResponse(BaseModel):
    """Response from pruning expired records."""

    success: bool
    pruned_count: int
    remaining_count: int | None = None
