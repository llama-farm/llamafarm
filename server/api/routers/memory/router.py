"""
Memory Router - API endpoints for the Embedded Trinity Memory System.

Provides unified access to:
- Vector Memory (ChromaDB) - Semantic search
- Time-Series Memory (DuckDB) - Telemetry, spatial queries
- Graph Memory (DuckDB) - Entity relationships
- Working Memory - Short-term buffer with TTL

Endpoints:
- POST /v1/memory/add - Add data to memory
- GET /v1/memory/query - Query unified context
- GET /v1/memory/stats - Get storage statistics
- GET /v1/memory/context - Get aggregated context
- POST /v1/memory/consolidate - Trigger memory synthesis
- DELETE /v1/memory/{uuid} - Cascade delete
- POST /v1/memory/prune - Prune expired records
"""

import logging
import re
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from services.memory_service import MemoryService

from .types import (
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryConsolidateRequest,
    MemoryConsolidateResponse,
    MemoryContextResponse,
    MemoryDeleteResponse,
    MemoryPruneRequest,
    MemoryPruneResponse,
    MemoryQueryResponse,
    MemoryRecord,
    MemoryStatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


# =============================================================================
# Add Data Endpoint
# =============================================================================


@router.post("/add", response_model=MemoryAddResponse)
async def add_to_memory(request: MemoryAddRequest) -> MemoryAddResponse:
    """Add data to the appropriate memory store.

    Routes data based on data_type:
    - text -> Vector Store (ChromaDB)
    - telemetry -> Time-Series Store (DuckDB)
    - chat, audio -> Working Memory (TTL buffer)
    - node, edge -> Graph Store (DuckDB)

    Example request (text):
    ```json
    {
        "data": "Important information to remember",
        "data_type": "text",
        "metadata": {"category": "reference"}
    }
    ```

    Example request (telemetry with location):
    ```json
    {
        "data": {"heart_rate": 72, "temperature": 98.6},
        "data_type": "telemetry",
        "latitude": 35.7800,
        "longitude": -78.6400,
        "metadata": {"device_id": "sensor-001"}
    }
    ```
    """
    result = MemoryService.add(
        data=request.data,
        data_type=request.data_type,
        metadata=request.metadata,
        timestamp=request.timestamp,
        latitude=request.latitude,
        longitude=request.longitude,
    )

    return MemoryAddResponse(**result)


# =============================================================================
# Query Endpoint
# =============================================================================


@router.get("/query", response_model=MemoryQueryResponse)
async def query_memory(
    start_time: datetime | None = Query(
        None, description="Start of time range (ISO format)"
    ),
    end_time: datetime | None = Query(
        None, description="End of time range (ISO format)"
    ),
    data_types: str | None = Query(
        None, description="Comma-separated list of data types to filter by"
    ),
    latitude: float | None = Query(
        None, description="Center latitude for spatial query"
    ),
    longitude: float | None = Query(
        None, description="Center longitude for spatial query"
    ),
    radius_m: float | None = Query(
        None, description="Radius in meters for spatial query"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
) -> MemoryQueryResponse:
    """Query unified context from all memory stores.

    Supports filtering by:
    - Time range (start_time, end_time)
    - Spatial location (latitude, longitude, radius_m)
    - Data types (comma-separated list)

    Example:
    ```
    GET /v1/memory/query?latitude=35.78&longitude=-78.64&radius_m=1000
    GET /v1/memory/query?data_types=chat,audio&limit=50
    GET /v1/memory/query?start_time=2024-01-01T00:00:00Z&end_time=2024-01-02T00:00:00Z
    ```
    """
    # Parse data_types
    parsed_data_types = None
    if data_types:
        parsed_data_types = [dt.strip() for dt in data_types.split(",")]

    result = MemoryService.query(
        start_time=start_time,
        end_time=end_time,
        data_types=parsed_data_types,
        latitude=latitude,
        longitude=longitude,
        radius_m=radius_m,
        limit=limit,
    )

    # Convert results to MemoryRecord objects
    records = []
    for r in result.get("results", []):
        records.append(
            MemoryRecord(
                uuid=r.get("uuid") or r.get("id"),
                content=r.get("content") or r.get("data"),
                data_type=r.get("data_type"),
                store=r.get("store"),
                created_at=r.get("created_at"),
                metadata=r.get("metadata"),
                distance_m=r.get("distance_m"),
            )
        )

    return MemoryQueryResponse(
        results=records,
        total_count=result.get("total_count", len(records)),
    )


# =============================================================================
# Delete Endpoint
# =============================================================================


@router.delete("/{uuid}", response_model=MemoryDeleteResponse)
async def delete_from_memory(uuid: str) -> MemoryDeleteResponse:
    """Delete a record by UUID (cascade delete via LinkageTable).

    This performs a cascade delete, removing the record from all
    linked stores (vector, graph, time-series).

    Args:
        uuid: The concept UUID to delete

    Returns:
        Deletion result with list of affected stores
    """
    # Validate UUID format (prevent path traversal)
    if not re.match(r"^[a-zA-Z0-9_\-]+$", uuid):
        raise HTTPException(status_code=400, detail=f"Invalid UUID format: {uuid}")

    result = MemoryService.delete(uuid)

    if result is None:
        raise HTTPException(status_code=404, detail=f"Record not found: {uuid}")

    return MemoryDeleteResponse(**result)


# =============================================================================
# Consolidate Endpoint
# =============================================================================


@router.post("/consolidate", response_model=MemoryConsolidateResponse)
async def consolidate_memory(
    request: MemoryConsolidateRequest | None = None,
) -> MemoryConsolidateResponse:
    """Trigger memory consolidation cycle.

    The consolidator (the "hippocampus") performs:
    1. Reads raw data from Working Memory
    2. Synthesizes facts using LLM or rule-based extraction
    3. Creates graph nodes from extracted facts
    4. Prunes processed raw data

    Example request:
    ```json
    {
        "use_llm": false,
        "force": false
    }
    ```

    If use_llm is false, rule-based extraction is used (faster, no API calls).
    """
    use_llm = request.use_llm if request else False
    force = request.force if request else False

    result = MemoryService.consolidate(use_llm=use_llm, force=force)

    return MemoryConsolidateResponse(**result)


# =============================================================================
# Stats Endpoint
# =============================================================================


@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats() -> MemoryStatsResponse:
    """Get storage statistics from all memory stores.

    Returns statistics including:
    - Working Memory: record count, breakdown by type
    - Graph Store: node and edge counts
    - Time-Series Store: record count, time range
    - Linkage Table: total cross-store links

    Example response:
    ```json
    {
        "working_memory": {"total_records": 150, "by_type": {"chat": 100, "audio": 50}},
        "graph": {"total_nodes": 45, "total_edges": 78},
        "timeseries": {"total_records": 10000},
        "linkage": {"total_links": 250}
    }
    ```
    """
    result = MemoryService.get_stats()

    return MemoryStatsResponse(**result)


# =============================================================================
# Context Endpoint
# =============================================================================


@router.get("/context", response_model=MemoryContextResponse)
async def get_memory_context(
    recent_minutes: int = Query(10, ge=1, le=1440, description="How far back to look"),
    include_graph: bool = Query(True, description="Include graph relationships"),
    include_working_memory: bool = Query(True, description="Include working memory"),
    limit: int = Query(100, ge=1, le=1000, description="Max records per store"),
) -> MemoryContextResponse:
    """Get aggregated context from all memory stores.

    Returns recent data from all stores, useful for building
    context for LLM prompts or agent decision-making.

    Example:
    ```
    GET /v1/memory/context?recent_minutes=10&include_graph=true
    ```

    Response includes:
    - working_memory: Recent chat, audio, stream data
    - graph: Related entity relationships
    - timeseries: Recent telemetry data
    - summary: Brief description of context
    """
    result = MemoryService.get_context(
        recent_minutes=recent_minutes,
        include_graph=include_graph,
        include_working_memory=include_working_memory,
        limit=limit,
    )

    return MemoryContextResponse(**result)


# =============================================================================
# Prune Endpoint
# =============================================================================


@router.post("/prune", response_model=MemoryPruneResponse)
async def prune_memory(
    request: MemoryPruneRequest | None = None,
) -> MemoryPruneResponse:
    """Prune expired records from working memory.

    By default, removes records that have exceeded their TTL.
    Optionally, can prune records older than a specified number of hours.

    Example request:
    ```json
    {
        "older_than_hours": 24
    }
    ```
    """
    older_than_hours = request.older_than_hours if request else None

    result = MemoryService.prune(older_than_hours=older_than_hours)

    return MemoryPruneResponse(**result)
