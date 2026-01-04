"""
Per-Project Memory Router - API endpoints for per-project memory stores.

Phase 12: Per-Project Memory API Router

Provides per-project memory API at /v1/projects/{namespace}/{project}/memory/*:
- POST /memory/add - Add data to project memory
- GET /memory/query - Query unified context
- GET /memory/context - Get aggregated context
- DELETE /memory/{uuid} - Cascade delete
- POST /memory/clear/{table} - Clear specific table
- POST /memory/consolidate - Trigger consolidation
- POST /memory/prune - Prune expired records
- GET /memory/stats - Get storage statistics
"""

import logging
import re
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query

from api.errors import MemoryStoreNotFoundError
from services.memory_data_service import MemoryDataService

from .project_memory_types import (
    ProjectMemoryAddRequest,
    ProjectMemoryAddResponse,
    ProjectMemoryClearResponse,
    ProjectMemoryConsolidateRequest,
    ProjectMemoryConsolidateResponse,
    ProjectMemoryContextResponse,
    ProjectMemoryDeleteResponse,
    ProjectMemoryPruneResponse,
    ProjectMemoryQueryResponse,
    ProjectMemoryStatsResponse,
)

# Regex pattern for valid names (namespace, project, store_name)
# Alphanumeric, underscore, hyphen - must start with letter
SAFE_NAME_PATTERN = r"^[a-zA-Z][a-zA-Z0-9_-]*$"

# Type aliases for validated path and query parameters
NamespacePath = Annotated[
    str,
    Path(
        description="Project namespace",
        pattern=SAFE_NAME_PATTERN,
        max_length=64,
    ),
]

ProjectPath = Annotated[
    str,
    Path(
        description="Project name",
        pattern=SAFE_NAME_PATTERN,
        max_length=64,
    ),
]

# Note: default must be set with `=` in function signature, not in Query()
StoreNameQuery = Annotated[
    str | None,
    Query(
        description="Memory store name (uses default if not specified)",
        pattern=SAFE_NAME_PATTERN,
        max_length=64,
    ),
]

logger = logging.getLogger(__name__)

# Router uses per-project path pattern like RAG
router = APIRouter(
    prefix="/projects/{namespace}/{project}/memory",
    tags=["project-memory"],
)


# =============================================================================
# Add Data Endpoint
# =============================================================================


@router.post("/add", response_model=ProjectMemoryAddResponse)
async def add_to_project_memory(
    namespace: NamespacePath,
    project: ProjectPath,
    request: ProjectMemoryAddRequest = ...,
    store_name: StoreNameQuery = None,
) -> ProjectMemoryAddResponse:
    """Add data to a project's memory store.

    Routes data based on data_type:
    - text -> Vector Store (future)
    - telemetry -> Time-Series Store (DuckDB)
    - chat, audio -> Working Memory (TTL buffer)
    - node, edge -> Graph Store (DuckDB)

    Example request:
    ```json
    {
        "data": "Important information to remember",
        "data_type": "text",
        "metadata": {"category": "reference"}
    }
    ```
    """
    try:
        result = MemoryDataService.add(
            namespace=namespace,
            project=project,
            data=request.data,
            data_type=request.data_type,
            metadata=request.metadata,
            timestamp=request.timestamp,
            latitude=request.latitude,
            longitude=request.longitude,
            store_name=store_name,
        )

        return ProjectMemoryAddResponse(**result)

    except MemoryStoreNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Failed to add to project memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to add data to memory store"
        ) from None


# =============================================================================
# Query Endpoint
# =============================================================================


@router.get("/query", response_model=ProjectMemoryQueryResponse)
async def query_project_memory(
    namespace: NamespacePath,
    project: ProjectPath,
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
    store_name: StoreNameQuery = None,
) -> ProjectMemoryQueryResponse:
    """Query unified context from a project's memory store.

    Supports filtering by:
    - Time range (start_time, end_time)
    - Spatial location (latitude, longitude, radius_m)
    - Data types (comma-separated list)

    Example:
    ```
    GET /v1/projects/{ns}/{proj}/memory/query?limit=50
    ```
    """
    try:
        # Parse data_types
        parsed_data_types = None
        if data_types:
            parsed_data_types = [dt.strip() for dt in data_types.split(",")]

        result = MemoryDataService.query(
            namespace=namespace,
            project=project,
            start_time=start_time,
            end_time=end_time,
            data_types=parsed_data_types,
            latitude=latitude,
            longitude=longitude,
            radius_m=radius_m,
            recent_limit=limit,
            store_name=store_name,
        )

        return ProjectMemoryQueryResponse(**result)

    except MemoryStoreNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Failed to query project memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to query memory store"
        ) from None


# =============================================================================
# Context Endpoint
# =============================================================================


@router.get("/context", response_model=ProjectMemoryContextResponse)
async def get_project_memory_context(
    namespace: NamespacePath,
    project: ProjectPath,
    recent_minutes: int = Query(10, ge=1, le=1440, description="How far back to look"),
    include_graph: bool = Query(True, description="Include graph relationships"),
    include_working_memory: bool = Query(True, description="Include working memory"),
    limit: int = Query(100, ge=1, le=1000, description="Max records per store"),
    store_name: StoreNameQuery = None,
) -> ProjectMemoryContextResponse:
    """Get aggregated context from a project's memory store.

    Returns recent data from all store components, useful for building
    context for LLM prompts or agent decision-making.

    Example:
    ```
    GET /v1/projects/{ns}/{proj}/memory/context?recent_minutes=10
    ```
    """
    try:
        result = MemoryDataService.get_context(
            namespace=namespace,
            project=project,
            recent_minutes=recent_minutes,
            include_graph=include_graph,
            include_working_memory=include_working_memory,
            limit=limit,
            store_name=store_name,
        )

        return ProjectMemoryContextResponse(**result)

    except MemoryStoreNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Failed to get project memory context: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to get memory context"
        ) from None


# =============================================================================
# Delete Endpoint
# =============================================================================


@router.delete("/{uuid}", response_model=ProjectMemoryDeleteResponse)
async def delete_from_project_memory(
    namespace: NamespacePath,
    project: ProjectPath,
    uuid: str = Path(..., description="UUID of the record to delete"),
    store_name: StoreNameQuery = None,
) -> ProjectMemoryDeleteResponse:
    """Delete a record by UUID (cascade delete via LinkageTable).

    This performs a cascade delete, removing the record from all
    linked store components.

    Args:
        uuid: The concept UUID to delete

    Returns:
        Deletion result with list of affected stores
    """
    # Validate UUID format (prevent path traversal)
    if not re.match(r"^[a-zA-Z0-9_\-]+$", uuid):
        raise HTTPException(status_code=400, detail=f"Invalid UUID format: {uuid}")

    try:
        result = MemoryDataService.delete(
            namespace=namespace,
            project=project,
            uuid=uuid,
            store_name=store_name,
        )

        if result is None:
            raise HTTPException(status_code=404, detail=f"Record not found: {uuid}")

        return ProjectMemoryDeleteResponse(**result)

    except MemoryStoreNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete from project memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to delete from memory store"
        ) from None


# =============================================================================
# Clear Table Endpoint
# =============================================================================


@router.post("/clear/{table}", response_model=ProjectMemoryClearResponse)
async def clear_project_memory_table(
    namespace: NamespacePath,
    project: ProjectPath,
    table: str = Path(
        ...,
        description="Table to clear: working_memory, timeseries, graph, linkage, or all",
    ),
    store_name: StoreNameQuery = None,
) -> ProjectMemoryClearResponse:
    """Clear specific table or all tables in a project's memory store.

    Valid table names:
    - working_memory: Short-term buffer with TTL
    - timeseries: Time-series telemetry data
    - graph: Entity nodes and relationships
    - linkage: Cross-store UUID mappings
    - all: Clear all tables

    Example:
    ```
    POST /v1/projects/{ns}/{proj}/memory/clear/working_memory
    POST /v1/projects/{ns}/{proj}/memory/clear/all
    ```
    """
    try:
        result = MemoryDataService.clear_table(
            namespace=namespace,
            project=project,
            table=table,
            store_name=store_name,
        )

        return ProjectMemoryClearResponse(**result)

    except MemoryStoreNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Failed to clear project memory table: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to clear memory table"
        ) from None


# =============================================================================
# Consolidate Endpoint
# =============================================================================


@router.post("/consolidate", response_model=ProjectMemoryConsolidateResponse)
async def consolidate_project_memory(
    namespace: NamespacePath,
    project: ProjectPath,
    request: ProjectMemoryConsolidateRequest | None = None,
    store_name: StoreNameQuery = None,
) -> ProjectMemoryConsolidateResponse:
    """Trigger memory consolidation cycle for a project.

    The consolidator performs:
    1. Reads raw data from Working Memory
    2. Synthesizes facts using LLM or rule-based extraction
    3. Creates graph nodes from extracted facts
    4. Prunes processed raw data

    Example request:
    ```json
    {
        "use_llm": false
    }
    ```
    """
    try:
        use_llm = request.use_llm if request else False

        result = MemoryDataService.consolidate(
            namespace=namespace,
            project=project,
            use_llm=use_llm,
            store_name=store_name,
        )

        return ProjectMemoryConsolidateResponse(**result)

    except MemoryStoreNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Failed to consolidate project memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to consolidate memory store"
        ) from None


# =============================================================================
# Prune Endpoint
# =============================================================================


@router.post("/prune", response_model=ProjectMemoryPruneResponse)
async def prune_project_memory(
    namespace: NamespacePath,
    project: ProjectPath,
    store_name: StoreNameQuery = None,
) -> ProjectMemoryPruneResponse:
    """Prune expired records from a project's working memory.

    Removes records that have exceeded their TTL.
    """
    try:
        result = MemoryDataService.prune(
            namespace=namespace,
            project=project,
            store_name=store_name,
        )

        return ProjectMemoryPruneResponse(**result)

    except MemoryStoreNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Failed to prune project memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to prune memory store"
        ) from None


# =============================================================================
# Stats Endpoint
# =============================================================================


@router.get("/stats", response_model=ProjectMemoryStatsResponse)
async def get_project_memory_stats(
    namespace: NamespacePath,
    project: ProjectPath,
    store_name: StoreNameQuery = None,
) -> ProjectMemoryStatsResponse:
    """Get storage statistics from a project's memory store.

    Returns statistics including:
    - Working Memory: record count, breakdown by type
    - Graph Store: node and edge counts
    - Time-Series Store: record count, time range
    - Linkage Table: total cross-store links
    - Store path and total size

    Example response:
    ```json
    {
        "success": true,
        "working_memory": {"total_records": 150},
        "graph": {"node_count": 45, "edge_count": 78},
        "timeseries": {"record_count": 10000},
        "linkage": {"total_links": 250},
        "store_path": "/path/to/lf_data/memory/store_name",
        "total_size_bytes": 1048576
    }
    ```
    """
    try:
        result = MemoryDataService.get_stats(
            namespace=namespace,
            project=project,
            store_name=store_name,
        )

        return ProjectMemoryStatsResponse(**result)

    except MemoryStoreNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Failed to get project memory stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to get memory store statistics"
        ) from None
