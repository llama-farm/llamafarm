"""
FastAPI router for event logs endpoints.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Path as FastAPIPath, Query

from services.event_log_service import EventLogService

from .models import EventDetail, ListEventsResponse

router = APIRouter(
    prefix="/projects/{namespace}/{project_id}/event_logs",
    tags=["event_logs", "observability"],
)


@router.get(
    "",
    response_model=ListEventsResponse,
    summary="List event logs",
    description="List event logs for a project with optional filtering by type and time range",
)
async def list_events(
    namespace: str = FastAPIPath(..., description="Project namespace"),
    project_id: str = FastAPIPath(..., description="Project ID"),
    type: str | None = Query(
        None,
        description="Filter by event type (e.g., 'inference', 'rag_processing')",
        alias="type",
    ),
    start_time: datetime | None = Query(
        None,
        description="Filter events after this timestamp (ISO 8601 format)",
    ),
    end_time: datetime | None = Query(
        None,
        description="Filter events before this timestamp (ISO 8601 format)",
    ),
    limit: int = Query(
        10,
        ge=1,
        le=100,
        description="Maximum number of events to return (1-100)",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Number of events to skip for pagination",
    ),
):
    """
    List event logs for a project.

    Returns a paginated list of event summaries. Events are returned in reverse
    chronological order (most recent first).

    **Filtering:**
    - `type`: Filter by event type (e.g., "inference", "rag_processing")
    - `start_time`: Only return events after this timestamp
    - `end_time`: Only return events before this timestamp

    **Pagination:**
    - `limit`: Number of events per page (default: 10, max: 100)
    - `offset`: Number of events to skip

    **Example:**
    ```
    GET /v1/projects/default/my-project/event_logs?type=inference&limit=20
    ```
    """
    events, total = EventLogService.list_events(
        namespace=namespace,
        project=project_id,
        event_type=type,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )

    return ListEventsResponse(
        total=total,
        events=events,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{event_id}",
    response_model=EventDetail,
    summary="Get event details",
    description="Get full details of a specific event including all sub-events",
    responses={
        200: {"description": "Event details"},
        404: {"description": "Event not found"},
    },
)
async def get_event(
    namespace: str = FastAPIPath(..., description="Project namespace"),
    project_id: str = FastAPIPath(..., description="Project ID"),
    event_id: str = FastAPIPath(..., description="Event ID"),
):
    """
    Get full details of a specific event.

    Returns the complete event log including all sub-events with their
    timestamps, durations, and data.

    **Example:**
    ```
    GET /v1/projects/default/my-project/event_logs/evt_inference_20251029_221203_cd62dc
    ```

    **Response includes:**
    - Event metadata (ID, type, timestamps, status)
    - Config hash (for version tracking)
    - All sub-events with detailed timing and data
    - Error information (if event failed)
    """
    if not (event := EventLogService.get_event(
        namespace=namespace,
        project=project_id,
        event_id=event_id,
    )):
        raise HTTPException(
            status_code=404,
            detail=f"Event '{event_id}' not found in project '{namespace}/{project_id}'",
        )

    return event
