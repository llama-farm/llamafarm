"""RAG router for query endpoints."""

from pathlib import Path
from fastapi import APIRouter, HTTPException, Query

from core.logging import FastAPIStructLogger
from services.project_service import ProjectService

from .rag_health import RAGHealthResponse, handle_rag_health
from .rag_query import RAGQueryRequest, QueryResponse, handle_rag_query

logger = FastAPIStructLogger()

router = APIRouter(
    prefix="/projects/{namespace}/{project}/rag",
    tags=["rag"],
)


@router.post(
    "/query",
    operation_id="rag_query",
    tags=["mcp"],
    summary="Query the RAG system for semantic search",
    responses={200: {"model": QueryResponse}},
)
async def query_rag(namespace: str, project: str, request: RAGQueryRequest):
    """Query the RAG system for semantic search."""
    logger.bind(namespace=namespace, project=project)

    project_dir = ProjectService.get_project_dir(namespace, project)

    if not Path(project_dir).exists():
        raise HTTPException(
            status_code=404, detail=f"Project {namespace}/{project} not found"
        )

    # Get project configuration
    project_config = ProjectService.load_config(namespace, project)

    if not project_config.rag:
        raise HTTPException(
            status_code=400, detail="RAG not configured for this project"
        )

    # Use the handler function from rag_query.py
    return await handle_rag_query(request, project_config, str(project_dir))


@router.get("/health", response_model=RAGHealthResponse)
async def get_rag_health(
    namespace: str,
    project: str,
    database: str | None = Query(
        None, description="Specific database to check health for"
    ),
):
    """Get health status of the RAG system and database."""
    logger.bind(namespace=namespace, project=project, database=database)

    # Get project configuration
    project_obj = ProjectService.get_project(namespace, project)
    project_dir = ProjectService.get_project_dir(namespace, project)

    if not project_obj.config.rag:
        raise HTTPException(
            status_code=400, detail="RAG not configured for this project"
        )

    # Use the handler function from rag_health.py
    return await handle_rag_health(project_obj.config, str(project_dir), database)
