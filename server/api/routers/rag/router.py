"""RAG router for query endpoints."""

from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any

from core.logging import FastAPIStructLogger
from services.project_service import ProjectService
from .rag_query import QueryRequest, QueryResponse, handle_rag_query

logger = FastAPIStructLogger()

router = APIRouter(
    prefix="/projects/{namespace}/{project}/rag",
    tags=["rag"],
)


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    namespace: str,
    project: str,
    request: QueryRequest
):
    """Query the RAG system for semantic search."""
    logger.bind(namespace=namespace, project=project)
    
    # Get project configuration
    project_obj = ProjectService.get_project(namespace, project)
    project_dir = ProjectService.get_project_dir(namespace, project)
    
    if not project_obj.config.rag:
        raise HTTPException(status_code=400, detail="RAG not configured for this project")
    
    # Use the handler function from rag_query.py
    return await handle_rag_query(request, project_obj.config, str(project_dir))