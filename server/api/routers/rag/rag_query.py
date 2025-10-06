"""RAG Query endpoint for semantic search."""

import time
from typing import Any, Dict, List, Optional

import structlog
from config.datamodel import LlamaFarmConfig
from fastapi import HTTPException
from pydantic import BaseModel, Field

from services.rag_service import search_with_rag

logger = structlog.get_logger()


class QueryRequest(BaseModel):
    """RAG query request model."""

    query: str
    database: Optional[str] = None
    data_processing_strategy: Optional[str] = None
    retrieval_strategy: Optional[str] = None
    top_k: int = 5
    score_threshold: Optional[float] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    distance_metric: Optional[str] = None
    hybrid_alpha: Optional[float] = None
    rerank_model: Optional[str] = None
    query_expansion: bool = False
    max_tokens: Optional[int] = None


class QueryResult(BaseModel):
    """Single search result."""

    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None


class QueryResponse(BaseModel):
    """RAG query response model."""

    query: str
    results: List[QueryResult]
    total_results: int
    processing_time_ms: Optional[float] = None
    retrieval_strategy_used: str
    database_used: str


async def handle_rag_query(
    request: QueryRequest, project_config: LlamaFarmConfig, project_dir: str
) -> QueryResponse:
    """Handle RAG query request using Celery service."""
    start_time = time.time()

    # Determine which database to use
    database_name = request.database
    if not database_name and project_config.rag and project_config.rag.databases:
        # Use first database as default
        database_name = project_config.rag.databases[0].name
        logger.info(f"Using default database: {database_name}")

    if not database_name:
        raise HTTPException(
            status_code=400, detail="No database specified and no default available"
        )

    # Validate database exists
    database_exists = False
    if project_config.rag:
        for db in project_config.rag.databases:
            if db.name == database_name:
                database_exists = True
                break

    if not database_exists:
        raise HTTPException(
            status_code=404, detail=f"Database '{database_name}' not found"
        )

    try:
        # Use Celery service to perform search
        logger.info(
            f"Performing RAG search via Celery service: "
            f"query='{request.query[:100]}...', database='{database_name}'"
        )

        search_results = search_with_rag(
            project_dir=project_dir,
            database=database_name,
            query=request.query,
            top_k=request.top_k,
            retrieval_strategy=request.retrieval_strategy,
        )

        # Format results from Celery service response
        results = []
        for result in search_results:
            results.append(
                QueryResult(
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    metadata=result.get("metadata", {}),
                    chunk_id=result.get("id"),
                    document_id=result.get("metadata", {}).get("document_id"),
                )
            )

        processing_time = (time.time() - start_time) * 1000
        logger.info(
            f"Query completed in {processing_time:.2f}ms with {len(results)} results"
        )

        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time_ms=processing_time,
            retrieval_strategy_used=request.retrieval_strategy or "default",
            database_used=database_name,
        )

    except Exception as e:
        logger.error(f"Error during RAG query: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
