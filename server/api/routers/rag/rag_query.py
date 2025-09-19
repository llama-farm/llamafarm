from typing import Optional, List, Dict, Any
import time
import structlog
from fastapi import HTTPException
from pydantic import BaseModel, Field

from core.celery.tasks.rag_tasks import rag_query_task

logger = structlog.get_logger()

class QueryRequest(BaseModel):
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
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]
    total_results: int
    processing_time_ms: Optional[float] = None
    retrieval_strategy_used: str
    database_used: str


async def handle_rag_query(
    request: QueryRequest, project_config: LlamaFarmConfig, project_dir: str, namespace: str, project: str
) -> QueryResponse:
    start_time = time.time()

    database_name = request.database
    if not database_name and project_config.rag and project_config.rag.databases:
        database_name = project_config.rag.databases[0].name
        logger.info(f"Using default database: {database_name}")

    if not database_name:
        raise HTTPException(
            status_code=400, detail="No database specified and no default available"
        )

    database_config = None
    if project_config.rag:
        for db in project_config.rag.databases:
            if db.name == database_name:
                database_config = db
                break

    if not database_config:
        raise HTTPException(
            status_code=404, detail=f"Database '{database_name}' not found"
        )

    retrieval_strategy_name = request.retrieval_strategy
    if not retrieval_strategy_name:
        retrieval_strategy_name = database_config.default_retrieval_strategy

        if not retrieval_strategy_name and database_config.retrieval_strategies:
            retrieval_strategy_name = database_config.retrieval_strategies[0].name

    if not retrieval_strategy_name:
        raise HTTPException(
            status_code=400,
            detail="No retrieval strategy specified and no default available",
        )

    retrieval_config = None
    for strategy in database_config.retrieval_strategies:
        if strategy.name == retrieval_strategy_name:
            retrieval_config = strategy
            break

    if not retrieval_config:
        raise HTTPException(
            status_code=404,
            detail=f"Retrieval strategy '{retrieval_strategy_name}' not found",
        )

    try:
        request_data = {
            'query': request.query,
            'database': database_name,
            'retrieval_strategy': retrieval_strategy_name,
            'top_k': request.top_k,
            'score_threshold': request.score_threshold,
            'metadata_filters': request.metadata_filters,
            'distance_metric': request.distance_metric,
            'hybrid_alpha': request.hybrid_alpha,
            'rerank_model': request.rerank_model,
            'query_expansion': request.query_expansion,
            'max_tokens': request.max_tokens,
        }

        logger.info(f"Submitting RAG query to Celery worker: {request.query[:100]}...")

        result = rag_query_task.delay(namespace, project, request_data)

        query_result = result.get(timeout=30)

        results = []
        for item in query_result.get('results', []):
            results.append(
                QueryResult(
                    content=item['content'],
                    score=item['score'],
                    metadata=item['metadata'],
                    chunk_id=item['metadata'].get('chunk_id'),
                    document_id=item['metadata'].get('document_id'),
                )
            )

        processing_time = (time.time() - start_time) * 1000

        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time_ms=processing_time,
            retrieval_strategy_used=retrieval_strategy_name,
            database_used=database_name,
        )

    except Exception as e:
        logger.error(f"Error submitting RAG query: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit RAG query: {str(e)}"
        )
