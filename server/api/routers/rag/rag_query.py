"""RAG Query endpoint for semantic search."""

from typing import Optional, List, Dict, Any
import time
import structlog
from fastapi import HTTPException
from pydantic import BaseModel, Field

import sys
from pathlib import Path

# Add parent directories to path for imports
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import LlamaFarmConfig
from rag.core.factories import (
    VectorStoreFactory,
    EmbedderFactory,
    RetrievalStrategyFactory,
)

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
    """Handle RAG query request."""
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

    # Find the database configuration
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

    # Determine retrieval strategy
    retrieval_strategy_name = request.retrieval_strategy
    if not retrieval_strategy_name:
        # Use default retrieval strategy from database config
        retrieval_strategy_name = database_config.default_retrieval_strategy

        # Use first strategy as fallback
        if not retrieval_strategy_name and database_config.retrieval_strategies:
            retrieval_strategy_name = database_config.retrieval_strategies[0].name

    if not retrieval_strategy_name:
        raise HTTPException(
            status_code=400,
            detail="No retrieval strategy specified and no default available",
        )

    # Find the retrieval strategy configuration
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

    # Get embedder for the query
    embedder_name = database_config.default_embedding_strategy
    embedder_config = None
    for emb_strategy in database_config.embedding_strategies:
        if emb_strategy.name == embedder_name:
            embedder_config = emb_strategy
            break

    if not embedder_config:
        raise HTTPException(
            status_code=500, detail=f"Embedding strategy '{embedder_name}' not found"
        )

    try:
        # Initialize embedder - handle enum type
        embedder_type = embedder_config.type
        if hasattr(embedder_type, "value"):
            embedder_type = embedder_type.value
        embedder = EmbedderFactory.create(
            component_type=embedder_type, config=embedder_config.config
        )

        # Generate query embedding
        logger.info(f"Generating embedding for query: {request.query[:100]}...")
        query_embedding = embedder.embed_text(request.query)

        # Initialize store - handle enum type and resolve paths
        store_type = database_config.type
        if hasattr(store_type, "value"):
            store_type = store_type.value

        # Copy config and resolve persist_directory if it's relative
        store_config = database_config.config.copy() if database_config.config else {}
        if "persist_directory" in store_config and not store_config[
            "persist_directory"
        ].startswith("/"):
            # Make persist_directory absolute based on project_dir
            from pathlib import Path

            store_config["persist_directory"] = str(
                Path(project_dir) / store_config["persist_directory"]
            )
            logger.info(
                f"Resolved persist_directory to: {store_config['persist_directory']}"
            )

        store = VectorStoreFactory.create(
            component_type=store_type, config=store_config
        )

        # Initialize retriever without store in config (it's passed as a parameter)
        retriever_config = (
            retrieval_config.config.copy() if retrieval_config.config else {}
        )
        retriever_type = retrieval_config.type
        if hasattr(retriever_type, "value"):
            retriever_type = retriever_type.value
        retriever = RetrievalStrategyFactory.create(
            component_type=retriever_type, config=retriever_config
        )

        # Perform search - pass vector_store as parameter
        logger.info(
            f"Searching with strategy '{retrieval_strategy_name}' in database '{database_name}'"
        )
        search_results = retriever.retrieve(
            query_embedding=query_embedding,
            vector_store=store,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            metadata_filters=request.metadata_filters,
            distance_metric=request.distance_metric,
        )

        # Format results - search_results is a RetrievalResult object
        results = []
        for doc, score in zip(search_results.documents, search_results.scores):
            results.append(
                QueryResult(
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata or {},
                    chunk_id=doc.metadata.get("chunk_id") if doc.metadata else None,
                    document_id=doc.metadata.get("document_id")
                    if doc.metadata
                    else None,
                )
            )

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time_ms=processing_time,
            retrieval_strategy_used=retrieval_strategy_name,
            database_used=database_name,
        )

    except Exception as e:
        logger.error(f"Error performing RAG query: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to perform RAG query: {str(e)}"
        )
