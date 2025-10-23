"""RAG Query endpoint for semantic search."""

import time
from typing import Any

import structlog
from config.datamodel import LlamaFarmConfig
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from services.rag_service import search_with_rag

logger = structlog.get_logger()


class RAGQueryRequest(BaseModel):
    """RAG query request model.

    Crafted so AI agents and humans can construct precise retrieval requests. See
    examples for common usage patterns across semantic, keyword, and hybrid search.
    """

    # Model-level examples to guide agents/tools
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Simple semantic search",
                    "value": {
                        "query": "Which letters mention clinical trial data?",
                        "database": "main_db",
                        "retrieval_strategy": "semantic",
                        "top_k": 5,
                        "score_threshold": 0.65,
                    },
                },
                {
                    "summary": "Hybrid search with metadata filter",
                    "value": {
                        "query": (
                            "Summarize FDA warning letters from 2024 "
                            "mentioning manufacturing issues"
                        ),
                        "database": "main_db",
                        "retrieval_strategy": "hybrid",
                        "hybrid_alpha": 0.5,
                        "top_k": 8,
                        "metadata_filters": {
                            "document_type": "letter",
                            "year": 2024,
                        },
                    },
                },
                {
                    "summary": "BM25 with date range + reranking",
                    "value": {
                        "query": "Find sections discussing adverse events",
                        "database": "pharma_db",
                        "retrieval_strategy": "bm25",
                        "top_k": 10,
                        "metadata_filters": {
                            "date": {
                                "gte": "2024-01-01",
                                "lte": "2024-12-31",
                            }
                        },
                        "rerank_model": "bge-reranker-v2-m3",
                    },
                },
            ]
        }
    )

    query: str = Field(
        ...,
        min_length=1,
        description=(
            "Natural-language question or statement used to retrieve relevant chunks from the RAG database."
        ),
        examples=[
            "Which letters mention clinical trial data?",
            "Key risks described in the Q2 report",
        ],
    )

    database: str | None = Field(
        None,
        description=(
            "Target database name (from `rag.databases[].name` in `llamafarm.yaml`). "
            "If omitted, the server uses the first configured database."
        ),
        examples=["main_db", "pharma_db"],
    )

    data_processing_strategy: str | None = Field(
        None,
        description=(
            "Optional name from `rag.data_processing_strategies[].name` indicating how the data was parsed/chunked."
        ),
        examples=["pdf_ingest", "default"],
    )

    retrieval_strategy: str | None = Field(
        None,
        description=(
            "Retrieval algorithm to use. Common values: `semantic` (vector), `bm25` (keyword), "
            "`hybrid` (blend), `mmr` (diversified). If omitted, server default applies."
        ),
        examples=["semantic", "bm25", "hybrid", "mmr"],
    )

    top_k: int = Field(
        5,
        ge=1,
        le=100,
        description=(
            "Maximum number of results to return. Higher values increase recall at the cost of speed."
        ),
        examples=[3, 5, 10],
    )

    score_threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum similarity score (0–1) required to include a result. Higher -> stricter matches."
        ),
        examples=[0.7, 0.5],
    )

    metadata_filters: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional exact/range filters on metadata. Examples: {`document_type`: `letter`} or "
            "{`date`: {`gte`: `2024-01-01`, `lte`: `2024-12-31`}}."
        ),
        examples=[
            {"document_type": "letter"},
            {"year": 2024},
            {"date": {"gte": "2024-01-01", "lte": "2024-12-31"}},
        ],
    )

    distance_metric: str | None = Field(
        None,
        description=(
            "Vector distance metric for semantic search. Typical values: `cosine`, `euclidean`, `dot`."
        ),
        examples=["cosine", "euclidean"],
    )

    hybrid_alpha: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Blend factor for `hybrid` retrieval. 0 = keyword-only, 1 = semantic-only. Suggested: 0.5."
        ),
        examples=[0.5, 0.3, 0.7],
    )

    rerank_model: str | None = Field(
        None,
        description=(
            "Optional reranker (cross-encoder) to reorder top results, e.g., `bge-reranker-v2-m3`."
        ),
        examples=["bge-reranker-v2-m3"],
    )

    query_expansion: bool = Field(
        False,
        description=(
            "Enable query expansion (e.g., LLM rewrites/PRF) to improve recall on sparse queries."
        ),
        examples=[True, False],
    )

    max_tokens: int | None = Field(
        None,
        ge=1,
        le=8192,
        description=(
            "Optional token cap for downstream model responses (if used). Not all backends apply this."
        ),
        examples=[512, 1024],
    )


class QueryResult(BaseModel):
    """Single search result."""

    content: str
    score: float
    metadata: dict[str, Any]
    chunk_id: str | None = None
    document_id: str | None = None


class QueryResponse(BaseModel):
    """RAG query response model."""

    query: str
    results: list[QueryResult]
    total_results: int
    processing_time_ms: float | None = None
    retrieval_strategy_used: str
    database_used: str


async def handle_rag_query(
    request: RAGQueryRequest, project_config: LlamaFarmConfig, project_dir: str
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
            score_threshold=request.score_threshold,
            metadata_filters=request.metadata_filters,
            distance_metric=request.distance_metric,
            hybrid_alpha=request.hybrid_alpha,
            rerank_model=request.rerank_model,
            query_expansion=request.query_expansion,
            max_tokens=request.max_tokens,
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
        logger.error("Error during RAG query", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
