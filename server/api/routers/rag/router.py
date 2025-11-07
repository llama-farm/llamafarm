"""RAG router for query endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.logging import FastAPIStructLogger
from services.project_service import ProjectService

from .rag_health import RAGHealthResponse, handle_rag_health
from .rag_query import QueryResponse, RAGQueryRequest, handle_rag_query

logger = FastAPIStructLogger()

router = APIRouter(
    prefix="/projects/{namespace}/{project}/rag",
    tags=["rag"],
)


class EmbeddingStrategyInfo(BaseModel):
    """Information about an embedding strategy."""
    name: str
    type: str
    priority: int
    is_default: bool


class RetrievalStrategyInfo(BaseModel):
    name: str
    type: str
    is_default: bool


class DatabaseInfo(BaseModel):
    name: str
    type: str
    is_default: bool
    embedding_strategies: list[EmbeddingStrategyInfo]
    retrieval_strategies: list[RetrievalStrategyInfo]


class DatabasesResponse(BaseModel):
    databases: list[DatabaseInfo]
    default_database: str | None


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


@router.get("/databases", response_model=DatabasesResponse)
async def get_rag_databases(namespace: str, project: str):
    """Get list of RAG databases with their embedding and retrieval strategies."""
    logger.bind(namespace=namespace, project=project)

    # Get project configuration
    project_obj = ProjectService.get_project(namespace, project)

    if not project_obj.config.rag:
        raise HTTPException(
            status_code=400, detail="RAG not configured for this project"
        )

    rag_config = project_obj.config.rag

    # Build database list with embedding and retrieval strategies
    databases = []
    for db in rag_config.databases or []:
        # BUILD EMBEDDING STRATEGIES
        embedding_strategies = []
        default_embedding_name = None
        found_default_embedding = False

        # Determine default embedding strategy
        if hasattr(db, "default_embedding_strategy") and db.default_embedding_strategy:
            default_embedding_name = str(db.default_embedding_strategy)

        # Build embedding strategies list with exactly one default
        for emb_strategy in db.embedding_strategies or []:
            is_default_emb = False

            # Mark as default based on priority order
            if not found_default_embedding:
                if default_embedding_name and str(emb_strategy.name) == default_embedding_name:
                    is_default_emb = True
                    found_default_embedding = True
                elif not default_embedding_name and not embedding_strategies:
                    # First strategy is default if no explicit default found
                    is_default_emb = True
                    found_default_embedding = True

            # Extract strategy type
            strategy_type = str(emb_strategy.type)
            if hasattr(emb_strategy.type, "value"):
                strategy_type = emb_strategy.type.value

            embedding_strategies.append(
                EmbeddingStrategyInfo(
                    name=str(emb_strategy.name),
                    type=strategy_type,
                    priority=getattr(emb_strategy, "priority", 0),
                    is_default=is_default_emb,
                )
            )

        # BUILD RETRIEVAL STRATEGIES
        retrieval_strategies = []
        default_retrieval_name = None
        found_default_retrieval = False

        # Determine default retrieval strategy (priority: default_retrieval_strategy > strategy.default > first)
        if hasattr(db, "default_retrieval_strategy") and db.default_retrieval_strategy:
            default_retrieval_name = str(db.default_retrieval_strategy)

        # First pass: check if any strategy is explicitly marked as default
        if not default_retrieval_name:
            for strategy in db.retrieval_strategies or []:
                if hasattr(strategy, "default") and strategy.default:
                    default_retrieval_name = str(strategy.name)
                    break

        # Second pass: build strategy list with exactly one default
        for strategy in db.retrieval_strategies or []:
            is_default_ret = False

            # Mark as default based on priority order, ensuring only one default
            if not found_default_retrieval:
                if default_retrieval_name and str(strategy.name) == default_retrieval_name:
                    is_default_ret = True
                    found_default_retrieval = True
                elif not default_retrieval_name and not retrieval_strategies:
                    # First strategy is default if no explicit default found
                    is_default_ret = True
                    found_default_retrieval = True

            retrieval_strategies.append(
                RetrievalStrategyInfo(
                    name=str(strategy.name),
                    type=strategy.type.value
                    if hasattr(strategy.type, "value")
                    else str(strategy.type),
                    is_default=is_default_ret,
                )
            )

        # Check if this database is the default
        is_default_db = False
        if rag_config.default_database and str(db.name) == str(
            rag_config.default_database
        ):
            is_default_db = True
        elif not rag_config.default_database and not databases:
            # First database is default if no explicit default
            is_default_db = True

        databases.append(
            DatabaseInfo(
                name=str(db.name),
                type=db.type.value if hasattr(db.type, "value") else str(db.type),
                is_default=is_default_db,
                embedding_strategies=embedding_strategies,
                retrieval_strategies=retrieval_strategies,
            )
        )

    return DatabasesResponse(
        databases=databases,
        default_database=str(rag_config.default_database)
        if rag_config.default_database
        else None,
    )
