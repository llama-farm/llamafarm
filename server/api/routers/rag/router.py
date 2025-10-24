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


class RetrievalStrategyInfo(BaseModel):
    name: str
    type: str
    is_default: bool


class DatabaseInfo(BaseModel):
    name: str
    type: str
    is_default: bool
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
    """Get list of RAG databases and their retrieval strategies."""
    logger.bind(namespace=namespace, project=project)

    # Get project configuration
    project_obj = ProjectService.get_project(namespace, project)

    if not project_obj.config.rag:
        raise HTTPException(
            status_code=400, detail="RAG not configured for this project"
        )

    rag_config = project_obj.config.rag

    # Build database list with retrieval strategies
    databases = []
    for db in rag_config.databases or []:
        # Get retrieval strategies for this database
        strategies = []
        default_strategy_name = None
        found_default = False

        # Determine default strategy (priority: default_retrieval_strategy > strategy.default > first)
        if hasattr(db, "default_retrieval_strategy") and db.default_retrieval_strategy:
            default_strategy_name = str(db.default_retrieval_strategy)

        # First pass: check if any strategy is explicitly marked as default
        if not default_strategy_name:
            for strategy in db.retrieval_strategies or []:
                if hasattr(strategy, "default") and strategy.default:
                    default_strategy_name = str(strategy.name)
                    break

        # Second pass: build strategy list with exactly one default
        for strategy in db.retrieval_strategies or []:
            is_default = False

            # Mark as default based on priority order, ensuring only one default
            if not found_default:
                if default_strategy_name and strategy.name == default_strategy_name:
                    is_default = True
                    found_default = True
                elif not default_strategy_name and not strategies:
                    # First strategy is default if no explicit default found
                    is_default = True
                    found_default = True

            strategies.append(
                RetrievalStrategyInfo(
                    name=str(strategy.name),
                    type=strategy.type.value
                    if hasattr(strategy.type, "value")
                    else str(strategy.type),
                    is_default=is_default,
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
                retrieval_strategies=strategies,
            )
        )

    return DatabasesResponse(
        databases=databases,
        default_database=str(rag_config.default_database)
        if rag_config.default_database
        else None,
    )
