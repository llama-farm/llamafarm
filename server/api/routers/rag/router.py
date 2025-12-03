"""RAG router for query endpoints."""

from pathlib import Path
from typing import Any

from config.datamodel import Database, EmbeddingStrategy, RetrievalStrategy
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api.errors import DatabaseNotFoundError
from core.logging import FastAPIStructLogger
from services.database_service import DatabaseService
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


def _build_embedding_strategies(db, db_name: str) -> list[EmbeddingStrategyInfo]:
    """
    Build embedding strategies list with exactly one marked as default.

    Priority for default selection:
    1. Explicitly configured default_embedding_strategy
    2. First strategy in list (if strategies exist)
    3. Empty list (if no strategies configured)
    """
    embedding_strategies = []

    # Early return for empty strategies
    if not db.embedding_strategies:
        return embedding_strategies

    default_embedding_name = (
        str(db.default_embedding_strategy)
        if hasattr(db, "default_embedding_strategy") and db.default_embedding_strategy
        else None
    )

    found_default = False

    for emb_strategy in db.embedding_strategies:
        is_default = False

        # Mark as default based on priority order
        if not found_default:
            if (
                default_embedding_name
                and str(emb_strategy.name) == default_embedding_name
            ):
                is_default = True
                found_default = True
            elif not default_embedding_name and not embedding_strategies:
                # First strategy becomes default if none explicitly configured
                is_default = True
                found_default = True
                logger.info(
                    f"No default embedding strategy configured for database '{db_name}'. "
                    f"Using first strategy '{emb_strategy.name}' as default."
                )

        # Extract strategy type safely
        strategy_type = str(emb_strategy.type)
        if hasattr(emb_strategy.type, "value"):
            strategy_type = emb_strategy.type.value

        embedding_strategies.append(
            EmbeddingStrategyInfo(
                name=str(emb_strategy.name),
                type=strategy_type,
                priority=getattr(emb_strategy, "priority", 0),
                is_default=is_default,
            )
        )

    # Warn if configured default was not found and mark first strategy as default
    if default_embedding_name and not found_default:
        logger.warning(
            f"Configured default embedding strategy '{default_embedding_name}' "
            f"not found in database '{db_name}'. Using first strategy as default."
        )
        # Actually mark the first strategy as default
        if embedding_strategies:
            embedding_strategies[0].is_default = True

    return embedding_strategies


def _build_retrieval_strategies(db, db_name: str) -> list[RetrievalStrategyInfo]:
    """
    Build retrieval strategies list with exactly one marked as default.

    Priority for default selection:
    1. Explicitly configured default_retrieval_strategy
    2. First strategy with default=True attribute
    3. First strategy in list (if strategies exist)
    4. Empty list (if no strategies configured)
    """
    retrieval_strategies = []

    # Early return for empty strategies
    if not db.retrieval_strategies:
        return retrieval_strategies

    default_retrieval_name = (
        str(db.default_retrieval_strategy)
        if hasattr(db, "default_retrieval_strategy") and db.default_retrieval_strategy
        else None
    )

    # Check for multiple strategies marked as default (misconfiguration)
    if not default_retrieval_name:
        default_marked_strategies = [
            str(s.name)
            for s in db.retrieval_strategies
            if hasattr(s, "default") and s.default
        ]

        if len(default_marked_strategies) > 1:
            logger.warning(
                f"Multiple retrieval strategies marked as default in database '{db_name}': "
                f"{default_marked_strategies}. Using first one: '{default_marked_strategies[0]}'"
            )
            default_retrieval_name = default_marked_strategies[0]
        elif len(default_marked_strategies) == 1:
            default_retrieval_name = default_marked_strategies[0]

    found_default = False

    for strategy in db.retrieval_strategies:
        is_default = False

        # Mark as default based on priority order
        if not found_default:
            if default_retrieval_name and str(strategy.name) == default_retrieval_name:
                is_default = True
                found_default = True
            elif not default_retrieval_name and not retrieval_strategies:
                # First strategy becomes default if none explicitly configured
                is_default = True
                found_default = True
                logger.info(
                    f"No default retrieval strategy configured for database '{db_name}'. "
                    f"Using first strategy '{strategy.name}' as default."
                )

        # Extract strategy type safely
        strategy_type = str(strategy.type)
        if hasattr(strategy.type, "value"):
            strategy_type = strategy.type.value

        retrieval_strategies.append(
            RetrievalStrategyInfo(
                name=str(strategy.name),
                type=strategy_type,
                is_default=is_default,
            )
        )

    # Warn if configured default was not found and mark first strategy as default
    if default_retrieval_name and not found_default:
        logger.warning(
            f"Configured default retrieval strategy '{default_retrieval_name}' "
            f"not found in database '{db_name}'. Using first strategy as default."
        )
        # Actually mark the first strategy as default
        if retrieval_strategies:
            retrieval_strategies[0].is_default = True

    return retrieval_strategies


def _is_default_database(db_name: str, rag_config, databases: list) -> bool:
    """Determine if this database should be marked as default."""
    if rag_config.default_database and str(db_name) == str(rag_config.default_database):
        return True
    # First database is default if no explicit default configured
    return not rag_config.default_database and not databases


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

    # Build database list with strategies
    databases = []
    for db in rag_config.databases or []:
        db_name = str(db.name)

        # Build strategies using helper functions
        embedding_strategies = _build_embedding_strategies(db, db_name)
        retrieval_strategies = _build_retrieval_strategies(db, db_name)

        # Determine if this is the default database
        is_default_db = _is_default_database(db_name, rag_config, databases)

        # Extract database type safely
        db_type = db.type.value if hasattr(db.type, "value") else str(db.type)

        databases.append(
            DatabaseInfo(
                name=db_name,
                type=db_type,
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


# ============================================================================
# Database CRUD Endpoints
# ============================================================================


class CreateDatabaseRequest(BaseModel):
    """Request model for creating a new database."""

    name: str = Field(
        ..., description="Unique database identifier", pattern=r"^[a-z][a-z0-9_]*$"
    )
    type: str = Field(..., description="Database type (ChromaStore, QdrantStore)")
    config: dict[str, Any] | None = Field(
        None, description="Database-specific configuration"
    )
    embedding_strategies: list[dict[str, Any]] | None = Field(
        None, description="Embedding strategies for this database"
    )
    retrieval_strategies: list[dict[str, Any]] | None = Field(
        None, description="Retrieval strategies for this database"
    )
    default_embedding_strategy: str | None = Field(
        None, description="Name of default embedding strategy"
    )
    default_retrieval_strategy: str | None = Field(
        None, description="Name of default retrieval strategy"
    )


class UpdateDatabaseRequest(BaseModel):
    """Request model for updating a database (partial update)."""

    config: dict[str, Any] | None = Field(
        None, description="Database-specific configuration"
    )
    embedding_strategies: list[dict[str, Any]] | None = Field(
        None, description="Embedding strategies for this database"
    )
    retrieval_strategies: list[dict[str, Any]] | None = Field(
        None, description="Retrieval strategies for this database"
    )
    default_embedding_strategy: str | None = Field(
        None, description="Name of default embedding strategy"
    )
    default_retrieval_strategy: str | None = Field(
        None, description="Name of default retrieval strategy"
    )


class DatabaseResponse(BaseModel):
    """Response model for a single database."""

    database: DatabaseInfo


class DatabaseDetailResponse(BaseModel):
    """Detailed response for a single database including raw config."""

    name: str
    type: str
    config: dict[str, Any] | None
    embedding_strategies: list[dict[str, Any]] | None
    retrieval_strategies: list[dict[str, Any]] | None
    default_embedding_strategy: str | None
    default_retrieval_strategy: str | None
    dependent_datasets: list[str]


def _database_to_info(db: Database, is_default: bool = False) -> DatabaseInfo:
    """Convert a Database model to DatabaseInfo response."""
    db_name = str(db.name)
    embedding_strategies = _build_embedding_strategies(db, db_name)
    retrieval_strategies = _build_retrieval_strategies(db, db_name)
    db_type = db.type.value if hasattr(db.type, "value") else str(db.type)

    return DatabaseInfo(
        name=db_name,
        type=db_type,
        is_default=is_default,
        embedding_strategies=embedding_strategies,
        retrieval_strategies=retrieval_strategies,
    )


def _database_to_detail(
    db: Database, dependent_datasets: list[str]
) -> DatabaseDetailResponse:
    """Convert a Database model to detailed response."""
    return DatabaseDetailResponse(
        name=db.name,
        type=db.type.value if hasattr(db.type, "value") else str(db.type),
        config=db.config,
        embedding_strategies=[s.model_dump() for s in (db.embedding_strategies or [])],
        retrieval_strategies=[s.model_dump() for s in (db.retrieval_strategies or [])],
        default_embedding_strategy=db.default_embedding_strategy,
        default_retrieval_strategy=db.default_retrieval_strategy,
        dependent_datasets=dependent_datasets,
    )


@router.post(
    "/databases",
    response_model=DatabaseResponse,
    status_code=201,
    operation_id="database_create",
    summary="Create a new RAG database",
)
async def create_database(
    namespace: str,
    project: str,
    request: CreateDatabaseRequest,
):
    """
    Create a new RAG database in the project configuration.

    The database will be added to the project's llamafarm.yaml config file.
    You must provide at least the name and type. Embedding and retrieval
    strategies can be added during creation or later via PATCH.
    """
    logger.bind(namespace=namespace, project=project, database=request.name)

    # Convert request to Database model
    try:
        # Build embedding strategies if provided
        embedding_strategies = None
        if request.embedding_strategies:
            embedding_strategies = [
                EmbeddingStrategy(**s) for s in request.embedding_strategies
            ]

        # Build retrieval strategies if provided
        retrieval_strategies = None
        if request.retrieval_strategies:
            retrieval_strategies = [
                RetrievalStrategy(**s) for s in request.retrieval_strategies
            ]

        database = Database(
            name=request.name,
            type=request.type,
            config=request.config,
            embedding_strategies=embedding_strategies,
            retrieval_strategies=retrieval_strategies,
            default_embedding_strategy=request.default_embedding_strategy,
            default_retrieval_strategy=request.default_retrieval_strategy,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid database configuration: {e}"
        ) from e

    try:
        created_db = DatabaseService.create_database(namespace, project, database)
    except ValueError as e:
        error_msg = str(e)
        # Return 409 Conflict for duplicate database names
        if "already exists" in error_msg:
            raise HTTPException(status_code=409, detail=error_msg) from e
        raise HTTPException(status_code=400, detail=error_msg) from e

    return DatabaseResponse(database=_database_to_info(created_db))


@router.get(
    "/databases/{database_name}",
    response_model=DatabaseDetailResponse,
    operation_id="database_get",
    summary="Get a single RAG database by name",
)
async def get_database(
    namespace: str,
    project: str,
    database_name: str,
):
    """
    Get detailed information about a specific RAG database.

    Returns the full configuration including all strategies and
    lists any datasets that depend on this database.
    """
    logger.bind(namespace=namespace, project=project, database=database_name)

    try:
        db = DatabaseService.get_database(namespace, project, database_name)
    except DatabaseNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Database '{database_name}' not found"
        ) from None

    dependent_datasets = DatabaseService.get_dependent_datasets(
        namespace, project, database_name
    )

    return _database_to_detail(db, dependent_datasets)


@router.patch(
    "/databases/{database_name}",
    response_model=DatabaseDetailResponse,
    operation_id="database_update",
    summary="Update a RAG database",
)
async def update_database(
    namespace: str,
    project: str,
    database_name: str,
    request: UpdateDatabaseRequest,
):
    """
    Update a RAG database's mutable fields.

    Only these fields can be updated:
    - config: Database-specific configuration
    - embedding_strategies: List of embedding strategies
    - retrieval_strategies: List of retrieval strategies
    - default_embedding_strategy: Name of default embedding strategy
    - default_retrieval_strategy: Name of default retrieval strategy

    The database name and type cannot be changed.
    """
    logger.bind(namespace=namespace, project=project, database=database_name)

    # Build typed strategies if provided
    embedding_strategies = None
    if request.embedding_strategies is not None:
        try:
            embedding_strategies = [
                EmbeddingStrategy(**s) for s in request.embedding_strategies
            ]
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid embedding strategy: {e}"
            ) from e

    retrieval_strategies = None
    if request.retrieval_strategies is not None:
        try:
            retrieval_strategies = [
                RetrievalStrategy(**s) for s in request.retrieval_strategies
            ]
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid retrieval strategy: {e}"
            ) from e

    try:
        updated_db = DatabaseService.update_database(
            namespace=namespace,
            project=project,
            name=database_name,
            config=request.config,
            embedding_strategies=embedding_strategies,
            retrieval_strategies=retrieval_strategies,
            default_embedding_strategy=request.default_embedding_strategy,
            default_retrieval_strategy=request.default_retrieval_strategy,
        )
    except DatabaseNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Database '{database_name}' not found"
        ) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    dependent_datasets = DatabaseService.get_dependent_datasets(
        namespace, project, database_name
    )

    return _database_to_detail(updated_db, dependent_datasets)


class DeleteDatabaseResponse(BaseModel):
    """Response model for database deletion."""

    message: str
    database: DatabaseDetailResponse
    collection_deleted: bool


@router.delete(
    "/databases/{database_name}",
    response_model=DeleteDatabaseResponse,
    operation_id="database_delete",
    summary="Delete a RAG database",
    responses={
        200: {"model": DeleteDatabaseResponse},
        404: {"description": "Database not found"},
        409: {"description": "Database has dependent datasets"},
    },
)
async def delete_database(
    namespace: str,
    project: str,
    database_name: str,
    delete_collection: bool = Query(
        True,
        description="Whether to delete the underlying vector store collection. "
        "Set to false to only remove from config.",
    ),
):
    """
    Delete a RAG database from the project.

    This will:
    1. Check for dependent datasets - fails if any exist
    2. Delete the vector store collection (ChromaDB/Qdrant) if delete_collection=true
    3. Remove the database from the project config

    **Important**: You must first delete or reassign any datasets that use this database.
    The endpoint will return a 409 Conflict error listing the dependent datasets if any exist.
    """
    logger.bind(namespace=namespace, project=project, database=database_name)

    # Get the database details before deletion for the response
    try:
        db = DatabaseService.get_database(namespace, project, database_name)
    except DatabaseNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Database '{database_name}' not found"
        ) from None

    dependent_datasets = DatabaseService.get_dependent_datasets(
        namespace, project, database_name
    )

    # Build the response detail before deletion
    db_detail = _database_to_detail(db, dependent_datasets)

    # Attempt deletion
    try:
        DatabaseService.delete_database(
            namespace=namespace,
            project=project,
            name=database_name,
            delete_collection=delete_collection,
        )
    except DatabaseNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Database '{database_name}' not found"
        ) from None
    except ValueError as e:
        error_msg = str(e)
        # Check if this is a dependent datasets error (409 Conflict)
        if "dataset(s) depend on it" in error_msg:
            raise HTTPException(status_code=409, detail=error_msg) from e
        # Other validation errors (400 Bad Request)
        raise HTTPException(status_code=400, detail=error_msg) from e

    return DeleteDatabaseResponse(
        message=f"Database '{database_name}' deleted successfully",
        database=db_detail,
        collection_deleted=delete_collection,
    )
