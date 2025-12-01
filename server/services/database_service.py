"""Service for managing RAG databases within projects."""

from pathlib import Path

from config.datamodel import Database, EmbeddingStrategy, RetrievalStrategy

from api.errors import DatabaseNotFoundError
from core.logging import FastAPIStructLogger
from services.project_service import ProjectService

logger = FastAPIStructLogger()


class DatabaseService:
    """Service for managing RAG databases within projects."""

    @classmethod
    def list_databases(cls, namespace: str, project: str) -> list[Database]:
        """List all databases for a given project."""
        project_config = ProjectService.load_config(namespace, project)
        if not project_config.rag:
            return []
        return project_config.rag.databases or []

    @classmethod
    def get_database(cls, namespace: str, project: str, name: str) -> Database:
        """
        Get a single database by name.

        Raises:
            DatabaseNotFoundError: If database with given name is not found
        """
        databases = cls.list_databases(namespace, project)
        for db in databases:
            if db.name == name:
                return db
        raise DatabaseNotFoundError(name)

    @classmethod
    def create_database(
        cls,
        namespace: str,
        project: str,
        database: Database,
    ) -> Database:
        """
        Create a new database in the project.

        Raises:
            ValueError: If database with same name already exists
        """
        project_config = ProjectService.load_config(namespace, project)

        # Ensure RAG config exists
        if not project_config.rag:
            raise ValueError("RAG not configured for this project")

        existing_databases = project_config.rag.databases or []

        # Check if database already exists
        for db in existing_databases:
            if db.name == database.name:
                raise ValueError(f"Database '{database.name}' already exists")

        # Add the new database to the project config
        existing_databases.append(database)
        project_config.rag.databases = existing_databases
        ProjectService.save_config(namespace, project, project_config)

        logger.info(
            "Created database",
            namespace=namespace,
            project=project,
            database=database.name,
            type=database.type.value if hasattr(database.type, "value") else str(database.type),
        )

        return database

    @classmethod
    def update_database(
        cls,
        namespace: str,
        project: str,
        name: str,
        config: dict | None = None,
        embedding_strategies: list[EmbeddingStrategy] | None = None,
        retrieval_strategies: list[RetrievalStrategy] | None = None,
        default_embedding_strategy: str | None = None,
        default_retrieval_strategy: str | None = None,
    ) -> Database:
        """
        Update a database's mutable fields.

        Only these fields can be updated:
        - config: Database-specific configuration
        - embedding_strategies: List of embedding strategies
        - retrieval_strategies: List of retrieval strategies
        - default_embedding_strategy: Name of default embedding strategy
        - default_retrieval_strategy: Name of default retrieval strategy

        Name and type are immutable.

        Raises:
            DatabaseNotFoundError: If database with given name is not found
            ValueError: If validation fails (e.g., default strategy doesn't exist)
        """
        project_config = ProjectService.load_config(namespace, project)

        if not project_config.rag:
            raise ValueError("RAG not configured for this project")

        existing_databases = project_config.rag.databases or []

        # Find the database to update
        db_index = None
        for i, db in enumerate(existing_databases):
            if db.name == name:
                db_index = i
                break

        if db_index is None:
            raise DatabaseNotFoundError(name)

        db = existing_databases[db_index]

        # Update mutable fields
        if config is not None:
            db.config = config

        if embedding_strategies is not None:
            db.embedding_strategies = embedding_strategies

        if retrieval_strategies is not None:
            db.retrieval_strategies = retrieval_strategies

        if default_embedding_strategy is not None:
            # Validate that the strategy exists
            strategy_names = [s.name for s in (db.embedding_strategies or [])]
            if default_embedding_strategy not in strategy_names:
                raise ValueError(
                    f"Embedding strategy '{default_embedding_strategy}' not found. "
                    f"Available: {strategy_names}"
                )
            db.default_embedding_strategy = default_embedding_strategy

        if default_retrieval_strategy is not None:
            # Validate that the strategy exists
            strategy_names = [s.name for s in (db.retrieval_strategies or [])]
            if default_retrieval_strategy not in strategy_names:
                raise ValueError(
                    f"Retrieval strategy '{default_retrieval_strategy}' not found. "
                    f"Available: {strategy_names}"
                )
            db.default_retrieval_strategy = default_retrieval_strategy

        # Save updated config
        existing_databases[db_index] = db
        project_config.rag.databases = existing_databases
        ProjectService.save_config(namespace, project, project_config)

        logger.info(
            "Updated database",
            namespace=namespace,
            project=project,
            database=name,
        )

        return db

    @classmethod
    def get_dependent_datasets(
        cls, namespace: str, project: str, database_name: str
    ) -> list[str]:
        """
        Get list of dataset names that depend on a database.

        Useful for checking before deletion.
        """
        project_config = ProjectService.load_config(namespace, project)
        datasets = project_config.datasets or []
        return [ds.name for ds in datasets if ds.database == database_name]

    @classmethod
    def _delete_vector_store_collection(
        cls,
        database: Database,
        project_dir: str,
    ) -> tuple[bool, str | None]:
        """
        Delete the vector store collection for a database.

        Returns:
            Tuple of (success, error_message)
        """
        db_type = database.type.value if hasattr(database.type, "value") else str(database.type)
        collection_name = (database.config or {}).get("collection_name", "documents")

        if db_type == "ChromaStore":
            return cls._delete_chroma_collection(database, project_dir, collection_name)
        elif db_type == "QdrantStore":
            return cls._delete_qdrant_collection(database, project_dir, collection_name)
        else:
            # Unknown store type - just log warning and continue
            logger.warning(
                "Unknown store type, skipping collection deletion",
                store_type=db_type,
                database=database.name,
            )
            return True, None

    @classmethod
    def _delete_chroma_collection(
        cls,
        database: Database,
        project_dir: str,
        collection_name: str,
    ) -> tuple[bool, str | None]:
        """Delete a ChromaDB collection."""
        try:
            import chromadb

            config = database.config or {}
            host = config.get("host")
            port = config.get("port")

            if host and port:
                # HTTP client mode
                client = chromadb.HttpClient(host=host, port=port)
            else:
                # Persistent client mode
                persist_dir = config.get(
                    "persist_directory",
                    str(Path(project_dir) / "lf_data" / "stores" / database.name),
                )
                client = chromadb.PersistentClient(path=persist_dir)

            # Delete the collection
            try:
                client.delete_collection(name=collection_name)
                logger.info(
                    "Deleted ChromaDB collection",
                    collection=collection_name,
                    database=database.name,
                )
            except ValueError as e:
                # Collection doesn't exist - that's fine
                if "does not exist" in str(e).lower():
                    logger.info(
                        "Collection does not exist, nothing to delete",
                        collection=collection_name,
                        database=database.name,
                    )
                else:
                    raise

            return True, None

        except ImportError:
            logger.warning("chromadb not installed, skipping collection deletion")
            return True, None
        except Exception as e:
            error_msg = f"Failed to delete ChromaDB collection: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    @classmethod
    def _delete_qdrant_collection(
        cls,
        database: Database,
        project_dir: str,
        collection_name: str,
    ) -> tuple[bool, str | None]:
        """Delete a Qdrant collection."""
        try:
            from qdrant_client import QdrantClient

            config = database.config or {}
            host = config.get("host", "localhost")
            port = config.get("port", 6333)

            client = QdrantClient(host=host, port=port)
            client.delete_collection(collection_name=collection_name)

            logger.info(
                "Deleted Qdrant collection",
                collection=collection_name,
                database=database.name,
            )
            return True, None

        except ImportError:
            logger.warning("qdrant-client not installed, skipping collection deletion")
            return True, None
        except Exception as e:
            # Collection might not exist
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                logger.info(
                    "Collection does not exist, nothing to delete",
                    collection=collection_name,
                    database=database.name,
                )
                return True, None
            error_msg = f"Failed to delete Qdrant collection: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    @classmethod
    def delete_database(
        cls,
        namespace: str,
        project: str,
        name: str,
        delete_collection: bool = True,
    ) -> Database:
        """
        Delete a database from the project.

        This will:
        1. Check for dependent datasets (fails if any exist)
        2. Delete the vector store collection (if delete_collection=True)
        3. Remove from config

        Args:
            namespace: Project namespace
            project: Project name
            name: Database name to delete
            delete_collection: Whether to delete the underlying vector store collection

        Raises:
            DatabaseNotFoundError: If database doesn't exist
            ValueError: If datasets depend on this database
        """
        project_config = ProjectService.load_config(namespace, project)

        if not project_config.rag:
            raise ValueError("RAG not configured for this project")

        existing_databases = project_config.rag.databases or []

        # Find the database
        db_to_delete = None
        db_index = None
        for i, db in enumerate(existing_databases):
            if db.name == name:
                db_to_delete = db
                db_index = i
                break

        if db_to_delete is None:
            raise DatabaseNotFoundError(name)

        # Check for dependent datasets
        dependent_datasets = cls.get_dependent_datasets(namespace, project, name)
        if dependent_datasets:
            raise ValueError(
                f"Cannot delete database '{name}': {len(dependent_datasets)} dataset(s) depend on it. "
                f"Delete or reassign these datasets first: {dependent_datasets}"
            )

        # Delete the vector store collection
        if delete_collection:
            project_dir = ProjectService.get_project_dir(namespace, project)
            success, error = cls._delete_vector_store_collection(db_to_delete, project_dir)
            if not success:
                raise ValueError(f"Failed to delete vector store collection: {error}")

        # Remove from config
        existing_databases.pop(db_index)
        project_config.rag.databases = existing_databases

        # If this was the default database, clear the default
        if project_config.rag.default_database == name:
            project_config.rag.default_database = None

        ProjectService.save_config(namespace, project, project_config)

        logger.info(
            "Deleted database",
            namespace=namespace,
            project=project,
            database=name,
            collection_deleted=delete_collection,
        )

        return db_to_delete
