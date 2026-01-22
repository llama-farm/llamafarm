"""
Tests for DatabaseService.

This module contains comprehensive tests for the DatabaseService class,
including unit tests for all public methods and edge cases.
"""

from unittest.mock import Mock, patch

import pytest
from config.datamodel import (
    Database,
    DatabaseEmbeddingStrategy,
    DatabaseEmbeddingType,
    DatabaseRetrievalStrategy,
    DatabaseRetrievalType,
    DatabaseType,
)

from api.errors import DatabaseNotFoundError
from services.database_service import DatabaseService
from services.project_service import ProjectService


class TestDatabaseService:
    """Test cases for DatabaseService class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock project config with databases
        self.mock_rag_config = Mock()
        self.mock_rag_config.databases = [
            Database(
                name="main_db",
                type=DatabaseType.ChromaStore,
                config={"collection_name": "documents"},
                embedding_strategies=[
                    DatabaseEmbeddingStrategy(
                        name="default_embeddings",
                        type=DatabaseEmbeddingType.OllamaEmbedder,
                        config={"model": "nomic-embed-text"},
                    )
                ],
                retrieval_strategies=[
                    DatabaseRetrievalStrategy(
                        name="basic_search",
                        type=DatabaseRetrievalType.BasicSimilarityStrategy,
                        config={"top_k": 10},
                        default=True,
                    )
                ],
                default_embedding_strategy="default_embeddings",
                default_retrieval_strategy="basic_search",
            ),
            Database(
                name="secondary_db",
                type=DatabaseType.ChromaStore,
                config={"collection_name": "secondary"},
                embedding_strategies=[
                    DatabaseEmbeddingStrategy(
                        name="secondary_embeddings",
                        type=DatabaseEmbeddingType.OllamaEmbedder,
                        config={"model": "nomic-embed-text"},
                    )
                ],
                retrieval_strategies=[
                    DatabaseRetrievalStrategy(
                        name="secondary_search",
                        type=DatabaseRetrievalType.BasicSimilarityStrategy,
                        config={},
                        default=True,
                    )
                ],
            ),
        ]
        self.mock_rag_config.default_database = "main_db"

        self.mock_project_config = Mock()
        self.mock_project_config.rag = self.mock_rag_config
        self.mock_project_config.datasets = []
        self.mock_project_config.components = None

    # =========================================================================
    # list_databases tests
    # =========================================================================

    @patch.object(ProjectService, "load_config")
    def test_list_databases_returns_all_databases(self, mock_load_config):
        """Test that list_databases returns all configured databases."""
        mock_load_config.return_value = self.mock_project_config

        result = DatabaseService.list_databases("test_ns", "test_proj")

        assert len(result) == 2
        assert result[0].name == "main_db"
        assert result[1].name == "secondary_db"

    @patch.object(ProjectService, "load_config")
    def test_list_databases_returns_empty_when_no_rag(self, mock_load_config):
        """Test that list_databases returns empty list when RAG not configured."""
        self.mock_project_config.rag = None
        mock_load_config.return_value = self.mock_project_config

        result = DatabaseService.list_databases("test_ns", "test_proj")

        assert result == []

    @patch.object(ProjectService, "load_config")
    def test_list_databases_returns_empty_when_no_databases(self, mock_load_config):
        """Test that list_databases returns empty list when no databases configured."""
        self.mock_rag_config.databases = None
        mock_load_config.return_value = self.mock_project_config

        result = DatabaseService.list_databases("test_ns", "test_proj")

        assert result == []

    # =========================================================================
    # get_database tests
    # =========================================================================

    @patch.object(ProjectService, "load_config")
    def test_get_database_returns_correct_database(self, mock_load_config):
        """Test that get_database returns the correct database by name."""
        mock_load_config.return_value = self.mock_project_config

        result = DatabaseService.get_database("test_ns", "test_proj", "main_db")

        assert result.name == "main_db"
        assert result.type == DatabaseType.ChromaStore

    @patch.object(ProjectService, "load_config")
    def test_get_database_raises_not_found(self, mock_load_config):
        """Test that get_database raises DatabaseNotFoundError for unknown database."""
        mock_load_config.return_value = self.mock_project_config

        with pytest.raises(DatabaseNotFoundError):
            DatabaseService.get_database("test_ns", "test_proj", "nonexistent_db")

    # =========================================================================
    # create_database tests
    # =========================================================================

    @patch("services.database_service.ComponentResolver")
    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_create_database_success(
        self, mock_load_config, mock_save_config, mock_resolver_class
    ):
        """Test that create_database adds new database to config."""
        mock_load_config.return_value = self.mock_project_config

        new_db = Database(
            name="new_db",
            type=DatabaseType.ChromaStore,
            config={"collection_name": "new_collection"},
            embedding_strategies=[
                DatabaseEmbeddingStrategy(
                    name="test_embeddings",
                    type=DatabaseEmbeddingType.OllamaEmbedder,
                    config={"model": "nomic-embed-text"},
                )
            ],
            retrieval_strategies=[
                DatabaseRetrievalStrategy(
                    name="test_retrieval",
                    type=DatabaseRetrievalType.BasicSimilarityStrategy,
                    config={"top_k": 10},
                    default=True,
                )
            ],
        )

        # Mock the resolver to return a config with the new database resolved
        mock_resolved_config = Mock()
        mock_resolved_config.rag = Mock()
        mock_resolved_config.rag.databases = self.mock_rag_config.databases + [new_db]
        mock_resolver_class.return_value.resolve_config.return_value = mock_resolved_config

        result = DatabaseService.create_database("test_ns", "test_proj", new_db)

        assert result.name == "new_db"
        mock_save_config.assert_called_once()

    @patch("services.database_service.ComponentResolver")
    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_create_database_preserves_component_references(
        self, mock_load_config, mock_save_config, mock_resolver_class
    ):
        """Creating a new database should not flatten existing references."""
        existing_db = Database(
            name="existing_db",
            type=DatabaseType.ChromaStore,
            config={"collection_name": "existing"},
            embedding_strategy="fast_embed",
            retrieval_strategy="fast_retrieve",
        )

        mock_rag_config = Mock()
        mock_rag_config.databases = [existing_db]
        mock_project_config = Mock()
        mock_project_config.rag = mock_rag_config

        new_db = Database(
            name="new_db",
            type=DatabaseType.ChromaStore,
            config={"collection_name": "new_collection"},
            embedding_strategy="fast_embed",
            retrieval_strategy="fast_retrieve",
        )

        resolved_db = Database(
            name="new_db",
            type=DatabaseType.ChromaStore,
            config={"collection_name": "new_collection"},
            embedding_strategies=[
                DatabaseEmbeddingStrategy(
                    name="fast_embed",
                    type=DatabaseEmbeddingType.OllamaEmbedder,
                    config={"model": "nomic-embed-text"},
                )
            ],
            retrieval_strategies=[
                DatabaseRetrievalStrategy(
                    name="fast_retrieve",
                    type=DatabaseRetrievalType.BasicSimilarityStrategy,
                    config={"top_k": 10},
                    default=True,
                )
            ],
            default_embedding_strategy="fast_embed",
            default_retrieval_strategy="fast_retrieve",
        )

        temp_config = Mock()
        temp_config.rag = Mock()
        temp_config.rag.databases = [new_db]
        mock_project_config.model_copy.return_value = temp_config

        mock_resolved_config = Mock()
        mock_resolved_config.rag = Mock()
        mock_resolved_config.rag.databases = [resolved_db]
        mock_resolver_class.return_value.resolve_config.return_value = mock_resolved_config

        mock_load_config.return_value = mock_project_config

        result = DatabaseService.create_database("test_ns", "test_proj", new_db)

        assert result == resolved_db
        mock_save_config.assert_called_once()

        saved_config = mock_save_config.call_args.args[2]
        saved_existing = next(
            db for db in saved_config.rag.databases if db.name == "existing_db"
        )
        assert saved_existing.embedding_strategy == "fast_embed"
        assert saved_existing.embedding_strategies is None
        assert saved_existing.retrieval_strategy == "fast_retrieve"
        assert saved_existing.retrieval_strategies is None

        saved_new = next(db for db in saved_config.rag.databases if db.name == "new_db")
        assert saved_new.embedding_strategies
        assert saved_new.retrieval_strategies

    @patch.object(ProjectService, "load_config")
    def test_create_database_raises_on_duplicate(self, mock_load_config):
        """Test that create_database raises ValueError for duplicate name."""
        mock_load_config.return_value = self.mock_project_config

        duplicate_db = Database(
            name="main_db",  # Already exists
            type=DatabaseType.ChromaStore,
        )

        with pytest.raises(ValueError, match="already exists"):
            DatabaseService.create_database("test_ns", "test_proj", duplicate_db)

    @patch.object(ProjectService, "load_config")
    def test_create_database_raises_when_no_rag(self, mock_load_config):
        """Test that create_database raises ValueError when RAG not configured."""
        self.mock_project_config.rag = None
        mock_load_config.return_value = self.mock_project_config

        new_db = Database(name="new_db", type=DatabaseType.ChromaStore)

        with pytest.raises(ValueError, match="RAG not configured"):
            DatabaseService.create_database("test_ns", "test_proj", new_db)

    # =========================================================================
    # update_database tests
    # =========================================================================

    @patch("services.database_service.ComponentResolver")
    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_update_database_config(
        self, mock_load_config, mock_save_config, mock_resolver_class
    ):
        """Test that update_database updates config field."""
        mock_load_config.return_value = self.mock_project_config

        # Create updated database with new config
        updated_db = Database(
            name="main_db",
            type=DatabaseType.ChromaStore,
            config={"collection_name": "updated_collection"},
            embedding_strategies=self.mock_rag_config.databases[0].embedding_strategies,
            retrieval_strategies=self.mock_rag_config.databases[0].retrieval_strategies,
        )

        # Mock the resolver to return the updated database
        mock_resolved_config = Mock()
        mock_resolved_config.rag = Mock()
        mock_resolved_config.rag.databases = [updated_db, self.mock_rag_config.databases[1]]
        mock_resolver_class.return_value.resolve_config.return_value = mock_resolved_config

        result = DatabaseService.update_database(
            "test_ns",
            "test_proj",
            "main_db",
            config={"collection_name": "updated_collection"},
        )

        assert result.config == {"collection_name": "updated_collection"}
        mock_save_config.assert_called_once()

    @patch("services.database_service.ComponentResolver")
    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_update_database_strategies(
        self, mock_load_config, mock_save_config, mock_resolver_class
    ):
        """Test that update_database updates strategies."""
        mock_load_config.return_value = self.mock_project_config

        new_strategies = [
            DatabaseRetrievalStrategy(
                name="new_strategy",
                type=DatabaseRetrievalType.CrossEncoderRerankedStrategy,
                config={"model_name": "reranker"},
            )
        ]

        # Create updated database with new strategies
        updated_db = Database(
            name="main_db",
            type=DatabaseType.ChromaStore,
            config=self.mock_rag_config.databases[0].config,
            embedding_strategies=self.mock_rag_config.databases[0].embedding_strategies,
            retrieval_strategies=new_strategies,
        )

        # Mock the resolver to return the updated database
        mock_resolved_config = Mock()
        mock_resolved_config.rag = Mock()
        mock_resolved_config.rag.databases = [updated_db, self.mock_rag_config.databases[1]]
        mock_resolver_class.return_value.resolve_config.return_value = mock_resolved_config

        result = DatabaseService.update_database(
            "test_ns",
            "test_proj",
            "main_db",
            retrieval_strategies=new_strategies,
        )

        assert len(result.retrieval_strategies) == 1
        assert result.retrieval_strategies[0].name == "new_strategy"

    @patch("services.database_service.ComponentResolver")
    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_update_database_default_strategy(
        self, mock_load_config, mock_save_config, mock_resolver_class
    ):
        """Test that update_database validates default strategy references."""
        mock_load_config.return_value = self.mock_project_config

        # Create updated database with new default strategy
        updated_db = Database(
            name="main_db",
            type=DatabaseType.ChromaStore,
            config=self.mock_rag_config.databases[0].config,
            embedding_strategies=self.mock_rag_config.databases[0].embedding_strategies,
            retrieval_strategies=self.mock_rag_config.databases[0].retrieval_strategies,
            default_retrieval_strategy="basic_search",
        )

        # Mock the resolver to return the updated database
        mock_resolved_config = Mock()
        mock_resolved_config.rag = Mock()
        mock_resolved_config.rag.databases = [updated_db, self.mock_rag_config.databases[1]]
        mock_resolver_class.return_value.resolve_config.return_value = mock_resolved_config

        result = DatabaseService.update_database(
            "test_ns",
            "test_proj",
            "main_db",
            default_retrieval_strategy="basic_search",
        )

        assert result.default_retrieval_strategy == "basic_search"

    @patch("services.database_service.ComponentResolver")
    @patch.object(ProjectService, "load_config")
    def test_update_database_raises_on_invalid_default(
        self, mock_load_config, mock_resolver_class
    ):
        """Test that update_database raises ValueError for invalid default strategy."""
        mock_load_config.return_value = self.mock_project_config

        # Create a resolved database with an invalid default (strategy doesn't exist)
        db_with_invalid_default = Database(
            name="main_db",
            type=DatabaseType.ChromaStore,
            config=self.mock_rag_config.databases[0].config,
            embedding_strategies=self.mock_rag_config.databases[0].embedding_strategies,
            retrieval_strategies=self.mock_rag_config.databases[0].retrieval_strategies,
            default_retrieval_strategy="nonexistent_strategy",  # Invalid default
        )

        # Mock the resolver to return the database with invalid default
        mock_resolved_config = Mock()
        mock_resolved_config.rag = Mock()
        mock_resolved_config.rag.databases = [
            db_with_invalid_default,
            self.mock_rag_config.databases[1],
        ]
        mock_resolver_class.return_value.resolve_config.return_value = mock_resolved_config

        with pytest.raises(ValueError, match="not found"):
            DatabaseService.update_database(
                "test_ns",
                "test_proj",
                "main_db",
                default_retrieval_strategy="nonexistent_strategy",
            )

    @patch("services.database_service.ComponentResolver")
    @patch.object(ProjectService, "load_config")
    def test_update_database_raises_when_strategy_update_orphans_default(
        self, mock_load_config, mock_resolver_class
    ):
        """Test that updating strategies fails if it would orphan an existing default.

        This tests the scenario where:
        1. Database has default_retrieval_strategy="basic_search"
        2. User updates retrieval_strategies to remove "basic_search"
        3. Validation should fail because the existing default no longer exists
        """
        mock_load_config.return_value = self.mock_project_config

        # Create new strategies that don't include "basic_search"
        new_strategies = [
            DatabaseRetrievalStrategy(
                name="new_strategy",
                type=DatabaseRetrievalType.CrossEncoderRerankedStrategy,
                config={"model_name": "reranker"},
            )
        ]

        # The resolved database still has default_retrieval_strategy="basic_search"
        # but that strategy no longer exists in the list
        db_with_orphaned_default = Database(
            name="main_db",
            type=DatabaseType.ChromaStore,
            config=self.mock_rag_config.databases[0].config,
            embedding_strategies=self.mock_rag_config.databases[0].embedding_strategies,
            retrieval_strategies=new_strategies,  # "basic_search" removed
            default_retrieval_strategy="basic_search",  # Still points to removed strategy
        )

        mock_resolved_config = Mock()
        mock_resolved_config.rag = Mock()
        mock_resolved_config.rag.databases = [
            db_with_orphaned_default,
            self.mock_rag_config.databases[1],
        ]
        mock_resolver_class.return_value.resolve_config.return_value = mock_resolved_config

        with pytest.raises(ValueError, match="basic_search.*not found"):
            DatabaseService.update_database(
                "test_ns",
                "test_proj",
                "main_db",
                retrieval_strategies=new_strategies,  # User only updates strategies, not default
            )

    @patch.object(ProjectService, "load_config")
    def test_update_database_raises_not_found(self, mock_load_config):
        """Test that update_database raises DatabaseNotFoundError for unknown database."""
        mock_load_config.return_value = self.mock_project_config

        with pytest.raises(DatabaseNotFoundError):
            DatabaseService.update_database(
                "test_ns",
                "test_proj",
                "nonexistent_db",
                config={"new": "config"},
            )

    # =========================================================================
    # get_dependent_datasets tests
    # =========================================================================

    @patch.object(ProjectService, "load_config")
    def test_get_dependent_datasets_finds_dependencies(self, mock_load_config):
        """Test that get_dependent_datasets finds datasets using a database."""
        from config.datamodel import Dataset

        self.mock_project_config.datasets = [
            Dataset(
                name="dataset1",
                data_processing_strategy="universal",
                database="main_db",
            ),
            Dataset(
                name="dataset2",
                data_processing_strategy="universal",
                database="main_db",
            ),
            Dataset(
                name="dataset3",
                data_processing_strategy="universal",
                database="secondary_db",
            ),
        ]
        mock_load_config.return_value = self.mock_project_config

        result = DatabaseService.get_dependent_datasets(
            "test_ns", "test_proj", "main_db"
        )

        assert len(result) == 2
        assert "dataset1" in result
        assert "dataset2" in result
        assert "dataset3" not in result

    @patch.object(ProjectService, "load_config")
    def test_get_dependent_datasets_returns_empty_when_none(self, mock_load_config):
        """Test that get_dependent_datasets returns empty list when no dependencies."""
        self.mock_project_config.datasets = []
        mock_load_config.return_value = self.mock_project_config

        result = DatabaseService.get_dependent_datasets(
            "test_ns", "test_proj", "main_db"
        )

        assert result == []

    # =========================================================================
    # delete_database tests
    # =========================================================================

    @patch.object(ProjectService, "get_project_dir")
    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_delete_database_success(
        self, mock_load_config, mock_save_config, mock_get_project_dir
    ):
        """Test that delete_database removes database from config."""
        mock_load_config.return_value = self.mock_project_config
        mock_get_project_dir.return_value = "/tmp/project"

        # Mock the collection deletion to succeed
        with patch.object(
            DatabaseService,
            "_delete_vector_store_collection",
            return_value=(True, None),
        ):
            result = DatabaseService.delete_database(
                "test_ns", "test_proj", "secondary_db"
            )

        assert result.name == "secondary_db"
        mock_save_config.assert_called_once()

    @patch.object(ProjectService, "load_config")
    def test_delete_database_raises_when_has_dependencies(self, mock_load_config):
        """Test that delete_database raises ValueError when datasets depend on it."""
        from config.datamodel import Dataset

        self.mock_project_config.datasets = [
            Dataset(
                name="dependent_dataset",
                data_processing_strategy="universal",
                database="main_db",
            )
        ]
        mock_load_config.return_value = self.mock_project_config

        with pytest.raises(ValueError, match="dataset\\(s\\) depend on it"):
            DatabaseService.delete_database("test_ns", "test_proj", "main_db")

    @patch.object(ProjectService, "load_config")
    def test_delete_database_raises_not_found(self, mock_load_config):
        """Test that delete_database raises DatabaseNotFoundError for unknown database."""
        mock_load_config.return_value = self.mock_project_config

        with pytest.raises(DatabaseNotFoundError):
            DatabaseService.delete_database("test_ns", "test_proj", "nonexistent_db")

    @patch.object(ProjectService, "get_project_dir")
    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_delete_database_clears_default_when_deleted(
        self, mock_load_config, mock_save_config, mock_get_project_dir
    ):
        """Test that delete_database clears default_database when deleting the default."""
        mock_load_config.return_value = self.mock_project_config
        mock_get_project_dir.return_value = "/tmp/project"

        # Make main_db the default (it already is in setup)
        self.mock_rag_config.default_database = "secondary_db"

        with patch.object(
            DatabaseService,
            "_delete_vector_store_collection",
            return_value=(True, None),
        ):
            DatabaseService.delete_database("test_ns", "test_proj", "secondary_db")

        # Verify default_database was cleared
        assert self.mock_rag_config.default_database is None

    @patch.object(ProjectService, "get_project_dir")
    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_delete_database_skip_collection_deletion(
        self, mock_load_config, mock_save_config, mock_get_project_dir
    ):
        """Test that delete_database can skip collection deletion."""
        mock_load_config.return_value = self.mock_project_config
        mock_get_project_dir.return_value = "/tmp/project"

        with patch.object(
            DatabaseService, "_delete_vector_store_collection"
        ) as mock_delete:
            DatabaseService.delete_database(
                "test_ns", "test_proj", "secondary_db", delete_collection=False
            )

            # Should not have called collection deletion
            mock_delete.assert_not_called()


class TestDatabaseServiceCollectionDeletion:
    """Test cases for database collection deletion methods."""

    def test_delete_vector_store_collection_success(self):
        """Test vector store collection deletion success case using RAG abstraction."""
        db = Database(
            name="test_db",
            type=DatabaseType.ChromaStore,
            config={"collection_name": "test_collection"},
        )

        # Mock the VectorStoreFactory and vector store
        mock_vector_store = Mock()
        mock_vector_store.delete_collection.return_value = True

        mock_factory = Mock()
        mock_factory.list_available.return_value = ["ChromaStore", "QdrantStore"]
        mock_factory.create.return_value = mock_vector_store

        with patch(
            "services.database_service._import_rag_factory", return_value=mock_factory
        ):
            success, error = DatabaseService._delete_vector_store_collection(
                db, "/tmp/project"
            )

            assert success is True
            assert error is None
            mock_vector_store.delete_collection.assert_called_once()

    def test_delete_vector_store_collection_not_exists(self):
        """Test vector store collection deletion when collection doesn't exist."""
        db = Database(
            name="test_db",
            type=DatabaseType.ChromaStore,
            config={"collection_name": "nonexistent"},
        )

        # Mock the factory to raise ValueError indicating collection doesn't exist
        mock_factory = Mock()
        mock_factory.list_available.return_value = ["ChromaStore"]
        mock_factory.create.side_effect = ValueError("Collection does not exist")

        with patch(
            "services.database_service._import_rag_factory", return_value=mock_factory
        ):
            success, error = DatabaseService._delete_vector_store_collection(
                db, "/tmp/project"
            )

            # Should still succeed (collection already gone)
            assert success is True
            assert error is None

    def test_delete_vector_store_unknown_type(self):
        """Test that unknown store types are handled gracefully."""
        db = Mock()
        db.type.value = "UnknownStore"
        db.name = "test_db"
        db.config = {"collection_name": "test"}

        # Mock the factory to NOT include the unknown store type
        mock_factory = Mock()
        mock_factory.list_available.return_value = ["ChromaStore", "QdrantStore"]

        with patch(
            "services.database_service._import_rag_factory", return_value=mock_factory
        ):
            success, error = DatabaseService._delete_vector_store_collection(
                db, "/tmp/project"
            )

            # Should succeed with warning (no-op for unknown types)
            assert success is True
            assert error is None

    def test_delete_vector_store_collection_failure(self):
        """Test vector store collection deletion failure case."""
        db = Database(
            name="test_db",
            type=DatabaseType.ChromaStore,
            config={"collection_name": "test_collection"},
        )

        # Mock the vector store to return False (deletion failed)
        mock_vector_store = Mock()
        mock_vector_store.delete_collection.return_value = False

        mock_factory = Mock()
        mock_factory.list_available.return_value = ["ChromaStore"]
        mock_factory.create.return_value = mock_vector_store

        with patch(
            "services.database_service._import_rag_factory", return_value=mock_factory
        ):
            success, error = DatabaseService._delete_vector_store_collection(
                db, "/tmp/project"
            )

            assert success is False
            assert error is not None
            assert "returned False" in error

    def test_delete_vector_store_import_error(self):
        """Test that import errors are handled gracefully."""
        db = Database(
            name="test_db",
            type=DatabaseType.ChromaStore,
            config={"collection_name": "test_collection"},
        )

        # Mock the factory to raise ImportError
        with patch(
            "services.database_service._import_rag_factory",
            side_effect=ImportError("chromadb not installed"),
        ):
            success, error = DatabaseService._delete_vector_store_collection(
                db, "/tmp/project"
            )

            # Should succeed (skip deletion when dependencies not installed)
            assert success is True
            assert error is None
