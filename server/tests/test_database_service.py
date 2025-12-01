"""
Tests for DatabaseService.

This module contains comprehensive tests for the DatabaseService class,
including unit tests for all public methods and edge cases.
"""

from unittest.mock import Mock, patch

import pytest
from config.datamodel import (
    Database,
    EmbeddingStrategy,
    LlamaFarmConfig,
    PromptMessage,
    PromptSet,
    RetrievalStrategy,
    Runtime,
    Type,
    Type1,
    Type2,
    Version,
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
                type=Type.ChromaStore,
                config={"collection_name": "documents"},
                embedding_strategies=[
                    EmbeddingStrategy(
                        name="default_embeddings",
                        type=Type1.OllamaEmbedder,
                        config={"model": "nomic-embed-text"},
                    )
                ],
                retrieval_strategies=[
                    RetrievalStrategy(
                        name="basic_search",
                        type=Type2.BasicSimilarityStrategy,
                        config={"top_k": 10},
                        default=True,
                    )
                ],
                default_embedding_strategy="default_embeddings",
                default_retrieval_strategy="basic_search",
            ),
            Database(
                name="secondary_db",
                type=Type.ChromaStore,
                config={"collection_name": "secondary"},
                embedding_strategies=[
                    EmbeddingStrategy(
                        name="secondary_embeddings",
                        type=Type1.OllamaEmbedder,
                        config={"model": "nomic-embed-text"},
                    )
                ],
                retrieval_strategies=[
                    RetrievalStrategy(
                        name="secondary_search",
                        type=Type2.BasicSimilarityStrategy,
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
        assert result.type == Type.ChromaStore

    @patch.object(ProjectService, "load_config")
    def test_get_database_raises_not_found(self, mock_load_config):
        """Test that get_database raises DatabaseNotFoundError for unknown database."""
        mock_load_config.return_value = self.mock_project_config

        with pytest.raises(DatabaseNotFoundError):
            DatabaseService.get_database("test_ns", "test_proj", "nonexistent_db")

    # =========================================================================
    # create_database tests
    # =========================================================================

    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_create_database_success(self, mock_load_config, mock_save_config):
        """Test that create_database adds new database to config."""
        mock_load_config.return_value = self.mock_project_config

        new_db = Database(
            name="new_db",
            type=Type.ChromaStore,
            config={"collection_name": "new_collection"},
        )

        result = DatabaseService.create_database("test_ns", "test_proj", new_db)

        assert result.name == "new_db"
        mock_save_config.assert_called_once()

    @patch.object(ProjectService, "load_config")
    def test_create_database_raises_on_duplicate(self, mock_load_config):
        """Test that create_database raises ValueError for duplicate name."""
        mock_load_config.return_value = self.mock_project_config

        duplicate_db = Database(
            name="main_db",  # Already exists
            type=Type.ChromaStore,
        )

        with pytest.raises(ValueError, match="already exists"):
            DatabaseService.create_database("test_ns", "test_proj", duplicate_db)

    @patch.object(ProjectService, "load_config")
    def test_create_database_raises_when_no_rag(self, mock_load_config):
        """Test that create_database raises ValueError when RAG not configured."""
        self.mock_project_config.rag = None
        mock_load_config.return_value = self.mock_project_config

        new_db = Database(name="new_db", type=Type.ChromaStore)

        with pytest.raises(ValueError, match="RAG not configured"):
            DatabaseService.create_database("test_ns", "test_proj", new_db)

    # =========================================================================
    # update_database tests
    # =========================================================================

    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_update_database_config(self, mock_load_config, mock_save_config):
        """Test that update_database updates config field."""
        mock_load_config.return_value = self.mock_project_config

        result = DatabaseService.update_database(
            "test_ns",
            "test_proj",
            "main_db",
            config={"collection_name": "updated_collection"},
        )

        assert result.config == {"collection_name": "updated_collection"}
        mock_save_config.assert_called_once()

    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_update_database_strategies(self, mock_load_config, mock_save_config):
        """Test that update_database updates strategies."""
        mock_load_config.return_value = self.mock_project_config

        new_strategies = [
            RetrievalStrategy(
                name="new_strategy",
                type=Type2.CrossEncoderRerankedStrategy,
                config={"model_name": "reranker"},
            )
        ]

        result = DatabaseService.update_database(
            "test_ns",
            "test_proj",
            "main_db",
            retrieval_strategies=new_strategies,
        )

        assert len(result.retrieval_strategies) == 1
        assert result.retrieval_strategies[0].name == "new_strategy"

    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_update_database_default_strategy(self, mock_load_config, mock_save_config):
        """Test that update_database validates default strategy references."""
        mock_load_config.return_value = self.mock_project_config

        result = DatabaseService.update_database(
            "test_ns",
            "test_proj",
            "main_db",
            default_retrieval_strategy="basic_search",
        )

        assert result.default_retrieval_strategy == "basic_search"

    @patch.object(ProjectService, "load_config")
    def test_update_database_raises_on_invalid_default(self, mock_load_config):
        """Test that update_database raises ValueError for invalid default strategy."""
        mock_load_config.return_value = self.mock_project_config

        with pytest.raises(ValueError, match="not found"):
            DatabaseService.update_database(
                "test_ns",
                "test_proj",
                "main_db",
                default_retrieval_strategy="nonexistent_strategy",
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

        result = DatabaseService.get_dependent_datasets("test_ns", "test_proj", "main_db")

        assert len(result) == 2
        assert "dataset1" in result
        assert "dataset2" in result
        assert "dataset3" not in result

    @patch.object(ProjectService, "load_config")
    def test_get_dependent_datasets_returns_empty_when_none(self, mock_load_config):
        """Test that get_dependent_datasets returns empty list when no dependencies."""
        self.mock_project_config.datasets = []
        mock_load_config.return_value = self.mock_project_config

        result = DatabaseService.get_dependent_datasets("test_ns", "test_proj", "main_db")

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
            DatabaseService, "_delete_vector_store_collection", return_value=(True, None)
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
            DatabaseService, "_delete_vector_store_collection", return_value=(True, None)
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
            type=Type.ChromaStore,
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
            type=Type.ChromaStore,
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
            type=Type.ChromaStore,
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
            type=Type.ChromaStore,
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
