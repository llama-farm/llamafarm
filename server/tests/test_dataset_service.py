"""
Tests for DatasetService.

This module contains comprehensive tests for the DatasetService class,
including unit tests for all public methods and edge cases.
"""

from unittest.mock import AsyncMock, patch

import pytest
from config.datamodel import (
    Dataset,
    LlamaFarmConfig,
    Model,
    PromptMessage,
    PromptSet,
    Provider,
    Runtime,
    Version,
)
from fastapi import UploadFile

from api.errors import DatasetNotFoundError
from services.data_service import MetadataFileContent
from services.dataset_service import DatasetService
from services.project_service import ProjectService


class TestDatasetService:
    """Test cases for DatasetService class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock project config with datasets
        self.mock_project_config = LlamaFarmConfig(
            version=Version.v1,
            name="test_project",
            namespace="test_namespace",
            prompts=[
                PromptSet(
                    name="default",
                    messages=[
                        PromptMessage(
                            role="system", content="You are a helpful assistant."
                        )
                    ],
                )
            ],
            rag={
                "databases": [
                    {
                        "name": "custom_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "custom_embedding",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "custom_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "custom_strategy",
                        "description": "Custom strategy for testing behavior",
                        "parsers": [
                            {
                                "type": "CSVParser_LlamaIndex",
                                "config": {},
                                "file_extensions": [".csv"],
                            }
                        ],
                    }
                ],
            },
            datasets=[
                Dataset(
                    name="dataset1",
                    data_processing_strategy="auto",
                    database="custom_db",
                    files=["file1.pdf", "file2.pdf"],
                ),
                Dataset(
                    name="dataset2",
                    data_processing_strategy="custom_strategy",
                    database="custom_db",
                    files=["data.csv"],
                ),
            ],
            runtime=Runtime(
                models=[
                    Model(
                        name="default",
                        provider=Provider.openai,
                        model="llama3.1:8b",
                        api_key="ollama",
                        base_url="http://localhost:11434/v1",
                        model_api_parameters={
                            "temperature": 0.5,
                        },
                    )
                ]
            ),
        )

        # Mock project config without datasets (rag requires >=1 strategy)
        self.mock_empty_project_config = LlamaFarmConfig(
            version=Version.v1,
            name="empty_project",
            namespace="test_namespace",
            prompts=[
                PromptSet(
                    name="default",
                    messages=[
                        PromptMessage(
                            role="system", content="You are a helpful assistant."
                        )
                    ],
                )
            ],
            rag={
                "databases": [
                    {
                        "name": "default_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "default_embedding",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "default_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "default_processing",
                        "description": "Default data processing strategy",
                        "parsers": [
                            {
                                "type": "CSVParser_LlamaIndex",
                                "config": {},
                                "file_extensions": [".csv"],
                            }
                        ],
                    }
                ],
            },
            datasets=[],
            runtime=Runtime(
                models=[
                    Model(
                        name="default",
                        provider=Provider.openai,
                        model="llama3.1:8b",
                        api_key="ollama",
                        base_url="http://localhost:11434/v1",
                        model_api_parameters={
                            "temperature": 0.5,
                        },
                    )
                ]
            ),
        )

    @patch.object(ProjectService, "load_config")
    def test_list_datasets_success(self, mock_load_config):
        """Test listing datasets successfully."""
        mock_load_config.return_value = self.mock_project_config

        datasets = DatasetService.list_datasets("test_namespace", "test_project")

        assert len(datasets) == 2
        assert datasets[0].name == "dataset1"
        assert datasets[0].data_processing_strategy == "auto"
        assert datasets[0].database == "custom_db"
        assert datasets[1].name == "dataset2"
        assert datasets[1].data_processing_strategy == "custom_strategy"
        assert datasets[1].database == "custom_db"

        mock_load_config.assert_called_once_with("test_namespace", "test_project")

    @patch.object(ProjectService, "load_config")
    def test_list_datasets_empty(self, mock_load_config):
        """Test listing datasets when no datasets exist."""
        mock_load_config.return_value = self.mock_empty_project_config

        datasets = DatasetService.list_datasets("test_namespace", "test_project")

        assert len(datasets) == 0
        assert datasets == []

    @patch.object(ProjectService, "load_config")
    def test_list_datasets_no_datasets_key(self, mock_load_config):
        """Test listing datasets when datasets list is empty."""
        config_without_datasets = LlamaFarmConfig(
            version=Version.v1,
            name="test_project",
            namespace="test_namespace",
            prompts=[
                PromptSet(
                    name="default",
                    messages=[
                        PromptMessage(
                            role="system", content="You are a helpful assistant."
                        )
                    ],
                )
            ],
            rag={
                "databases": [
                    {
                        "name": "default_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "default_embedding",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "default_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "default_processing",
                        "description": "Default data processing strategy",
                        "parsers": [
                            {
                                "type": "CSVParser_LlamaIndex",
                                "config": {},
                                "file_extensions": [".csv"],
                            }
                        ],
                    }
                ],
            },
            datasets=[],
            runtime=Runtime(
                models=[
                    Model(
                        name="default",
                        provider=Provider.openai,
                        model="llama3.1:8b",
                        api_key="ollama",
                        base_url="http://localhost:11434/v1",
                        model_api_parameters={
                            "temperature": 0.5,
                        },
                    )
                ]
            ),
        )
        mock_load_config.return_value = config_without_datasets

        datasets = DatasetService.list_datasets("test_namespace", "test_project")

        assert len(datasets) == 0
        assert datasets == []

    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_create_dataset_success(self, mock_load_config, mock_save_config):
        """Test creating a dataset successfully."""
        mock_load_config.return_value = self.mock_project_config.model_copy()
        mock_save_config.return_value = None

        dataset = DatasetService.create_dataset(
            "test_namespace",
            "test_project",
            "new_dataset",
            "custom_strategy",
            "custom_db",
        )

        assert dataset.name == "new_dataset"
        assert dataset.data_processing_strategy == "custom_strategy"
        assert dataset.database == "custom_db"

        # Verify save_config was called with updated config
        mock_save_config.assert_called_once()
        call_args = mock_save_config.call_args[0]
        assert call_args[0] == "test_namespace"
        assert call_args[1] == "test_project"
        updated_config = call_args[2]
        assert len(updated_config.datasets) == 3
        assert updated_config.datasets[-1].name == "new_dataset"

    @patch.object(ProjectService, "load_config")
    def test_create_dataset_duplicate_name(self, mock_load_config):
        """Test creating a dataset with a name that already exists."""
        mock_load_config.return_value = self.mock_project_config

        with pytest.raises(ValueError, match="Dataset dataset1 already exists"):
            DatasetService.create_dataset(
                "test_namespace",
                "test_project",
                "dataset1",  # This name already exists
                "custom_strategy",
                "custom_db",
            )

    @patch.object(ProjectService, "load_config")
    def test_create_dataset_unsupported_data_processing_strategy(
        self, mock_load_config
    ):
        """Test creating a dataset with an unsupported RAG strategy."""
        mock_load_config.return_value = self.mock_project_config

        with pytest.raises(
            ValueError,
            match="RAG data processing strategy unsupported_strategy not supported",
        ):
            DatasetService.create_dataset(
                "test_namespace",
                "test_project",
                "new_dataset",
                "unsupported_strategy",
                "custom_db",
            )

    @patch.object(ProjectService, "load_config")
    def test_create_dataset_unsupported_database(self, mock_load_config):
        """Test creating a dataset with an unsupported database."""
        mock_load_config.return_value = self.mock_project_config

        with pytest.raises(
            ValueError, match="RAG database unsupported_database not supported"
        ):
            DatasetService.create_dataset(
                "test_namespace",
                "test_project",
                "new_dataset",
                "custom_strategy",
                "unsupported_database",
            )

    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_create_dataset_with_custom_strategy(
        self, mock_load_config, mock_save_config
    ):
        """Test creating a dataset with a custom RAG strategy."""
        mock_load_config.return_value = self.mock_project_config.model_copy()

        dataset = DatasetService.create_dataset(
            "test_namespace",
            "test_project",
            "new_dataset",
            "custom_strategy",
            "custom_db",
        )

        assert dataset.name == "new_dataset"
        assert dataset.data_processing_strategy == "custom_strategy"
        assert dataset.database == "custom_db"
        mock_save_config.assert_called_once()

    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    def test_delete_dataset_success(self, mock_load_config, mock_save_config):
        """Test deleting a dataset successfully."""
        mock_load_config.return_value = self.mock_project_config.model_copy()
        mock_save_config.return_value = None

        deleted_dataset = DatasetService.delete_dataset(
            "test_namespace", "test_project", "dataset1"
        )

        assert deleted_dataset.name == "dataset1"

        # Verify save_config was called with updated config
        mock_save_config.assert_called_once()
        call_args = mock_save_config.call_args[0]
        updated_config = call_args[2]
        assert len(updated_config.datasets) == 1
        assert updated_config.datasets[0].name == "dataset2"

    @patch.object(ProjectService, "load_config")
    def test_delete_dataset_not_found(self, mock_load_config):
        """Test deleting a dataset that doesn't exist."""
        mock_load_config.return_value = self.mock_project_config

        with pytest.raises(ValueError, match="Dataset nonexistent_dataset not found"):
            DatasetService.delete_dataset(
                "test_namespace", "test_project", "nonexistent_dataset"
            )

    @patch.object(ProjectService, "load_config")
    def test_delete_dataset_empty_list(self, mock_load_config):
        """Test deleting a dataset when no datasets exist."""
        mock_load_config.return_value = self.mock_empty_project_config

        with pytest.raises(ValueError, match="Dataset any_dataset not found"):
            DatasetService.delete_dataset(
                "test_namespace", "test_project", "any_dataset"
            )

    @patch.object(ProjectService, "load_config")
    def test_get_supported_data_processing_strategies_with_custom_strategies(
        self, mock_load_config
    ):
        """Test getting supported data processing strategies including custom ones."""
        mock_load_config.return_value = self.mock_project_config

        strategies = DatasetService.get_supported_data_processing_strategies(
            "test_namespace", "test_project"
        )

        # universal_rag is always included as the built-in default
        expected_strategies = ["universal_rag", "custom_strategy"]
        assert strategies == expected_strategies

    @patch.object(ProjectService, "load_config")
    def test_get_supported_data_processing_strategies_no_rag_config(
        self, mock_load_config
    ):
        """Test getting supported data processing strategies when RAG config is missing."""
        config_no_rag = LlamaFarmConfig(
            version=Version.v1,
            name="test_project",
            namespace="test_namespace",
            prompts=[
                PromptSet(
                    name="default",
                    messages=[
                        PromptMessage(
                            role="system", content="You are a helpful assistant."
                        )
                    ],
                )
            ],
            rag=None,
            datasets=[],
            runtime=Runtime(
                models=[
                    Model(
                        name="default",
                        provider=Provider.openai,
                        model="llama3.1:8b",
                        api_key="ollama",
                        model_api_parameters={
                            "temperature": 0.5,
                        },
                    )
                ]
            ),
        )
        mock_load_config.return_value = config_no_rag

        strategies = DatasetService.get_supported_data_processing_strategies(
            "test_namespace", "test_project"
        )

        # universal_rag is always available as the built-in default, even without RAG config
        expected_strategies = ["universal_rag"]
        assert strategies == expected_strategies

    @patch.object(ProjectService, "load_config")
    def test_create_dataset_no_existing_datasets(self, mock_load_config):
        """Test creating a dataset when no datasets exist yet."""
        config_no_datasets = LlamaFarmConfig(
            version=Version.v1,
            name="test_project",
            namespace="test_namespace",
            prompts=[
                PromptSet(
                    name="default",
                    messages=[
                        PromptMessage(
                            role="system", content="You are a helpful assistant."
                        )
                    ],
                )
            ],
            rag={
                "databases": [
                    {
                        "name": "default_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "default_embedding",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "default_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "default_processing",
                        "description": "Default data processing strategy",
                        "parsers": [
                            {
                                "type": "CSVParser_LlamaIndex",
                                "config": {},
                                "file_extensions": [".csv"],
                            }
                        ],
                    }
                ],
            },
            datasets=[],
            runtime=Runtime(
                models=[
                    Model(
                        name="default",
                        provider=Provider.openai,
                        model="llama3.1:8b",
                        api_key="ollama",
                        base_url="http://localhost:11434/v1",
                        model_api_parameters={
                            "temperature": 0.5,
                        },
                    )
                ]
            ),
        )
        mock_load_config.return_value = config_no_datasets

        with patch.object(ProjectService, "save_config") as mock_save_config:
            dataset = DatasetService.create_dataset(
                "test_namespace",
                "test_project",
                "first_dataset",
                "default_processing",
                "default_db",
            )

            assert dataset.name == "first_dataset"
            assert dataset.data_processing_strategy == "default_processing"
            assert dataset.database == "default_db"

            # Verify the config was updated correctly
            call_args = mock_save_config.call_args[0]
            updated_config = call_args[2]
            assert len(updated_config.datasets) == 1
            assert updated_config.datasets[0].name == "first_dataset"

    @patch("services.dataset_service.DataService.add_data_file")
    @patch("services.dataset_service.DataService.hash_data")
    @patch("services.dataset_service.DataService.get_data_file_metadata_by_hash")
    @patch.object(ProjectService, "save_config")
    @patch.object(ProjectService, "load_config")
    @pytest.mark.asyncio
    async def test_add_file_to_dataset_success(
        self,
        mock_load_config,
        mock_save_config,
        mock_get_metadata,
        mock_hash_data,
        mock_add_data_file,
    ):
        """Test successfully adding a file to a dataset."""
        mock_load_config.return_value = self.mock_project_config.model_copy()
        mock_hash_data.return_value = "new_file_hash"
        mock_get_metadata.side_effect = FileNotFoundError  # Simulate file not existing

        # Mock MetadataFileContent response from add_data_file
        mock_metadata = MetadataFileContent(
            original_file_name="new_file.pdf",
            resolved_file_name="new_file_123.pdf",
            size=100,
            mime_type="application/pdf",
            hash="new_file_hash",
            timestamp=1234567890.0,
        )
        mock_add_data_file.return_value = mock_metadata

        mock_file = AsyncMock(spec=UploadFile)
        mock_file.read.return_value = b"content"
        mock_file.seek = AsyncMock(return_value=None)

        success, result = await DatasetService.add_file_to_dataset(
            "test_namespace", "test_project", "dataset1", mock_file
        )

        assert success is True
        assert result == mock_metadata
        mock_file.seek.assert_awaited()
        mock_file.seek.assert_awaited_with(0)
        # With current implementation, save_config is not called in add_file_to_dataset
        # mock_save_config.assert_called_once()

        # Verify file was added to dataset
        # call_args = mock_save_config.call_args[0]
        # updated_config = call_args[2]
        # target_dataset = next(d for d in updated_config.datasets if d.name == "dataset1")
        # assert "new_file_hash" in target_dataset.files

    @patch.object(ProjectService, "load_config")
    @pytest.mark.asyncio
    async def test_add_file_to_dataset_not_found(self, mock_load_config):
        """Test adding a file to a non-existent dataset."""
        mock_load_config.return_value = self.mock_project_config
        mock_file = AsyncMock(spec=UploadFile)

        with pytest.raises(DatasetNotFoundError):
            await DatasetService.add_file_to_dataset(
                "test_namespace", "test_project", "nonexistent_dataset", mock_file
            )

    @patch("services.dataset_service.DataService.hash_data")
    @patch("services.dataset_service.DataService.get_data_file_metadata_by_hash")
    @patch.object(ProjectService, "load_config")
    @pytest.mark.asyncio
    async def test_add_file_to_dataset_duplicate(
        self, mock_load_config, mock_get_metadata, mock_hash_data
    ):
        """Test adding a duplicate file to a dataset."""
        mock_load_config.return_value = self.mock_project_config
        # Hash matches existing file
        mock_hash_data.return_value = "file1.pdf"  # file1.pdf is in dataset1

        existing_metadata = MetadataFileContent(
            original_file_name="existing.pdf",
            resolved_file_name="existing_123.pdf",
            size=100,
            mime_type="application/pdf",
            hash="file1.pdf",
            timestamp=1234567890.0,
        )
        mock_get_metadata.return_value = existing_metadata

        mock_file = AsyncMock(spec=UploadFile)
        mock_file.read.return_value = b"content"
        mock_file.seek = AsyncMock(return_value=None)

        success, result = await DatasetService.add_file_to_dataset(
            "test_namespace", "test_project", "dataset1", mock_file
        )

        assert success is False
        assert result == existing_metadata
        mock_file.seek.assert_awaited()
        mock_file.seek.assert_awaited_with(0)

    @pytest.mark.asyncio
    @patch("services.dataset_service.DataService.delete_data_file")
    @patch("services.dataset_service.delete_file_from_rag")
    @patch.object(ProjectService, "get_project_dir")
    @patch.object(ProjectService, "load_config")
    async def test_remove_file_from_dataset_success(
        self,
        mock_load_config,
        mock_get_project_dir,
        mock_delete_rag,
        mock_delete_data_file,
    ):
        """Test successfully removing a file from a dataset."""
        mock_load_config.return_value = self.mock_project_config.model_copy()
        mock_get_project_dir.return_value = "/mock/project/dir"
        mock_delete_rag.return_value = {"status": "success", "deleted_count": 5}

        await DatasetService.remove_file_from_dataset(
            "test_namespace", "test_project", "dataset1", "file1.pdf"
        )

        mock_delete_rag.assert_awaited_once_with(
            project_dir="/mock/project/dir",
            database_name="custom_db",
            file_hash="file1.pdf",
        )
        mock_delete_data_file.assert_called_once_with(
            namespace="test_namespace",
            project_id="test_project",
            dataset="dataset1",
            file_hash="file1.pdf",
        )

    @pytest.mark.asyncio
    @patch.object(ProjectService, "load_config")
    async def test_remove_file_from_dataset_dataset_not_found(self, mock_load_config):
        """Test removing a file from a non-existent dataset."""
        mock_load_config.return_value = self.mock_project_config

        with pytest.raises(DatasetNotFoundError):
            await DatasetService.remove_file_from_dataset(
                "test_namespace", "test_project", "nonexistent_dataset", "hash"
            )


# Integration test for the full workflow
class TestDatasetServiceIntegration:
    """Integration tests for DatasetService workflows."""

    def test_full_dataset_lifecycle(self):
        """Test complete dataset lifecycle: list, create, list, delete, list."""
        # Use a simpler approach with controlled state
        with (
            patch.object(ProjectService, "load_config") as mock_load_config,
            patch.object(ProjectService, "save_config") as mock_save_config,
        ):
            # Track the current state of datasets
            current_datasets = []

            def mock_load_side_effect(namespace, project):
                return LlamaFarmConfig(
                    version=Version.v1,
                    name="test_project",
                    namespace=namespace,
                    prompts=[
                        PromptSet(
                            name="default",
                            messages=[
                                PromptMessage(
                                    role="system",
                                    content="You are a helpful assistant.",
                                )
                            ],
                        )
                    ],
                    rag={
                        "databases": [
                            {
                                "name": "test_db",
                                "type": "ChromaStore",
                                "config": {},
                                "embedding_strategies": [
                                    {
                                        "name": "test_embedding",
                                        "type": "OllamaEmbedder",
                                        "config": {"model": "nomic-embed-text"},
                                    }
                                ],
                                "retrieval_strategies": [
                                    {
                                        "name": "test_retrieval",
                                        "type": "BasicSimilarityStrategy",
                                        "config": {},
                                        "default": True,
                                    }
                                ],
                            }
                        ],
                        "data_processing_strategies": [
                            {
                                "name": "custom_strategy",
                                "description": "Custom strategy for testing behavior",
                                "parsers": [
                                    {
                                        "type": "CSVParser_LlamaIndex",
                                        "config": {},
                                        "file_extensions": [".csv"],
                                    }
                                ],
                            }
                        ],
                    },
                    datasets=current_datasets.copy(),
                    runtime=Runtime(
                        models=[
                            Model(
                                name="default",
                                provider=Provider.openai,
                                model="llama3.1:8b",
                                api_key="ollama",
                                base_url="http://localhost:11434/v1",
                                model_api_parameters={
                                    "temperature": 0.5,
                                },
                            )
                        ]
                    ),
                )

            def mock_save_side_effect(namespace, project, config):
                nonlocal current_datasets
                current_datasets = config.datasets.copy()

            mock_load_config.side_effect = mock_load_side_effect
            mock_save_config.side_effect = mock_save_side_effect

            # 1. List datasets (should be empty)
            datasets = DatasetService.list_datasets("ns", "proj")
            assert len(datasets) == 0

            # 2. Create dataset
            dataset = DatasetService.create_dataset(
                "ns", "proj", "test_dataset", "custom_strategy", "test_db"
            )
            assert dataset.name == "test_dataset"
            assert dataset.data_processing_strategy == "custom_strategy"
            assert dataset.database == "test_db"

            # 3. List datasets (should have one)
            datasets = DatasetService.list_datasets("ns", "proj")
            assert len(datasets) == 1
            assert datasets[0].name == "test_dataset"

            # 4. Delete dataset
            deleted = DatasetService.delete_dataset("ns", "proj", "test_dataset")
            assert deleted.name == "test_dataset"

            # 5. List datasets (should be empty again)
            datasets = DatasetService.list_datasets("ns", "proj")
            assert len(datasets) == 0

            # Verify save was called twice (create and delete)
            assert mock_save_config.call_count == 2
