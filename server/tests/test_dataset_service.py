"""
Tests for DatasetService.

This module contains comprehensive tests for the DatasetService class,
including unit tests for all public methods and edge cases.
"""

from unittest.mock import patch

import pytest
from config.datamodel import Dataset, LlamaFarmConfig, Version

from services.dataset_service import DEFAULT_RAG_STRATEGIES, DatasetService
from services.project_service import ProjectService


class TestDatasetService:
    """Test cases for DatasetService class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock project config with datasets
        self.mock_project_config = LlamaFarmConfig(
            version=Version.v1,
            name="test_project",
            prompts=[],
            rag={
                "strategies": [
                    {
                        "name": "custom_strategy",
                        "description": "Custom strategy for testing behavior",
                        "components": {
                            "parser": {"type": "CSVParser", "config": {}},
                            "extractors": [],
                            "embedder": {
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            },
                            "vector_store": {"type": "ChromaStore", "config": {}},
                            "retrieval_strategy": {
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                            },
                        },
                    }
                ]
            },
            datasets=[
                Dataset(
                    name="dataset1",
                    rag_strategy="auto",
                    files=["file1.pdf", "file2.pdf"],
                ),
                Dataset(
                    name="dataset2", rag_strategy="custom_strategy", files=["data.csv"]
                ),
            ],
            models=[],
        )

        # Mock project config without datasets (rag requires >=1 strategy)
        self.mock_empty_project_config = LlamaFarmConfig(
            version=Version.v1,
            name="empty_project",
            prompts=[],
            rag={
                "strategies": [
                    {
                        "name": "default",
                        "description": "Default strategy configuration",
                        "components": {
                            "parser": {"type": "CSVParser", "config": {}},
                            "extractors": [],
                            "embedder": {
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            },
                            "vector_store": {"type": "ChromaStore", "config": {}},
                            "retrieval_strategy": {
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                            },
                        },
                    }
                ]
            },
            datasets=[],
            models=[],
        )

    @patch.object(ProjectService, "load_config")
    def test_list_datasets_success(self, mock_load_config):
        """Test listing datasets successfully."""
        mock_load_config.return_value = self.mock_project_config

        datasets = DatasetService.list_datasets("test_namespace", "test_project")

        assert len(datasets) == 2
        assert datasets[0].name == "dataset1"
        assert datasets[0].rag_strategy == "auto"
        assert datasets[0].files == ["file1.pdf", "file2.pdf"]
        assert datasets[1].name == "dataset2"
        assert datasets[1].rag_strategy == "custom_strategy"
        assert datasets[1].files == ["data.csv"]

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
            prompts=[],
            rag={
                "strategies": [
                    {
                        "name": "default",
                        "description": "Default strategy configuration",
                        "components": {
                            "parser": {"type": "CSVParser", "config": {}},
                            "extractors": [],
                            "embedder": {
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            },
                            "vector_store": {"type": "ChromaStore", "config": {}},
                            "retrieval_strategy": {
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                            },
                        },
                    }
                ]
            },
            datasets=[],
            models=[],
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
            "test_namespace", "test_project", "new_dataset", "auto"
        )

        assert dataset.name == "new_dataset"
        assert dataset.rag_strategy == "auto"
        assert dataset.files == []

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
                "auto",
            )

    @patch.object(ProjectService, "load_config")
    def test_create_dataset_unsupported_rag_strategy(self, mock_load_config):
        """Test creating a dataset with an unsupported RAG strategy."""
        mock_load_config.return_value = self.mock_project_config

        with pytest.raises(
            ValueError, match="RAG strategy unsupported_strategy not supported"
        ):
            DatasetService.create_dataset(
                "test_namespace", "test_project", "new_dataset", "unsupported_strategy"
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
            "custom_dataset",
            "custom_strategy",  # This is in the custom rag strategies list
        )

        assert dataset.name == "custom_dataset"
        assert dataset.rag_strategy == "custom_strategy"
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
        assert deleted_dataset.rag_strategy == "auto"
        assert deleted_dataset.files == ["file1.pdf", "file2.pdf"]

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
    def test_get_supported_rag_strategies_with_custom_strategies(
        self, mock_load_config
    ):
        """Test getting supported RAG strategies including custom ones."""
        mock_load_config.return_value = self.mock_project_config

        strategies = DatasetService.get_supported_rag_strategies(
            "test_namespace", "test_project"
        )

        expected_strategies = DEFAULT_RAG_STRATEGIES + ["custom_strategy"]
        assert strategies == expected_strategies

    @patch.object(ProjectService, "load_config")
    def test_get_supported_rag_strategies_default_only(self, mock_load_config):
        """Test getting supported RAG strategies with only default strategies."""
        config_no_custom = self.mock_project_config.model_copy()
        config_no_custom.rag = {
            "strategies": [
                {
                    "name": "default",
                    "description": "Default strategy configuration",
                    "components": {
                        "parser": {"type": "CSVParser", "config": {}},
                        "extractors": [],
                        "embedder": {
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text"},
                        },
                        "vector_store": {"type": "ChromaStore", "config": {}},
                        "retrieval_strategy": {
                            "type": "BasicSimilarityStrategy",
                            "config": {},
                        },
                    },
                }
            ]
        }
        mock_load_config.return_value = config_no_custom

        strategies = DatasetService.get_supported_rag_strategies(
            "test_namespace", "test_project"
        )

        expected_strategies = DEFAULT_RAG_STRATEGIES + ["default"]
        assert strategies == expected_strategies

    @patch.object(ProjectService, "load_config")
    def test_get_supported_rag_strategies_no_rag_config(self, mock_load_config):
        """Test getting supported RAG strategies when RAG config is missing."""
        config_no_rag = LlamaFarmConfig(
            version=Version.v1,
            name="test_project",
            prompts=[],
            rag={
                "strategies": [
                    {
                        "name": "default",
                        "description": "Default strategy configuration",
                        "components": {
                            "parser": {"type": "CSVParser", "config": {}},
                            "extractors": [],
                            "embedder": {
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            },
                            "vector_store": {"type": "ChromaStore", "config": {}},
                            "retrieval_strategy": {
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                            },
                        },
                    }
                ]
            },
            datasets=[],
            models=[],
        )
        mock_load_config.return_value = config_no_rag

        strategies = DatasetService.get_supported_rag_strategies(
            "test_namespace", "test_project"
        )

        expected_strategies = DEFAULT_RAG_STRATEGIES + ["default"]
        assert strategies == expected_strategies

    @patch.object(ProjectService, "load_config")
    def test_get_supported_rag_strategies_no_strategies_key(self, mock_load_config):
        """Test getting supported RAG strategies when rag_strategies key is missing from RAG config."""
        config_no_strategies_key = LlamaFarmConfig(
            version=Version.v1,
            name="test_project",
            prompts=[],
            rag={
                "strategies": [
                    {
                        "name": "default",
                        "description": "Default strategy configuration",
                        "components": {
                            "parser": {"type": "CSVParser", "config": {}},
                            "extractors": [],
                            "embedder": {
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            },
                            "vector_store": {"type": "ChromaStore", "config": {}},
                            "retrieval_strategy": {
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                            },
                        },
                    }
                ]
            },
            datasets=[],
            models=[],
        )
        mock_load_config.return_value = config_no_strategies_key

        strategies = DatasetService.get_supported_rag_strategies(
            "test_namespace", "test_project"
        )

        expected_strategies = DEFAULT_RAG_STRATEGIES + ["default"]
        assert strategies == expected_strategies

    @patch.object(ProjectService, "load_config")
    def test_create_dataset_no_existing_datasets(self, mock_load_config):
        """Test creating a dataset when no datasets exist yet."""
        config_no_datasets = LlamaFarmConfig(
            version=Version.v1,
            name="test_project",
            prompts=[],
            rag={
                "strategies": [
                    {
                        "name": "default",
                        "description": "Default strategy configuration",
                        "components": {
                            "parser": {"type": "CSVParser", "config": {}},
                            "extractors": [],
                            "embedder": {
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            },
                            "vector_store": {"type": "ChromaStore", "config": {}},
                            "retrieval_strategy": {
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                            },
                        },
                    }
                ]
            },
            datasets=[],
            models=[],
        )
        mock_load_config.return_value = config_no_datasets

        with patch.object(ProjectService, "save_config") as mock_save_config:
            dataset = DatasetService.create_dataset(
                "test_namespace", "test_project", "first_dataset", "auto"
            )

            assert dataset.name == "first_dataset"
            assert dataset.rag_strategy == "auto"
            assert dataset.files == []

            # Verify the config was updated correctly
            call_args = mock_save_config.call_args[0]
            updated_config = call_args[2]
            assert len(updated_config.datasets) == 1
            assert updated_config.datasets[0].name == "first_dataset"


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
                    prompts=[],
                    rag={
                        "strategies": [
                            {
                                "name": "custom_strategy",
                                "description": "Custom strategy for testing behavior",
                                "components": {
                                    "parser": {"type": "CSVParser", "config": {}},
                                    "extractors": [],
                                    "embedder": {
                                        "type": "OllamaEmbedder",
                                        "config": {"model": "nomic-embed-text"},
                                    },
                                    "vector_store": {
                                        "type": "ChromaStore",
                                        "config": {},
                                    },
                                    "retrieval_strategy": {
                                        "type": "BasicSimilarityStrategy",
                                        "config": {},
                                    },
                                },
                            }
                        ]
                    },
                    datasets=current_datasets.copy(),
                    models=[],
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
                "ns", "proj", "test_dataset", "auto"
            )
            assert dataset.name == "test_dataset"

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
