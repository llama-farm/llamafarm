"""
Tests for DatasetService.

This module contains comprehensive tests for the DatasetService class,
including unit tests for all public methods and edge cases.
"""

from unittest.mock import patch

import pytest

from services.dataset_service import DEFAULT_PARSERS, DatasetService
from services.project_service import ProjectService


class TestDatasetService:
    """Test cases for DatasetService class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock project config with datasets
        self.mock_project_config = {
            "name": "test_project",
            "datasets": [
                {
                    "name": "dataset1",
                    "parser": "pdf",
                    "files": ["file1.pdf", "file2.pdf"]
                },
                {
                    "name": "dataset2", 
                    "parser": "csv",
                    "files": ["data.csv"]
                }
            ],
            "rag": {
                "parsers": ["custom_parser"]
            }
        }

        # Mock project config without datasets
        self.mock_empty_project_config = {
            "name": "empty_project",
            "datasets": [],
            "rag": {
                "parsers": []
            }
        }

    @patch.object(ProjectService, 'load_config')
    def test_list_datasets_success(self, mock_load_config):
        """Test listing datasets successfully."""
        mock_load_config.return_value = self.mock_project_config
        
        datasets = DatasetService.list_datasets("test_namespace", "test_project")
        
        assert len(datasets) == 2
        assert datasets[0]["name"] == "dataset1"
        assert datasets[0]["parser"] == "pdf"
        assert datasets[0]["files"] == ["file1.pdf", "file2.pdf"]
        assert datasets[1]["name"] == "dataset2"
        assert datasets[1]["parser"] == "csv"
        assert datasets[1]["files"] == ["data.csv"]
        
        mock_load_config.assert_called_once_with("test_namespace", "test_project")

    @patch.object(ProjectService, 'load_config')
    def test_list_datasets_empty(self, mock_load_config):
        """Test listing datasets when no datasets exist."""
        mock_load_config.return_value = self.mock_empty_project_config
        
        datasets = DatasetService.list_datasets("test_namespace", "test_project")
        
        assert len(datasets) == 0
        assert datasets == []

    @patch.object(ProjectService, 'load_config')
    def test_list_datasets_no_datasets_key(self, mock_load_config):
        """Test listing datasets when datasets key doesn't exist in config."""
        config_without_datasets = {"name": "test_project"}
        mock_load_config.return_value = config_without_datasets
        
        datasets = DatasetService.list_datasets("test_namespace", "test_project")
        
        assert len(datasets) == 0
        assert datasets == []

    @patch.object(ProjectService, 'save_config')
    @patch.object(ProjectService, 'load_config')
    def test_create_dataset_success(self, mock_load_config, mock_save_config):
        """Test creating a dataset successfully."""
        mock_load_config.return_value = self.mock_project_config.copy()
        mock_save_config.return_value = None
        
        dataset = DatasetService.create_dataset(
            "test_namespace", 
            "test_project", 
            "new_dataset", 
            "pdf"
        )
        
        assert dataset["name"] == "new_dataset"
        assert dataset["parser"] == "pdf"
        assert dataset["files"] == []
        
        # Verify save_config was called with updated config
        mock_save_config.assert_called_once()
        call_args = mock_save_config.call_args[0]
        assert call_args[0] == "test_namespace"
        assert call_args[1] == "test_project"
        updated_config = call_args[2]
        assert len(updated_config["datasets"]) == 3
        assert updated_config["datasets"][-1]["name"] == "new_dataset"

    @patch.object(ProjectService, 'load_config')
    def test_create_dataset_duplicate_name(self, mock_load_config):
        """Test creating a dataset with a name that already exists."""
        mock_load_config.return_value = self.mock_project_config
        
        with pytest.raises(ValueError, match="Dataset dataset1 already exists"):
            DatasetService.create_dataset(
                "test_namespace",
                "test_project", 
                "dataset1",  # This name already exists
                "pdf"
            )

    @patch.object(ProjectService, 'load_config')
    def test_create_dataset_unsupported_parser(self, mock_load_config):
        """Test creating a dataset with an unsupported parser."""
        mock_load_config.return_value = self.mock_project_config
        
        with pytest.raises(ValueError, match="Parser unsupported_parser not supported"):
            DatasetService.create_dataset(
                "test_namespace",
                "test_project",
                "new_dataset",
                "unsupported_parser"
            )

    @patch.object(ProjectService, 'save_config')
    @patch.object(ProjectService, 'load_config')
    def test_create_dataset_with_custom_parser(self, mock_load_config, mock_save_config):
        """Test creating a dataset with a custom parser."""
        mock_load_config.return_value = self.mock_project_config.copy()
        
        dataset = DatasetService.create_dataset(
            "test_namespace",
            "test_project",
            "custom_dataset",
            "custom_parser"  # This is in the custom parsers list
        )
        
        assert dataset["name"] == "custom_dataset"
        assert dataset["parser"] == "custom_parser"
        mock_save_config.assert_called_once()

    @patch.object(ProjectService, 'save_config')
    @patch.object(ProjectService, 'load_config')
    def test_delete_dataset_success(self, mock_load_config, mock_save_config):
        """Test deleting a dataset successfully."""
        mock_load_config.return_value = self.mock_project_config.copy()
        mock_save_config.return_value = None
        
        deleted_dataset = DatasetService.delete_dataset(
            "test_namespace",
            "test_project", 
            "dataset1"
        )
        
        assert deleted_dataset["name"] == "dataset1"
        assert deleted_dataset["parser"] == "pdf"
        assert deleted_dataset["files"] == ["file1.pdf", "file2.pdf"]
        
        # Verify save_config was called with updated config
        mock_save_config.assert_called_once()
        call_args = mock_save_config.call_args[0]
        updated_config = call_args[2]
        assert len(updated_config["datasets"]) == 1
        assert updated_config["datasets"][0]["name"] == "dataset2"

    @patch.object(ProjectService, 'load_config')
    def test_delete_dataset_not_found(self, mock_load_config):
        """Test deleting a dataset that doesn't exist."""
        mock_load_config.return_value = self.mock_project_config
        
        with pytest.raises(ValueError, match="Dataset nonexistent_dataset not found"):
            DatasetService.delete_dataset(
                "test_namespace",
                "test_project",
                "nonexistent_dataset"
            )

    @patch.object(ProjectService, 'load_config') 
    def test_delete_dataset_empty_list(self, mock_load_config):
        """Test deleting a dataset when no datasets exist."""
        mock_load_config.return_value = self.mock_empty_project_config
        
        with pytest.raises(ValueError, match="Dataset any_dataset not found"):
            DatasetService.delete_dataset(
                "test_namespace",
                "test_project",
                "any_dataset"
            )

    @patch.object(ProjectService, 'load_config')
    def test_get_supported_parsers_with_custom_parsers(self, mock_load_config):
        """Test getting supported parsers including custom ones."""
        mock_load_config.return_value = self.mock_project_config
        
        parsers = DatasetService.get_supported_parsers("test_namespace", "test_project")
        
        expected_parsers = DEFAULT_PARSERS + ["custom_parser"]
        assert parsers == expected_parsers

    @patch.object(ProjectService, 'load_config')
    def test_get_supported_parsers_default_only(self, mock_load_config):
        """Test getting supported parsers with only default parsers."""
        config_no_custom = self.mock_project_config.copy()
        config_no_custom["rag"]["parsers"] = []
        mock_load_config.return_value = config_no_custom
        
        parsers = DatasetService.get_supported_parsers("test_namespace", "test_project")
        
        expected_parsers = DEFAULT_PARSERS
        assert parsers == expected_parsers

    @patch.object(ProjectService, 'load_config')
    def test_get_supported_parsers_no_rag_config(self, mock_load_config):
        """Test getting supported parsers when RAG config is missing."""
        config_no_rag = {"name": "test_project", "datasets": []}
        mock_load_config.return_value = config_no_rag
        
        parsers = DatasetService.get_supported_parsers("test_namespace", "test_project")
        
        expected_parsers = DEFAULT_PARSERS
        assert parsers == expected_parsers

    @patch.object(ProjectService, 'load_config')
    def test_get_supported_parsers_no_parsers_key(self, mock_load_config):
        """Test getting supported parsers when parsers key is missing from RAG config."""
        config_no_parsers_key = {
            "name": "test_project",
            "datasets": [],
            "rag": {}
        }
        mock_load_config.return_value = config_no_parsers_key
        
        parsers = DatasetService.get_supported_parsers("test_namespace", "test_project")
        
        expected_parsers = DEFAULT_PARSERS
        assert parsers == expected_parsers

    @patch.object(ProjectService, 'load_config')
    def test_create_dataset_no_existing_datasets(self, mock_load_config):
        """Test creating a dataset when no datasets exist yet."""
        config_no_datasets = {"name": "test_project", "rag": {"parsers": []}}
        mock_load_config.return_value = config_no_datasets
        
        with patch.object(ProjectService, 'save_config') as mock_save_config:
            dataset = DatasetService.create_dataset(
                "test_namespace",
                "test_project",
                "first_dataset",
                "pdf"
            )
            
            assert dataset["name"] == "first_dataset"
            assert dataset["parser"] == "pdf"
            assert dataset["files"] == []
            
            # Verify the config was updated correctly
            call_args = mock_save_config.call_args[0]
            updated_config = call_args[2]
            assert len(updated_config["datasets"]) == 1
            assert updated_config["datasets"][0]["name"] == "first_dataset"


# Integration test for the full workflow
class TestDatasetServiceIntegration:
    """Integration tests for DatasetService workflows."""

    def test_full_dataset_lifecycle(self):
        """Test complete dataset lifecycle: list, create, list, delete, list."""
        # Use a simpler approach with controlled state
        with patch.object(ProjectService, 'load_config') as mock_load_config, \
             patch.object(ProjectService, 'save_config') as mock_save_config:
            
            # Track the current state of datasets
            current_datasets = []
            
            def mock_load_side_effect(namespace, project):
                return {
                    "name": "test_project",
                    "datasets": current_datasets.copy(),
                    "rag": {"parsers": ["custom_parser"]}
                }
            
            def mock_save_side_effect(namespace, project, config):
                nonlocal current_datasets
                current_datasets = config["datasets"].copy()
                
            mock_load_config.side_effect = mock_load_side_effect
            mock_save_config.side_effect = mock_save_side_effect
            
            # 1. List datasets (should be empty)
            datasets = DatasetService.list_datasets("ns", "proj")
            assert len(datasets) == 0
            
            # 2. Create dataset
            dataset = DatasetService.create_dataset("ns", "proj", "test_dataset", "pdf")
            assert dataset["name"] == "test_dataset"
            
            # 3. List datasets (should have one)
            datasets = DatasetService.list_datasets("ns", "proj") 
            assert len(datasets) == 1
            assert datasets[0]["name"] == "test_dataset"
            
            # 4. Delete dataset
            deleted = DatasetService.delete_dataset("ns", "proj", "test_dataset")
            assert deleted["name"] == "test_dataset"
            
            # 5. List datasets (should be empty again)
            datasets = DatasetService.list_datasets("ns", "proj")
            assert len(datasets) == 0
            
            # Verify save was called twice (create and delete)
            assert mock_save_config.call_count == 2