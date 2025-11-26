"""
Tests for dataset metadata error handling.

This module contains comprehensive tests for error scenarios in dataset service
metadata handling, focusing on resilience and graceful error recovery.
"""

from unittest.mock import patch

import pytest
from config.datamodel import (
    Dataset,
    LlamaFarmConfig,
    PromptSet,
    PromptMessage,
    Provider,
    Runtime,
    Version,
    Model,
)

from services.data_service import MetadataFileContent
from services.dataset_service import DatasetService
from services.project_service import ProjectService


class TestDatasetMetadataErrorHandling:
    """Test cases for dataset metadata error handling."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Sample file hashes
        self.file_hash_valid = (
            "valid123456789abcdefvalid123456789abcdefvalid123456789abcdefvalid"
        )
        self.file_hash_missing = (
            "missing456789abcdefmissing456789abcdefmissing456789abcdefmissing"
        )
        self.file_hash_corrupted = (
            "corrupt789abcdefcorrupt789abcdefcorrupt789abcdefcorrupt789abcdef"
        )

        # Valid metadata
        self.valid_metadata = MetadataFileContent(
            original_file_name="valid_document.pdf",
            resolved_file_name="valid_document_1640995200.0.pdf",
            size=1024000,
            mime_type="application/pdf",
            hash=self.file_hash_valid,
            timestamp=1640995200.0,
        )

        # Mock project config with problematic files
        self.mock_project_config_with_errors = LlamaFarmConfig(
            version=Version.v1,
            name="error_test_project",
            namespace="error_test_namespace",
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
                        "name": "error_test_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "error_test_embedding",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "error_test_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "error_test_strategy",
                        "description": "Error test strategy",
                        "parsers": [
                            {
                                "type": "PDFParser_LlamaIndex",
                                "config": {},
                                "file_extensions": [".pdf"],
                            }
                        ],
                    }
                ],
            },
            datasets=[
                Dataset(
                    name="dataset_with_missing_metadata",
                    data_processing_strategy="error_test_strategy",
                    database="error_test_db",
                    files=[self.file_hash_valid, self.file_hash_missing],
                ),
                Dataset(
                    name="dataset_with_corrupted_metadata",
                    data_processing_strategy="error_test_strategy",
                    database="error_test_db",
                    files=[self.file_hash_valid, self.file_hash_corrupted],
                ),
                Dataset(
                    name="dataset_all_missing",
                    data_processing_strategy="error_test_strategy",
                    database="error_test_db",
                    files=[self.file_hash_missing, self.file_hash_corrupted],
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

    @patch.object(ProjectService, "load_config")
    @patch.object(DatasetService, "list_dataset_files")
    def test_missing_metadata_file_graceful_handling(
        self, mock_list_files, mock_load_config
    ):
        """Test graceful handling when metadata file is missing for some file hashes."""
        mock_load_config.return_value = self.mock_project_config_with_errors

        def list_files_side_effect(namespace, project_id, dataset):
            if dataset == "dataset_with_missing_metadata":
                return [self.valid_metadata]
            if dataset == "dataset_with_corrupted_metadata":
                return [self.valid_metadata]
            return []

        mock_list_files.side_effect = list_files_side_effect

        # Execute
        datasets_with_details = DatasetService.list_datasets_with_file_details(
            "error_test_namespace", "error_test_project"
        )

        # Verify results - processing should continue for other files
        assert len(datasets_with_details) == 3

        # First dataset should only have metadata for valid file
        dataset_mixed = next(
            d
            for d in datasets_with_details
            if d.name == "dataset_with_missing_metadata"
        )
        assert len(dataset_mixed.details.files_metadata) == 1
        assert dataset_mixed.details.files_metadata[0].hash == self.file_hash_valid

        # Second dataset should also only have metadata for valid file
        dataset_corrupted = next(
            d
            for d in datasets_with_details
            if d.name == "dataset_with_corrupted_metadata"
        )
        assert len(dataset_corrupted.details.files_metadata) == 1
        assert dataset_corrupted.details.files_metadata[0].hash == self.file_hash_valid

        # Third dataset should have no file metadata (all files failed)
        dataset_all_missing = next(
            d for d in datasets_with_details if d.name == "dataset_all_missing"
        )
        assert len(dataset_all_missing.details.files_metadata) == 0

        # No warnings expected in filesystem-based approach; missing metadata simply reduces results

    @patch.object(ProjectService, "load_config")
    @patch.object(DatasetService, "list_dataset_files")
    def test_corrupted_metadata_file_handling(
        self, mock_list_files, mock_load_config
    ):
        """Test handling of corrupted JSON in metadata file."""
        mock_load_config.return_value = self.mock_project_config_with_errors

        # Configure metadata lookup to raise JSON decode errors for corrupted files
        def list_files_side_effect(namespace, project_id, dataset):
            if dataset == "dataset_with_corrupted_metadata":
                raise ValueError("Invalid JSON in metadata file")
            if dataset == "dataset_with_missing_metadata":
                return [self.valid_metadata]
            return []

        mock_list_files.side_effect = list_files_side_effect

        # Execute - should handle ValueError gracefully (though current implementation doesn't)
        # Note: Current implementation only catches FileNotFoundError, not ValueError
        # This test documents the current behavior - ValueError propagates up
        with pytest.raises(ValueError, match="Invalid JSON in metadata file"):
            DatasetService.list_datasets_with_file_details(
                "error_test_namespace", "error_test_project"
            )

    @patch.object(ProjectService, "load_config")
    @patch.object(DatasetService, "list_dataset_files")
    def test_permission_denied_metadata_access(
        self, mock_list_files, mock_load_config
    ):
        """Test handling of permission issues when accessing metadata files."""
        mock_load_config.return_value = self.mock_project_config_with_errors

        # Configure metadata lookup to raise permission errors
        def list_files_side_effect(namespace, project_id, dataset):
            if dataset == "dataset_with_missing_metadata":
                raise PermissionError("Permission denied accessing metadata directory")
            if dataset == "dataset_with_corrupted_metadata":
                return [self.valid_metadata]
            return []

        mock_list_files.side_effect = list_files_side_effect

        # Execute - should handle permission errors gracefully
        # Note: Current implementation only catches FileNotFoundError
        # This test documents the current behavior
        with pytest.raises(PermissionError):
            DatasetService.list_datasets_with_file_details(
                "error_test_namespace", "error_test_project"
            )

    @patch.object(ProjectService, "load_config")
    @patch.object(DatasetService, "list_dataset_files")
    def test_file_hash_in_dataset_but_no_raw_file(
        self, mock_list_files, mock_load_config
    ):
        """Test when dataset references file hash but raw file doesn't exist."""
        mock_load_config.return_value = self.mock_project_config_with_errors

        def list_files_side_effect(namespace, project_id, dataset):
            if dataset in (
                "dataset_with_missing_metadata",
                "dataset_with_corrupted_metadata",
            ):
                return [self.valid_metadata]
            return []

        mock_list_files.side_effect = list_files_side_effect

        # Execute
        datasets_with_details = DatasetService.list_datasets_with_file_details(
            "error_test_namespace", "error_test_project"
        )

        # Should handle gracefully - only include files with accessible metadata
        assert len(datasets_with_details) == 3

        # Only valid files should have metadata
        total_metadata_files = sum(
            len(dataset.details.files_metadata) for dataset in datasets_with_details
        )
        # We expect 2 valid files total: 1 in dataset_with_missing_metadata and 1 in dataset_with_corrupted_metadata
        # The third dataset (dataset_all_missing) has no valid files
        assert total_metadata_files == 2

    @patch.object(ProjectService, "load_config")
    @patch.object(DatasetService, "list_dataset_files")
    def test_metadata_file_invalid_schema(self, mock_list_files, mock_load_config):
        """Test handling when metadata file doesn't match MetadataFileContent schema."""
        mock_load_config.return_value = self.mock_project_config_with_errors

        # Create invalid metadata that would fail Pydantic validation
        from pydantic_core import ValidationError as CoreValidationError

        def list_files_side_effect(namespace, project_id, dataset):
            if dataset == "dataset_with_corrupted_metadata":
                raise CoreValidationError.from_exception_data(
                    "ValidationError",
                    [
                        {
                            "type": "string_type",
                            "loc": ("test",),
                            "msg": "Invalid metadata schema",
                            "input": {},
                        }
                    ],
                )
            if dataset == "dataset_with_missing_metadata":
                return [self.valid_metadata]
            return []

        mock_list_files.side_effect = list_files_side_effect

        # Execute - should handle validation errors
        # Note: Current implementation doesn't explicitly catch ValidationError
        with pytest.raises(CoreValidationError):
            DatasetService.list_datasets_with_file_details(
                "error_test_namespace", "error_test_project"
            )

    @patch.object(ProjectService, "load_config")
    @patch.object(DatasetService, "list_dataset_files")
    def test_dataset_files_list_contains_invalid_hashes(
        self, mock_list_files, mock_load_config
    ):
        """Test handling when dataset.files contains malformed hash values."""
        # Create config with invalid hashes
        config_with_invalid_hashes = LlamaFarmConfig(
            version=Version.v1,
            name="invalid_hash_project",
            namespace="error_test_namespace",
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
                        "name": "error_test_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "error_test_embedding",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "error_test_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "error_test_strategy",
                        "description": "Error test strategy",
                        "parsers": [
                            {
                                "type": "PDFParser_LlamaIndex",
                                "config": {},
                                "file_extensions": [".pdf"],
                            }
                        ],
                    }
                ],
            },
            datasets=[
                Dataset(
                    name="dataset_with_invalid_hashes",
                    data_processing_strategy="error_test_strategy",
                    database="error_test_db",
                    files=[
                        "invalid_hash",  # Too short
                        "another_invalid_hash_that_is_malformed",  # Wrong format
                        "",  # Empty string
                        self.file_hash_valid,  # One valid hash
                    ],
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

        mock_load_config.return_value = config_with_invalid_hashes

        mock_list_files.return_value = [self.valid_metadata]

        datasets_with_details = DatasetService.list_datasets_with_file_details(
            "error_test_namespace", "invalid_hash_project"
        )

        assert len(datasets_with_details) == 1
        dataset = datasets_with_details[0]
        assert len(dataset.details.files_metadata) == 1
        assert dataset.details.files_metadata[0].hash == self.file_hash_valid

    @patch.object(ProjectService, "load_config")
    @patch.object(DatasetService, "list_dataset_files")
    def test_metadata_listing_unavailable(self, mock_list_files, mock_load_config):
        """Test behavior when dataset metadata listing is completely unavailable."""
        mock_load_config.return_value = self.mock_project_config_with_errors
        mock_list_files.side_effect = Exception("Metadata service is down")

        with pytest.raises(Exception, match="Metadata service is down"):
            DatasetService.list_datasets_with_file_details(
                "error_test_namespace", "error_test_project"
            )

    def test_project_service_load_config_failure(self):
        """Test behavior when ProjectService.load_config() fails."""
        with patch.object(ProjectService, "load_config") as mock_load_config:
            # Simulate project service failure
            mock_load_config.side_effect = Exception(
                "Project configuration is corrupted"
            )

            # Execute - should propagate configuration errors
            with pytest.raises(Exception, match="Project configuration is corrupted"):
                DatasetService.list_datasets_with_file_details(
                    "error_test_namespace", "error_test_project"
                )

    @patch.object(ProjectService, "load_config")
    @patch.object(DatasetService, "list_dataset_files")
    def test_mixed_error_scenarios_resilience(
        self, mock_list_files, mock_load_config
    ):
        """Test system resilience with multiple error types occurring simultaneously."""
        # Create config with various problematic scenarios
        mixed_error_config = LlamaFarmConfig(
            version=Version.v1,
            name="mixed_errors_project",
            namespace="error_test_namespace",
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
                        "name": "error_test_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "error_test_embedding",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "error_test_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "error_test_strategy",
                        "description": "Error test strategy",
                        "parsers": [
                            {
                                "type": "PDFParser_LlamaIndex",
                                "config": {},
                                "file_extensions": [".pdf"],
                            }
                        ],
                    }
                ],
            },
            datasets=[
                Dataset(
                    name="dataset_mixed_errors",
                    data_processing_strategy="error_test_strategy",
                    database="error_test_db",
                    files=[
                        self.file_hash_valid,  # This should work
                        self.file_hash_missing,  # FileNotFoundError
                        self.file_hash_corrupted,  # Will cause different error
                        "short_hash",  # Invalid hash format
                        "",  # Empty hash
                    ],
                ),
                Dataset(
                    name="dataset_empty_files",
                    data_processing_strategy="error_test_strategy",
                    database="error_test_db",
                    files=[],  # No files at all
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

        mock_load_config.return_value = mixed_error_config

        def list_files_side_effect(namespace, project_id, dataset):
            if dataset == "dataset_mixed_errors":
                return [self.valid_metadata]
            return []

        mock_list_files.side_effect = list_files_side_effect

        # Execute - should handle mixed errors gracefully
        datasets_with_details = DatasetService.list_datasets_with_file_details(
            "error_test_namespace", "mixed_errors_project"
        )

        # Verify resilient behavior
        assert len(datasets_with_details) == 2

        # First dataset should only have the valid file
        dataset_mixed = next(
            d for d in datasets_with_details if d.name == "dataset_mixed_errors"
        )
        assert len(dataset_mixed.details.files_metadata) == 1
        assert dataset_mixed.details.files_metadata[0].hash == self.file_hash_valid

        # Second dataset should have no files
        dataset_empty = next(
            d for d in datasets_with_details if d.name == "dataset_empty_files"
        )
        assert len(dataset_empty.details.files_metadata) == 0

        # No warnings expected; metadata listing simply returns available files

    def test_empty_files_list_handling(self):
        """Test handling of datasets with empty files lists."""
        # This is actually a normal case, not an error, but worth testing
        empty_config = LlamaFarmConfig(
            version=Version.v1,
            name="empty_files_project",
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
                        "name": "test_strategy",
                        "description": "Test strategy",
                        "parsers": [
                            {
                                "type": "PDFParser_LlamaIndex",
                                "config": {},
                                "file_extensions": [".pdf"],
                            }
                        ],
                    }
                ],
            },
            datasets=[
                Dataset(
                    name="dataset_no_files",
                    data_processing_strategy="test_strategy",
                    database="test_db",
                    files=[],
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

        with patch.object(ProjectService, "load_config") as mock_load_config:
            mock_load_config.return_value = empty_config

            # Execute
            datasets_with_details = DatasetService.list_datasets_with_file_details(
                "test_namespace", "empty_files_project"
            )

            # Should handle empty files lists without errors
            assert len(datasets_with_details) == 1
            dataset = datasets_with_details[0]
            assert dataset.name == "dataset_no_files"
            assert len(dataset.details.files_metadata) == 0
            assert isinstance(dataset.details.files_metadata, list)


class TestErrorHandlingIntegration:
    """Integration tests for error handling across the entire system."""

    @patch.object(ProjectService, "load_config")
    @patch.object(DatasetService, "list_dataset_files")
    def test_production_like_error_scenario(
        self, mock_list_files, mock_load_config
    ):
        """Test a production-like scenario with various real-world error conditions."""
        # Simulate a realistic production scenario
        production_hashes = [
            "prod1234567890abcdefprod1234567890abcdefprod1234567890abcdefprod12",  # Valid
            "prod5678901234abcdefprod5678901234abcdefprod5678901234abcdefprod56",  # Valid
            "missing890abcdefmissing890abcdefmissing890abcdefmissing890abcdefmiss",  # Missing metadata
            "corrupt123456789abcdefcorrupt123456789abcdefcorrupt123456789abcdefcor",  # Corrupted file
        ]

        # Valid metadata instances
        metadata_1 = MetadataFileContent(
            original_file_name="production_doc_1.pdf",
            resolved_file_name="production_doc_1_1700000000.0.pdf",
            size=5000000,
            mime_type="application/pdf",
            hash=production_hashes[0],
            timestamp=1700000000.0,
        )

        metadata_2 = MetadataFileContent(
            original_file_name="production_data.csv",
            resolved_file_name="production_data_1700000100.0.csv",
            size=250000,
            mime_type="text/csv",
            hash=production_hashes[1],
            timestamp=1700000100.0,
        )

        # Production-like config
        production_config = LlamaFarmConfig(
            version=Version.v1,
            name="production_project",
            namespace="production_namespace",
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
                        "name": "production_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "production_embedding",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "production_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "production_strategy",
                        "description": "Production strategy",
                        "parsers": [
                            {
                                "type": "PDFParser_LlamaIndex",
                                "config": {},
                                "file_extensions": [".pdf"],
                            },
                            {
                                "type": "CSVParser_LlamaIndex",
                                "config": {},
                                "file_extensions": [".csv"],
                            },
                        ],
                    }
                ],
            },
            datasets=[
                Dataset(
                    name="critical_documents",
                    data_processing_strategy="production_strategy",
                    database="production_db",
                    files=production_hashes,  # Mix of valid and problematic files
                ),
                Dataset(
                    name="user_uploads",
                    data_processing_strategy="production_strategy",
                    database="production_db",
                    files=[
                        production_hashes[1],
                        production_hashes[2],
                    ],  # One valid, one missing
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

        mock_load_config.return_value = production_config

        # Simulate realistic production errors
        def metadata_side_effect(namespace, project_id, dataset, file_content_hash):
            if file_content_hash == production_hashes[0]:
                return metadata_1
            elif file_content_hash == production_hashes[1]:
                return metadata_2
            elif file_content_hash == production_hashes[2]:
                # Simulate file system issue (common in production)
                raise FileNotFoundError(
                    f"Metadata file was deleted or moved: {file_content_hash}"
                )
            elif file_content_hash == production_hashes[3]:
                # Simulate disk corruption (rare but happens)
                raise FileNotFoundError(
                    f"Metadata file is corrupted: {file_content_hash}"
                )
            else:
                raise FileNotFoundError(f"Unexpected file hash: {file_content_hash}")

        def list_files_side_effect(namespace, project_id, dataset):
            if dataset == "critical_documents":
                return [metadata_1, metadata_2]
            if dataset == "user_uploads":
                return [metadata_2]
            return []

        mock_list_files.side_effect = list_files_side_effect

        # Execute - should handle production errors gracefully
        datasets_with_details = DatasetService.list_datasets_with_file_details(
            "production_namespace", "production_project"
        )

        # Verify production-grade resilience
        assert len(datasets_with_details) == 2

        # Critical documents dataset should have partial success
        critical_dataset = next(
            d for d in datasets_with_details if d.name == "critical_documents"
        )
        assert (
            len(critical_dataset.details.files_metadata) == 2
        )  # Only 2 of 4 files succeeded

        # Verify successful files have correct metadata
        successful_hashes = [
            meta.hash for meta in critical_dataset.details.files_metadata
        ]
        assert production_hashes[0] in successful_hashes
        assert production_hashes[1] in successful_hashes
        assert production_hashes[2] not in successful_hashes  # Failed files excluded
        assert production_hashes[3] not in successful_hashes

        # User uploads dataset should have partial success
        user_dataset = next(
            d for d in datasets_with_details if d.name == "user_uploads"
        )
        assert (
            len(user_dataset.details.files_metadata) == 1
        )  # Only 1 of 2 files succeeded
        assert user_dataset.details.files_metadata[0].hash == production_hashes[1]

        # Verify system remains functional despite errors
        total_successful_files = sum(
            len(dataset.details.files_metadata) for dataset in datasets_with_details
        )
        assert total_successful_files == 3  # 2 + 1 successful files across datasets
