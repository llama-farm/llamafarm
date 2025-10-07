"""
Tests for DatasetService extra details functionality.

This module contains comprehensive tests for the DatasetService class
focusing on the list_datasets_with_file_details method and file metadata handling.
"""

from unittest.mock import patch

import pytest
from config.datamodel import (
    Dataset,
    LlamaFarmConfig,
    Prompt,
    Provider,
    Runtime,
    Version,
    Model,
)

from services.data_service import MetadataFileContent
from services.dataset_service import (
    DatasetService,
    DatasetWithFileDetails,
    DatasetDetails,
)
from services.project_service import ProjectService


class TestDatasetServiceExtraDetails:
    """Test cases for DatasetService extra details functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Sample file hashes and metadata
        self.file_hash_1 = (
            "abcd1234567890abcd1234567890abcd1234567890abcd1234567890abcd1234"
        )
        self.file_hash_2 = (
            "efgh5678901234efgh5678901234efgh5678901234efgh5678901234efgh5678"
        )
        self.file_hash_3 = (
            "ijkl9012345678ijkl9012345678ijkl9012345678ijkl9012345678ijkl9012"
        )

        # Sample metadata objects
        self.metadata_1 = MetadataFileContent(
            original_file_name="document1.pdf",
            resolved_file_name="document1_1640995200.0.pdf",
            size=1024000,
            mime_type="application/pdf",
            hash=self.file_hash_1,
            timestamp=1640995200.0,
        )

        self.metadata_2 = MetadataFileContent(
            original_file_name="data.csv",
            resolved_file_name="data_1640995300.0.csv",
            size=2048,
            mime_type="text/csv",
            hash=self.file_hash_2,
            timestamp=1640995300.0,
        )

        self.metadata_3 = MetadataFileContent(
            original_file_name="readme.txt",
            resolved_file_name="readme_1640995400.0.txt",
            size=512,
            mime_type="text/plain",
            hash=self.file_hash_3,
            timestamp=1640995400.0,
        )

        # Mock project config with datasets containing files
        self.mock_project_config = LlamaFarmConfig(
            version=Version.v1,
            name="test_project",
            namespace="test_namespace",
            prompts=[
                Prompt(
                    role="system",
                    content="You are a helpful assistant.",
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
                        "description": "Test strategy for testing behavior",
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
                    name="dataset_with_files",
                    data_processing_strategy="test_strategy",
                    database="test_db",
                    files=[self.file_hash_1, self.file_hash_2],
                ),
                Dataset(
                    name="dataset_single_file",
                    data_processing_strategy="test_strategy",
                    database="test_db",
                    files=[self.file_hash_3],
                ),
                Dataset(
                    name="dataset_empty",
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

        # Mock project config with no datasets
        self.mock_empty_project_config = LlamaFarmConfig(
            version=Version.v1,
            name="empty_project",
            namespace="test_namespace",
            prompts=[
                Prompt(
                    role="system",
                    content="You are a helpful assistant.",
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
                        "description": "Test strategy for testing behavior",
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
    @patch("services.data_service.DataService.get_data_file_metadata_by_hash")
    def test_list_datasets_with_file_details_success(
        self, mock_get_metadata, mock_load_config
    ):
        """Test successful retrieval of datasets with complete file metadata."""
        # Setup mocks
        mock_load_config.return_value = self.mock_project_config

        # Configure metadata lookup to return appropriate metadata for each hash
        def metadata_side_effect(namespace, project_id, file_content_hash):
            if file_content_hash == self.file_hash_1:
                return self.metadata_1
            elif file_content_hash == self.file_hash_2:
                return self.metadata_2
            elif file_content_hash == self.file_hash_3:
                return self.metadata_3
            else:
                raise FileNotFoundError(f"Metadata not found for {file_content_hash}")

        mock_get_metadata.side_effect = metadata_side_effect

        # Execute
        datasets_with_details = DatasetService.list_datasets_with_file_details(
            "test_namespace", "test_project"
        )

        # Verify results
        assert len(datasets_with_details) == 3

        # Check first dataset (2 files)
        dataset_1 = datasets_with_details[0]
        assert isinstance(dataset_1, DatasetWithFileDetails)
        assert dataset_1.name == "dataset_with_files"
        assert dataset_1.data_processing_strategy == "test_strategy"
        assert dataset_1.database == "test_db"
        assert dataset_1.files == [self.file_hash_1, self.file_hash_2]
        assert isinstance(dataset_1.details, DatasetDetails)
        assert len(dataset_1.details.files_metadata) == 2

        # Verify file metadata for first dataset
        file_meta_1 = dataset_1.details.files_metadata[0]
        assert file_meta_1.hash == self.file_hash_1
        assert file_meta_1.original_file_name == "document1.pdf"
        assert file_meta_1.resolved_file_name == "document1_1640995200.0.pdf"
        assert file_meta_1.size == 1024000
        assert file_meta_1.mime_type == "application/pdf"
        assert file_meta_1.timestamp == 1640995200.0

        file_meta_2 = dataset_1.details.files_metadata[1]
        assert file_meta_2.hash == self.file_hash_2
        assert file_meta_2.original_file_name == "data.csv"
        assert file_meta_2.mime_type == "text/csv"

        # Check second dataset (1 file)
        dataset_2 = datasets_with_details[1]
        assert dataset_2.name == "dataset_single_file"
        assert len(dataset_2.details.files_metadata) == 1
        assert dataset_2.details.files_metadata[0].hash == self.file_hash_3

        # Check third dataset (no files)
        dataset_3 = datasets_with_details[2]
        assert dataset_3.name == "dataset_empty"
        assert len(dataset_3.details.files_metadata) == 0

        # Verify metadata service was called correctly
        assert mock_get_metadata.call_count == 3

        # Check that the method was called with the expected file hashes
        called_hashes = []
        for call in mock_get_metadata.call_args_list:
            # Extract file_content_hash from kwargs or args
            if "file_content_hash" in call.kwargs:
                called_hashes.append(call.kwargs["file_content_hash"])
            elif len(call.args) >= 3:
                called_hashes.append(
                    call.args[2]
                )  # file_content_hash is 3rd positional arg

        assert self.file_hash_1 in called_hashes
        assert self.file_hash_2 in called_hashes
        assert self.file_hash_3 in called_hashes

    @patch.object(ProjectService, "load_config")
    @patch("services.data_service.DataService.get_data_file_metadata_by_hash")
    def test_list_datasets_with_file_details_multiple_file_types(
        self, mock_get_metadata, mock_load_config
    ):
        """Test datasets containing multiple files with different metadata types."""
        # Create config with files of various types
        config_with_various_files = self.mock_project_config.model_copy()
        config_with_various_files.datasets = [
            Dataset(
                name="mixed_files_dataset",
                data_processing_strategy="test_strategy",
                database="test_db",
                files=[self.file_hash_1, self.file_hash_2, self.file_hash_3],
            )
        ]

        mock_load_config.return_value = config_with_various_files

        # Configure metadata with different file types
        def metadata_side_effect(namespace, project_id, file_content_hash):
            if file_content_hash == self.file_hash_1:
                return self.metadata_1  # PDF
            elif file_content_hash == self.file_hash_2:
                return self.metadata_2  # CSV
            elif file_content_hash == self.file_hash_3:
                return self.metadata_3  # TXT

        mock_get_metadata.side_effect = metadata_side_effect

        # Execute
        datasets_with_details = DatasetService.list_datasets_with_file_details(
            "test_namespace", "test_project"
        )

        # Verify
        assert len(datasets_with_details) == 1
        dataset = datasets_with_details[0]
        assert len(dataset.details.files_metadata) == 3

        # Verify different file types are handled correctly
        mime_types = [meta.mime_type for meta in dataset.details.files_metadata]
        assert "application/pdf" in mime_types
        assert "text/csv" in mime_types
        assert "text/plain" in mime_types

        # Verify file extensions are preserved
        original_names = [
            meta.original_file_name for meta in dataset.details.files_metadata
        ]
        assert "document1.pdf" in original_names
        assert "data.csv" in original_names
        assert "readme.txt" in original_names

    @patch.object(ProjectService, "load_config")
    def test_list_datasets_with_file_details_empty_datasets(self, mock_load_config):
        """Test datasets with no files (empty files list)."""
        mock_load_config.return_value = self.mock_empty_project_config

        # Execute
        datasets_with_details = DatasetService.list_datasets_with_file_details(
            "test_namespace", "test_project"
        )

        # Verify
        assert len(datasets_with_details) == 0
        assert datasets_with_details == []

    @patch.object(ProjectService, "load_config")
    def test_list_datasets_with_file_details_no_datasets(self, mock_load_config):
        """Test when project has no datasets at all."""
        mock_load_config.return_value = self.mock_empty_project_config

        # Execute
        datasets_with_details = DatasetService.list_datasets_with_file_details(
            "test_namespace", "test_project"
        )

        # Verify
        assert len(datasets_with_details) == 0
        assert isinstance(datasets_with_details, list)

    @patch.object(ProjectService, "load_config")
    @patch("services.data_service.DataService.get_data_file_metadata_by_hash")
    @patch("services.dataset_service.logger")
    def test_list_datasets_with_file_details_missing_metadata(
        self, mock_logger, mock_get_metadata, mock_load_config
    ):
        """Test behavior when metadata file is missing for a file hash."""
        mock_load_config.return_value = self.mock_project_config

        # Configure metadata lookup to fail for second file
        def metadata_side_effect(namespace, project_id, file_content_hash):
            if file_content_hash == self.file_hash_1:
                return self.metadata_1
            elif file_content_hash == self.file_hash_2:
                raise FileNotFoundError(f"Metadata not found for {file_content_hash}")
            elif file_content_hash == self.file_hash_3:
                return self.metadata_3

        mock_get_metadata.side_effect = metadata_side_effect

        # Execute
        datasets_with_details = DatasetService.list_datasets_with_file_details(
            "test_namespace", "test_project"
        )

        # Verify results - should continue processing other files
        assert len(datasets_with_details) == 3

        # First dataset should only have metadata for first file (second file metadata missing)
        dataset_1 = datasets_with_details[0]
        assert len(dataset_1.details.files_metadata) == 1  # Only one file has metadata
        assert dataset_1.details.files_metadata[0].hash == self.file_hash_1

        # Second dataset should have metadata for its file
        dataset_2 = datasets_with_details[1]
        assert len(dataset_2.details.files_metadata) == 1
        assert dataset_2.details.files_metadata[0].hash == self.file_hash_3

        # Third dataset should have no files
        dataset_3 = datasets_with_details[2]
        assert len(dataset_3.details.files_metadata) == 0

        # Verify warning was logged for missing metadata
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "File metadata not found for hash" in warning_call
        assert self.file_hash_2 in warning_call

    def test_file_metadata_fields_completeness(self):
        """Test that all required MetadataFileContent fields are present."""
        metadata = self.metadata_1

        # Verify all required fields exist
        assert hasattr(metadata, "hash")
        assert hasattr(metadata, "original_file_name")
        assert hasattr(metadata, "resolved_file_name")
        assert hasattr(metadata, "size")
        assert hasattr(metadata, "mime_type")
        assert hasattr(metadata, "timestamp")

        # Verify field values are correct types
        assert isinstance(metadata.hash, str)
        assert isinstance(metadata.original_file_name, str)
        assert isinstance(metadata.resolved_file_name, str)
        assert isinstance(metadata.size, int)
        assert isinstance(metadata.mime_type, str)
        assert isinstance(metadata.timestamp, float)

    def test_file_metadata_type_accuracy(self):
        """Test correct data types for all metadata fields."""
        metadata = self.metadata_1

        # Test string fields
        assert isinstance(metadata.hash, str)
        assert len(metadata.hash) == 64  # SHA256 hash length
        assert isinstance(metadata.original_file_name, str)
        assert metadata.original_file_name.endswith(".pdf")
        assert isinstance(metadata.resolved_file_name, str)
        assert isinstance(metadata.mime_type, str)

        # Test numeric fields
        assert isinstance(metadata.size, int)
        assert metadata.size > 0
        assert isinstance(metadata.timestamp, float)
        assert metadata.timestamp > 0

    @patch.object(ProjectService, "load_config")
    @patch("services.data_service.DataService.get_data_file_metadata_by_hash")
    def test_dataset_with_file_details_structure(
        self, mock_get_metadata, mock_load_config
    ):
        """Test DatasetWithFileDetails contains all expected fields."""
        mock_load_config.return_value = self.mock_project_config

        def metadata_side_effect(namespace, project_id, file_content_hash):
            if file_content_hash == self.file_hash_1:
                return self.metadata_1
            elif file_content_hash == self.file_hash_2:
                return self.metadata_2
            elif file_content_hash == self.file_hash_3:
                return self.metadata_3
            else:
                raise FileNotFoundError(f"Metadata not found for {file_content_hash}")

        mock_get_metadata.side_effect = metadata_side_effect

        # Execute
        datasets_with_details = DatasetService.list_datasets_with_file_details(
            "test_namespace", "test_project"
        )

        # Verify structure of DatasetWithFileDetails
        dataset = datasets_with_details[0]

        # Verify it has all original Dataset fields
        assert hasattr(dataset, "name")
        assert hasattr(dataset, "data_processing_strategy")
        assert hasattr(dataset, "database")
        assert hasattr(dataset, "files")

        # Verify it has the additional details field
        assert hasattr(dataset, "details")
        assert isinstance(dataset.details, DatasetDetails)

        # Verify DatasetDetails structure
        assert hasattr(dataset.details, "files_metadata")
        assert isinstance(dataset.details.files_metadata, list)

        # Verify files_metadata contains MetadataFileContent objects
        for file_meta in dataset.details.files_metadata:
            assert isinstance(file_meta, MetadataFileContent)


# Integration test for the full workflow
class TestDatasetServiceExtraDetailsIntegration:
    """Integration tests for DatasetService extra details workflows."""

    @patch.object(ProjectService, "load_config")
    @patch("services.data_service.DataService.get_data_file_metadata_by_hash")
    def test_comprehensive_dataset_metadata_workflow(
        self, mock_get_metadata, mock_load_config
    ):
        """Test complete workflow with multiple datasets and files."""
        # Create comprehensive test data
        file_hash_a = "aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111"
        file_hash_b = "bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222"
        file_hash_c = "cccc3333cccc3333cccc3333cccc3333cccc3333cccc3333cccc3333cccc3333"

        metadata_a = MetadataFileContent(
            original_file_name="large_doc.pdf",
            resolved_file_name="large_doc_1700000000.0.pdf",
            size=5000000,
            mime_type="application/pdf",
            hash=file_hash_a,
            timestamp=1700000000.0,
        )

        metadata_b = MetadataFileContent(
            original_file_name="analysis.xlsx",
            resolved_file_name="analysis_1700000100.0.xlsx",
            size=150000,
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            hash=file_hash_b,
            timestamp=1700000100.0,
        )

        metadata_c = MetadataFileContent(
            original_file_name="notes.md",
            resolved_file_name="notes_1700000200.0.md",
            size=2048,
            mime_type="text/markdown",
            hash=file_hash_c,
            timestamp=1700000200.0,
        )

        # Create complex project config
        complex_config = LlamaFarmConfig(
            version=Version.v1,
            name="complex_project",
            namespace="test_namespace",
            prompts=[
                Prompt(
                    role="system",
                    content="You are a helpful assistant.",
                )
            ],
            rag={
                "databases": [
                    {
                        "name": "primary_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "primary_embedding",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "primary_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "comprehensive_strategy",
                        "description": "Comprehensive strategy for multiple file types",
                        "parsers": [
                            {
                                "type": "PDFParser_LlamaIndex",
                                "config": {},
                                "file_extensions": [".pdf"],
                            },
                            {
                                "type": "CSVParser_LlamaIndex",
                                "config": {},
                                "file_extensions": [".csv", ".xlsx"],
                            },
                        ],
                    }
                ],
            },
            datasets=[
                Dataset(
                    name="documents_dataset",
                    data_processing_strategy="comprehensive_strategy",
                    database="primary_db",
                    files=[file_hash_a, file_hash_c],  # PDF and Markdown
                ),
                Dataset(
                    name="analytics_dataset",
                    data_processing_strategy="comprehensive_strategy",
                    database="primary_db",
                    files=[file_hash_b],  # Excel file
                ),
                Dataset(
                    name="mixed_dataset",
                    data_processing_strategy="comprehensive_strategy",
                    database="primary_db",
                    files=[file_hash_a, file_hash_b, file_hash_c],  # All files
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

        mock_load_config.return_value = complex_config

        # Configure metadata lookup
        def metadata_side_effect(namespace, project_id, file_content_hash):
            if file_content_hash == file_hash_a:
                return metadata_a
            elif file_content_hash == file_hash_b:
                return metadata_b
            elif file_content_hash == file_hash_c:
                return metadata_c
            else:
                raise FileNotFoundError(f"Metadata not found for {file_content_hash}")

        mock_get_metadata.side_effect = metadata_side_effect

        # Execute
        datasets_with_details = DatasetService.list_datasets_with_file_details(
            "test_namespace", "complex_project"
        )

        # Comprehensive verification
        assert len(datasets_with_details) == 3

        # Verify documents_dataset (2 files: PDF + Markdown)
        docs_dataset = next(
            d for d in datasets_with_details if d.name == "documents_dataset"
        )
        assert len(docs_dataset.details.files_metadata) == 2
        file_types = [meta.mime_type for meta in docs_dataset.details.files_metadata]
        assert "application/pdf" in file_types
        assert "text/markdown" in file_types

        # Verify analytics_dataset (1 file: Excel)
        analytics_dataset = next(
            d for d in datasets_with_details if d.name == "analytics_dataset"
        )
        assert len(analytics_dataset.details.files_metadata) == 1
        assert (
            analytics_dataset.details.files_metadata[0].mime_type
            == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert (
            analytics_dataset.details.files_metadata[0].original_file_name
            == "analysis.xlsx"
        )

        # Verify mixed_dataset (3 files: all types)
        mixed_dataset = next(
            d for d in datasets_with_details if d.name == "mixed_dataset"
        )
        assert len(mixed_dataset.details.files_metadata) == 3

        # Verify all unique files are represented
        all_hashes = [meta.hash for meta in mixed_dataset.details.files_metadata]
        assert file_hash_a in all_hashes
        assert file_hash_b in all_hashes
        assert file_hash_c in all_hashes

        # Verify metadata service was called for all unique files
        assert mock_get_metadata.call_count == 6  # 2 + 1 + 3 = 6 total calls
