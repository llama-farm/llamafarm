"""
Tests for datasets API extra details functionality.

This module contains comprehensive tests for the datasets API endpoints
focusing on the include_extra_details parameter and file metadata handling.
"""

from unittest.mock import patch, Mock

import pytest
from fastapi.testclient import TestClient
from config.datamodel import (
    Dataset,
    LlamaFarmConfig,
    Prompt,
    Provider,
    Runtime,
    Version,
    Model,
)

from main import app
from services.dataset_service import DatasetService, DatasetWithFileDetails
from services.data_service import MetadataFileContent
from services.project_service import ProjectService

# Configure pytest for async tests
pytest_plugins = ("pytest_asyncio",)

client = TestClient(app)


class TestDatasetsAPIExtraDetails:
    """Test cases for datasets API extra details functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Sample file hashes and metadata
        self.file_hash_1 = (
            "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        )
        self.file_hash_2 = (
            "fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321"
        )

        # Sample metadata objects
        self.metadata_1 = MetadataFileContent(
            original_file_name="api_test_doc.pdf",
            resolved_file_name="api_test_doc_1640995200.0.pdf",
            size=2048000,
            mime_type="application/pdf",
            hash=self.file_hash_1,
            timestamp=1640995200.0,
        )

        self.metadata_2 = MetadataFileContent(
            original_file_name="api_test_data.csv",
            resolved_file_name="api_test_data_1640995300.0.csv",
            size=4096,
            mime_type="text/csv",
            hash=self.file_hash_2,
            timestamp=1640995300.0,
        )

        # Mock project config for API tests
        self.mock_project_config = LlamaFarmConfig(
            version=Version.v1,
            name="api_test_project",
            namespace="api_test_namespace",
            prompts=[
                Prompt(
                    role="system",
                    content="You are a helpful assistant.",
                )
            ],
            rag={
                "databases": [
                    {
                        "name": "api_test_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "api_test_embedding",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "api_test_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "api_test_strategy",
                        "description": "API test strategy",
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
                    name="api_dataset_with_files",
                    data_processing_strategy="api_test_strategy",
                    database="api_test_db",
                    files=[self.file_hash_1, self.file_hash_2],
                ),
                Dataset(
                    name="api_dataset_empty",
                    data_processing_strategy="api_test_strategy",
                    database="api_test_db",
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
            name="api_empty_project",
            namespace="api_test_namespace",
            prompts=[
                Prompt(
                    role="system",
                    content="You are a helpful assistant.",
                )
            ],
            rag={
                "databases": [
                    {
                        "name": "api_test_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "api_test_embedding",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "api_test_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "api_test_strategy",
                        "description": "API test strategy",
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
    def test_list_datasets_include_extra_details_true(
        self, mock_get_metadata, mock_load_config
    ):
        """Test API with include_extra_details=True (default)."""
        # Setup mocks
        mock_load_config.return_value = self.mock_project_config

        def metadata_side_effect(namespace, project_id, file_content_hash):
            if file_content_hash == self.file_hash_1:
                return self.metadata_1
            elif file_content_hash == self.file_hash_2:
                return self.metadata_2
            else:
                raise FileNotFoundError(f"Metadata not found for {file_content_hash}")

        mock_get_metadata.side_effect = metadata_side_effect

        # Execute API call with include_extra_details=True
        response = client.get(
            "/v1/projects/api_test_namespace/api_test_project/datasets/",
            params={"include_extra_details": True},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "total" in data
        assert "datasets" in data
        assert data["total"] == 2
        assert len(data["datasets"]) == 2

        # Verify first dataset has detailed information
        dataset_with_files = next(
            d for d in data["datasets"] if d["name"] == "api_dataset_with_files"
        )
        assert "details" in dataset_with_files
        assert "files_metadata" in dataset_with_files["details"]
        assert len(dataset_with_files["details"]["files_metadata"]) == 2

        # Verify file metadata structure
        file_meta = dataset_with_files["details"]["files_metadata"][0]
        assert "hash" in file_meta
        assert "original_file_name" in file_meta
        assert "resolved_file_name" in file_meta
        assert "size" in file_meta
        assert "mime_type" in file_meta
        assert "timestamp" in file_meta

        # Verify metadata values
        if file_meta["hash"] == self.file_hash_1:
            assert file_meta["original_file_name"] == "api_test_doc.pdf"
            assert file_meta["mime_type"] == "application/pdf"
            assert file_meta["size"] == 2048000

        # Verify second dataset (empty)
        dataset_empty = next(
            d for d in data["datasets"] if d["name"] == "api_dataset_empty"
        )
        assert "details" in dataset_empty
        assert "files_metadata" in dataset_empty["details"]
        assert len(dataset_empty["details"]["files_metadata"]) == 0

    @patch.object(ProjectService, "load_config")
    def test_list_datasets_include_extra_details_false(self, mock_load_config):
        """Test API with include_extra_details=False."""
        mock_load_config.return_value = self.mock_project_config

        # Execute API call with include_extra_details=False
        response = client.get(
            "/v1/projects/api_test_namespace/api_test_project/datasets/",
            params={"include_extra_details": False},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "total" in data
        assert "datasets" in data
        assert data["total"] == 2
        assert len(data["datasets"]) == 2

        # Verify datasets contain basic information only (no details field)
        for dataset in data["datasets"]:
            assert "name" in dataset
            assert "data_processing_strategy" in dataset
            assert "database" in dataset
            assert "files" in dataset
            assert "details" not in dataset  # Key verification: no details field

        # Verify specific dataset content
        dataset_with_files = next(
            d for d in data["datasets"] if d["name"] == "api_dataset_with_files"
        )
        assert dataset_with_files["files"] == [self.file_hash_1, self.file_hash_2]
        assert dataset_with_files["data_processing_strategy"] == "api_test_strategy"
        assert dataset_with_files["database"] == "api_test_db"

    @patch.object(ProjectService, "load_config")
    @patch("services.data_service.DataService.get_data_file_metadata_by_hash")
    def test_list_datasets_parameter_default_behavior(
        self, mock_get_metadata, mock_load_config
    ):
        """Test API without specifying include_extra_details parameter (should default to True)."""
        # Setup mocks
        mock_load_config.return_value = self.mock_project_config

        def metadata_side_effect(namespace, project_id, file_content_hash):
            if file_content_hash == self.file_hash_1:
                return self.metadata_1
            elif file_content_hash == self.file_hash_2:
                return self.metadata_2
            else:
                raise FileNotFoundError(f"Metadata not found for {file_content_hash}")

        mock_get_metadata.side_effect = metadata_side_effect

        # Execute API call without include_extra_details parameter
        response = client.get(
            "/v1/projects/api_test_namespace/api_test_project/datasets/"
        )

        # Verify response (should behave like include_extra_details=True)
        assert response.status_code == 200
        data = response.json()

        # Verify default behavior includes details
        assert "total" in data
        assert "datasets" in data
        assert len(data["datasets"]) > 0

        # Verify that details are included by default
        dataset_with_files = next(
            d for d in data["datasets"] if d["name"] == "api_dataset_with_files"
        )
        assert "details" in dataset_with_files
        assert "files_metadata" in dataset_with_files["details"]

    @patch.object(ProjectService, "load_config")
    @patch("services.data_service.DataService.get_data_file_metadata_by_hash")
    def test_api_response_structure_with_details(
        self, mock_get_metadata, mock_load_config
    ):
        """Test API response structure when include_extra_details=True."""
        mock_load_config.return_value = self.mock_project_config

        def metadata_side_effect(namespace, project_id, file_content_hash):
            if file_content_hash == self.file_hash_1:
                return self.metadata_1
            elif file_content_hash == self.file_hash_2:
                return self.metadata_2
            else:
                raise FileNotFoundError(f"Metadata not found for {file_content_hash}")

        mock_get_metadata.side_effect = metadata_side_effect

        response = client.get(
            "/v1/projects/api_test_namespace/api_test_project/datasets/",
            params={"include_extra_details": True},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify top-level structure
        assert isinstance(data, dict)
        assert set(data.keys()) == {"total", "datasets"}
        assert isinstance(data["total"], int)
        assert isinstance(data["datasets"], list)

        # Verify dataset structure with details
        if len(data["datasets"]) > 0:
            dataset = data["datasets"][0]
            expected_keys = {
                "name",
                "data_processing_strategy",
                "database",
                "files",
                "details",
            }
            assert set(dataset.keys()) == expected_keys

            # Verify details structure
            details = dataset["details"]
            assert isinstance(details, dict)
            assert "files_metadata" in details
            assert isinstance(details["files_metadata"], list)

            # Verify file metadata structure if files exist
            if len(details["files_metadata"]) > 0:
                file_meta = details["files_metadata"][0]
                expected_meta_keys = {
                    "hash",
                    "original_file_name",
                    "resolved_file_name",
                    "size",
                    "mime_type",
                    "timestamp",
                }
                assert set(file_meta.keys()) == expected_meta_keys

    @patch.object(ProjectService, "load_config")
    def test_api_response_structure_without_details(self, mock_load_config):
        """Test API response structure when include_extra_details=False."""
        mock_load_config.return_value = self.mock_project_config

        response = client.get(
            "/v1/projects/api_test_namespace/api_test_project/datasets/",
            params={"include_extra_details": False},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify top-level structure
        assert isinstance(data, dict)
        assert set(data.keys()) == {"total", "datasets"}
        assert isinstance(data["total"], int)
        assert isinstance(data["datasets"], list)

        # Verify dataset structure without details
        if len(data["datasets"]) > 0:
            dataset = data["datasets"][0]
            expected_keys = {"name", "data_processing_strategy", "database", "files"}
            assert set(dataset.keys()) == expected_keys
            assert "details" not in dataset

    @patch.object(ProjectService, "load_config")
    @patch("services.data_service.DataService.get_data_file_metadata_by_hash")
    def test_api_file_metadata_serialization(self, mock_get_metadata, mock_load_config):
        """Test that MetadataFileContent serializes correctly in API response."""
        mock_load_config.return_value = self.mock_project_config
        mock_get_metadata.return_value = self.metadata_1

        response = client.get(
            "/v1/projects/api_test_namespace/api_test_project/datasets/",
            params={"include_extra_details": True},
        )

        assert response.status_code == 200
        data = response.json()

        # Find dataset with files
        dataset_with_files = next(
            d for d in data["datasets"] if len(d["details"]["files_metadata"]) > 0
        )

        # Verify metadata serialization
        file_meta = dataset_with_files["details"]["files_metadata"][0]

        # Verify all fields are present and correctly typed in JSON
        assert isinstance(file_meta["hash"], str)
        assert isinstance(file_meta["original_file_name"], str)
        assert isinstance(file_meta["resolved_file_name"], str)
        assert isinstance(file_meta["size"], int)
        assert isinstance(file_meta["mime_type"], str)
        assert isinstance(file_meta["timestamp"], float)

        # Verify specific values
        assert file_meta["hash"] == self.metadata_1.hash
        assert file_meta["original_file_name"] == self.metadata_1.original_file_name
        assert file_meta["resolved_file_name"] == self.metadata_1.resolved_file_name
        assert file_meta["size"] == self.metadata_1.size
        assert file_meta["mime_type"] == self.metadata_1.mime_type
        assert file_meta["timestamp"] == self.metadata_1.timestamp

    @patch.object(ProjectService, "load_config")
    def test_list_datasets_empty_project_with_details(self, mock_load_config):
        """Test API with include_extra_details=True on empty project."""
        mock_load_config.return_value = self.mock_empty_project_config

        response = client.get(
            "/v1/projects/api_test_namespace/api_empty_project/datasets/",
            params={"include_extra_details": True},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 0
        assert data["datasets"] == []

    @patch.object(ProjectService, "load_config")
    def test_list_datasets_empty_project_without_details(self, mock_load_config):
        """Test API with include_extra_details=False on empty project."""
        mock_load_config.return_value = self.mock_empty_project_config

        response = client.get(
            "/v1/projects/api_test_namespace/api_empty_project/datasets/",
            params={"include_extra_details": False},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 0
        assert data["datasets"] == []

    @patch.object(ProjectService, "load_config")
    @patch("services.data_service.DataService.get_data_file_metadata_by_hash")
    @patch("services.dataset_service.logger")
    def test_api_handles_metadata_errors_gracefully(
        self, mock_logger, mock_get_metadata, mock_load_config
    ):
        """Test that API handles metadata retrieval errors gracefully."""
        mock_load_config.return_value = self.mock_project_config

        # Configure metadata lookup to fail for some files
        def metadata_side_effect(namespace, project_id, file_content_hash):
            if file_content_hash == self.file_hash_1:
                return self.metadata_1
            elif file_content_hash == self.file_hash_2:
                raise FileNotFoundError(f"Metadata not found for {file_content_hash}")

        mock_get_metadata.side_effect = metadata_side_effect

        response = client.get(
            "/v1/projects/api_test_namespace/api_test_project/datasets/",
            params={"include_extra_details": True},
        )

        # API should still return 200 despite metadata errors
        assert response.status_code == 200
        data = response.json()

        # Should have datasets in response
        assert data["total"] == 2
        assert len(data["datasets"]) == 2

        # Dataset with files should only have metadata for successful lookups
        dataset_with_files = next(
            d for d in data["datasets"] if d["name"] == "api_dataset_with_files"
        )
        # Should only have 1 file metadata (for file_hash_1) since file_hash_2 failed
        assert len(dataset_with_files["details"]["files_metadata"]) == 1
        assert (
            dataset_with_files["details"]["files_metadata"][0]["hash"]
            == self.file_hash_1
        )

        # Verify warning was logged
        mock_logger.warning.assert_called()


# Integration tests for end-to-end workflows
class TestDatasetsAPIExtraDetailsIntegration:
    """Integration tests for datasets API extra details workflows."""

    @patch.object(ProjectService, "load_config")
    @patch("services.data_service.DataService.get_data_file_metadata_by_hash")
    def test_complex_api_workflow_with_multiple_datasets(
        self, mock_get_metadata, mock_load_config
    ):
        """Test complex API workflow with multiple datasets and various file types."""
        # Create complex test scenario
        hash_pdf = "pdf1234567890abcdefpdf1234567890abcdefpdf1234567890abcdefpdf123"
        hash_csv = "csv1234567890abcdefcsv1234567890abcdefcsv1234567890abcdefcsv123"
        hash_txt = "txt1234567890abcdeftxt1234567890abcdeftxt1234567890abcdeftxt123"

        metadata_pdf = MetadataFileContent(
            original_file_name="complex_document.pdf",
            resolved_file_name="complex_document_1700000000.0.pdf",
            size=10000000,
            mime_type="application/pdf",
            hash=hash_pdf,
            timestamp=1700000000.0,
        )

        metadata_csv = MetadataFileContent(
            original_file_name="complex_data.csv",
            resolved_file_name="complex_data_1700000100.0.csv",
            size=500000,
            mime_type="text/csv",
            hash=hash_csv,
            timestamp=1700000100.0,
        )

        metadata_txt = MetadataFileContent(
            original_file_name="complex_notes.txt",
            resolved_file_name="complex_notes_1700000200.0.txt",
            size=1024,
            mime_type="text/plain",
            hash=hash_txt,
            timestamp=1700000200.0,
        )

        complex_config = LlamaFarmConfig(
            version=Version.v1,
            name="complex_api_project",
            namespace="complex_api_namespace",
            prompts=[
                Prompt(
                    role="system",
                    content="You are a helpful assistant.",
                )
            ],
            rag={
                "databases": [
                    {
                        "name": "complex_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "complex_embedding",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "complex_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "complex_strategy",
                        "description": "Complex multi-format strategy",
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
                    name="documents",
                    data_processing_strategy="complex_strategy",
                    database="complex_db",
                    files=[hash_pdf, hash_txt],
                ),
                Dataset(
                    name="data",
                    data_processing_strategy="complex_strategy",
                    database="complex_db",
                    files=[hash_csv],
                ),
                Dataset(
                    name="everything",
                    data_processing_strategy="complex_strategy",
                    database="complex_db",
                    files=[hash_pdf, hash_csv, hash_txt],
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

        def metadata_side_effect(namespace, project_id, file_content_hash):
            if file_content_hash == hash_pdf:
                return metadata_pdf
            elif file_content_hash == hash_csv:
                return metadata_csv
            elif file_content_hash == hash_txt:
                return metadata_txt
            else:
                raise FileNotFoundError(f"Metadata not found for {file_content_hash}")

        mock_get_metadata.side_effect = metadata_side_effect

        # Test with details
        response_with_details = client.get(
            "/v1/projects/complex_api_namespace/complex_api_project/datasets/",
            params={"include_extra_details": True},
        )

        assert response_with_details.status_code == 200
        data_with_details = response_with_details.json()

        # Verify comprehensive response
        assert data_with_details["total"] == 3
        assert len(data_with_details["datasets"]) == 3

        # Verify each dataset has correct file metadata
        documents_dataset = next(
            d for d in data_with_details["datasets"] if d["name"] == "documents"
        )
        assert len(documents_dataset["details"]["files_metadata"]) == 2

        data_dataset = next(
            d for d in data_with_details["datasets"] if d["name"] == "data"
        )
        assert len(data_dataset["details"]["files_metadata"]) == 1
        assert data_dataset["details"]["files_metadata"][0]["mime_type"] == "text/csv"

        everything_dataset = next(
            d for d in data_with_details["datasets"] if d["name"] == "everything"
        )
        assert len(everything_dataset["details"]["files_metadata"]) == 3

        # Test without details for comparison
        response_without_details = client.get(
            "/v1/projects/complex_api_namespace/complex_api_project/datasets/",
            params={"include_extra_details": False},
        )

        assert response_without_details.status_code == 200
        data_without_details = response_without_details.json()

        # Verify basic response structure
        assert data_without_details["total"] == 3
        assert len(data_without_details["datasets"]) == 3

        # Verify no details in response
        for dataset in data_without_details["datasets"]:
            assert "details" not in dataset
            assert "files" in dataset

        # Verify that file hashes are still present in basic response
        everything_basic = next(
            d for d in data_without_details["datasets"] if d["name"] == "everything"
        )
        assert len(everything_basic["files"]) == 3
        assert hash_pdf in everything_basic["files"]
        assert hash_csv in everything_basic["files"]
        assert hash_txt in everything_basic["files"]
