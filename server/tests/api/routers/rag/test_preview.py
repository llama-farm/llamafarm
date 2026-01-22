"""API endpoint tests for document preview - TDD Red Phase.

All tests written FIRST and will fail until implementation is complete.
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api.main import llama_farm_api


def _client() -> TestClient:
    """Create test client."""
    app = llama_farm_api()
    return TestClient(app)


class TestPreviewEndpoint:
    """API endpoint tests for document preview."""

    def test_preview_endpoint_returns_200(self, mocker):
        """POST /rag/databases/{db}/preview returns 200."""
        # Mock ProjectService
        mock_database = Mock()
        mock_database.name = "default"

        mock_rag_config = Mock()
        mock_rag_config.databases = [mock_database]

        mock_project = Mock()
        mock_project.config.rag = mock_rag_config

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        # Mock DatabaseService.get_database to return the mock database
        mock_db_service = mocker.patch("api.routers.rag.preview.DatabaseService")
        mock_db_service.get_database.return_value = mock_database

        # Mock the preview handler - must match PreviewResult.to_dict() format
        mock_preview_response = {
            "original_text": "This is the document content.",
            "chunks": [
                {
                    "chunk_index": 0,
                    "content": "This is the document",
                    "start_position": 0,
                    "end_position": 20,
                    "char_count": 20,
                    "word_count": 4,
                }
            ],
            "file_info": {
                "filename": "test.txt",
                "size": 30,
                "content_type": "text/plain",
            },
            "parser_used": "TextParser_Python",
            "chunk_strategy": "characters",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "total_chunks": 1,
            "avg_chunk_size": 20.0,
            "total_size_with_overlaps": 20,
            "warnings": [],
        }

        mock_handle_preview = mocker.patch("api.routers.rag.preview.handle_preview")
        mock_handle_preview.return_value = mock_preview_response

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/default/preview",
            json={"file_hash": "abc123"},
        )

        assert resp.status_code == 200

    def test_preview_response_structure(self, mocker):
        """Response contains required fields."""
        mock_database = Mock()
        mock_database.name = "default"

        mock_rag_config = Mock()
        mock_rag_config.databases = [mock_database]

        mock_project = Mock()
        mock_project.config.rag = mock_rag_config

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        # Mock DatabaseService.get_database to return the mock database
        mock_db_service = mocker.patch("api.routers.rag.preview.DatabaseService")
        mock_db_service.get_database.return_value = mock_database

        mock_preview_response = {
            "original_text": "Test content for preview.",
            "chunks": [
                {
                    "chunk_index": 0,
                    "content": "Test content",
                    "start_position": 0,
                    "end_position": 12,
                    "char_count": 12,
                    "word_count": 2,
                }
            ],
            "file_info": {
                "filename": "test.txt",
                "size": 25,
                "content_type": "text/plain",
            },
            "parser_used": "TextParser_Python",
            "chunk_strategy": "characters",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "total_chunks": 1,
            "avg_chunk_size": 12.0,
            "total_size_with_overlaps": 12,
            "warnings": [],
        }

        mock_handle_preview = mocker.patch("api.routers.rag.preview.handle_preview")
        mock_handle_preview.return_value = mock_preview_response

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/default/preview",
            json={"file_hash": "abc123"},
        )

        assert resp.status_code == 200
        data = resp.json()

        # Verify required fields
        assert "original_text" in data
        assert "chunks" in data
        assert "total_chunks" in data
        assert "chunk_size" in data
        assert "chunk_overlap" in data
        assert "parser_used" in data
        assert "avg_chunk_size" in data

    def test_preview_with_file_upload(self, mocker):
        """Preview works with base64 file upload."""
        import base64

        mock_database = Mock()
        mock_database.name = "default"

        mock_rag_config = Mock()
        mock_rag_config.databases = [mock_database]

        mock_project = Mock()
        mock_project.config.rag = mock_rag_config

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        # Mock DatabaseService.get_database to return the mock database
        mock_db_service = mocker.patch("api.routers.rag.preview.DatabaseService")
        mock_db_service.get_database.return_value = mock_database

        mock_preview_response = {
            "original_text": "Uploaded content",
            "chunks": [
                {
                    "chunk_index": 0,
                    "content": "Uploaded content",
                    "start_position": 0,
                    "end_position": 16,
                    "char_count": 16,
                    "word_count": 2,
                }
            ],
            "file_info": {
                "filename": "uploaded.txt",
                "size": 16,
                "content_type": "text/plain",
            },
            "parser_used": "TextParser_Python",
            "chunk_strategy": "characters",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "total_chunks": 1,
            "avg_chunk_size": 16.0,
            "total_size_with_overlaps": 16,
            "warnings": [],
        }

        mock_handle_preview = mocker.patch("api.routers.rag.preview.handle_preview")
        mock_handle_preview.return_value = mock_preview_response

        # Encode file content as base64
        file_content = base64.b64encode(b"Uploaded content").decode("utf-8")

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/default/preview",
            json={
                "file_content": file_content,
                "filename": "uploaded.txt",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "uploaded.txt"

    def test_preview_with_overrides(self, mocker):
        """Chunk setting overrides are applied."""
        mock_database = Mock()
        mock_database.name = "default"

        mock_rag_config = Mock()
        mock_rag_config.databases = [mock_database]

        mock_project = Mock()
        mock_project.config.rag = mock_rag_config

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        # Mock DatabaseService.get_database to return the mock database
        mock_db_service = mocker.patch("api.routers.rag.preview.DatabaseService")
        mock_db_service.get_database.return_value = mock_database

        # Response with overridden settings
        mock_preview_response = {
            "original_text": "Test",
            "chunks": [],
            "file_info": {
                "filename": "test.txt",
                "size": 4,
                "content_type": "text/plain",
            },
            "parser_used": "TextParser_Python",
            "chunk_strategy": "sentences",  # Overridden
            "chunk_size": 1000,  # Overridden
            "chunk_overlap": 200,  # Overridden
            "total_chunks": 0,
            "avg_chunk_size": 0.0,
            "total_size_with_overlaps": 0,
            "warnings": [],
        }

        mock_handle_preview = mocker.patch("api.routers.rag.preview.handle_preview")
        mock_handle_preview.return_value = mock_preview_response

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/default/preview",
            json={
                "file_hash": "abc123",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "chunk_strategy": "sentences",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["chunk_size"] == 1000
        assert data["chunk_overlap"] == 200
        assert data["chunk_strategy"] == "sentences"

    def test_preview_invalid_database_404(self, mocker):
        """Non-existent database returns 404."""
        from api.errors import DatabaseNotFoundError

        mock_database = Mock()
        mock_database.name = "existing_db"

        mock_rag_config = Mock()
        mock_rag_config.databases = [mock_database]

        mock_project = Mock()
        mock_project.config.rag = mock_rag_config

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        # Mock DatabaseService.get_database to raise DatabaseNotFoundError
        mock_db_service = mocker.patch("api.routers.rag.preview.DatabaseService")
        mock_db_service.get_database.side_effect = DatabaseNotFoundError("nonexistent_db")

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/nonexistent_db/preview",
            json={"file_hash": "abc123"},
        )

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_preview_invalid_file_400(self, mocker):
        """Invalid file reference returns 400."""
        mock_database = Mock()
        mock_database.name = "default"

        mock_rag_config = Mock()
        mock_rag_config.databases = [mock_database]

        mock_project = Mock()
        mock_project.config.rag = mock_rag_config

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        # Mock DatabaseService.get_database to return the mock database
        mock_db_service = mocker.patch("api.routers.rag.preview.DatabaseService")
        mock_db_service.get_database.return_value = mock_database

        # Mock preview handler to raise an error for invalid file
        mock_handle_preview = mocker.patch("api.routers.rag.preview.handle_preview")
        mock_handle_preview.side_effect = ValueError("File not found: invalid_hash")

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/default/preview",
            json={"file_hash": "invalid_hash"},
        )

        assert resp.status_code == 400

    def test_preview_requires_file_reference(self, mocker):
        """Request must include either file_hash or file_content."""
        mock_database = Mock()
        mock_database.name = "default"

        mock_rag_config = Mock()
        mock_rag_config.databases = [mock_database]

        mock_project = Mock()
        mock_project.config.rag = mock_rag_config

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/default/preview",
            json={},  # No file reference
        )

        # Should return 422 (validation error) or 400
        assert resp.status_code in [400, 422]

    def test_preview_rag_not_configured_400(self, mocker):
        """Returns 400 when RAG is not configured."""
        mock_project = Mock()
        mock_project.config.rag = None

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/default/preview",
            json={"file_hash": "abc123"},
        )

        assert resp.status_code == 400
        assert "RAG not configured" in resp.json()["detail"]

    def test_preview_project_not_found_404(self, mocker):
        """Returns 404 when project doesn't exist."""
        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.side_effect = FileNotFoundError(
            "Project not found"
        )

        client = _client()
        resp = client.post(
            "/v1/projects/nonexistent/project/rag/databases/default/preview",
            json={"file_hash": "abc123"},
        )

        assert resp.status_code == 404

    def test_preview_chunk_statistics(self, mocker):
        """Response includes chunk statistics like RAGPlay."""
        mock_database = Mock()
        mock_database.name = "default"

        mock_rag_config = Mock()
        mock_rag_config.databases = [mock_database]

        mock_project = Mock()
        mock_project.config.rag = mock_rag_config

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        # Mock DatabaseService.get_database to return the mock database
        mock_db_service = mocker.patch("api.routers.rag.preview.DatabaseService")
        mock_db_service.get_database.return_value = mock_database

        mock_preview_response = {
            "original_text": "A" * 100 + "B" * 100 + "C" * 100,
            "chunks": [
                {"chunk_index": 0, "content": "A" * 100, "start_position": 0, "end_position": 100, "char_count": 100, "word_count": 1},
                {"chunk_index": 1, "content": "B" * 100, "start_position": 90, "end_position": 190, "char_count": 100, "word_count": 1},
                {"chunk_index": 2, "content": "C" * 100, "start_position": 180, "end_position": 280, "char_count": 100, "word_count": 1},
            ],
            "file_info": {
                "filename": "test.txt",
                "size": 300,
                "content_type": "text/plain",
            },
            "parser_used": "TextParser_Python",
            "chunk_strategy": "characters",
            "chunk_size": 100,
            "chunk_overlap": 10,
            "total_chunks": 3,
            "avg_chunk_size": 100.0,
            "total_size_with_overlaps": 300,
            "warnings": [],
        }

        mock_handle_preview = mocker.patch("api.routers.rag.preview.handle_preview")
        mock_handle_preview.return_value = mock_preview_response

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/default/preview",
            json={"file_hash": "abc123"},
        )

        assert resp.status_code == 200
        data = resp.json()

        # Verify statistics fields (like RAGPlay)
        assert "total_chunks" in data
        assert data["total_chunks"] == 3
        assert "avg_chunk_size" in data
        assert data["avg_chunk_size"] == 100.0
        assert "total_size_with_overlaps" in data
        assert data["total_size_with_overlaps"] == 300


class TestPreviewRequestValidation:
    """Tests for request model validation."""

    def test_request_with_dataset_id_and_file_hash(self, mocker):
        """Request can specify dataset_id and file_hash."""
        mock_database = Mock()
        mock_database.name = "default"

        mock_rag_config = Mock()
        mock_rag_config.databases = [mock_database]

        mock_project = Mock()
        mock_project.config.rag = mock_rag_config

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        # Mock DatabaseService.get_database to return the mock database
        mock_db_service = mocker.patch("api.routers.rag.preview.DatabaseService")
        mock_db_service.get_database.return_value = mock_database

        mock_preview_response = {
            "original_text": "Test",
            "chunks": [],
            "file_info": {
                "filename": "test.txt",
                "size": 4,
                "content_type": "text/plain",
            },
            "parser_used": "TextParser_Python",
            "chunk_strategy": "characters",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "total_chunks": 0,
            "avg_chunk_size": 0.0,
            "total_size_with_overlaps": 0,
            "warnings": [],
        }

        mock_handle_preview = mocker.patch("api.routers.rag.preview.handle_preview")
        mock_handle_preview.return_value = mock_preview_response

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/default/preview",
            json={
                "dataset_id": "my-dataset",
                "file_hash": "abc123def456",
            },
        )

        assert resp.status_code == 200

        # Verify the handler was called with the dataset context
        mock_handle_preview.assert_called_once()

    def test_chunk_size_override_validation(self, mocker):
        """Chunk size override must be positive."""
        mock_database = Mock()
        mock_database.name = "default"

        mock_rag_config = Mock()
        mock_rag_config.databases = [mock_database]

        mock_project = Mock()
        mock_project.config.rag = mock_rag_config

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/default/preview",
            json={
                "file_hash": "abc123",
                "chunk_size": -100,  # Invalid
            },
        )

        # Should fail validation
        assert resp.status_code == 422

    def test_chunk_overlap_must_be_less_than_chunk_size(self, mocker):
        """Chunk overlap must be less than chunk size."""
        mock_database = Mock()
        mock_database.name = "default"

        mock_rag_config = Mock()
        mock_rag_config.databases = [mock_database]

        mock_project = Mock()
        mock_project.config.rag = mock_rag_config

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        # Mock handler to validate internally
        mock_handle_preview = mocker.patch("api.routers.rag.preview.handle_preview")
        mock_handle_preview.side_effect = ValueError(
            "chunk_overlap must be less than chunk_size"
        )

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/default/preview",
            json={
                "file_hash": "abc123",
                "chunk_size": 100,
                "chunk_overlap": 150,  # Greater than chunk_size
            },
        )

        # FastAPI/Pydantic returns 422 for validation errors
        assert resp.status_code == 422

    def test_valid_chunk_strategy_values(self, mocker):
        """Chunk strategy must be a valid value."""
        mock_database = Mock()
        mock_database.name = "default"

        mock_rag_config = Mock()
        mock_rag_config.databases = [mock_database]

        mock_project = Mock()
        mock_project.config.rag = mock_rag_config

        mock_project_service = mocker.patch("api.routers.rag.preview.ProjectService")
        mock_project_service.get_project.return_value = mock_project
        mock_project_service.get_project_dir.return_value = "/fake/path"

        client = _client()
        resp = client.post(
            "/v1/projects/test-ns/test-project/rag/databases/default/preview",
            json={
                "file_hash": "abc123",
                "chunk_strategy": "invalid_strategy",
            },
        )

        # Should fail validation
        assert resp.status_code == 422
