"""Tests for Files router endpoints (file upload, list, get, delete)."""

import io
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class MockStoredFile:
    """Mock stored file object."""

    def __init__(
        self,
        file_id="file-123",
        filename="test.txt",
        content_type="text/plain",
        size=100,
        page_images=None,
    ):
        self.id = file_id
        self.filename = filename
        self.content_type = content_type
        self.size = size
        self.created_at = "2024-01-01T00:00:00Z"
        self.page_images = page_images


@pytest.fixture
def mock_file_handler():
    """Mock the file handler functions."""
    with (
        patch("routers.files.router.store_file") as mock_store,
        patch("routers.files.router.list_files") as mock_list,
        patch("routers.files.router.get_file") as mock_get,
        patch("routers.files.router.delete_file") as mock_delete,
        patch("routers.files.router.get_file_images") as mock_images,
    ):
        mock_store.return_value = MockStoredFile()
        mock_list.return_value = [
            {
                "id": "file-123",
                "filename": "test.txt",
                "content_type": "text/plain",
                "size": 100,
            }
        ]
        mock_get.return_value = MockStoredFile()
        mock_delete.return_value = True
        mock_images.return_value = ["base64encodedimage1", "base64encodedimage2"]

        yield {
            "store": mock_store,
            "list": mock_list,
            "get": mock_get,
            "delete": mock_delete,
            "images": mock_images,
        }


@pytest.fixture
def test_app(mock_file_handler):
    """Create a test FastAPI app with the files router."""
    from routers.files import router

    app = FastAPI()
    app.include_router(router)

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestFileUploadEndpoint:
    """Test POST /v1/files endpoint."""

    def test_upload_file_success(self, client, mock_file_handler):
        """Test POST /v1/files successfully uploads a file."""
        content = b"Hello, world!"

        response = client.post(
            "/v1/files",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "file-123"
        assert data["object"] == "file"
        assert data["filename"] == "test.txt"

    def test_upload_file_with_pdf_options(self, client, mock_file_handler):
        """Test POST /v1/files with PDF conversion options."""
        content = b"%PDF-1.4 fake pdf content"

        response = client.post(
            "/v1/files",
            files={"file": ("document.pdf", io.BytesIO(content), "application/pdf")},
            data={"convert_pdf": "true", "pdf_dpi": "200"},
        )

        assert response.status_code == 200
        # Verify store_file was called with correct parameters
        mock_file_handler["store"].assert_called_once()

    def test_upload_file_too_large(self, client, mock_file_handler):
        """Test POST /v1/files rejects files that are too large."""
        # Create content larger than MAX_UPLOAD_SIZE
        from routers.files.router import MAX_UPLOAD_SIZE

        content = b"x" * (MAX_UPLOAD_SIZE + 1)

        response = client.post(
            "/v1/files",
            files={
                "file": ("large.bin", io.BytesIO(content), "application/octet-stream")
            },
        )

        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()


class TestFileListEndpoint:
    """Test GET /v1/files endpoint."""

    def test_list_files_success(self, client, mock_file_handler):
        """Test GET /v1/files returns list of uploaded files."""
        response = client.get("/v1/files")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "file-123"

    def test_list_files_empty(self, client, mock_file_handler):
        """Test GET /v1/files returns empty list when no files."""
        mock_file_handler["list"].return_value = []

        response = client.get("/v1/files")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 0


class TestFileGetEndpoint:
    """Test GET /v1/files/{file_id} endpoint."""

    def test_get_file_success(self, client, mock_file_handler):
        """Test GET /v1/files/{file_id} returns file metadata."""
        response = client.get("/v1/files/file-123")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "file-123"
        assert data["object"] == "file"
        assert data["filename"] == "test.txt"

    def test_get_file_not_found(self, client, mock_file_handler):
        """Test GET /v1/files/{file_id} returns 404 for unknown file."""
        mock_file_handler["get"].return_value = None

        response = client.get("/v1/files/unknown-file")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestFileImagesEndpoint:
    """Test GET /v1/files/{file_id}/images endpoint."""

    def test_get_file_images_success(self, client, mock_file_handler):
        """Test GET /v1/files/{file_id}/images returns base64 images."""
        response = client.get("/v1/files/file-123/images")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["file_id"] == "file-123"
        assert len(data["data"]) == 2
        assert data["data"][0]["index"] == 0
        assert "base64" in data["data"][0]

    def test_get_file_images_not_found(self, client, mock_file_handler):
        """Test GET /v1/files/{file_id}/images returns 404 for unknown file."""
        mock_file_handler["get"].return_value = None

        response = client.get("/v1/files/unknown-file/images")

        assert response.status_code == 404

    def test_get_file_images_no_images(self, client, mock_file_handler):
        """Test GET /v1/files/{file_id}/images returns 400 when file has no images."""
        mock_file_handler["images"].return_value = None

        response = client.get("/v1/files/file-123/images")

        assert response.status_code == 400
        assert "cannot be converted" in response.json()["detail"].lower()


class TestFileDeleteEndpoint:
    """Test DELETE /v1/files/{file_id} endpoint."""

    def test_delete_file_success(self, client, mock_file_handler):
        """Test DELETE /v1/files/{file_id} deletes file."""
        response = client.delete("/v1/files/file-123")

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert data["id"] == "file-123"

    def test_delete_file_not_found(self, client, mock_file_handler):
        """Test DELETE /v1/files/{file_id} returns 404 for unknown file."""
        mock_file_handler["delete"].return_value = False

        response = client.delete("/v1/files/unknown-file")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
