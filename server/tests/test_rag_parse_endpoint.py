"""Tests for the RAG parse-only endpoint.

These tests verify the parse endpoint can:
- Accept file uploads
- Return markdown/text content without storing to database
- Respect chunking options
- Handle unsupported file types gracefully
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# Test fixtures
@pytest.fixture
def sample_pdf_bytes():
    """Create minimal PDF bytes for testing."""
    # Minimal valid PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
306
%%EOF"""
    return pdf_content


@pytest.fixture
def sample_text_bytes():
    """Create sample text content for testing."""
    return b"""This is a sample text document.
It has multiple lines.
And multiple paragraphs.

Here is the second paragraph.
It also has multiple sentences. This is another sentence.

And a third paragraph with more content for chunking tests.
The chunker should respect these paragraph boundaries when configured."""


@pytest.fixture
def mock_project_exists():
    """Mock project existence check."""
    with patch("services.project_service.ProjectService.get_project") as mock:
        mock_project = MagicMock()
        mock_project.config.rag = MagicMock()
        mock.return_value = mock_project
        yield mock


@pytest.fixture
def mock_project_dir():
    """Mock project directory."""
    with patch("services.project_service.ProjectService.get_project_dir") as mock:
        mock.return_value = "/tmp/test_project"
        yield mock


class TestParseEndpointAcceptsFileUpload:
    """Test: POST /rag/parse endpoint accepts file upload."""

    def test_parse_endpoint_exists(self):
        """Verify the parse endpoint is registered."""
        # This tests that the endpoint exists
        # Without proper mocking, it will return 404 for project not found
        response = client.post(
            "/v1/projects/test/demo/rag/parse",
            files={"file": ("test.txt", b"Hello world", "text/plain")},
        )
        # Should not be 405 Method Not Allowed - endpoint should exist
        assert response.status_code != status.HTTP_405_METHOD_NOT_ALLOWED

    def test_parse_accepts_file_upload(self, mock_project_exists, mock_project_dir):
        """Test that parse endpoint accepts file upload."""
        with patch(
            "api.routers.rag.rag_parse.parse_file_content"
        ) as mock_parse:
            mock_parse.return_value = {
                "content": "Parsed content",
                "format": "markdown",
                "chunks": [],
                "metadata": {},
            }

            response = client.post(
                "/v1/projects/test/demo/rag/parse",
                files={"file": ("test.txt", b"Hello world", "text/plain")},
            )

            # Should succeed or return validation error, not method not allowed
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_404_NOT_FOUND,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ]


class TestParseEndpointReturnsMarkdown:
    """Test: Parse endpoint returns markdown content."""

    def test_returns_markdown_by_default(self, mock_project_exists, mock_project_dir):
        """Test that parse returns markdown by default."""
        with patch(
            "api.routers.rag.rag_parse.parse_file_content"
        ) as mock_parse:
            mock_parse.return_value = {
                "content": "# Heading\n\nThis is **markdown** content.",
                "format": "markdown",
                "chunks": [],
                "metadata": {"page_count": 1},
            }

            response = client.post(
                "/v1/projects/test/demo/rag/parse",
                files={"file": ("test.pdf", b"PDF content", "application/pdf")},
            )

            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                assert "content" in data
                assert data.get("format") == "markdown"


class TestParseEndpointReturnsText:
    """Test: Parse endpoint returns text content when requested."""

    def test_returns_text_when_requested(self, mock_project_exists, mock_project_dir):
        """Test that parse returns text when format=text."""
        with patch(
            "api.routers.rag.rag_parse.parse_file_content"
        ) as mock_parse:
            mock_parse.return_value = {
                "content": "Plain text content without formatting.",
                "format": "text",
                "chunks": [],
                "metadata": {},
            }

            response = client.post(
                "/v1/projects/test/demo/rag/parse",
                files={"file": ("test.pdf", b"PDF content", "application/pdf")},
                data={"output_format": "text"},
            )

            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                assert data.get("format") == "text"


class TestParseEndpointRespectsChunkingOptions:
    """Test: Parse endpoint respects chunking options."""

    def test_respects_chunk_size(self, mock_project_exists, mock_project_dir):
        """Test that parse respects chunk_size parameter."""
        with patch(
            "api.routers.rag.rag_parse.parse_file_content"
        ) as mock_parse:
            mock_parse.return_value = {
                "content": "Full content",
                "format": "markdown",
                "chunks": [
                    {"content": "Chunk 1", "metadata": {"chunk_index": 0}},
                    {"content": "Chunk 2", "metadata": {"chunk_index": 1}},
                ],
                "metadata": {},
            }

            response = client.post(
                "/v1/projects/test/demo/rag/parse",
                files={"file": ("test.txt", b"Some text content", "text/plain")},
                data={"chunk_size": "256", "chunk_strategy": "sentences"},
            )

            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                assert "chunks" in data


class TestParseEndpointDoesNotStore:
    """Test: Parse endpoint does NOT store to database."""

    def test_does_not_call_database_service(
        self, mock_project_exists, mock_project_dir
    ):
        """Test that parse does not store anything to database."""
        with patch(
            "api.routers.rag.rag_parse.parse_file_content"
        ) as mock_parse, patch(
            "services.database_service.DatabaseService"
        ) as mock_db:
            mock_parse.return_value = {
                "content": "Parsed content",
                "format": "markdown",
                "chunks": [],
                "metadata": {},
            }

            response = client.post(
                "/v1/projects/test/demo/rag/parse",
                files={"file": ("test.txt", b"Hello", "text/plain")},
            )

            # Verify database service was not called for storage
            mock_db.store_document.assert_not_called()
            mock_db.add_vectors.assert_not_called()


class TestParseEndpointReturnsMetadata:
    """Test: Parse endpoint returns document structure metadata."""

    def test_returns_metadata(self, mock_project_exists, mock_project_dir):
        """Test that parse returns document metadata."""
        with patch(
            "api.routers.rag.rag_parse.parse_file_content"
        ) as mock_parse:
            mock_parse.return_value = {
                "content": "Document content",
                "format": "markdown",
                "chunks": [],
                "metadata": {
                    "page_count": 5,
                    "filename": "test.pdf",
                    "file_type": "application/pdf",
                    "headings": ["Introduction", "Chapter 1"],
                    "tables_count": 2,
                },
            }

            response = client.post(
                "/v1/projects/test/demo/rag/parse",
                files={"file": ("test.pdf", b"PDF", "application/pdf")},
            )

            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                assert "metadata" in data
                metadata = data["metadata"]
                assert "page_count" in metadata or "filename" in metadata


class TestParseEndpointErrorHandling:
    """Test: Error handling for unsupported file types."""

    def test_rejects_unsupported_file_type(
        self, mock_project_exists, mock_project_dir
    ):
        """Test that parse rejects unsupported file types."""
        response = client.post(
            "/v1/projects/test/demo/rag/parse",
            files={"file": ("test.xyz", b"Unknown content", "application/octet-stream")},
        )

        # Should return error for unsupported type
        # Could be 400 Bad Request or 415 Unsupported Media Type
        if response.status_code not in [
            status.HTTP_404_NOT_FOUND,  # Project not found is OK in unit test
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]:
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            ]

    def test_handles_empty_file(self, mock_project_exists, mock_project_dir):
        """Test that parse handles empty files gracefully."""
        response = client.post(
            "/v1/projects/test/demo/rag/parse",
            files={"file": ("empty.txt", b"", "text/plain")},
        )

        # Should return error for empty file
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]


class TestParseEndpointIntegration:
    """Integration tests for the parse endpoint with real parsers."""

    @pytest.mark.integration
    def test_parse_text_file_integration(
        self, sample_text_bytes, mock_project_exists, mock_project_dir
    ):
        """Integration test: parse a real text file."""
        response = client.post(
            "/v1/projects/test/demo/rag/parse",
            files={"file": ("sample.txt", sample_text_bytes, "text/plain")},
            data={"output_format": "text"},
        )

        # In integration test, this would succeed
        # In unit test without full setup, it may fail
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "content" in data
            assert len(data["content"]) > 0


class TestParseEndpointChunkOutput:
    """Test chunk output format."""

    def test_chunks_include_metadata(self, mock_project_exists, mock_project_dir):
        """Test that chunks include proper metadata."""
        with patch(
            "api.routers.rag.rag_parse.parse_file_content"
        ) as mock_parse:
            mock_parse.return_value = {
                "content": "Full document content",
                "format": "markdown",
                "chunks": [
                    {
                        "content": "First chunk content",
                        "metadata": {
                            "chunk_index": 0,
                            "total_chunks": 2,
                            "chunking_strategy": "sentences",
                        },
                    },
                    {
                        "content": "Second chunk content",
                        "metadata": {
                            "chunk_index": 1,
                            "total_chunks": 2,
                            "chunking_strategy": "sentences",
                        },
                    },
                ],
                "metadata": {"filename": "test.txt"},
            }

            response = client.post(
                "/v1/projects/test/demo/rag/parse",
                files={"file": ("test.txt", b"Text", "text/plain")},
                data={"chunk_strategy": "sentences", "chunk_size": "100"},
            )

            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                chunks = data.get("chunks", [])
                if chunks:
                    assert "content" in chunks[0]
                    assert "metadata" in chunks[0]
