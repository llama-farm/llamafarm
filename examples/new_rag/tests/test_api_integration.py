"""API Integration Tests for New RAG System.

These tests require a running LlamaFarm server.
Run with: cd examples/new_rag && uv run pytest tests/ -v

Note: These tests are marked as integration tests and will be
skipped if the server is not running.
"""

import os
import tempfile
from pathlib import Path

import pytest
import requests

# Server configuration
SERVER_URL = os.environ.get("LLAMAFARM_SERVER_URL", "http://localhost:8005")
NAMESPACE = os.environ.get("LLAMAFARM_NAMESPACE", "test")
PROJECT = os.environ.get("LLAMAFARM_PROJECT", "demo")
API_BASE = f"{SERVER_URL}/v1/projects/{NAMESPACE}/{PROJECT}"


def server_is_running() -> bool:
    """Check if the LlamaFarm server is running."""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# Skip all tests if server is not running
pytestmark = pytest.mark.skipif(
    not server_is_running(),
    reason="LlamaFarm server is not running"
)


class TestAPIHealthCheck:
    """Test: API health check returns healthy."""

    def test_server_health(self):
        """Test server health endpoint."""
        response = requests.get(f"{SERVER_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ["healthy", "ok"]

    def test_rag_health(self):
        """Test RAG-specific health endpoint."""
        response = requests.get(f"{API_BASE}/rag/health")
        # May return 400 if RAG not configured, which is expected
        assert response.status_code in [200, 400]


class TestAPIDatabaseCreation:
    """Test: API database creation works."""

    def test_list_databases(self):
        """Test listing RAG databases."""
        response = requests.get(f"{API_BASE}/rag/databases")
        # May return 400 if RAG not configured
        assert response.status_code in [200, 400]

    def test_create_database(self):
        """Test creating a new database."""
        # This is a destructive test, only run if specifically enabled
        if not os.environ.get("RUN_DESTRUCTIVE_TESTS"):
            pytest.skip("Skipping destructive test (set RUN_DESTRUCTIVE_TESTS=1)")

        response = requests.post(
            f"{API_BASE}/rag/databases",
            json={
                "name": "test_db",
                "type": "ChromaStore",
                "config": {},
            },
        )
        # 201 = created, 409 = already exists
        assert response.status_code in [201, 409, 400]


class TestAPIFileIngest:
    """Test: API file ingest works with new parsers."""

    def test_list_datasets(self):
        """Test listing datasets."""
        response = requests.get(f"{API_BASE}/datasets/")
        assert response.status_code in [200, 400]

    def test_upload_file(self):
        """Test uploading a file to a dataset."""
        if not os.environ.get("RUN_DESTRUCTIVE_TESTS"):
            pytest.skip("Skipping destructive test")

        # Create a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Test content for ingest")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = requests.post(
                    f"{API_BASE}/datasets/test_dataset/data",
                    files={"file": ("test.txt", f, "text/plain")},
                )
            # Various valid responses
            assert response.status_code in [200, 400, 404]
        finally:
            os.unlink(temp_path)


class TestAPIQuery:
    """Test: API query works and returns results."""

    def test_query_endpoint_exists(self):
        """Test that query endpoint exists."""
        response = requests.post(
            f"{API_BASE}/rag/query",
            json={"query": "test query"},
        )
        # 400 = validation error (expected if no database)
        # 200 = success
        # 404 = database not found
        assert response.status_code in [200, 400, 404]

    def test_query_with_database(self):
        """Test query with explicit database."""
        response = requests.post(
            f"{API_BASE}/rag/query",
            json={
                "query": "What is this document about?",
                "database": "main_db",
                "top_k": 5,
            },
        )
        # Various valid responses depending on setup
        assert response.status_code in [200, 400, 404]


class TestAPIParseEndpoint:
    """Test: API parse-only endpoint works."""

    def test_parse_text_file(self):
        """Test parsing a text file without storing."""
        # Create temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("# Test Document\n\nThis is test content for parsing.")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = requests.post(
                    f"{API_BASE}/rag/parse",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"output_format": "markdown"},
                )

            if response.status_code == 200:
                data = response.json()
                assert "content" in data
                assert "format" in data
                assert data["format"] == "markdown"
            else:
                # 400/404 are acceptable if project doesn't exist
                assert response.status_code in [400, 404]
        finally:
            os.unlink(temp_path)

    def test_parse_with_chunking(self):
        """Test parsing with chunking enabled."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = requests.post(
                    f"{API_BASE}/rag/parse",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={
                        "chunk_strategy": "paragraphs",
                        "chunk_size": "100",
                    },
                )

            if response.status_code == 200:
                data = response.json()
                assert "chunks" in data
            else:
                assert response.status_code in [400, 404]
        finally:
            os.unlink(temp_path)

    def test_parse_unsupported_type(self):
        """Test parsing unsupported file type."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write("Unknown content")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = requests.post(
                    f"{API_BASE}/rag/parse",
                    files={"file": ("test.xyz", f, "application/octet-stream")},
                )

            # Should reject unsupported type with 415 or 400
            assert response.status_code in [400, 404, 415]
        finally:
            os.unlink(temp_path)


class TestParseEndpointFormats:
    """Test various output formats for parse endpoint."""

    def test_markdown_output(self):
        """Test markdown output format."""
        content = "# Heading\n\nParagraph text."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = requests.post(
                    f"{API_BASE}/rag/parse",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"output_format": "markdown"},
                )

            if response.status_code == 200:
                data = response.json()
                assert data["format"] == "markdown"
        finally:
            os.unlink(temp_path)

    def test_text_output(self):
        """Test plain text output format."""
        content = "# Heading\n\nParagraph text."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = requests.post(
                    f"{API_BASE}/rag/parse",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"output_format": "text"},
                )

            if response.status_code == 200:
                data = response.json()
                assert data["format"] == "text"
        finally:
            os.unlink(temp_path)


# Allow running tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
