"""Tests for DocumentManager functionality."""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import Document
from core.document_manager import DocumentDeletionManager, DocumentManager


class TestDocumentDeletionManager:
    """Test DocumentDeletionManager functionality."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing."""
        store = MagicMock()
        return store

    def test_delete_by_file_hash_success(self, mock_vector_store):
        """Test successful deletion by file_hash."""
        # Setup mock to return documents
        mock_docs = [
            Document(
                id="chunk1", content="Content 1", metadata={"file_hash": "abc123"}
            ),
            Document(
                id="chunk2", content="Content 2", metadata={"file_hash": "abc123"}
            ),
        ]
        mock_vector_store.get_documents_by_metadata.return_value = mock_docs
        mock_vector_store.delete_documents.return_value = 2

        manager = DocumentDeletionManager(mock_vector_store)
        result = manager.delete_by_file_hash("abc123")

        assert result["file_hash"] == "abc123"
        assert result["deleted_count"] == 2
        assert result["error"] is None

        # Verify correct calls were made
        mock_vector_store.get_documents_by_metadata.assert_called_once_with(
            {"file_hash": "abc123"}
        )
        mock_vector_store.delete_documents.assert_called_once_with(["chunk1", "chunk2"])

    def test_delete_by_file_hash_no_documents(self, mock_vector_store):
        """Test deletion when no documents match."""
        mock_vector_store.get_documents_by_metadata.return_value = []

        manager = DocumentDeletionManager(mock_vector_store)
        result = manager.delete_by_file_hash("nonexistent")

        assert result["file_hash"] == "nonexistent"
        assert result["deleted_count"] == 0
        assert result["error"] is None

        # delete_documents should not be called
        mock_vector_store.delete_documents.assert_not_called()

    def test_delete_by_file_hash_error(self, mock_vector_store):
        """Test handling of errors during deletion."""
        mock_vector_store.get_documents_by_metadata.side_effect = Exception(
            "Database error"
        )

        manager = DocumentDeletionManager(mock_vector_store)
        result = manager.delete_by_file_hash("abc123")

        assert result["file_hash"] == "abc123"
        assert result["deleted_count"] == 0
        assert "Database error" in result["error"]


class TestDocumentManager:
    """Test DocumentManager functionality."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing."""
        store = MagicMock()
        return store

    def test_delete_by_file_hash_delegates(self, mock_vector_store):
        """Test that delete_by_file_hash delegates to deletion manager."""
        mock_docs = [
            Document(id="doc1", content="Test", metadata={"file_hash": "hash123"})
        ]
        mock_vector_store.get_documents_by_metadata.return_value = mock_docs
        mock_vector_store.delete_documents.return_value = 1

        manager = DocumentManager(mock_vector_store)
        result = manager.delete_by_file_hash("hash123")

        assert result["deleted_count"] == 1
        mock_vector_store.get_documents_by_metadata.assert_called_once()
        mock_vector_store.delete_documents.assert_called_once()


class TestDocumentManagerIntegration:
    """Integration tests with real ChromaStore."""

    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def real_store(self, temp_directory):
        """Create a real ChromaStore for integration testing."""
        from components.stores.chroma_store.chroma_store import ChromaStore

        config = {"collection_name": "test_deletion"}
        return ChromaStore("test_store", config, project_dir=Path(temp_directory))

    def test_delete_by_file_hash_integration(self, real_store):
        """Test delete_by_file_hash with real ChromaStore."""
        # Add documents with file_hash metadata
        docs = [
            Document(
                id="file1_chunk1",
                content="First chunk of file 1",
                metadata={"file_hash": "file1_hash"},
                embeddings=[0.1] * 500,
            ),
            Document(
                id="file1_chunk2",
                content="Second chunk of file 1",
                metadata={"file_hash": "file1_hash"},
                embeddings=[0.2] * 500,
            ),
            Document(
                id="file2_chunk1",
                content="First chunk of file 2",
                metadata={"file_hash": "file2_hash"},
                embeddings=[0.3] * 500,
            ),
        ]
        real_store.add_documents(docs)

        # Create DocumentManager and delete by file_hash
        manager = DocumentManager(real_store)
        result = manager.delete_by_file_hash("file1_hash")

        assert result["deleted_count"] == 2
        assert result["error"] is None

        # Verify file1 documents are gone
        remaining = real_store.get_documents_by_metadata({"file_hash": "file1_hash"})
        assert len(remaining) == 0

        # Verify file2 document still exists
        file2_docs = real_store.get_documents_by_metadata({"file_hash": "file2_hash"})
        assert len(file2_docs) == 1


if __name__ == "__main__":
    pytest.main([__file__])
