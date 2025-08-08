"""Tests for vector stores."""

import pytest
import tempfile
import shutil
from pathlib import Path

from core.base import Document
from components.stores.chroma_store import ChromaStore


class TestChromaStore:
    """Test ChromaDB vector store."""

    @pytest.fixture
    def temp_chroma_dir(self):
        """Create temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_store(self, temp_chroma_dir):
        """Create test ChromaDB store."""
        config = {
            "collection_name": "test_collection",
            "persist_directory": temp_chroma_dir,
        }
        store = ChromaStore(config=config)
        yield store
        try:
            store.delete_collection()
        except:
            pass

    @pytest.mark.chromadb
    def test_store_initialization(self, temp_chroma_dir):
        """Test store initialization."""
        config = {"collection_name": "test_init", "persist_directory": temp_chroma_dir}

        store = ChromaStore(config=config)
        assert store.collection_name == "test_init"
        assert store.persist_directory == temp_chroma_dir

        # Clean up
        store.delete_collection()

    @pytest.mark.chromadb
    def test_validation(self, test_store):
        """Test config validation."""
        result = test_store.validate_config()
        assert result is True

    @pytest.mark.chromadb
    def test_add_and_search_documents(self, test_store, sample_documents):
        """Test adding and searching documents."""
        # Add documents
        success = test_store.add_documents(sample_documents)
        assert success is True

        # Search by embedding
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = test_store.search(query_embedding=query_embedding, top_k=3)

        assert len(results) <= 3
        assert len(results) > 0

        # First result should be the most similar (exact match)
        assert results[0].id == "doc1"

    @pytest.mark.chromadb
    def test_collection_info(self, test_store, sample_documents):
        """Test getting collection information."""
        # Add some documents first
        test_store.add_documents(sample_documents)

        info = test_store.get_collection_info()

        assert "name" in info
        assert "count" in info
        assert info["name"] == "test_collection"
        assert info["count"] == len(sample_documents)

    @pytest.mark.chromadb
    def test_delete_collection(self, temp_chroma_dir):
        """Test collection deletion."""
        config = {
            "collection_name": "test_delete",
            "persist_directory": temp_chroma_dir,
        }

        store = ChromaStore(config=config)

        # Add a document
        doc = Document(
            id="test_doc", content="Test content", embeddings=[0.1, 0.2, 0.3, 0.4]
        )
        store.add_documents([doc])

        # Delete collection
        result = store.delete_collection()
        assert result is True

    @pytest.mark.chromadb
    def test_metadata_serialization(self, test_store):
        """Test metadata serialization for ChromaDB."""
        doc = Document(
            id="meta_test",
            content="Test content",
            metadata={
                "string_field": "test",
                "int_field": 42,
                "float_field": 3.14,
                "bool_field": True,
                "list_field": ["a", "b", "c"],
                "complex_field": {"nested": "value"},
            },
            embeddings=[0.1, 0.2, 0.3, 0.4],
        )

        success = test_store.add_documents([doc])
        assert success is True

        # Search to get the document back
        results = test_store.search(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=1)
        assert len(results) == 1

        result_doc = results[0]
        metadata = result_doc.metadata

        # Check serialization worked
        assert metadata["string_field"] == "test"
        assert metadata["int_field"] == 42
        assert metadata["float_field"] == 3.14
        assert metadata["bool_field"] is True
        assert metadata["list_field"] == "a,b,c"  # Lists become comma-separated
        assert "complex_field" in metadata  # Complex objects become strings

    @pytest.mark.chromadb
    def test_search_with_filters(self, test_store, sample_documents):
        """Test search with metadata filtering."""
        # Add documents
        test_store.add_documents(sample_documents)

        # Search with filter
        query_embedding = [0.5, 0.5, 0.5, 0.5]
        results = test_store.search(
            query_embedding=query_embedding, top_k=5, where={"priority": "high"}
        )

        # Should only return documents with high priority
        for result in results:
            assert result.metadata.get("priority") == "high"
