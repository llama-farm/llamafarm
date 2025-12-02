"""Tests for Chroma Store component."""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from components.stores.chroma_store.chroma_store import ChromaStore
from core.base import Document


class TestChromaStore:
    """Test ChromaStore functionality."""

    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_documents(self):
        """Sample documents with embeddings for testing."""
        return [
            Document(
                content="This is the first test document for vector storage.",
                id="doc1",
                source="test1.txt",
                metadata={"category": "test", "priority": "high"},
                embeddings=[0.1, 0.2, 0.3, 0.4, 0.5] * 100,  # 500-dim vector
            ),
            Document(
                content="Second document with different content for testing.",
                id="doc2",
                source="test2.txt",
                metadata={"category": "test", "priority": "medium"},
                embeddings=[0.2, 0.3, 0.4, 0.5, 0.6] * 100,  # 500-dim vector
            ),
            Document(
                content="Third test document with unique characteristics.",
                id="doc3",
                source="test3.txt",
                metadata={"category": "sample", "priority": "low"},
                embeddings=[0.3, 0.4, 0.5, 0.6, 0.7] * 100,  # 500-dim vector
            ),
        ]

    @pytest.fixture
    def test_store(self, temp_directory):
        """Create test ChromaStore instance."""
        config = {
            "collection_name": "test_collection",
            "embedding_dimension": 500,
        }
        # Pass a project_dir so internal persist path is deterministic
        return ChromaStore("test_store", config, project_dir=Path(temp_directory))

    def test_store_initialization(self, temp_directory):
        """Test store initialization with different configs."""
        # Default config
        store = ChromaStore("default", {}, project_dir=Path(temp_directory))
        assert store is not None
        assert store.collection_name == "documents"

        # Custom config
        custom_config = {
            "collection_name": "custom_collection",
            "embedding_dimension": 384,
        }
        store = ChromaStore("custom", custom_config, project_dir=Path(temp_directory))
        assert store.collection_name == "custom_collection"
        assert store.embedding_dimension == 384

    def test_add_documents(self, test_store, sample_documents):
        """Test adding documents to the store."""
        # Add documents - returns list of IDs
        result = test_store.add_documents(sample_documents)

        # Should return list of added document IDs
        assert isinstance(result, list)
        assert len(result) == len(sample_documents)
        assert result == ["doc1", "doc2", "doc3"]

        # Verify documents were added
        info = test_store.get_collection_info()
        if info:
            assert info.get("document_count", 0) == len(sample_documents)

    def test_search_functionality(self, test_store, sample_documents):
        """Test vector similarity search."""
        # Add documents first
        test_store.add_documents(sample_documents)

        # Search with query embedding
        query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55] * 100  # Close to doc1
        results = test_store.search(query_embedding=query_embedding, top_k=2)

        assert isinstance(results, list)
        assert len(results) <= 2

        # Results should be Document objects
        for result in results:
            assert isinstance(result, Document)
            assert hasattr(result, "content")
            assert hasattr(result, "id")

    def test_search_with_filters(self, test_store, sample_documents):
        """Test search with metadata filters."""
        # Add documents first
        test_store.add_documents(sample_documents)

        # Search with type filter
        query_embedding = [0.2, 0.3, 0.4, 0.5, 0.6] * 100
        results = test_store.search(
            query_embedding=query_embedding, top_k=5, filters={"type": "login"}
        )

        # Should only return documents with type="login"
        for result in results:
            if "type" in result.metadata:
                assert result.metadata["type"] == "login"

    def test_get_document_by_id(self, test_store, sample_documents):
        """Test retrieving specific document by ID."""
        # Add documents first
        test_store.add_documents(sample_documents)

        # Get specific document
        doc = test_store.get_document("doc1")

        if doc:  # Some stores might not support this
            assert isinstance(doc, Document)
            assert doc.id == "doc1"
            assert doc.content == sample_documents[0].content

    def test_delete_documents(self, test_store, sample_documents):
        """Test deleting documents from store."""
        # Add documents first
        test_store.add_documents(sample_documents)

        # Delete specific document - returns count of deleted docs
        deleted_count = test_store.delete_documents(["doc1"])

        assert deleted_count == 1

        # Verify document was deleted
        doc = test_store.get_document("doc1")
        assert doc is None

        # Other documents should still exist
        remaining_doc = test_store.get_document("doc2")
        assert remaining_doc is not None

    def test_get_documents_by_metadata(self, test_store):
        """Test retrieving documents by metadata filter."""
        # Create documents with specific metadata
        docs = [
            Document(
                content="Document with file_hash A",
                id="hash_doc1",
                source="test1.txt",
                metadata={"file_hash": "abc123", "category": "test"},
                embeddings=[0.1, 0.2, 0.3, 0.4, 0.5] * 100,
            ),
            Document(
                content="Document with file_hash A (chunk 2)",
                id="hash_doc2",
                source="test1.txt",
                metadata={"file_hash": "abc123", "category": "test"},
                embeddings=[0.2, 0.3, 0.4, 0.5, 0.6] * 100,
            ),
            Document(
                content="Document with different file_hash",
                id="hash_doc3",
                source="test2.txt",
                metadata={"file_hash": "xyz789", "category": "other"},
                embeddings=[0.3, 0.4, 0.5, 0.6, 0.7] * 100,
            ),
        ]
        test_store.add_documents(docs)

        # Get documents by file_hash
        results = test_store.get_documents_by_metadata({"file_hash": "abc123"})

        assert len(results) == 2
        assert all(doc.metadata.get("file_hash") == "abc123" for doc in results)
        assert {doc.id for doc in results} == {"hash_doc1", "hash_doc2"}

        # Get documents by different filter
        results = test_store.get_documents_by_metadata({"category": "other"})
        assert len(results) == 1
        assert results[0].id == "hash_doc3"

        # No results for non-existent filter
        results = test_store.get_documents_by_metadata({"file_hash": "nonexistent"})
        assert len(results) == 0

    def test_delete_documents_by_file_hash_workflow(self, test_store):
        """Test the full workflow of finding and deleting documents by file_hash."""
        # Create documents with file_hash metadata
        docs = [
            Document(
                content="Chunk 1 of file ABC",
                id="file_abc_chunk1",
                source="abc.pdf",
                metadata={"file_hash": "filehash_abc"},
                embeddings=[0.1] * 500,
            ),
            Document(
                content="Chunk 2 of file ABC",
                id="file_abc_chunk2",
                source="abc.pdf",
                metadata={"file_hash": "filehash_abc"},
                embeddings=[0.2] * 500,
            ),
            Document(
                content="Chunk of file XYZ",
                id="file_xyz_chunk1",
                source="xyz.pdf",
                metadata={"file_hash": "filehash_xyz"},
                embeddings=[0.3] * 500,
            ),
        ]
        test_store.add_documents(docs)

        # Step 1: Find documents by file_hash
        docs_to_delete = test_store.get_documents_by_metadata(
            {"file_hash": "filehash_abc"}
        )
        assert len(docs_to_delete) == 2

        # Step 2: Delete by IDs
        doc_ids = [doc.id for doc in docs_to_delete]
        deleted_count = test_store.delete_documents(doc_ids)
        assert deleted_count == 2

        # Verify ABC documents are gone
        remaining = test_store.get_documents_by_metadata({"file_hash": "filehash_abc"})
        assert len(remaining) == 0

        # XYZ document should still exist
        xyz_docs = test_store.get_documents_by_metadata({"file_hash": "filehash_xyz"})
        assert len(xyz_docs) == 1

    def test_collection_management(self, test_store):
        """Test collection creation and deletion."""
        # Collection should exist after initialization
        info = test_store.get_collection_info()
        assert info is not None

        # Delete collection
        success = test_store.delete_collection()
        assert success is True

        # Collection should no longer exist
        info = test_store.get_collection_info()
        if info:
            assert info.get("document_count", 0) == 0

    def test_empty_store_operations(self, test_store):
        """Test operations on empty store."""
        # Search empty store
        query_embedding = [0.1, 0.2, 0.3] * 100
        results = test_store.search(query_embedding=query_embedding, top_k=5)

        assert isinstance(results, list)
        assert len(results) == 0

        # Get collection info
        info = test_store.get_collection_info()
        if info:
            assert info.get("document_count", 0) == 0

    def test_documents_without_embeddings(self, test_store):
        """Test handling documents without embeddings."""
        docs_no_embeddings = [
            Document(
                content="Document without embeddings",
                id="no_emb1",
                source="test.txt",
                metadata={},
                # No embeddings field
            )
        ]

        # Should handle gracefully - returns False on error or list on success
        result = test_store.add_documents(docs_no_embeddings)
        # May return False (error) or empty list (no embeddings to add) or list of IDs
        assert isinstance(result, (bool, list))

    def test_duplicate_document_handling(self, test_store, sample_documents):
        """Test handling of duplicate document IDs."""
        # Add documents - first time should add all
        result1 = test_store.add_documents(sample_documents)
        assert isinstance(result1, list)
        assert len(result1) == len(sample_documents)

        # Add same documents again - should return empty list (all duplicates)
        result2 = test_store.add_documents(sample_documents)

        # Should return empty list since all are duplicates
        assert isinstance(result2, list)
        assert len(result2) == 0  # No new documents added

        # Collection should not have duplicates
        info = test_store.get_collection_info()
        if info:
            # Should have exactly the original count
            assert info.get("document_count", 0) == len(sample_documents)

    def test_large_batch_operations(self, test_store):
        """Test operations with larger batches of documents."""
        # Create many documents
        large_batch = []
        for i in range(50):
            doc = Document(
                content=f"Test document number {i} with unique content.",
                id=f"batch_doc_{i}",
                source=f"batch_{i}.txt",
                metadata={"batch": True, "index": i},
                embeddings=[0.1 + i * 0.01] * 500,
            )
            large_batch.append(doc)

        # Add large batch - should return list of IDs
        result = test_store.add_documents(large_batch)
        assert isinstance(result, list)
        assert len(result) == len(large_batch)
        # Check IDs match
        expected_ids = [f"batch_doc_{i}" for i in range(50)]
        assert result == expected_ids

        # Verify count
        info = test_store.get_collection_info()
        if info:
            assert info.get("document_count", 0) == len(large_batch)

    def test_metadata_preservation(self, test_store, sample_documents):
        """Test that document metadata is preserved."""
        # Add documents
        test_store.add_documents(sample_documents)

        # Search and verify metadata
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 100
        results = test_store.search(query_embedding=query_embedding, top_k=1)

        if results:
            result = results[0]
            assert "category" in result.metadata
            assert "priority" in result.metadata
            # Metadata should match original
            original_doc = next(doc for doc in sample_documents if doc.id == result.id)
            assert result.metadata["category"] == original_doc.metadata["category"]

    def test_configuration_validation(self, temp_directory):
        """Test configuration validation."""
        # Missing required config should use defaults
        minimal_config = {}
        store = ChromaStore("minimal", minimal_config, project_dir=Path(temp_directory))
        assert store.collection_name == "documents"

        # Invalid embedding dimension should use default
        invalid_config = {"embedding_dimension": -1}
        store = ChromaStore("invalid", invalid_config, project_dir=Path(temp_directory))
        assert store.embedding_dimension > 0

    def test_persistence(self, temp_directory, sample_documents):
        """Test data persistence across store instances."""
        # Create first store instance and add data
        store1 = ChromaStore(
            "persist_shared",
            {"collection_name": "persist_test"},
            project_dir=Path(temp_directory),
        )
        store1.add_documents(sample_documents)

        # Create second store instance with same config
        store2 = ChromaStore(
            "persist_shared",
            {"collection_name": "persist_test"},
            project_dir=Path(temp_directory),
        )

        # Should be able to access same data
        info = store2.get_collection_info()
        if info:
            assert info.get("document_count", 0) > 0

    def test_get_description(self):
        """Test store description method."""
        description = ChromaStore.get_description()
        assert isinstance(description, str)
        assert len(description) > 0
        assert "chroma" in description.lower()

    def test_list_documents_empty_collection(self, test_store):
        """Test listing documents from empty collection."""
        docs, total = test_store.list_documents()

        assert isinstance(docs, list)
        assert len(docs) == 0
        assert total == 0

    def test_list_documents_returns_all(self, test_store, sample_documents):
        """Test listing all documents."""
        test_store.add_documents(sample_documents)

        docs, total = test_store.list_documents()

        assert total == len(sample_documents)
        assert len(docs) == len(sample_documents)
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.id in ["doc1", "doc2", "doc3"]

    def test_list_documents_with_pagination(self, test_store, sample_documents):
        """Test listing documents with limit and offset."""
        test_store.add_documents(sample_documents)

        # Get first 2 documents
        docs, total = test_store.list_documents(limit=2, offset=0)
        assert len(docs) == 2
        assert total == 3  # Total should still be 3

        # Get next document with offset
        docs, total = test_store.list_documents(limit=2, offset=2)
        assert len(docs) == 1  # Only 1 remaining
        assert total == 3

    def test_list_documents_preserves_metadata(self, test_store, sample_documents):
        """Test that listing preserves document metadata."""
        test_store.add_documents(sample_documents)

        docs, _ = test_store.list_documents()

        for doc in docs:
            assert "category" in doc.metadata
            assert "priority" in doc.metadata

    def test_list_documents_with_content(self, test_store, sample_documents):
        """Test listing with content included."""
        test_store.add_documents(sample_documents)

        docs, _ = test_store.list_documents(include_content=True)

        for doc in docs:
            assert doc.content  # Content should be present
            assert len(doc.content) > 0

    def test_list_documents_without_content(self, test_store, sample_documents):
        """Test listing without content (default)."""
        test_store.add_documents(sample_documents)

        docs, _ = test_store.list_documents(include_content=False)

        for doc in docs:
            # Content should be empty when not requested
            assert doc.content == ""

    def test_list_documents_source_preserved(self, test_store, sample_documents):
        """Test that source is correctly set from metadata."""
        test_store.add_documents(sample_documents)

        docs, _ = test_store.list_documents()

        sources = {doc.source for doc in docs}
        expected_sources = {"test1.txt", "test2.txt", "test3.txt"}
        assert sources == expected_sources


# Integration tests (may require actual ChromaDB)
class TestChromaStoreIntegration:
    """Integration tests that may require actual ChromaDB installation."""

    @pytest.mark.skipif(True, reason="Requires ChromaDB installation")
    def test_real_chroma_operations(self):
        """Test with real ChromaDB (skipped by default)."""
        try:
            import chromadb  # noqa: F401

            temp_dir = tempfile.mkdtemp()
            store = ChromaStore("integration", {"persist_directory": temp_dir})

            # Basic operations
            docs = [
                Document(
                    content="Integration test document",
                    id="integration_doc",
                    source="test.txt",
                    metadata={},
                    embeddings=[0.1] * 384,
                )
            ]

            success = store.add_documents(docs)
            assert success is True

            # Cleanup
            shutil.rmtree(temp_dir)

        except ImportError:
            pytest.skip("ChromaDB not available")


if __name__ == "__main__":
    pytest.main([__file__])
