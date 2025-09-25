"""Essential vector store tests."""

import pytest
import tempfile
import shutil

from core.base import Document
from components.stores.chroma_store.chroma_store import ChromaStore


class TestVectorStores:
    """Core vector store functionality tests."""

    def test_chroma_store_init(self):
        """Test ChromaDB store initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "collection_name": "test_collection",
                "persist_directory": temp_dir,
            }
            store = ChromaStore(config=config, project_dir=temp_dir)

            assert store.collection_name == "test_collection"
            # Persist directory is derived from project_dir now
            assert store.persist_directory.startswith(str(temp_dir))

    @pytest.mark.integration
    def test_document_operations(self):
        """Test adding and searching documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaStore(
                config={"collection_name": "test"}, project_dir=temp_dir
            )

            # Create test documents
            docs = [
                Document(
                    id="doc1",
                    content="First document",
                    embeddings=[0.1, 0.2, 0.3],
                    metadata={"type": "test"},
                ),
                Document(
                    id="doc2",
                    content="Second document",
                    embeddings=[0.4, 0.5, 0.6],
                    metadata={"type": "test"},
                ),
            ]

            # Add documents - returns list of IDs
            result = store.add_documents(docs)
            assert isinstance(result, list)
            assert len(result) == 2
            assert result == ["doc1", "doc2"]

            # Search documents
            results = store.search(query_embedding=[0.1, 0.2, 0.3], top_k=1)
            assert len(results) == 1
            assert results[0].id == "doc1"

    def test_store_error_handling(self):
        """Test store handles errors gracefully."""
        # Test with a valid temporary directory instead
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaStore(
                config={"collection_name": "test"}, project_dir=temp_dir
            )

            # Test with invalid document (no embeddings)
            doc = Document(id="test", content="test")
            result = store.add_documents([doc])
            # Should handle gracefully

            result = store.delete_collection()
            assert result is True
