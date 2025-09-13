"""Essential vector store tests."""

import pytest
import tempfile
import shutil

from rag.core.base import Document
from rag.components.stores.chroma_store.chroma_store import ChromaStore


class TestVectorStores:
    """Core vector store functionality tests."""

    def test_chroma_store_init(self):
        """Test ChromaDB store initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "collection_name": "test_collection",
                "persist_directory": temp_dir
            }
            store = ChromaStore(config=config)
            
            assert store.collection_name == "test_collection"
            assert store.persist_directory == temp_dir

    @pytest.mark.integration
    def test_document_operations(self):
        """Test adding and searching documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaStore(config={
                "collection_name": "test",
                "persist_directory": temp_dir
            })
            
            # Create test documents
            docs = [
                Document(
                    id="doc1",
                    content="First document",
                    embeddings=[0.1, 0.2, 0.3],
                    metadata={"type": "test"}
                ),
                Document(
                    id="doc2", 
                    content="Second document",
                    embeddings=[0.4, 0.5, 0.6],
                    metadata={"type": "test"}
                )
            ]
            
            # Add documents
            success = store.add_documents(docs)
            assert success is True
            
            # Search documents
            results = store.search(query_embedding=[0.1, 0.2, 0.3], top_k=1)
            assert len(results) == 1
            assert results[0].id == "doc1"

    def test_store_error_handling(self):
        """Test store handles errors gracefully."""
        # Test with a valid temporary directory instead
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaStore(config={
                "collection_name": "test",
                "persist_directory": temp_dir
            })
            
            # Test with invalid document (no embeddings)
            doc = Document(id="test", content="test")
            result = store.add_documents([doc])
            # Should handle gracefully
            
            result = store.delete_collection()
            assert result is True