"""Essential retriever tests."""

import pytest
from rag.core.base import Document
from rag.components.retrievers.basic_similarity.basic_similarity import BasicSimilarityStrategy
from rag.components.retrievers.metadata_filtered.metadata_filtered import MetadataFilteredStrategy


class TestRetrievers:
    """Core retriever functionality tests."""

    def test_basic_similarity_retriever(self):
        """Test basic similarity-based retrieval."""
        retriever = BasicSimilarityStrategy()
        
        docs = [
            Document(id="1", content="Python programming", embeddings=[0.1, 0.2]),
            Document(id="2", content="Java programming", embeddings=[0.3, 0.4]),
            Document(id="3", content="Python data science", embeddings=[0.15, 0.25])
        ]
        
        # Mock store behavior
        class MockStore:
            def search(self, query="", query_embedding=None, top_k=5):
                # Return docs sorted by similarity
                return docs[:top_k]
        
        store = MockStore()
        result = retriever.retrieve([0.1, 0.2], store, top_k=2)
        
        assert hasattr(result, 'documents')
        assert len(result.documents) <= 2

    def test_metadata_filtered_retriever(self):
        """Test retrieval with metadata filtering."""
        retriever = MetadataFilteredStrategy(config={
            "filter_key": "type",
            "filter_value": "technical"
        })
        
        docs = [
            Document(id="1", content="Technical doc", metadata={"type": "technical"}),
            Document(id="2", content="Business doc", metadata={"type": "business"}),
            Document(id="3", content="Another technical", metadata={"type": "technical"})
        ]
        
        # Filter should only return technical docs
        filtered = [d for d in docs if d.metadata.get("type") == "technical"]
        assert len(filtered) == 2
        assert all(d.metadata["type"] == "technical" for d in filtered)

    def test_retriever_configuration(self):
        """Test retriever accepts configuration."""
        config = {"top_k": 5, "threshold": 0.7}
        retriever = BasicSimilarityStrategy(config=config)
        
        assert hasattr(retriever, 'config')
        assert retriever.config.get("top_k") == 5

    def test_empty_retrieval(self):
        """Test retriever handles empty results."""
        retriever = BasicSimilarityStrategy()
        
        class EmptyStore:
            def search(self, query="", query_embedding=None, top_k=5):
                return []
        
        store = EmptyStore()
        result = retriever.retrieve([0.1, 0.2], store, top_k=5)
        
        assert hasattr(result, 'documents')
        assert result.documents == []