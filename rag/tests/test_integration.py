"""Essential integration tests for the RAG pipeline."""

import pytest
import tempfile
from pathlib import Path

from rag.core.base import Document, Pipeline
from rag.components.parsers.text.python_parser import TextParser_Python
from rag.components.embedders.ollama_embedder.ollama_embedder import OllamaEmbedder
from rag.components.stores.chroma_store.chroma_store import ChromaStore
from rag.core.strategies.handler import SchemaHandler


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.skip(reason="Requires complex setup with embedders and stores")
    @pytest.mark.integration
    def test_full_pipeline_text_processing(self):
        """Test complete pipeline from text to vector store."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pipeline components
            parser = TextParser_Python()
            # Use mock embedder for testing
            class MockEmbedder:
                def embed(self, texts):
                    return [[0.1, 0.2, 0.3] for _ in texts]
            
            embedder = MockEmbedder()
            store = ChromaStore(config={
                "collection_name": "test_integration",
                "persist_directory": temp_dir
            })
            
            # Process text
            text = "This is a test document for integration testing."
            parsed = parser.parse(text)
            
            assert len(parsed.documents) > 0
            
            # Generate embeddings
            texts = [doc.content for doc in parsed.documents]
            embeddings = embedder.embed(texts)
            
            assert len(embeddings) == len(texts)
            
            # Store documents
            for doc, emb in zip(parsed.documents, embeddings):
                doc.embeddings = emb
            
            success = store.add_documents(parsed.documents)
            assert success is True

    @pytest.mark.skip(reason="Requires complex setup with vector stores")
    @pytest.mark.integration
    def test_document_retrieval(self):
        """Test document storage and retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaStore(config={
                "collection_name": "test_retrieval",
                "persist_directory": temp_dir
            })
            
            # Add test documents
            docs = [
                Document(
                    id="1",
                    content="Python programming language",
                    embeddings=[0.1, 0.2, 0.3]
                ),
                Document(
                    id="2",
                    content="JavaScript web development",
                    embeddings=[0.4, 0.5, 0.6]
                ),
                Document(
                    id="3",
                    content="Machine learning with Python",
                    embeddings=[0.15, 0.25, 0.35]
                )
            ]
            
            store.add_documents(docs)
            
            # Search for similar documents
            results = store.search(query_embedding=[0.12, 0.22, 0.32], top_k=2)
            
            assert len(results) == 2
            # Most similar should be doc 1 or doc 3 (both about Python)
            assert results[0].id in ["1", "3"]

    @pytest.mark.skip(reason="Requires complex setup with strategies and stores")
    @pytest.mark.integration
    def test_strategy_execution(self):
        """Test strategy-based processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test strategy config
            test_config = {
                "databases": {
                    "test_db": {
                        "vector_store": {
                            "type": "chroma",
                            "config": {
                                "collection_name": "test",
                                "persist_directory": temp_dir
                            }
                        },
                        "embedding_strategies": {
                            "default": {
                                "type": "sentence_transformer",
                                "config": {
                                    "model_name": "all-MiniLM-L6-v2"
                                }
                            }
                        }
                    }
                },
                "data_processing_strategies": {
                    "test_strategy": {
                        "parsers": {
                            "text": {
                                "type": "text",
                                "config": {}
                            }
                        }
                    }
                }
            }
            
            # Save config to file
            import yaml
            config_path = Path(temp_dir) / "test_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            # Load and use strategy
            handler = SchemaHandler(str(config_path))
            strategies = handler.get_available_strategies()
            
            assert len(strategies) > 0

    def test_error_recovery(self):
        """Test pipeline handles errors gracefully."""
        parser = TextParser_Python()
        
        # Test with non-existent file
        result = parser.parse("/non/existent/file.txt")
        assert result.documents == []
        assert len(result.errors) > 0
        
        # Test with empty file path
        result = parser.parse("")
        assert result is not None
        assert result.documents == []

    def test_metadata_preservation(self):
        """Test metadata is preserved through pipeline."""
        doc = Document(
            content="Test content",
            metadata={"source": "test", "author": "tester"},
            id="test-1"
        )
        
        # Verify metadata is preserved
        assert doc.metadata["source"] == "test"
        assert doc.metadata["author"] == "tester"
        
        doc_dict = doc.to_dict()
        assert doc_dict["metadata"]["source"] == "test"