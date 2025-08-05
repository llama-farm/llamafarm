"""Tests for document_manager module."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

from core.document_manager import (
    DocumentManager,
    DocumentLifecycleManager,
    DocumentHashManager,
    DocumentDeletionManager,
    DocumentMetadata,
    DeletionStrategy,
    UpdateStrategy
)
from core.base import Document


class TestDocumentHashManager:
    """Test document hash generation functionality."""
    
    def test_initialization(self):
        """Test hash manager initialization."""
        manager = DocumentHashManager()
        assert manager.hash_algorithm == "sha256"
        
        manager_md5 = DocumentHashManager("md5")
        assert manager_md5.hash_algorithm == "md5"
    
    def test_generate_content_hash(self):
        """Test content hash generation."""
        manager = DocumentHashManager()
        
        # Test basic hashing
        hash1 = manager.generate_content_hash("Hello World")
        assert hash1.startswith("sha256:")
        assert len(hash1) > 10
        
        # Test that same content gives same hash
        hash2 = manager.generate_content_hash("Hello World")
        assert hash1 == hash2
        
        # Test that different content gives different hash
        hash3 = manager.generate_content_hash("Hello Python")
        assert hash1 != hash3
        
        # Test normalization
        hash4 = manager.generate_content_hash("  Hello   World  ")
        hash5 = manager.generate_content_hash("hello world")
        assert hash4 == hash5  # Normalized to same
    
    def test_generate_file_hash(self):
        """Test file hash generation."""
        manager = DocumentHashManager()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content for hashing")
            temp_path = Path(f.name)
        
        try:
            # Test successful hash
            file_hash = manager.generate_file_hash(temp_path)
            assert file_hash.startswith("sha256:")
            assert len(file_hash) > 10
            
            # Test non-existent file
            fake_path = Path("/nonexistent/file.txt")
            hash_result = manager.generate_file_hash(fake_path)
            assert hash_result == ""  # Should return empty string on error
            
        finally:
            temp_path.unlink()
    
    def test_generate_metadata_hash(self):
        """Test metadata hash generation."""
        manager = DocumentHashManager()
        
        metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "created_at": "2024-01-01",
            "tags": ["test", "document"]
        }
        
        hash1 = manager.generate_metadata_hash(metadata)
        assert hash1.startswith("sha256:")
        
        # Test that timestamps are excluded
        metadata["updated_at"] = "2024-01-02"
        hash2 = manager.generate_metadata_hash(metadata)
        assert hash1 == hash2  # Should be same since timestamps excluded
        
        # Test that other changes affect hash
        metadata["author"] = "Different Author"
        hash3 = manager.generate_metadata_hash(metadata)
        assert hash1 != hash3
    
    def test_generate_composite_hash(self):
        """Test composite hash generation."""
        manager = DocumentHashManager()
        
        hash1 = manager.generate_composite_hash("part1", "part2", "part3")
        assert hash1.startswith("sha256:")
        
        # Different order should give different hash
        hash2 = manager.generate_composite_hash("part2", "part1", "part3")
        assert hash1 != hash2


class TestDocumentMetadata:
    """Test DocumentMetadata dataclass."""
    
    def test_initialization(self):
        """Test metadata initialization."""
        metadata = DocumentMetadata(
            doc_id="doc123",
            chunk_id="chunk456",
            filename="test.txt"
        )
        
        assert metadata.doc_id == "doc123"
        assert metadata.chunk_id == "chunk456"
        assert metadata.filename == "test.txt"
        assert metadata.chunk_index == 0
        assert metadata.total_chunks == 1
        assert metadata.is_active is True
        assert metadata.version == 1
        
        # Check default list initialization
        assert metadata.related_docs == []
        assert metadata.tags == []
        assert metadata.keywords == []
        
        # Check timestamps were set
        assert metadata.created_at != ""
        assert metadata.updated_at == metadata.created_at
        assert metadata.indexed_at == metadata.created_at


class TestDocumentLifecycleManager:
    """Test document lifecycle management."""
    
    @pytest.fixture
    def manager(self):
        """Create lifecycle manager instance."""
        config = {
            "hash_algorithm": "sha256",
            "retention_policy": {
                "default_ttl_days": 30
            },
            "enable_versioning": True,
            "enable_soft_delete": True
        }
        return DocumentLifecycleManager(config)
    
    def test_initialization(self, manager):
        """Test lifecycle manager initialization."""
        assert manager.hash_manager is not None
        assert manager.retention_policy["default_ttl_days"] == 30
        assert manager.enable_versioning is True
        assert manager.enable_soft_delete is True
    
    def test_enhance_document_metadata(self, manager):
        """Test document metadata enhancement."""
        doc = Document(
            content="Test document content",
            metadata={"existing": "metadata"}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("File content")
            temp_path = Path(f.name)
        
        try:
            enhanced_doc = manager.enhance_document_metadata(doc, temp_path)
            
            # Check basic metadata was added
            assert "doc_id" in enhanced_doc.metadata
            assert enhanced_doc.metadata["filename"] == temp_path.name
            assert enhanced_doc.metadata["file_size"] > 0
            assert enhanced_doc.metadata["version"] == 1
            assert enhanced_doc.metadata["is_active"] is True
            
            # Check hashes were generated
            assert "file_hash" in enhanced_doc.metadata
            assert "chunk_hash" in enhanced_doc.metadata
            assert "metadata_hash" in enhanced_doc.metadata
            
            # Check timestamps
            assert "created_at" in enhanced_doc.metadata
            assert "updated_at" in enhanced_doc.metadata
            assert "indexed_at" in enhanced_doc.metadata
            
            # Check expiration was set
            assert "expires_at" in enhanced_doc.metadata
            
            # Check existing metadata preserved
            assert enhanced_doc.metadata["existing"] == "metadata"
            
        finally:
            temp_path.unlink()
    
    def test_generate_summary(self, manager):
        """Test summary generation."""
        content = "This is the first sentence. This is the second sentence. This is the third sentence."
        summary = manager._generate_summary(content, max_length=50)
        
        assert len(summary) <= 50
        assert summary.startswith("This is the first sentence.")
    
    def test_extract_keywords(self, manager):
        """Test keyword extraction."""
        content = "Python programming is great. Python is versatile. Programming in Python is fun."
        keywords = manager._extract_keywords(content, max_keywords=5)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert "python" in keywords
        assert "programming" in keywords
    
    def test_calculate_content_statistics(self, manager):
        """Test content statistics calculation."""
        content = "This is a test. It has two sentences."
        stats = manager._calculate_content_statistics(content)
        
        assert stats["word_count"] == 8
        assert stats["character_count"] == len(content)
        assert stats["sentence_count"] == 2
        assert stats["reading_time_minutes"] > 0
        assert stats["avg_sentence_length"] == 4.0
    
    def test_auto_classify_content(self, manager):
        """Test content classification."""
        assert manager._auto_classify_content("There is an error in the system") == "technical_issue"
        assert manager._auto_classify_content("I forgot my password") == "access_issue"
        assert manager._auto_classify_content("Please check my invoice") == "billing_issue"
        assert manager._auto_classify_content("General inquiry") == "general"
    
    def test_analyze_sentiment(self, manager):
        """Test sentiment analysis."""
        assert manager._analyze_sentiment("This is great and excellent!") == "positive"
        assert manager._analyze_sentiment("This is terrible and awful") == "negative"
        assert manager._analyze_sentiment("This is okay") == "neutral"


class TestDocumentDeletionManager:
    """Test document deletion functionality."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock()
        store.update_metadata = Mock(return_value=5)
        store.delete_documents = Mock(return_value=3)
        return store
    
    @pytest.fixture
    def manager(self, mock_vector_store):
        """Create deletion manager instance."""
        config = {
            "enable_soft_delete": True,
            "retention_policy": {
                "archive_location": "/archive"
            }
        }
        return DocumentDeletionManager(mock_vector_store, config)
    
    def test_initialization(self, manager):
        """Test deletion manager initialization."""
        assert manager.enable_soft_delete is True
        assert manager.archive_location == "/archive"
    
    def test_delete_by_time(self, manager, mock_vector_store):
        """Test time-based deletion."""
        result = manager.delete_by_time(30, DeletionStrategy.SOFT_DELETE)
        
        assert result["strategy"] == "soft_delete"
        assert result["deleted_count"] == 5
        assert len(result["errors"]) == 0
        
        # Verify soft delete was called
        mock_vector_store.update_metadata.assert_called_once()
        call_args = mock_vector_store.update_metadata.call_args[0]
        assert "created_at" in call_args[0]
        assert call_args[1]["is_active"] is False
    
    def test_delete_by_document(self, manager, mock_vector_store):
        """Test document ID based deletion."""
        doc_ids = ["doc1", "doc2", "doc3"]
        result = manager.delete_by_document(doc_ids, DeletionStrategy.HARD_DELETE)
        
        assert result["strategy"] == "hard_delete"
        assert result["deleted_count"] == 3
        
        # Verify hard delete was called
        mock_vector_store.delete_documents.assert_called_once()
        call_args = mock_vector_store.delete_documents.call_args[0][0]
        assert call_args["doc_id"]["$in"] == doc_ids
    
    def test_delete_expired_documents(self, manager, mock_vector_store):
        """Test expired document deletion."""
        result = manager.delete_expired_documents(DeletionStrategy.SOFT_DELETE)
        
        assert result["strategy"] == "soft_delete"
        assert result["deleted_count"] == 5
        
        # Verify expiration filter was used
        call_args = mock_vector_store.update_metadata.call_args[0][0]
        assert "expires_at" in call_args
    
    def test_deletion_error_handling(self, manager, mock_vector_store):
        """Test deletion error handling."""
        mock_vector_store.update_metadata.side_effect = Exception("Test error")
        
        result = manager.delete_by_time(30, DeletionStrategy.SOFT_DELETE)
        
        assert result["deleted_count"] == 0
        assert len(result["errors"]) == 1
        assert "Test error" in result["errors"][0]


class TestDocumentManager:
    """Test main document manager."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock()
        store.get_collection_info = Mock(return_value={"count": 100})
        return store
    
    @pytest.fixture
    def manager(self, mock_vector_store):
        """Create document manager instance."""
        config = {
            "hash_algorithm": "sha256",
            "retention_policy": {
                "default_ttl_days": 30
            },
            "enable_versioning": True,
            "enable_soft_delete": True
        }
        return DocumentManager(mock_vector_store, config)
    
    def test_initialization(self, manager):
        """Test document manager initialization."""
        assert manager.lifecycle_manager is not None
        assert manager.deletion_manager is not None
        assert manager.hash_manager is not None
    
    def test_process_documents(self, manager):
        """Test document processing."""
        docs = [
            Document(content="Document 1", metadata={}),
            Document(content="Document 2", metadata={})
        ]
        
        processed_docs = manager.process_documents(docs)
        
        assert len(processed_docs) == 2
        for doc in processed_docs:
            assert "doc_id" in doc.metadata
            assert "chunk_hash" in doc.metadata
            assert "created_at" in doc.metadata
    
    def test_find_duplicates(self, manager):
        """Test duplicate detection."""
        docs = [
            Document(content="Same content", metadata={"chunk_hash": "hash1"}, id="doc1"),
            Document(content="Same content", metadata={"chunk_hash": "hash1"}, id="doc2"),
            Document(content="Different", metadata={"chunk_hash": "hash2"}, id="doc3"),
        ]
        
        duplicates = manager.find_duplicates(docs)
        
        assert "hash1" in duplicates
        assert len(duplicates["hash1"]) == 2
        assert "doc1" in duplicates["hash1"]
        assert "doc2" in duplicates["hash1"]
        assert "hash2" not in duplicates  # Only duplicates returned
    
    def test_get_document_stats(self, manager, mock_vector_store):
        """Test statistics retrieval with fallback method."""
        # Remove get_collection_stats attribute to test fallback
        if hasattr(mock_vector_store, 'get_collection_stats'):
            delattr(mock_vector_store, 'get_collection_stats')
        
        stats = manager.get_document_stats()
        
        assert stats["total_documents"] == 100
        assert stats["active_documents"] == 100
        
        # Test with enhanced stats
        mock_vector_store.get_collection_stats = Mock(return_value={
            "total_documents": 150,
            "active_documents": 140,
            "deleted_documents": 10
        })
        
        stats = manager.get_document_stats()
        assert stats["total_documents"] == 150
        assert stats["active_documents"] == 140
        assert stats["deleted_documents"] == 10
    
    def test_get_document_stats_error_handling(self, manager, mock_vector_store):
        """Test error handling in stats retrieval."""
        mock_vector_store.get_collection_info.side_effect = Exception("Test error")
        mock_vector_store.get_collection_stats = None
        
        stats = manager.get_document_stats()
        
        # Should return default values on error
        assert stats["total_documents"] == 0
        assert stats["active_documents"] == 0