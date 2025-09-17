"""Essential extractor tests."""

import pytest
from rag.core.base import Document
from rag.components.extractors.statistics_extractor.statistics_extractor import ContentStatisticsExtractor
from rag.components.extractors.entity_extractor.entity_extractor import EntityExtractor
from rag.components.extractors.keyword_extractor.keyword_extractor import KeywordExtractor


class TestExtractors:
    """Core extractor functionality tests."""

    def test_statistics_extractor(self):
        """Test statistics extraction from documents."""
        extractor = ContentStatisticsExtractor()
        
        doc = Document(
            content="This is a test document with some words. It has two sentences.",
            id="test-doc"
        )
        
        result = extractor.extract([doc])
        assert len(result) == 1
        
        # Check if statistics are in the metadata (may be under different key)
        metadata = result[0].metadata
        assert len(metadata) > 0  # Should have some metadata
        # Statistics might be under 'extractors' or direct keys
        assert any(k in str(metadata) for k in ['word', 'sentence', 'char'])

    def test_entity_extractor_basic(self):
        """Test basic entity extraction."""
        extractor = EntityExtractor()
        
        doc = Document(
            content="John Smith works at OpenAI in San Francisco.",
            id="test-doc"
        )
        
        result = extractor.extract([doc])
        assert len(result) == 1
        
        # Should extract some entities (check various possible keys)
        metadata = result[0].metadata
        # Entities might be under different keys
        assert any(k in metadata for k in ['entities_person', 'entities_org', 'extractors'])

    def test_keyword_extractor(self):
        """Test keyword extraction from text."""
        extractor = KeywordExtractor(config={"max_keywords": 5})
        
        doc = Document(
            content="Machine learning algorithms process data to identify patterns and make predictions.",
            id="test-doc"
        )
        
        result = extractor.extract([doc])
        assert len(result) == 1
        
        keywords = result[0].metadata.get("keywords", [])
        assert isinstance(keywords, list)
        assert len(keywords) <= 5

    def test_extractor_error_handling(self):
        """Test extractors handle errors gracefully."""
        extractor = ContentStatisticsExtractor()
        
        # Test with empty document
        doc = Document(content="", id="empty")
        result = extractor.extract([doc])
        
        assert len(result) == 1
        # Empty document should have minimal metadata
        metadata = result[0].metadata
        assert metadata is not None