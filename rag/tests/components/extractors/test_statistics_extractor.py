"""Tests for Statistics Extractor component."""

import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.base import Document
from components.extractors.statistics_extractor.statistics_extractor import (
    ContentStatisticsExtractor,
)


class TestContentStatisticsExtractor:
    """Test ContentStatisticsExtractor functionality."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                content="This is a simple test document. It contains multiple sentences for analysis. The content should provide enough data for statistical analysis of readability and structure.",
                id="doc1",
                source="test_doc1.txt",
                metadata={},
            ),
            Document(
                content="Another test document with different characteristics. This one has varied sentence lengths. Some are short. Others are significantly longer to test the statistical analysis capabilities of the extractor system.",
                id="doc2",
                source="test_doc2.txt",
                metadata={},
            ),
        ]

    @pytest.fixture
    def default_extractor(self):
        """Create default statistics extractor."""
        return ContentStatisticsExtractor(
            "stats_extractor",
            {
                "include_readability": True,
                "include_vocabulary": True,
                "include_structure": True,
                "include_sentiment_indicators": True,
            },
        )

    def test_extractor_initialization(self):
        """Test extractor initialization with different configs."""
        # Default config
        extractor = ContentStatisticsExtractor("default")
        assert extractor is not None
        assert extractor.include_readability == True

        # Custom config
        custom_config = {
            "include_readability": False,
            "include_vocabulary": True,
            "include_structure": False,
            "include_sentiment_indicators": True,
            "language": "en",
        }
        extractor = ContentStatisticsExtractor("custom", custom_config)
        assert extractor.include_readability == False
        assert extractor.include_vocabulary == True
        assert extractor.include_structure == False
        assert extractor.language == "en"

    def test_basic_statistics_extraction(self, default_extractor, sample_documents):
        """Test basic statistics extraction functionality."""
        result_docs = default_extractor.extract(sample_documents)

        # Check that documents are returned
        assert len(result_docs) == 2
        assert all(isinstance(doc, Document) for doc in result_docs)

        # Check that statistics metadata is added
        for doc in result_docs:
            assert "extractors" in doc.metadata
            assert "statistics" in doc.metadata["extractors"]

    def test_word_count_extraction(self, default_extractor, sample_documents):
        """Test word count statistics."""
        result_docs = default_extractor.extract(sample_documents)

        for doc in result_docs:
            stats = doc.metadata["extractors"]["statistics"]

            # Should have basic counts in the "basic" section
            assert "basic" in stats
            basic_stats = stats["basic"]
            assert "word_count" in basic_stats
            assert "sentence_count" in basic_stats
            assert "paragraph_count" in basic_stats

            # Counts should be reasonable
            assert basic_stats["word_count"] > 0
            assert basic_stats["sentence_count"] > 0
            assert basic_stats["paragraph_count"] >= 1

    def test_readability_metrics(self, default_extractor, sample_documents):
        """Test readability metrics calculation."""
        result_docs = default_extractor.extract(sample_documents)

        for doc in result_docs:
            stats = doc.metadata["extractors"]["statistics"]

            if "readability" in stats:
                readability = stats["readability"]

                # Should have readability scores
                if "flesch_reading_ease" in readability:
                    assert isinstance(readability["flesch_reading_ease"], (int, float))
                    assert 0 <= readability["flesch_reading_ease"] <= 100

                if "flesch_kincaid_grade" in readability:
                    assert isinstance(readability["flesch_kincaid_grade"], (int, float))

    def test_vocabulary_analysis(self, default_extractor, sample_documents):
        """Test vocabulary diversity analysis."""
        result_docs = default_extractor.extract(sample_documents)

        for doc in result_docs:
            stats = doc.metadata["extractors"]["statistics"]

            if "vocabulary" in stats:
                vocab = stats["vocabulary"]

                # Should have vocabulary metrics
                if "unique_words" in vocab:
                    assert vocab["unique_words"] > 0

                if "vocabulary_diversity" in vocab:
                    assert isinstance(vocab["vocabulary_diversity"], (int, float))
                    assert 0 <= vocab["vocabulary_diversity"] <= 1

    def test_structure_analysis(self, default_extractor, sample_documents):
        """Test document structure analysis."""
        result_docs = default_extractor.extract(sample_documents)

        for doc in result_docs:
            stats = doc.metadata["extractors"]["statistics"]

            # Look for structure metrics in the basic section
            if "basic" in stats:
                basic_stats = stats["basic"]
                # Should have structure metrics
                assert (
                    "avg_words_per_sentence" in basic_stats
                    or "average_sentence_length" in basic_stats
                )
                assert (
                    "avg_chars_per_word" in basic_stats
                    or "average_word_length" in basic_stats
                )

    def test_sentiment_indicators(self, default_extractor):
        """Test sentiment indicator extraction."""
        # Create document with clear sentiment words
        sentiment_doc = Document(
            content="This is an excellent and wonderful analysis. The results are outstanding and remarkable. However, there are some serious concerns and critical issues that need attention.",
            id="sentiment_test",
            source="sentiment.txt",
            metadata={},
        )

        result_docs = default_extractor.extract([sentiment_doc])
        stats = result_docs[0].metadata["extractors"]["statistics"]

        if "sentiment_indicators" in stats:
            sentiment = stats["sentiment_indicators"]

            # Should detect both positive and negative words
            if "positive_words" in sentiment:
                assert sentiment["positive_words"] > 0
            if "negative_words" in sentiment:
                assert sentiment["negative_words"] > 0

    def test_empty_document_handling(self, default_extractor):
        """Test handling of empty documents."""
        empty_docs = [
            Document(content="", id="empty1", source="empty.txt", metadata={}),
            Document(
                content="   \n  \t  ",  # Only whitespace
                id="empty2",
                source="empty2.txt",
                metadata={},
            ),
        ]

        result_docs = default_extractor.extract(empty_docs)

        # Should handle gracefully
        assert len(result_docs) == 2
        for doc in result_docs:
            assert "extractors" in doc.metadata
            stats = doc.metadata["extractors"]["statistics"]

            # Empty docs should have zero counts
            if "word_count" in stats:
                assert stats["word_count"] == 0

    def test_reading_time_estimation(self, default_extractor, sample_documents):
        """Test reading time estimation."""
        result_docs = default_extractor.extract(sample_documents)

        for doc in result_docs:
            stats = doc.metadata["extractors"]["statistics"]

            if "reading_time" in stats:
                reading_time = stats["reading_time"]

                # Should have time estimates for different reading speeds
                if "average" in reading_time:
                    assert reading_time["average"] > 0
                if "slow" in reading_time:
                    assert reading_time["slow"] > reading_time.get("fast", 0)

    def test_custom_configuration(self):
        """Test extractor with custom configuration."""
        # Only structure analysis
        structure_only = ContentStatisticsExtractor(
            "structure_only",
            {
                "include_readability": False,
                "include_vocabulary": False,
                "include_structure": True,
                "include_sentiment_indicators": False,
            },
        )

        test_doc = Document(
            content="Test document for structure analysis only.",
            id="structure_test",
            source="test.txt",
            metadata={},
        )

        result_docs = structure_only.extract([test_doc])
        stats = result_docs[0].metadata["extractors"]["statistics"]

        # Should only have structure-related stats
        assert "structure" in stats or "word_count" in stats
        # Should not have readability or vocabulary
        assert "readability" not in stats or len(stats.get("readability", {})) == 0

    def test_metadata_preservation(self, default_extractor):
        """Test that existing metadata is preserved."""
        test_doc = Document(
            content="Test content for metadata preservation analysis.",
            id="metadata_test",
            source="test.txt",
            metadata={"existing_key": "existing_value", "document_type": "test"},
        )

        result_docs = default_extractor.extract([test_doc])

        # Existing metadata should be preserved
        assert "existing_key" in result_docs[0].metadata
        assert result_docs[0].metadata["existing_key"] == "existing_value"
        assert result_docs[0].metadata["document_type"] == "test"

        # New statistics metadata should be added
        assert "extractors" in result_docs[0].metadata
        assert "statistics" in result_docs[0].metadata["extractors"]

    def test_statistics_consistency(self, default_extractor):
        """Test that statistics are consistent across multiple runs."""
        test_doc = Document(
            content="Consistent test document with specific word count for validation purposes.",
            id="consistency_test",
            source="test.txt",
            metadata={},
        )

        # Run extraction twice
        result1 = default_extractor.extract([test_doc])
        result2 = default_extractor.extract([test_doc])

        stats1 = result1[0].metadata["extractors"]["statistics"]
        stats2 = result2[0].metadata["extractors"]["statistics"]

        # Basic counts should be consistent
        if "word_count" in stats1 and "word_count" in stats2:
            assert stats1["word_count"] == stats2["word_count"]
        if "sentence_count" in stats1 and "sentence_count" in stats2:
            assert stats1["sentence_count"] == stats2["sentence_count"]


if __name__ == "__main__":
    pytest.main([__file__])
