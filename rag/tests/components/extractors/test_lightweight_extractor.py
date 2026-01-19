"""Tests for LightweightExtractor."""

import pytest

from core.base import Document


class TestLightweightExtractor:
    """Tests for LightweightExtractor using YAKE + GLiNER."""

    @pytest.fixture
    def extractor(self):
        """Create a LightweightExtractor instance."""
        from components.extractors.lightweight import LightweightExtractor
        return LightweightExtractor()

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        return Document(
            content="""
            Machine learning is a subset of artificial intelligence (AI) that enables
            systems to learn and improve from experience without being explicitly programmed.
            Google and Microsoft are leading companies in AI research.
            The technology has applications in healthcare, finance, and transportation.
            Dr. John Smith from Stanford University published groundbreaking research in 2023.
            """,
            metadata={"source": "test"},
            id="test_doc_1"
        )

    @pytest.fixture
    def short_document(self):
        """Create a short document for testing."""
        return Document(
            content="This is a short test document.",
            metadata={"source": "test"},
            id="test_doc_2"
        )

    def test_extractor_initialization(self, extractor):
        """Test that the extractor initializes correctly."""
        assert extractor.name == "LightweightExtractor"
        assert extractor.keyword_count == 5
        assert extractor.summary_sentences == 3

    def test_extract_keywords(self, extractor, sample_document):
        """Test keyword extraction with YAKE."""
        if not extractor._yake_available:
            pytest.skip("YAKE not available")

        documents = extractor.extract([sample_document])

        assert len(documents) == 1
        doc = documents[0]

        assert "keywords" in doc.metadata
        keywords = doc.metadata["keywords"]
        assert isinstance(keywords, list)
        assert len(keywords) <= extractor.keyword_count

    def test_generate_summary(self, extractor, sample_document):
        """Test summary generation."""
        documents = extractor.extract([sample_document])

        assert len(documents) == 1
        doc = documents[0]

        assert "summary" in doc.metadata
        summary = doc.metadata["summary"]
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert len(summary) <= 500  # Max summary length

    def test_extraction_methods_tracked(self, extractor, sample_document):
        """Test that extraction methods are tracked in metadata."""
        documents = extractor.extract([sample_document])

        assert len(documents) == 1
        doc = documents[0]

        assert "extraction_methods" in doc.metadata
        methods = doc.metadata["extraction_methods"]
        assert isinstance(methods, list)
        assert "sentence_summary" in methods

    def test_extract_multiple_documents(self, extractor, sample_document, short_document):
        """Test extraction on multiple documents."""
        documents = extractor.extract([sample_document, short_document])

        assert len(documents) == 2

        for doc in documents:
            assert "summary" in doc.metadata
            assert "extractor" in doc.metadata
            assert doc.metadata["extractor"] == "LightweightExtractor"

    def test_original_content_preserved(self, extractor, sample_document):
        """Test that original content is preserved."""
        original_content = sample_document.content
        documents = extractor.extract([sample_document])

        assert len(documents) == 1
        assert documents[0].content == original_content

    def test_original_metadata_preserved(self, extractor, sample_document):
        """Test that original metadata is preserved."""
        documents = extractor.extract([sample_document])

        assert len(documents) == 1
        assert documents[0].metadata["source"] == "test"

    def test_document_id_preserved(self, extractor, sample_document):
        """Test that document ID is preserved."""
        documents = extractor.extract([sample_document])

        assert len(documents) == 1
        assert documents[0].id == "test_doc_1"

    def test_empty_document_list(self, extractor):
        """Test extraction on empty document list."""
        documents = extractor.extract([])
        assert len(documents) == 0

    def test_short_document_summary(self, extractor, short_document):
        """Test summary generation for short documents."""
        documents = extractor.extract([short_document])

        assert len(documents) == 1
        doc = documents[0]

        assert "summary" in doc.metadata
        # Short document summary should be the content itself
        assert len(doc.metadata["summary"]) > 0

    def test_validate_config(self, extractor):
        """Test configuration validation."""
        assert extractor.validate_config() is True

    def test_invalid_keyword_count(self):
        """Test that invalid keyword_count raises error."""
        from components.extractors.lightweight import LightweightExtractor

        extractor = LightweightExtractor(config={"keyword_count": 0})
        with pytest.raises(ValueError):
            extractor.validate_config()

    def test_invalid_summary_sentences(self):
        """Test that invalid summary_sentences raises error."""
        from components.extractors.lightweight import LightweightExtractor

        extractor = LightweightExtractor(config={"summary_sentences": 0})
        with pytest.raises(ValueError):
            extractor.validate_config()

    def test_get_dependencies(self, extractor):
        """Test get_dependencies returns required packages."""
        deps = extractor.get_dependencies()
        assert isinstance(deps, list)
        assert "yake" in deps

    def test_process_interface(self, extractor, sample_document):
        """Test the process() interface returns ProcessingResult."""
        result = extractor.process([sample_document])

        assert hasattr(result, 'documents')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'metrics')
        assert len(result.documents) == 1


class TestLightweightExtractorGLiNER:
    """Tests for GLiNER entity extraction (optional)."""

    @pytest.fixture
    def extractor_with_gliner(self):
        """Create extractor with GLiNER enabled."""
        from components.extractors.lightweight import LightweightExtractor
        return LightweightExtractor(config={"use_gliner": True})

    @pytest.fixture
    def sample_document_with_entities(self):
        """Create a document with clear entities."""
        return Document(
            content="""
            Apple Inc. is headquartered in Cupertino, California.
            Tim Cook is the CEO of Apple.
            The company was founded by Steve Jobs in 1976.
            Microsoft and Google are competitors in the technology industry.
            """,
            metadata={"source": "test"},
            id="test_entities"
        )

    def test_gliner_available_check(self, extractor_with_gliner):
        """Test GLiNER availability check."""
        # Just check the flag exists
        assert hasattr(extractor_with_gliner, '_gliner_available')

    def test_entities_extracted_if_gliner_available(
        self, extractor_with_gliner, sample_document_with_entities
    ):
        """Test entity extraction if GLiNER is available."""
        if not extractor_with_gliner._gliner_available:
            pytest.skip("GLiNER not available")

        documents = extractor_with_gliner.extract([sample_document_with_entities])

        assert len(documents) == 1
        doc = documents[0]

        assert "entities" in doc.metadata
        entities = doc.metadata["entities"]
        assert isinstance(entities, list)

        # Check entity structure
        if len(entities) > 0:
            entity = entities[0]
            assert "text" in entity
            assert "label" in entity
            assert "score" in entity

    def test_gliner_disabled(self):
        """Test that GLiNER can be disabled."""
        from components.extractors.lightweight import LightweightExtractor

        extractor = LightweightExtractor(config={"use_gliner": False})
        assert extractor.use_gliner is False


class TestLightweightExtractorRegistry:
    """Test extractor registry integration."""

    def test_extractor_registered(self):
        """Test that LightweightExtractor is in the registry."""
        from components.extractors import registry

        assert "LightweightExtractor" in registry.list_extractors()

    def test_create_from_registry(self):
        """Test creating extractor from registry."""
        from components.extractors import registry

        extractor = registry.create("LightweightExtractor")
        assert extractor is not None
        assert extractor.name == "LightweightExtractor"

    def test_create_with_config(self):
        """Test creating extractor with custom config."""
        from components.extractors import registry

        extractor = registry.create(
            "LightweightExtractor",
            config={"keyword_count": 10, "summary_sentences": 5}
        )

        assert extractor is not None
        assert extractor.keyword_count == 10
        assert extractor.summary_sentences == 5
