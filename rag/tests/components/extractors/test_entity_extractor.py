"""Tests for Entity Extractor component.

Phase 18: Enhanced tests for:
- Entity and Relationship dataclasses
- Graph store integration
- LLM relationship extraction
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from components.extractors.entity_extractor.entity_extractor import (
    Entity,
    EntityExtractor,
    Relationship,
    extract_entities_to_graph,
)
from core.base import Document


class TestEntityDataclass:
    """Test Entity dataclass functionality."""

    def test_entity_basic_creation(self):
        """Test basic Entity creation."""
        entity = Entity(
            name="John Smith",
            entity_type="PERSON",
            source_doc="doc1",
        )
        assert entity.name == "John Smith"
        assert entity.entity_type == "PERSON"
        assert entity.source_doc == "doc1"
        assert entity.confidence == 1.0  # Default
        assert entity.method == "spacy"  # Default

    def test_entity_with_all_fields(self):
        """Test Entity with all fields specified."""
        entity = Entity(
            name="Apple Inc.",
            entity_type="ORG",
            source_doc="doc2",
            span_start=10,
            span_end=20,
            confidence=0.95,
            method="regex",
            properties={"ticker": "AAPL"},
        )
        assert entity.span_start == 10
        assert entity.span_end == 20
        assert entity.confidence == 0.95
        assert entity.method == "regex"
        assert entity.properties["ticker"] == "AAPL"

    def test_entity_normalized_name(self):
        """Test Entity normalized_name property."""
        entity = Entity(name="  John SMITH  ", entity_type="PERSON", source_doc="doc")
        assert entity.normalized_name == "john smith"

    def test_entity_id_generation(self):
        """Test Entity ID generation is consistent."""
        entity1 = Entity(name="John Smith", entity_type="PERSON", source_doc="doc1")
        entity2 = Entity(name="john smith", entity_type="PERSON", source_doc="doc2")

        # Same normalized name + type should produce same ID
        assert entity1.entity_id == entity2.entity_id

    def test_entity_id_unique_for_different_types(self):
        """Test Entity IDs are different for different entity types."""
        entity1 = Entity(name="Apple", entity_type="ORG", source_doc="doc")
        entity2 = Entity(name="Apple", entity_type="PRODUCT", source_doc="doc")

        assert entity1.entity_id != entity2.entity_id

    def test_entity_to_dict(self):
        """Test Entity.to_dict() method."""
        entity = Entity(
            name="Test Entity",
            entity_type="ORG",
            source_doc="doc1",
            span_start=5,
            span_end=15,
            confidence=0.9,
            method="spacy",
            properties={"key": "value"},
        )
        result = entity.to_dict()

        assert result["name"] == "Test Entity"
        assert result["type"] == "ORG"
        assert result["source_doc"] == "doc1"
        assert result["span_start"] == 5
        assert result["span_end"] == 15
        assert result["confidence"] == 0.9
        assert result["method"] == "spacy"
        assert result["properties"] == {"key": "value"}
        assert "id" in result  # entity_id should be included


class TestRelationshipDataclass:
    """Test Relationship dataclass functionality."""

    def test_relationship_basic_creation(self):
        """Test basic Relationship creation."""
        rel = Relationship(
            source_entity="John Smith",
            target_entity="Acme Corp",
            relationship_type="works_at",
            source_doc="doc1",
        )
        assert rel.source_entity == "John Smith"
        assert rel.target_entity == "Acme Corp"
        assert rel.relationship_type == "works_at"
        assert rel.source_doc == "doc1"
        assert rel.confidence == 0.8  # Default

    def test_relationship_with_properties(self):
        """Test Relationship with properties."""
        rel = Relationship(
            source_entity="Company A",
            target_entity="Company B",
            relationship_type="acquired",
            source_doc="doc2",
            confidence=0.95,
            properties={"date": "2024-01-01", "value": "1B"},
        )
        assert rel.confidence == 0.95
        assert rel.properties["date"] == "2024-01-01"

    def test_relationship_to_dict(self):
        """Test Relationship.to_dict() method."""
        rel = Relationship(
            source_entity="A",
            target_entity="B",
            relationship_type="related_to",
            source_doc="doc1",
            confidence=0.75,
            properties={"note": "test"},
        )
        result = rel.to_dict()

        assert result["source"] == "A"
        assert result["target"] == "B"
        assert result["type"] == "related_to"
        assert result["source_doc"] == "doc1"
        assert result["confidence"] == 0.75
        assert result["properties"] == {"note": "test"}


class TestEntityExtractor:
    """Test EntityExtractor functionality."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                content="John Smith works at Apple Inc. in San Francisco. He joined the company in 2020 and has been leading the AI research team.",
                id="doc1",
                source="test_source.txt",
                metadata={},
            ),
            Document(
                content="Microsoft announced a new partnership with OpenAI. The collaboration will focus on developing advanced language models for enterprise applications.",
                id="doc2",
                source="test_source2.txt",
                metadata={},
            ),
        ]

    @pytest.fixture
    def default_extractor(self):
        """Create default entity extractor."""
        return EntityExtractor(
            "test_extractor",
            {
                "entity_types": ["PERSON", "ORG", "GPE", "DATE"],
                "use_fallback": True,
                "min_entity_length": 2,
            },
        )

    def test_extractor_initialization(self):
        """Test extractor initialization with different configs."""
        # Default config
        extractor = EntityExtractor()
        assert extractor is not None
        assert isinstance(extractor.entity_types, set)
        assert "PERSON" in extractor.entity_types
        assert "ORG" in extractor.entity_types
        assert "GPE" in extractor.entity_types

        # Custom config
        custom_config = {
            "entity_types": ["PERSON", "ORG"],
            "use_fallback": False,
            "min_entity_length": 3,
        }
        extractor = EntityExtractor("custom_extractor", custom_config)
        assert extractor.entity_types == set(["PERSON", "ORG"])
        assert extractor.use_fallback is False
        assert extractor.min_entity_length == 3

    def test_extractor_with_llm_client(self):
        """Test extractor initialization with LLM client."""
        mock_llm = MagicMock()
        extractor = EntityExtractor(
            "llm_extractor",
            {"extract_relationships": True},
            llm_client=mock_llm,
        )
        assert extractor.llm_client is mock_llm
        assert extractor.extract_relationships is True

    def test_entity_extraction_basic(self, default_extractor, sample_documents):
        """Test basic entity extraction functionality."""
        # Process documents
        result_docs = default_extractor.extract(sample_documents)

        # Check that documents are returned
        assert len(result_docs) == 2
        assert all(isinstance(doc, Document) for doc in result_docs)

        # Check that extractors metadata is added
        for doc in result_docs:
            assert "extractors" in doc.metadata
            assert "entities" in doc.metadata["extractors"]

    def test_entity_extraction_returns_entity_objects(
        self, default_extractor, sample_documents
    ):
        """Test that extract_entities returns Entity objects."""
        entities = default_extractor.extract_entities(sample_documents[0])

        assert isinstance(entities, list)
        for entity in entities:
            assert isinstance(entity, Entity)

    def test_entity_extraction_content(self, default_extractor, sample_documents):
        """Test that entities are properly extracted from content."""
        result_docs = default_extractor.extract(sample_documents)

        # Check first document entities
        doc1_entities = result_docs[0].metadata["extractors"]["entities"]
        assert isinstance(doc1_entities, list)
        assert len(doc1_entities) > 0

        # Each entity should have required fields
        for entity in doc1_entities:
            assert "name" in entity
            assert "type" in entity
            assert "id" in entity

    def test_entities_by_type(self, default_extractor, sample_documents):
        """Test entities_by_type metadata is populated."""
        result_docs = default_extractor.extract(sample_documents)

        # Check entities_by_type structure
        doc1 = result_docs[0]
        assert "entities_by_type" in doc1.metadata["extractors"]

        entities_by_type = doc1.metadata["extractors"]["entities_by_type"]
        assert isinstance(entities_by_type, dict)

        # Each type should have list of entities with text
        for _entity_type, entity_list in entities_by_type.items():
            assert isinstance(entity_list, list)
            for entity in entity_list:
                assert "text" in entity
                assert "confidence" in entity

    def test_empty_document_handling(self, default_extractor):
        """Test handling of empty documents."""
        empty_docs = [
            Document(content="", id="empty1", source="empty.txt", metadata={}),
            Document(
                content="   ",  # Only whitespace
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
            entities = doc.metadata["extractors"]["entities"]
            assert isinstance(entities, list)
            assert len(entities) == 0

    def test_extract_entities_from_text(self, default_extractor):
        """Test extract_entities_from_text convenience method."""
        text = "Bill Gates founded Microsoft in Seattle."
        entities = default_extractor.extract_entities_from_text(text, "test_source")

        assert isinstance(entities, list)
        for entity in entities:
            assert isinstance(entity, Entity)
            assert entity.source_doc == "test_source"

    def test_fallback_extraction(self):
        """Test fallback extraction when spaCy is not available."""
        # Force fallback mode
        extractor = EntityExtractor(
            "fallback_extractor",
            {"entity_types": ["PERSON", "ORG"], "use_fallback": True},
        )

        test_doc = Document(
            content="Barack Obama worked at the White House. He was the 44th President.",
            id="fallback_test",
            source="test.txt",
            metadata={},
        )

        result_docs = extractor.extract([test_doc])

        # Should still extract some entities using fallback
        assert len(result_docs) == 1
        assert "extractors" in result_docs[0].metadata
        entities = result_docs[0].metadata["extractors"]["entities"]
        assert isinstance(entities, list)

    def test_entity_deduplication(self, default_extractor):
        """Test entity deduplication within document."""
        test_doc = Document(
            content="John Smith met John Smith at the office. John Smith was early.",
            id="dedup_test",
            source="test.txt",
            metadata={},
        )

        entities = default_extractor.extract_entities(test_doc)

        # Should deduplicate by normalized name + type
        person_entities = [e for e in entities if e.entity_type == "PERSON"]
        names = [e.normalized_name for e in person_entities]
        unique_names = list(set(names))

        # There should be fewer or equal entities after dedup
        assert len(names) == len(unique_names)

    def test_entity_cache(self, default_extractor, sample_documents):
        """Test entity caching across documents."""
        # Clear cache first
        default_extractor.clear_cache()
        assert len(default_extractor.get_cached_entities()) == 0

        # Extract from multiple documents
        default_extractor.extract(sample_documents)

        # Cache should have entities
        cached = default_extractor.get_cached_entities()
        assert len(cached) > 0

    def test_metadata_preservation(self, default_extractor):
        """Test that existing metadata is preserved."""
        test_doc = Document(
            content="Test content with entities.",
            id="metadata_test",
            source="test.txt",
            metadata={"existing_key": "existing_value"},
        )

        result_docs = default_extractor.extract([test_doc])

        # Existing metadata should be preserved
        assert "existing_key" in result_docs[0].metadata
        assert result_docs[0].metadata["existing_key"] == "existing_value"

        # New extractor metadata should be added
        assert "extractors" in result_docs[0].metadata
        assert "entities" in result_docs[0].metadata["extractors"]

    def test_get_supported_entities(self, default_extractor):
        """Test get_supported_entities returns descriptions."""
        supported = default_extractor.get_supported_entities()

        assert isinstance(supported, dict)
        assert "PERSON" in supported
        assert "ORG" in supported
        assert isinstance(supported["PERSON"], str)


class TestRelationshipExtraction:
    """Test LLM-based relationship extraction."""

    def test_relationship_extraction_without_llm(self):
        """Test that relationship extraction gracefully handles missing LLM."""
        extractor = EntityExtractor(
            config={"extract_relationships": True},
            llm_client=None,  # No LLM
        )

        doc = Document(
            content="John works at Apple.",
            id="test",
        )
        entities = extractor.extract_entities(doc)

        # Should return empty list without LLM
        relationships = extractor.extract_relationships_llm(doc, entities)
        assert relationships == []

    def test_relationship_extraction_with_mock_llm(self):
        """Test relationship extraction with mocked LLM."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """John Smith | works_at | Apple Inc.
Apple Inc. | located_in | San Francisco"""

        extractor = EntityExtractor(
            config={"extract_relationships": True},
            llm_client=mock_llm,
        )

        doc = Document(
            content="John Smith works at Apple Inc. in San Francisco.",
            id="test",
        )
        entities = [
            Entity(name="John Smith", entity_type="PERSON", source_doc="test"),
            Entity(name="Apple Inc.", entity_type="ORG", source_doc="test"),
            Entity(name="San Francisco", entity_type="GPE", source_doc="test"),
        ]

        relationships = extractor.extract_relationships_llm(doc, entities)

        assert len(relationships) == 2
        assert relationships[0].source_entity == "John Smith"
        assert relationships[0].target_entity == "Apple Inc."
        assert relationships[0].relationship_type == "works_at"

    def test_relationship_extraction_too_few_entities(self):
        """Test relationship extraction with < 2 entities."""
        mock_llm = MagicMock()
        extractor = EntityExtractor(
            config={"extract_relationships": True},
            llm_client=mock_llm,
        )

        doc = Document(content="Just one entity.", id="test")
        entities = [Entity(name="One", entity_type="ORG", source_doc="test")]

        relationships = extractor.extract_relationships_llm(doc, entities)

        assert relationships == []
        mock_llm.generate.assert_not_called()

    def test_parse_relationships_handles_malformed_input(self):
        """Test _parse_relationships handles malformed LLM output."""
        extractor = EntityExtractor()

        # Test various malformed inputs
        result = extractor._parse_relationships("", "doc")
        assert result == []

        result = extractor._parse_relationships("no pipes here", "doc")
        assert result == []

        result = extractor._parse_relationships("only | one | pipe", "doc")
        assert len(result) == 1  # Should work with exactly 3 parts

        result = extractor._parse_relationships("too | many | pipe | parts", "doc")
        assert result == []


class TestExtractEntitiesToGraph:
    """Test extract_entities_to_graph integration function."""

    def test_extract_to_graph_basic(self):
        """Test basic graph extraction."""
        mock_graph_store = MagicMock()
        mock_graph_store.add_node.return_value = "node_123"

        doc = Document(
            content="John Smith works at Apple Inc.",
            id="doc1",
        )

        result = extract_entities_to_graph(
            document=doc,
            graph_store=mock_graph_store,
            config={"use_fallback": True},
        )

        assert "document_id" in result
        assert result["document_id"] == "doc1"
        assert "entities_extracted" in result
        assert "nodes_created" in result
        assert "entity_types" in result
        assert "entities" in result

    def test_extract_to_graph_with_linkage(self):
        """Test graph extraction with linkage table."""
        mock_graph_store = MagicMock()
        mock_graph_store.add_node.return_value = "node_123"

        mock_linkage = MagicMock()

        doc = Document(
            content="Test Entity Name here.",
            id="doc1",
        )

        result = extract_entities_to_graph(
            document=doc,
            graph_store=mock_graph_store,
            linkage_table=mock_linkage,
            config={"use_fallback": True},
        )

        # If any nodes were created, linkage should be called
        if result["nodes_created"] > 0:
            mock_linkage.link.assert_called()

    def test_extract_to_graph_with_relationships(self):
        """Test graph extraction with LLM relationship extraction."""
        mock_graph_store = MagicMock()
        mock_graph_store.add_node.return_value = "node_123"
        mock_graph_store.add_edge.return_value = "edge_456"

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Entity A | related_to | Entity B"

        doc = Document(
            content="Entity A and Entity B are related.",
            id="doc1",
        )

        result = extract_entities_to_graph(
            document=doc,
            graph_store=mock_graph_store,
            config={"use_fallback": True, "extract_relationships": True},
            llm_client=mock_llm,
        )

        assert "edges_created" in result


class TestEntityExtractorDependencies:
    """Test dependency handling."""

    def test_get_dependencies_with_fallback(self):
        """Test dependencies when fallback is enabled."""
        extractor = EntityExtractor(config={"use_fallback": True})
        deps = extractor.get_dependencies()
        assert "spacy" not in deps

    def test_get_dependencies_without_fallback(self):
        """Test dependencies when fallback is disabled."""
        extractor = EntityExtractor(config={"use_fallback": False})
        deps = extractor.get_dependencies()
        assert "spacy" in deps

    def test_validate_dependencies_with_fallback(self):
        """Test validation always passes with fallback."""
        extractor = EntityExtractor(config={"use_fallback": True})
        assert extractor.validate_dependencies() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
