"""Entity extraction using spaCy and other local NLP libraries.

Phase 18: Entity Extraction Pipeline

Enhanced with:
- Entity and Relationship dataclasses for structured output
- Optional LLM-based relationship extraction
- Graph store integration for automatic node/edge creation
- Entity deduplication across documents
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from components.extractors.base import BaseExtractor
from core.base import Document
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.extractors.entity_extractor.entity_extractor")


@dataclass
class Entity:
    """Represents an extracted named entity.

    Attributes:
        name: The entity text as extracted
        entity_type: Type of entity (PERSON, ORG, GPE, etc.)
        source_doc: ID of the source document
        span_start: Character offset where entity starts
        span_end: Character offset where entity ends
        confidence: Extraction confidence score (0-1)
        method: Extraction method used (spacy, regex, llm)
        properties: Additional entity properties
    """

    name: str
    entity_type: str
    source_doc: str
    span_start: int = 0
    span_end: int = 0
    confidence: float = 1.0
    method: str = "spacy"
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_name(self) -> str:
        """Get normalized entity name for deduplication."""
        return self.name.strip().lower()

    @property
    def entity_id(self) -> str:
        """Generate a unique ID based on normalized name and type."""
        key = f"{self.entity_type}:{self.normalized_name}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary format."""
        return {
            "name": self.name,
            "type": self.entity_type,
            "id": self.entity_id,
            "source_doc": self.source_doc,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "confidence": self.confidence,
            "method": self.method,
            "properties": self.properties,
        }


@dataclass
class Relationship:
    """Represents a relationship between two entities.

    Attributes:
        source_entity: Name or ID of the source entity
        target_entity: Name or ID of the target entity
        relationship_type: Type of relationship (e.g., works_at, located_in)
        source_doc: ID of the source document
        confidence: Extraction confidence score (0-1)
        properties: Additional relationship properties
    """

    source_entity: str
    target_entity: str
    relationship_type: str
    source_doc: str
    confidence: float = 0.8
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert relationship to dictionary format."""
        return {
            "source": self.source_entity,
            "target": self.target_entity,
            "type": self.relationship_type,
            "source_doc": self.source_doc,
            "confidence": self.confidence,
            "properties": self.properties,
        }


class EntityExtractor(BaseExtractor):
    """
    Entity extraction using spaCy for named entity recognition.

    Extracts persons, organizations, locations, dates, money, etc.
    Falls back to regex patterns if spaCy is not available.

    Enhanced Features (Phase 18):
    - Structured Entity/Relationship dataclasses
    - Optional LLM-based relationship extraction
    - Graph store integration
    - Entity deduplication across documents

    Configuration:
        model: spaCy model name (default: en_core_web_sm)
        entity_types: List of entity types to extract
        use_fallback: Whether to use regex fallback if spaCy unavailable
        min_entity_length: Minimum character length for entities
        extract_relationships: Whether to use LLM for relationships
        deduplicate: Whether to deduplicate entities across documents
    """

    def __init__(
        self,
        name: str = "EntityExtractor",
        config: dict[str, Any] | None = None,
        llm_client: Any = None,
    ):
        super().__init__(name, config)

        # LLM client for relationship extraction
        self.llm_client = llm_client

        # Configuration
        self.model_name = self.config.get("model", "en_core_web_sm")
        # Handle entity_types - use default if None or not specified
        default_entity_types = [
            "PERSON",
            "ORG",
            "GPE",
            "DATE",
            "TIME",
            "MONEY",
            "PERCENT",
            "PRODUCT",
            "EVENT",
            "LAW",
            "LANGUAGE",
            "NORP",
            "LOC",
            "FAC",
            "WORK_OF_ART",
        ]
        entity_types_config = self.config.get("entity_types")
        self.entity_types = set(
            entity_types_config
            if entity_types_config is not None
            else default_entity_types
        )
        self.use_fallback = self.config.get("use_fallback", True)
        self.min_entity_length = self.config.get("min_entity_length", 2)
        self.extract_relationships = self.config.get("extract_relationships", False)
        self.deduplicate = self.config.get("deduplicate", True)

        # Try to load spaCy model
        self.nlp = None
        self._load_spacy_model()

        # Regex patterns for fallback
        self.regex_patterns = self._initialize_regex_patterns()

        # Entity deduplication cache (for cross-document deduplication)
        self._entity_cache: dict[str, Entity] = {}

    def _load_spacy_model(self) -> None:
        """Load spaCy model if available."""
        try:
            import spacy

            self.nlp = spacy.load(self.model_name)
            self.logger.info(f"Loaded spaCy model: {self.model_name}")
        except ImportError:
            self.logger.warning("spaCy not available, will use regex fallback")
        except OSError:
            self.logger.warning(
                f"spaCy model {self.model_name} not found, will use regex fallback"
            )

    def _initialize_regex_patterns(self) -> dict[str, re.Pattern]:
        """Initialize regex patterns for entity extraction fallback."""
        patterns = {}

        # Email addresses
        patterns["EMAIL"] = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )

        # Phone numbers (US format)
        patterns["PHONE"] = re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b"
        )

        # URLs
        patterns["URL"] = re.compile(
            r"https?://[^\s<>\"'(){}[\]]+(?:[^\s<>\"'(){}[\].,;!?])"
        )

        # Currency amounts
        patterns["MONEY"] = re.compile(
            r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|usd)\b"
        )

        # Percentages
        patterns["PERCENT"] = re.compile(
            r"\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*percent\b"
        )

        # Dates (various formats)
        patterns["DATE"] = re.compile(
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|"
            r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
            re.IGNORECASE,
        )

        # Times
        patterns["TIME"] = re.compile(
            r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b"
        )

        # Social Security Numbers (masked for privacy)
        patterns["SSN"] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

        # Credit Card Numbers (basic pattern)
        patterns["CREDIT_CARD"] = re.compile(
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        )

        return patterns

    def extract(self, documents: list[Document]) -> list[Document]:
        """Extract entities from documents and add to metadata.

        Args:
            documents: List of documents to process

        Returns:
            Documents with entities added to metadata
        """
        for doc in documents:
            try:
                # Extract entities
                entities = self.extract_entities(doc)

                # Add to metadata
                if "extractors" not in doc.metadata:
                    doc.metadata["extractors"] = {}

                doc.metadata["extractors"]["entities"] = [e.to_dict() for e in entities]

                # Also add simplified lists for easy access
                entities_by_type: dict[str, list[dict[str, Any]]] = {}
                for entity in entities:
                    if entity.entity_type not in entities_by_type:
                        entities_by_type[entity.entity_type] = []
                    entities_by_type[entity.entity_type].append(
                        {
                            "text": entity.name,
                            "start": entity.span_start,
                            "end": entity.span_end,
                            "confidence": entity.confidence,
                            "method": entity.method,
                        }
                    )

                doc.metadata["extractors"]["entities_by_type"] = entities_by_type

                # Add entity name lists by type for backward compatibility
                for entity_type, entity_list in entities_by_type.items():
                    doc.metadata[f"entities_{entity_type.lower()}"] = [
                        e["text"] for e in entity_list
                    ]

                # Extract relationships if enabled
                if (
                    self.extract_relationships
                    and self.llm_client
                    and len(entities) >= 2
                ):
                    relationships = self.extract_relationships_llm(doc, entities)
                    doc.metadata["extractors"]["relationships"] = [
                        r.to_dict() for r in relationships
                    ]

                self.logger.debug(
                    f"Extracted {len(entities)} entities from document {doc.id}"
                )

            except Exception as e:
                self.logger.error(
                    f"Entity extraction failed for document {doc.id}: {e}"
                )

        return documents

    def extract_entities(self, document: Document) -> list[Entity]:
        """Extract named entities from a single document.

        Args:
            document: Document to extract entities from

        Returns:
            List of extracted Entity objects
        """
        content = document.content
        if not content or not isinstance(content, str):
            return []

        if self.nlp:
            entities = self._extract_spacy_entities(content, document.id)
        else:
            entities = self._extract_regex_entities(content, document.id)

        # Deduplicate if enabled
        if self.deduplicate:
            entities = self._deduplicate_entities(entities)

        return entities

    def extract_entities_from_text(
        self, text: str, source_id: str = "unknown"
    ) -> list[Entity]:
        """Extract entities from raw text (convenience method).

        Args:
            text: Text to extract entities from
            source_id: Source identifier for the text

        Returns:
            List of extracted Entity objects
        """
        doc = Document(id=source_id, content=text)
        return self.extract_entities(doc)

    def _extract_spacy_entities(self, text: str, source_doc: str) -> list[Entity]:
        """Extract entities using spaCy."""
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if (
                ent.label_ in self.entity_types
                and len(ent.text.strip()) >= self.min_entity_length
            ):
                entity = Entity(
                    name=ent.text.strip(),
                    entity_type=ent.label_,
                    source_doc=source_doc,
                    span_start=ent.start_char,
                    span_end=ent.end_char,
                    confidence=getattr(ent, "confidence", 1.0)
                    if hasattr(ent, "confidence")
                    else 1.0,
                    method="spacy",
                    properties={
                        "kb_id": ent.kb_id_
                        if hasattr(ent, "kb_id_") and ent.kb_id_
                        else None,
                    },
                )
                entities.append(entity)

                # Update cache
                self._entity_cache[entity.entity_id] = entity

        return entities

    def _extract_regex_entities(self, text: str, source_doc: str) -> list[Entity]:
        """Extract entities using regex patterns as fallback."""
        entities = []

        for entity_type, pattern in self.regex_patterns.items():
            matches = pattern.finditer(text)

            for match in matches:
                entity_text = match.group().strip()

                if len(entity_text) >= self.min_entity_length:
                    entity = Entity(
                        name=entity_text,
                        entity_type=entity_type,
                        source_doc=source_doc,
                        span_start=match.start(),
                        span_end=match.end(),
                        confidence=0.8,  # Lower confidence for regex
                        method="regex",
                    )
                    entities.append(entity)

                    # Update cache
                    self._entity_cache[entity.entity_id] = entity

        # Add capitalization-based entities
        cap_entities = self._extract_capitalized_entities(text, source_doc)
        for entity in cap_entities:
            self._entity_cache[entity.entity_id] = entity
        entities.extend(cap_entities)

        return entities

    def _extract_capitalized_entities(self, text: str, source_doc: str) -> list[Entity]:
        """Extract potential entities based on capitalization patterns."""
        entities = []

        # Pattern for potential person names (Title Case words)
        person_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
        person_matches = person_pattern.finditer(text)

        for match in person_matches:
            name = match.group().strip()
            # Simple heuristics to avoid false positives
            if (
                len(name.split()) >= 2
                and not any(
                    word.lower() in ["the", "this", "that", "and", "or"]
                    for word in name.split()
                )
                and len(name) >= self.min_entity_length
            ):
                entities.append(
                    Entity(
                        name=name,
                        entity_type="PERSON",
                        source_doc=source_doc,
                        span_start=match.start(),
                        span_end=match.end(),
                        confidence=0.6,
                        method="capitalization",
                    )
                )

        # Pattern for potential organizations (Inc, Corp, LLC, etc.)
        org_pattern = re.compile(
            r"\b[A-Z][a-zA-Z\s&]+(?:Inc|Corp|Corporation|LLC|Ltd|Limited|Company|Co)\b"
        )
        org_matches = org_pattern.finditer(text)

        for match in org_matches:
            org = match.group().strip()
            if len(org) >= self.min_entity_length:
                entities.append(
                    Entity(
                        name=org,
                        entity_type="ORG",
                        source_doc=source_doc,
                        span_start=match.start(),
                        span_end=match.end(),
                        confidence=0.7,
                        method="pattern",
                    )
                )

        return entities

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """Remove duplicate entities based on normalized name and type.

        Args:
            entities: List of entities to deduplicate

        Returns:
            Deduplicated list of entities
        """
        seen: dict[str, Entity] = {}
        unique = []

        for entity in entities:
            if entity.entity_id not in seen:
                seen[entity.entity_id] = entity
                unique.append(entity)
            else:
                # Keep the one with higher confidence
                existing = seen[entity.entity_id]
                if entity.confidence > existing.confidence:
                    # Replace existing with higher confidence
                    unique = [e for e in unique if e.entity_id != entity.entity_id]
                    unique.append(entity)
                    seen[entity.entity_id] = entity

        return unique

    def extract_relationships_llm(
        self, document: Document, entities: list[Entity]
    ) -> list[Relationship]:
        """Extract relationships between entities using LLM.

        Args:
            document: Source document
            entities: Entities extracted from the document

        Returns:
            List of extracted Relationship objects
        """
        if not self.llm_client:
            return []

        if len(entities) < 2:
            return []

        prompt = self._build_relationship_prompt(document, entities)

        try:
            response = self.llm_client.generate(prompt)
            return self._parse_relationships(response, document.id)
        except Exception as e:
            self.logger.error(f"LLM relationship extraction failed: {e}")
            return []

    def _build_relationship_prompt(
        self, document: Document, entities: list[Entity]
    ) -> str:
        """Build prompt for LLM relationship extraction."""
        entity_list = "\n".join([f"- {e.name} ({e.entity_type})" for e in entities])

        # Limit content to first 2000 chars
        content_snippet = document.content[:2000] if document.content else ""

        return f"""Analyze the following text and identify relationships between the entities listed.

Text:
{content_snippet}

Entities:
{entity_list}

For each relationship found, respond in the format:
SOURCE_ENTITY | RELATIONSHIP_TYPE | TARGET_ENTITY

Example:
John Smith | works_at | Acme Corp
Acme Corp | located_in | New York

Only include clear, explicit relationships from the text. Respond with just the relationships, one per line.
"""

    def _parse_relationships(
        self, response: str, source_doc: str
    ) -> list[Relationship]:
        """Parse LLM response into Relationship objects."""
        relationships = []

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3:
                continue

            source, rel_type, target = parts
            relationships.append(
                Relationship(
                    source_entity=source,
                    target_entity=target,
                    relationship_type=rel_type.lower().replace(" ", "_"),
                    source_doc=source_doc,
                    confidence=0.8,
                )
            )

        return relationships

    def get_cached_entities(self) -> dict[str, Entity]:
        """Get all cached entities (for deduplication across documents)."""
        return self._entity_cache.copy()

    def clear_cache(self) -> None:
        """Clear the entity deduplication cache."""
        self._entity_cache.clear()

    def get_dependencies(self) -> list[str]:
        """Get dependencies - spaCy is optional."""
        dependencies = []
        if not self.use_fallback:
            dependencies.append("spacy")
        return dependencies

    def validate_dependencies(self) -> bool:
        """Validate dependencies - always returns True if fallback is enabled."""
        if self.use_fallback:
            return True

        return super().validate_dependencies()

    def get_available_models(self) -> list[str]:
        """Get list of available spaCy models."""
        try:
            import spacy

            return list(spacy.util.get_installed_models())
        except ImportError:
            return []

    def get_supported_entities(self) -> dict[str, str]:
        """Get mapping of supported entity types to descriptions."""
        return {
            "PERSON": "People, including fictional characters",
            "NORP": "Nationalities, religious or political groups",
            "FAC": "Buildings, airports, highways, bridges, etc.",
            "ORG": "Companies, agencies, institutions, etc.",
            "GPE": "Countries, cities, states",
            "LOC": "Non-GPE locations, mountain ranges, bodies of water",
            "PRODUCT": "Objects, vehicles, foods, etc. (not services)",
            "EVENT": "Named hurricanes, battles, wars, sports events, etc.",
            "WORK_OF_ART": "Titles of books, songs, etc.",
            "LAW": "Named documents made into laws",
            "LANGUAGE": "Any named language",
            "DATE": "Absolute or relative dates or periods",
            "TIME": "Times smaller than a day",
            "PERCENT": "Percentage, including '%'",
            "MONEY": "Monetary values, including unit",
            "QUANTITY": "Measurements, as of weight or distance",
            "ORDINAL": "First, second, etc.",
            "CARDINAL": "Numerals that do not fall under another type",
            "EMAIL": "Email addresses",
            "PHONE": "Phone numbers",
            "URL": "Web URLs",
            "SSN": "Social Security Numbers",
            "CREDIT_CARD": "Credit card numbers",
        }


def extract_entities_to_graph(
    document: Document,
    graph_store: Any,
    linkage_table: Any = None,
    config: dict[str, Any] | None = None,
    llm_client: Any = None,
) -> dict[str, Any]:
    """Convenience function to extract entities and add to graph store.

    This integrates the entity extraction pipeline with graph storage,
    creating nodes for entities and edges for relationships.

    Args:
        document: Document to extract entities from
        graph_store: GraphStore instance to add nodes/edges
        linkage_table: Optional LinkageTable for cross-store linking
        config: EntityExtractor configuration
        llm_client: Optional LLM client for relationship extraction

    Returns:
        Dictionary with extraction results including counts
    """
    extractor = EntityExtractor(config=config, llm_client=llm_client)

    # Extract entities
    entities = extractor.extract_entities(document)

    # Add to graph store
    node_ids = []
    for entity in entities:
        node_id = graph_store.add_node(
            name=entity.name,
            node_type=entity.entity_type.lower(),
            node_id=entity.entity_id,
            properties={
                "source_doc": entity.source_doc,
                "span_start": entity.span_start,
                "span_end": entity.span_end,
                "confidence": entity.confidence,
                "method": entity.method,
                **entity.properties,
            },
        )
        if node_id:
            node_ids.append(node_id)
            # Link to source document
            if linkage_table:
                linkage_table.link(
                    concept_uuid=document.id,
                    graph_node_id=node_id,
                )

    # Extract and add relationships if LLM available
    edge_count = 0
    if llm_client and config and config.get("extract_relationships"):
        relationships = extractor.extract_relationships_llm(document, entities)
        for rel in relationships:
            # Find entity IDs for source and target
            source_entity = next(
                (e for e in entities if e.name == rel.source_entity), None
            )
            target_entity = next(
                (e for e in entities if e.name == rel.target_entity), None
            )

            if source_entity and target_entity:
                edge_id = graph_store.add_edge(
                    source_id=source_entity.entity_id,
                    target_id=target_entity.entity_id,
                    relationship=rel.relationship_type,
                    properties={
                        "source_doc": rel.source_doc,
                        "confidence": rel.confidence,
                    },
                )
                if edge_id:
                    edge_count += 1

    return {
        "document_id": document.id,
        "entities_extracted": len(entities),
        "nodes_created": len(node_ids),
        "edges_created": edge_count,
        "entity_types": list(set(e.entity_type for e in entities)),
        "entities": [e.to_dict() for e in entities],
    }
