"""Entity extraction using spaCy and other local NLP libraries."""

import re
from typing import Dict, Any, List, Optional, Set

from components.extractors.base import BaseExtractor
from core.base import Document

from core.logging import RAGStructLogger
logger = RAGStructLogger("rag.components.extractors.entity_extractor.entity_extractor")


class EntityExtractor(BaseExtractor):
    """
    Entity extraction using spaCy for named entity recognition.

    Extracts persons, organizations, locations, dates, money, etc.
    Falls back to regex patterns if spaCy is not available.
    """

    def __init__(
        self, name: str = "EntityExtractor", config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, config)

        # Configuration
        self.model_name = self.config.get("model", "en_core_web_sm")
        self.entity_types = set(
            self.config.get(
                "entity_types",
                [
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
                ],
            )
        )
        self.use_fallback = self.config.get("use_fallback", True)
        self.min_entity_length = self.config.get("min_entity_length", 2)

        # Try to load spaCy model
        self.nlp = None
        self._load_spacy_model()

        # Regex patterns for fallback
        self.regex_patterns = self._initialize_regex_patterns()

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

    def _initialize_regex_patterns(self) -> Dict[str, re.Pattern]:
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

    def extract(self, documents: List[Document]) -> List[Document]:
        """Extract entities from documents."""
        for doc in documents:
            try:
                if self.nlp:
                    entities = self._extract_spacy_entities(doc.content)
                else:
                    entities = self._extract_regex_entities(doc.content)

                # Add to metadata
                if "extractors" not in doc.metadata:
                    doc.metadata["extractors"] = {}

                doc.metadata["extractors"]["entities"] = entities

                # Also add simplified lists for easy access
                for entity_type, entity_list in entities.items():
                    doc.metadata[f"entities_{entity_type.lower()}"] = [
                        entity["text"] for entity in entity_list
                    ]

                total_entities = sum(
                    len(entity_list) for entity_list in entities.values()
                )
                self.logger.debug(
                    f"Extracted {total_entities} entities from document {doc.id}"
                )

            except Exception as e:
                self.logger.error(
                    f"Entity extraction failed for document {doc.id}: {e}"
                )

        return documents

    def _extract_spacy_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using spaCy."""
        doc = self.nlp(text)
        entities = {}

        for ent in doc.ents:
            if (
                ent.label_ in self.entity_types
                and len(ent.text.strip()) >= self.min_entity_length
            ):
                entity_type = ent.label_

                if entity_type not in entities:
                    entities[entity_type] = []

                entity_info = {
                    "text": ent.text.strip(),
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(ent, "confidence", 1.0),
                    "method": "spacy",
                }

                # Avoid duplicates
                if not any(
                    e["text"].lower() == entity_info["text"].lower()
                    for e in entities[entity_type]
                ):
                    entities[entity_type].append(entity_info)

        return entities

    def _extract_regex_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using regex patterns as fallback."""
        entities = {}

        for entity_type, pattern in self.regex_patterns.items():
            matches = pattern.finditer(text)

            for match in matches:
                entity_text = match.group().strip()

                if len(entity_text) >= self.min_entity_length:
                    if entity_type not in entities:
                        entities[entity_type] = []

                    entity_info = {
                        "text": entity_text,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.8,  # Lower confidence for regex
                        "method": "regex",
                    }

                    # Avoid duplicates
                    if not any(
                        e["text"].lower() == entity_info["text"].lower()
                        for e in entities[entity_type]
                    ):
                        entities[entity_type].append(entity_info)

        # Add some basic named entity patterns
        entities.update(self._extract_capitalized_entities(text))

        return entities

    def _extract_capitalized_entities(
        self, text: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract potential entities based on capitalization patterns."""
        entities = {"PERSON": [], "ORG": []}

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
                entities["PERSON"].append(
                    {
                        "text": name,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.6,
                        "method": "capitalization",
                    }
                )

        # Pattern for potential organizations (Inc, Corp, LLC, etc.)
        org_pattern = re.compile(
            r"\b[A-Z][a-zA-Z\s&]+(?:Inc|Corp|Corporation|LLC|Ltd|Limited|Company|Co)\b"
        )
        org_matches = org_pattern.finditer(text)

        for match in org_matches:
            org = match.group().strip()
            if len(org) >= self.min_entity_length:
                entities["ORG"].append(
                    {
                        "text": org,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.7,
                        "method": "pattern",
                    }
                )

        return entities

    def get_dependencies(self) -> List[str]:
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

    def get_available_models(self) -> List[str]:
        """Get list of available spaCy models."""
        try:
            import spacy

            return list(spacy.util.get_installed_models())
        except ImportError:
            return []

    def get_supported_entities(self) -> Dict[str, str]:
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
