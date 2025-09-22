"""
Pattern Extractor

Extracts structured patterns from text using regular expressions.
Useful for extracting emails, phone numbers, URLs, IDs, and custom patterns.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from components.extractors.base import BaseExtractor


class PatternExtractor(BaseExtractor):
    """Extractor for finding structured patterns in text using regex."""

    # Predefined patterns for common data types
    PREDEFINED_PATTERNS = {
        "email": {
            "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "description": "Email addresses",
        },
        "phone": {
            "pattern": r"(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}",
            "description": "Phone numbers (US format)",
        },
        "url": {
            "pattern": r"https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?",
            "description": "URLs (HTTP/HTTPS)",
        },
        "ip_address": {
            "pattern": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            "description": "IPv4 addresses",
        },
        "social_security": {
            "pattern": r"\b\d{3}-?\d{2}-?\d{4}\b",
            "description": "Social Security Numbers",
        },
        "credit_card": {
            "pattern": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "description": "Credit card numbers",
        },
        "zip_code": {"pattern": r"\b\d{5}(?:-\d{4})?\b", "description": "US ZIP codes"},
        "isbn": {
            "pattern": r"(?:ISBN[-]?(?:13|10)?:?\s?)?(?:\d{1,3}[-]?\d{1,3}[-]?\d{1,3}[-]?\d{1,3}[-]?\d{1,3}|\d{10}|\d{13})",
            "description": "ISBN numbers",
        },
        "doi": {
            "pattern": r"10\.\d{4,}\/[-._;()\/:a-zA-Z0-9]+",
            "description": "DOI identifiers",
        },
        "uuid": {
            "pattern": r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
            "description": "UUID identifiers",
        },
        "hashtag": {"pattern": r"#\w+", "description": "Hashtags"},
        "mention": {"pattern": r"@\w+", "description": "Social media mentions"},
        "currency": {
            "pattern": r"[\$£€¥₹]\s?\d+(?:,\d{3})*(?:\.\d{2})?",
            "description": "Currency amounts",
        },
        "percentage": {"pattern": r"\d+(?:\.\d+)?%", "description": "Percentages"},
        "date_iso": {
            "pattern": r"\b\d{4}-\d{2}-\d{2}\b",
            "description": "ISO date format (YYYY-MM-DD)",
        },
        "time_24h": {
            "pattern": r"\b([01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?\b",
            "description": "24-hour time format",
        },
        "file_path": {
            "pattern": r'(?:[a-zA-Z]:)?[/\\]?(?:[^/\\<>:"|?*\n\r]+[/\\])*[^/\\<>:"|?*\n\r]*\.[a-zA-Z0-9]+',
            "description": "File paths",
        },
        "version": {
            "pattern": r"\bv?\d+(?:\.\d+)*(?:-[a-zA-Z0-9]+)*\b",
            "description": "Version numbers",
        },
    }

    def __init__(
        self, name: str = "PatternExtractor", config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pattern extractor.

        Args:
            name: Extractor name
            config: Extractor configuration
        """
        super().__init__(name=name, config=config)

        # Configuration options
        self.patterns = config.get("patterns", []) if config else []
        self.predefined_patterns = (
            config.get("predefined_patterns", []) if config else []
        )
        self.case_sensitive = config.get("case_sensitive", False) if config else False
        self.include_positions = (
            config.get("include_positions", False) if config else False
        )
        self.max_matches_per_pattern = (
            config.get("max_matches_per_pattern", 100) if config else 100
        )
        self.deduplicate_matches = (
            config.get("deduplicate_matches", True) if config else True
        )

        # Compile regex patterns
        self._compiled_patterns = self._compile_patterns()

    def extract(self, documents: List["Document"]) -> List["Document"]:
        """
        Extract patterns from documents.

        Args:
            documents: List of documents to process

        Returns:
            List of enhanced documents with pattern metadata
        """
        from core.base import Document

        enhanced_docs = []
        for doc in documents:
            pattern_data = self._extract_from_text(doc.content, doc.metadata)

            # Add pattern data to document metadata
            new_metadata = doc.metadata.copy()
            if "extractors" not in new_metadata:
                new_metadata["extractors"] = {}
            new_metadata["extractors"]["patterns"] = pattern_data

            enhanced_doc = Document(
                content=doc.content,
                metadata=new_metadata,
                id=doc.id,
                source=doc.source,
                embeddings=doc.embeddings,
            )
            enhanced_docs.append(enhanced_doc)

        return enhanced_docs

    def _extract_from_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract patterns from text.

        Args:
            text: Input text to analyze
            metadata: Optional metadata dictionary

        Returns:
            Dictionary containing extracted patterns
        """
        if not text or not text.strip():
            return {}

        try:
            results = {}
            all_matches = []

            # Apply each compiled pattern
            for pattern_name, pattern_info in self._compiled_patterns.items():
                matches = self._find_pattern_matches(text, pattern_info)

                if matches:
                    results[pattern_name] = {
                        "matches": matches,
                        "count": len(matches),
                        "description": pattern_info.get(
                            "description", "Custom pattern"
                        ),
                    }

                    # Add to all_matches for global statistics
                    for match in matches:
                        all_matches.append(
                            {
                                "pattern": pattern_name,
                                "value": match["value"],
                                "position": match.get("position", {}),
                            }
                        )

            # Add summary statistics
            if all_matches:
                results["_summary"] = {
                    "total_matches": len(all_matches),
                    "patterns_found": len(
                        [k for k in results.keys() if not k.startswith("_")]
                    ),
                    "pattern_types": list(results.keys()),
                }

                # Pattern density (matches per 100 characters)
                results["_summary"]["pattern_density"] = (
                    len(all_matches) / len(text)
                ) * 100

            return results

        except Exception as e:
            self.logger.error(f"Error extracting patterns: {e}")
            return {}

    def _compile_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Compile regex patterns for efficient matching."""
        compiled = {}

        # Add predefined patterns
        for pattern_name in self.predefined_patterns:
            if pattern_name in self.PREDEFINED_PATTERNS:
                pattern_info = self.PREDEFINED_PATTERNS[pattern_name].copy()
                flags = 0 if self.case_sensitive else re.IGNORECASE

                try:
                    pattern_info["compiled"] = re.compile(
                        pattern_info["pattern"], flags
                    )
                    compiled[pattern_name] = pattern_info
                except re.error as e:
                    self.logger.warning(
                        f"Failed to compile predefined pattern '{pattern_name}': {e}"
                    )

        # Add custom patterns
        for pattern_config in self.patterns:
            if (
                not isinstance(pattern_config, dict)
                or "name" not in pattern_config
                or "pattern" not in pattern_config
            ):
                continue

            pattern_name = pattern_config["name"]
            pattern_text = pattern_config["pattern"]
            description = pattern_config.get("description", "Custom pattern")

            flags = 0 if self.case_sensitive else re.IGNORECASE

            try:
                compiled[pattern_name] = {
                    "pattern": pattern_text,
                    "description": description,
                    "compiled": re.compile(pattern_text, flags),
                }
            except re.error as e:
                self.logger.warning(
                    f"Failed to compile custom pattern '{pattern_name}': {e}"
                )

        return compiled

    def _find_pattern_matches(
        self, text: str, pattern_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find all matches for a specific pattern."""
        compiled_pattern = pattern_info["compiled"]
        matches = []

        # Find all matches
        for match in compiled_pattern.finditer(text):
            match_data = {
                "value": match.group(0),
                "groups": match.groups() if match.groups() else [],
            }

            # Add position information if requested
            if self.include_positions:
                match_data["position"] = {
                    "start": match.start(),
                    "end": match.end(),
                    "line": text[: match.start()].count("\n") + 1,
                    "column": match.start() - text.rfind("\n", 0, match.start()),
                }

            matches.append(match_data)

            # Respect max matches limit
            if len(matches) >= self.max_matches_per_pattern:
                break

        # Deduplicate if requested
        if self.deduplicate_matches:
            matches = self._deduplicate_matches(matches)

        return matches

    def _deduplicate_matches(
        self, matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate matches based on value."""
        seen_values = set()
        unique_matches = []

        for match in matches:
            value = match["value"]
            if value not in seen_values:
                seen_values.add(value)
                unique_matches.append(match)

        return unique_matches

    def validate_pattern(self, pattern: str) -> Tuple[bool, str]:
        """
        Validate a regex pattern.

        Args:
            pattern: Regex pattern to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            re.compile(pattern)
            return True, "Pattern is valid"
        except re.error as e:
            return False, f"Invalid regex pattern: {e}"

    def get_available_predefined_patterns(self) -> Dict[str, str]:
        """Get list of available predefined patterns."""
        return {
            name: info["description"] for name, info in self.PREDEFINED_PATTERNS.items()
        }

    def test_pattern(self, pattern: str, test_text: str) -> List[str]:
        """
        Test a pattern against sample text.

        Args:
            pattern: Regex pattern to test
            test_text: Text to test against

        Returns:
            List of matches found
        """
        try:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            compiled = re.compile(pattern, flags)
            return compiled.findall(test_text)
        except re.error as e:
            self.logger.error(f"Error testing pattern: {e}")
            return []

    def get_dependencies(self) -> List[str]:
        """Get list of dependencies for this extractor."""
        return []  # No external dependencies

    def get_feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces."""
        features = []

        # Add predefined pattern names
        for pattern_name in self.predefined_patterns:
            if pattern_name in self.PREDEFINED_PATTERNS:
                features.append(pattern_name)

        # Add custom pattern names
        for pattern_config in self.patterns:
            if isinstance(pattern_config, dict) and "name" in pattern_config:
                features.append(pattern_config["name"])

        # Add summary features
        features.extend(["_summary"])

        return features

    @staticmethod
    def get_config_schema() -> Dict[str, Any]:
        """Get configuration schema for the extractor."""
        return {
            "type": "object",
            "properties": {
                "predefined_patterns": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(PatternExtractor.PREDEFINED_PATTERNS.keys()),
                    },
                    "default": [],
                    "description": "List of predefined patterns to use",
                },
                "patterns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "pattern": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["name", "pattern"],
                    },
                    "default": [],
                    "description": "Custom regex patterns to extract",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether pattern matching is case sensitive",
                },
                "include_positions": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include position information for matches",
                },
                "max_matches_per_pattern": {
                    "type": "integer",
                    "default": 100,
                    "minimum": 1,
                    "description": "Maximum matches to extract per pattern",
                },
                "deduplicate_matches": {
                    "type": "boolean",
                    "default": True,
                    "description": "Remove duplicate matches",
                },
            },
        }
