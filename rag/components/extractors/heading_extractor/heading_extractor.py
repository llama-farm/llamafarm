"""
Heading Extractor for identifying document structure and hierarchy.
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from core.base import Document

from components.extractors.base import BaseExtractor


class HeadingExtractor(BaseExtractor):
    """Extract document headings and structure information."""

    def __init__(
        self, name: str = "HeadingExtractor", config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize heading extractor.

        Args:
            name: Extractor name
            config: Extractor configuration
        """
        super().__init__(name=name, config=config or {})

        # Configuration options
        self.extract_markdown_headings = self.config.get(
            "extract_markdown_headings", True
        )
        self.extract_underlined_headings = self.config.get(
            "extract_underlined_headings", True
        )
        self.extract_numbered_headings = self.config.get(
            "extract_numbered_headings", True
        )
        self.extract_caps_headings = self.config.get("extract_caps_headings", True)
        self.min_heading_length = self.config.get("min_heading_length", 3)
        self.max_heading_length = self.config.get("max_heading_length", 200)
        self.generate_toc = self.config.get("generate_toc", True)
        self.detect_sections = self.config.get("detect_sections", True)

    def extract(self, documents: List["Document"]) -> List["Document"]:
        """
        Extract heading structure from documents.

        Args:
            documents: List of documents to process

        Returns:
            List of enhanced documents with heading metadata
        """
        from core.base import Document

        for doc in documents:
            try:
                heading_data = self._extract_from_text(doc.content, doc.metadata)

                # Add to extractors metadata
                if "extractors" not in doc.metadata:
                    doc.metadata["extractors"] = {}

                doc.metadata["extractors"]["headings"] = heading_data

                self.logger.debug(f"Extracted headings for document {doc.id}")

            except Exception as e:
                self.logger.error(
                    f"Heading extraction failed for document {doc.id}: {e}"
                )

        return documents

    def get_dependencies(self) -> List[str]:
        """Get required dependencies for this extractor."""
        return []  # No external dependencies

    def _extract_from_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract heading structure from text.

        Args:
            text: Input text to analyze
            metadata: Optional metadata context

        Returns:
            Dictionary containing extracted heading information
        """
        try:
            headings = []
            lines = text.split("\n")

            # Extract different types of headings
            if self.extract_markdown_headings:
                md_headings = self._extract_markdown_headings(lines)
                headings.extend(md_headings)

            if self.extract_underlined_headings:
                underlined_headings = self._extract_underlined_headings(lines)
                headings.extend(underlined_headings)

            if self.extract_numbered_headings:
                numbered_headings = self._extract_numbered_headings(lines)
                headings.extend(numbered_headings)

            if self.extract_caps_headings:
                caps_headings = self._extract_caps_headings(lines)
                headings.extend(caps_headings)

            # Remove duplicates and sort by line number
            headings = self._deduplicate_headings(headings)
            headings.sort(key=lambda x: x["line_number"])

            # Analyze heading hierarchy
            hierarchy = self._analyze_hierarchy(headings)

            # Generate table of contents
            toc = (
                self._generate_table_of_contents(headings)
                if self.generate_toc
                else None
            )

            # Detect document sections
            sections = (
                self._detect_document_sections(headings, lines)
                if self.detect_sections
                else None
            )

            # Generate statistics
            stats = self._generate_heading_stats(headings)

            return {
                "headings": headings,
                "heading_count": len(headings),
                "hierarchy": hierarchy,
                "table_of_contents": toc,
                "sections": sections,
                "statistics": stats,
                "extractor": "HeadingExtractor",
            }

        except Exception as e:
            logger.error(f"Error in heading extraction: {e}")
            return {"headings": [], "heading_count": 0, "error": str(e)}

    def _extract_markdown_headings(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract markdown-style headings (# ## ### etc.)."""
        headings = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check for markdown heading pattern
            match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()

                if self._is_valid_heading(text):
                    headings.append(
                        {
                            "text": text,
                            "level": level,
                            "type": "markdown",
                            "line_number": i + 1,
                            "raw_line": line,
                        }
                    )

        return headings

    def _extract_underlined_headings(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract underlined headings (text followed by === or ---)."""
        headings = []

        for i in range(len(lines) - 1):
            current_line = lines[i].strip()
            next_line = lines[i + 1].strip()

            # Check if next line is underline
            if current_line and self._is_valid_heading(current_line):
                if re.match(r"^=+$", next_line):
                    # Level 1 heading (underlined with =)
                    headings.append(
                        {
                            "text": current_line,
                            "level": 1,
                            "type": "underlined",
                            "line_number": i + 1,
                            "raw_line": lines[i],
                        }
                    )
                elif re.match(r"^-+$", next_line):
                    # Level 2 heading (underlined with -)
                    headings.append(
                        {
                            "text": current_line,
                            "level": 2,
                            "type": "underlined",
                            "line_number": i + 1,
                            "raw_line": lines[i],
                        }
                    )

        return headings

    def _extract_numbered_headings(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract numbered headings (1. 1.1. 1.1.1. etc.)."""
        headings = []

        numbered_pattern = re.compile(r"^((?:\d+\.)+)\s+(.+)$")

        for i, line in enumerate(lines):
            stripped = line.strip()

            match = numbered_pattern.match(stripped)
            if match:
                number_part = match.group(1)
                text = match.group(2).strip()

                if self._is_valid_heading(text):
                    # Calculate level based on number of dots
                    level = number_part.count(".")

                    headings.append(
                        {
                            "text": text,
                            "level": level,
                            "type": "numbered",
                            "number": number_part.rstrip("."),
                            "line_number": i + 1,
                            "raw_line": line,
                        }
                    )

        return headings

    def _extract_caps_headings(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract all-caps headings."""
        headings = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check if line is all caps and looks like a heading
            if (
                stripped
                and stripped.isupper()
                and self._is_valid_heading(stripped)
                and not self._looks_like_data(stripped)
            ):
                # Determine level based on context
                level = self._estimate_caps_heading_level(lines, i)

                headings.append(
                    {
                        "text": stripped,
                        "level": level,
                        "type": "caps",
                        "line_number": i + 1,
                        "raw_line": line,
                    }
                )

        return headings

    def _is_valid_heading(self, text: str) -> bool:
        """Check if text looks like a valid heading."""
        if not text:
            return False

        # Check length constraints
        if len(text) < self.min_heading_length or len(text) > self.max_heading_length:
            return False

        # Should not end with punctuation (except ? or :)
        if text.endswith((".", "!")) and not text.endswith((":", "?")):
            return False

        # Should not be mostly numbers or special characters
        alphanumeric_count = sum(c.isalnum() for c in text)
        if alphanumeric_count < len(text) * 0.5:
            return False

        return True

    def _looks_like_data(self, text: str) -> bool:
        """Check if text looks like data rather than a heading."""
        # Common data patterns
        data_patterns = [
            r"^\d+$",  # Just numbers
            r"^[A-Z]{2,3}\d+$",  # Code patterns like ABC123
            r"^\d+[.,]\d+$",  # Decimal numbers
            r".*[=:].*",  # Key-value pairs
        ]

        return any(re.match(pattern, text) for pattern in data_patterns)

    def _estimate_caps_heading_level(self, lines: List[str], line_index: int) -> int:
        """Estimate heading level for all-caps headings based on context."""
        # Look at surrounding context to estimate level
        # This is a heuristic approach

        # Check if it's at the beginning of document or after blank lines
        preceding_blank_lines = 0
        for i in range(line_index - 1, -1, -1):
            if not lines[i].strip():
                preceding_blank_lines += 1
            else:
                break

        # More blank lines suggest higher importance (lower level number)
        if preceding_blank_lines >= 2:
            return 1
        elif preceding_blank_lines >= 1:
            return 2
        else:
            return 3

    def _deduplicate_headings(
        self, headings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate headings that might be detected by multiple methods."""
        unique_headings = []
        seen_positions = set()

        for heading in headings:
            # Use line number and text as key for deduplication
            key = (heading["line_number"], heading["text"].lower())

            if key not in seen_positions:
                seen_positions.add(key)
                unique_headings.append(heading)
            else:
                # If duplicate, prefer more specific type
                existing_index = next(
                    i
                    for i, h in enumerate(unique_headings)
                    if h["line_number"] == heading["line_number"]
                    and h["text"].lower() == heading["text"].lower()
                )

                # Priority order: markdown > numbered > underlined > caps
                type_priority = {
                    "markdown": 4,
                    "numbered": 3,
                    "underlined": 2,
                    "caps": 1,
                }

                if type_priority.get(heading["type"], 0) > type_priority.get(
                    unique_headings[existing_index]["type"], 0
                ):
                    unique_headings[existing_index] = heading

        return unique_headings

    def _analyze_hierarchy(self, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the hierarchical structure of headings."""
        if not headings:
            return {}

        levels_used = sorted(set(h["level"] for h in headings))
        level_counts = {}

        for heading in headings:
            level = heading["level"]
            level_counts[level] = level_counts.get(level, 0) + 1

        # Find the structure pattern
        max_depth = max(levels_used) if levels_used else 0

        return {
            "max_depth": max_depth,
            "levels_used": levels_used,
            "level_counts": level_counts,
            "total_headings": len(headings),
            "has_consistent_numbering": self._has_consistent_numbering(headings),
        }

    def _has_consistent_numbering(self, headings: List[Dict[str, Any]]) -> bool:
        """Check if document has consistent numbering scheme."""
        numbered_headings = [h for h in headings if h["type"] == "numbered"]

        if len(numbered_headings) < 2:
            return len(numbered_headings) > 0  # True if has some numbered headings

        # Check if numbering follows a logical sequence
        # This is a simplified check
        return len(numbered_headings) >= len(headings) * 0.5

    def _generate_table_of_contents(
        self, headings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate a table of contents from headings."""
        toc = []

        for heading in headings:
            indent = "  " * (heading["level"] - 1)
            toc_entry = {
                "text": heading["text"],
                "level": heading["level"],
                "line_number": heading["line_number"],
                "formatted": f"{indent}- {heading['text']}",
            }
            toc.append(toc_entry)

        return toc

    def _detect_document_sections(
        self, headings: List[Dict[str, Any]], lines: List[str]
    ) -> List[Dict[str, Any]]:
        """Detect document sections based on headings."""
        if not headings:
            return []

        sections = []

        for i, heading in enumerate(headings):
            # Determine section boundaries
            start_line = heading["line_number"]

            # Find end line (next heading at same or higher level, or end of document)
            end_line = len(lines)
            for j in range(i + 1, len(headings)):
                next_heading = headings[j]
                if next_heading["level"] <= heading["level"]:
                    end_line = next_heading["line_number"] - 1
                    break

            # Extract section content
            section_lines = lines[start_line:end_line]
            content = "\n".join(line for line in section_lines if line.strip())

            section = {
                "title": heading["text"],
                "level": heading["level"],
                "start_line": start_line,
                "end_line": end_line,
                "content_preview": content[:200] + "..."
                if len(content) > 200
                else content,
                "word_count": len(content.split()) if content else 0,
                "line_count": end_line - start_line,
            }

            sections.append(section)

        return sections

    def _generate_heading_stats(self, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about headings."""
        if not headings:
            return {}

        type_counts = {}
        level_distribution = {}

        for heading in headings:
            # Count by type
            h_type = heading["type"]
            type_counts[h_type] = type_counts.get(h_type, 0) + 1

            # Count by level
            level = heading["level"]
            level_distribution[level] = level_distribution.get(level, 0) + 1

        avg_length = sum(len(h["text"]) for h in headings) / len(headings)

        return {
            "total_headings": len(headings),
            "types": type_counts,
            "level_distribution": level_distribution,
            "average_heading_length": round(avg_length, 1),
            "longest_heading": max(headings, key=lambda x: len(x["text"]))["text"],
            "shortest_heading": min(headings, key=lambda x: len(x["text"]))["text"],
        }

    @staticmethod
    def get_supported_types() -> List[str]:
        """Get list of supported heading types."""
        return ["markdown", "underlined", "numbered", "caps"]

    @staticmethod
    def get_description() -> str:
        """Get extractor description."""
        return "Extract document headings and hierarchical structure"
