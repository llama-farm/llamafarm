"""CSV parser for customer support tickets and other tabular data."""

import csv
import logging
from typing import Dict, Any, List
from pathlib import Path

from core.base import Parser, Document, ProcessingResult

logger = logging.getLogger(__name__)


class CSVParser(Parser):
    """Parser for CSV files with configurable field mapping."""

    def __init__(self, name: str = "CSVParser", config: Dict[str, Any] = None):
        super().__init__(name, config)
        config = config or {}
        self.content_fields = config.get("content_fields", ["subject", "body"])
        self.metadata_fields = config.get("metadata_fields", [])
        self.id_field = config.get("id_field")
        self.combine_content = config.get("combine_content", True)
        self.content_separator = config.get("content_separator", "\n\n")

    def validate_config(self) -> bool:
        """Validate configuration."""
        if not self.content_fields:
            raise ValueError("content_fields cannot be empty")
        return True

    def parse(self, source: str) -> ProcessingResult:
        """Parse CSV file into documents."""
        documents = []
        errors = []

        try:
            with open(source, "r", encoding="utf-8") as file:
                # Auto-detect delimiter
                sample = file.read(1024)
                file.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter

                reader = csv.DictReader(file, delimiter=delimiter)

                for row_idx, row in enumerate(reader, 1):
                    try:
                        doc = self._row_to_document(row, row_idx, source)
                        documents.append(doc)
                    except Exception as e:
                        errors.append(
                            {"row": row_idx, "error": str(e), "source": source}
                        )
                        logger.warning(f"Error processing row {row_idx}: {e}")

        except Exception as e:
            errors.append(
                {"error": f"Failed to parse CSV file: {str(e)}", "source": source}
            )
            logger.error(f"Failed to parse CSV file {source}: {e}")

        return ProcessingResult(
            documents=documents,
            errors=errors,
            metrics={
                "total_rows": len(documents) + len(errors),
                "parsed_successfully": len(documents),
                "parse_errors": len(errors),
            },
        )

    def _row_to_document(
        self, row: Dict[str, str], row_idx: int, source: str
    ) -> Document:
        """Convert CSV row to Document."""
        # Extract content fields
        content_parts = []
        for field in self.content_fields:
            if field in row and row[field]:
                content_parts.append(row[field].strip())

        if not content_parts:
            raise ValueError(
                f"No content found in specified fields: {self.content_fields}"
            )

        # Combine content
        if self.combine_content:
            content = self.content_separator.join(content_parts)
        else:
            content = content_parts[0]  # Use first content field only

        # Extract metadata
        metadata = {"source_file": Path(source).name, "row_number": row_idx}

        # Add specified metadata fields
        if self.metadata_fields:
            for field in self.metadata_fields:
                if field in row:
                    metadata[field] = row[field]
        else:
            # Include all fields not used for content as metadata
            for key, value in row.items():
                if key not in self.content_fields:
                    metadata[key] = value

        # Generate document ID
        doc_id = None
        if self.id_field and self.id_field in row:
            doc_id = row[self.id_field]
        else:
            doc_id = f"{Path(source).stem}_{row_idx}"

        return Document(content=content, metadata=metadata, id=doc_id, source=source)


class CustomerSupportCSVParser(CSVParser):
    """Specialized parser for customer support CSV files."""

    def __init__(
        self, name: str = "CustomerSupportCSVParser", config: Dict[str, Any] = None
    ):
        # Default configuration for customer support tickets
        default_config = {
            "content_fields": ["subject", "body", "answer"],
            "metadata_fields": ["type", "queue", "priority", "language"],
            "combine_content": True,
            "content_separator": "\n\n---\n\n",
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config)

    def _row_to_document(
        self, row: Dict[str, str], row_idx: int, source: str
    ) -> Document:
        """Enhanced document creation for customer support tickets."""
        doc = super()._row_to_document(row, row_idx, source)

        # Add tags as a list
        tags = []
        for i in range(1, 9):  # tag_1 through tag_8
            tag_field = f"tag_{i}"
            if tag_field in row and row[tag_field]:
                tags.append(row[tag_field])

        if tags:
            doc.metadata["tags"] = tags

        # Parse priority as numeric if possible
        if "priority" in doc.metadata:
            priority_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            priority = doc.metadata["priority"].lower()
            if priority in priority_map:
                doc.metadata["priority_numeric"] = priority_map[priority]

        return doc