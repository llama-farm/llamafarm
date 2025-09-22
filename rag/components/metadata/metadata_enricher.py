"""
Metadata enricher for adding comprehensive metadata to documents.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import mimetypes
import json

from core.base import Document
from .metadata_config import MetadataSchema, DocumentMetadata, CoreMetadataConfig

logger = logging.getLogger(__name__)


class MetadataEnricher:
    """Enrich documents with comprehensive metadata."""

    def __init__(self, schema: Optional[MetadataSchema] = None):
        """
        Initialize metadata enricher.

        Args:
            schema: Metadata schema configuration
        """
        self.schema = schema or MetadataSchema()
        self.logger = logger

    def enrich_documents(self, documents: List[Document]) -> List[Document]:
        """
        Enrich multiple documents with metadata.

        Args:
            documents: List of documents to enrich

        Returns:
            List of enriched documents
        """
        enriched_docs = []
        for doc in documents:
            try:
                enriched_doc = self.enrich_document(doc)
                enriched_docs.append(enriched_doc)
            except Exception as e:
                self.logger.error(f"Failed to enrich document {doc.id}: {e}")
                enriched_docs.append(doc)  # Return original if enrichment fails

        return enriched_docs

    def enrich_document(self, doc: Document) -> Document:
        """
        Enrich a single document with metadata.

        Args:
            doc: Document to enrich

        Returns:
            Enriched document
        """
        # Initialize metadata if not present
        if doc.metadata is None:
            doc.metadata = {}

        # Add core metadata
        self._add_core_metadata(doc)

        # Add content statistics
        if self.schema.core.calculate_stats:
            self._add_content_statistics(doc)

        # Add source information
        self._add_source_metadata(doc)

        # Add custom fields from schema
        for field, value in self.schema.custom_fields.items():
            if field not in doc.metadata:
                doc.metadata[field] = value

        # Validate required fields
        self._validate_required_fields(doc)

        # Convert to ChromaDB compatible format if needed
        if self.schema.chroma_compatible:
            doc.metadata = self._ensure_chroma_compatibility(doc.metadata)

        return doc

    def _add_core_metadata(self, doc: Document) -> None:
        """Add core metadata fields."""
        # Generate ID if needed
        if self.schema.core.generate_id and not doc.id:
            doc.id = self._generate_document_id(doc)
            doc.metadata["id"] = doc.id

        # Add content hash
        if self.schema.core.generate_hash:
            doc.metadata["content_hash"] = self._generate_content_hash(
                doc.content, self.schema.core.hash_algorithm
            )

        # Add timestamps
        if self.schema.core.add_timestamps:
            now = datetime.now()
            if self.schema.core.timestamp_format == "iso":
                doc.metadata["processing_timestamp"] = now.isoformat()
            else:
                doc.metadata["processing_timestamp"] = str(now)

            doc.metadata["processing_date"] = now.strftime("%Y-%m-%d")
            doc.metadata["processing_time"] = now.strftime("%H:%M:%S")

    def _add_content_statistics(self, doc: Document) -> None:
        """Add content statistics."""
        stats = {}

        if "word_count" in self.schema.core.stats_fields:
            stats["word_count"] = len(doc.content.split())

        if "char_count" in self.schema.core.stats_fields:
            stats["char_count"] = len(doc.content)

        if "line_count" in self.schema.core.stats_fields:
            stats["line_count"] = doc.content.count("\n") + 1

        if "sentence_count" in self.schema.core.stats_fields:
            # Simple sentence counting (can be improved)
            import re

            sentences = re.split(r"[.!?]+", doc.content)
            stats["sentence_count"] = len([s for s in sentences if s.strip()])

        if "paragraph_count" in self.schema.core.stats_fields:
            paragraphs = doc.content.split("\n\n")
            stats["paragraph_count"] = len([p for p in paragraphs if p.strip()])

        doc.metadata.update(stats)

    def _add_source_metadata(self, doc: Document) -> None:
        """Add source-related metadata."""
        if not doc.source:
            doc.metadata["source_type"] = "unknown"
            return

        # Detect source type
        if doc.source.startswith(("http://", "https://")):
            doc.metadata["source_type"] = "url"
            doc.metadata["source_domain"] = self._extract_domain(doc.source)
        else:
            # File source
            path = Path(doc.source)
            doc.metadata["source_type"] = "file"
            doc.metadata["file_name"] = path.name
            doc.metadata["file_extension"] = path.suffix.lower()

            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type:
                doc.metadata["mime_type"] = mime_type

            # Add file stats if file exists
            if path.exists():
                stat = path.stat()
                doc.metadata["file_size"] = stat.st_size
                doc.metadata["file_modified"] = datetime.fromtimestamp(
                    stat.st_mtime
                ).isoformat()
                doc.metadata["file_created"] = datetime.fromtimestamp(
                    stat.st_ctime
                ).isoformat()

    def _generate_document_id(self, doc: Document) -> str:
        """Generate unique document ID."""
        if self.schema.core.id_method == "content_hash":
            # Use content + source for ID
            content = f"{doc.source or 'unknown'}:{doc.content[:500]}"
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        elif self.schema.core.id_method == "uuid":
            import uuid

            return str(uuid.uuid4())
        else:
            # Fallback to timestamp-based ID
            import time

            return f"doc_{int(time.time() * 1000000)}"

    def _generate_content_hash(self, content: str, algorithm: str = "sha256") -> str:
        """Generate content hash."""
        if algorithm == "sha256":
            return hashlib.sha256(content.encode()).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(content.encode()).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(content.encode()).hexdigest()
        else:
            # Default to sha256
            return hashlib.sha256(content.encode()).hexdigest()

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return parsed.netloc

    def _validate_required_fields(self, doc: Document) -> None:
        """Validate that required fields are present."""
        missing_fields = []

        for field in self.schema.required_fields:
            if field == "id" and not doc.id:
                missing_fields.append(field)
            elif field != "id" and field not in doc.metadata:
                missing_fields.append(field)

        if missing_fields:
            self.logger.warning(
                f"Document {doc.id} missing required fields: {missing_fields}"
            )

    def _ensure_chroma_compatibility(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Union[str, int, float, bool, None]]:
        """
        Ensure metadata is compatible with ChromaDB.

        ChromaDB only accepts: str, int, float, bool, None
        """
        compatible_metadata = {}

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                compatible_metadata[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert lists to JSON string
                compatible_metadata[f"{key}_json"] = json.dumps(value)
                compatible_metadata[f"{key}_count"] = len(value)
                # Also store first few items as separate fields
                for i, item in enumerate(value[:3]):
                    if isinstance(item, (str, int, float, bool)):
                        compatible_metadata[f"{key}_{i}"] = item
            elif isinstance(value, dict):
                # Flatten nested dictionaries
                for sub_key, sub_value in value.items():
                    flat_key = f"{key}_{sub_key}"
                    if (
                        isinstance(sub_value, (str, int, float, bool))
                        or sub_value is None
                    ):
                        compatible_metadata[flat_key] = sub_value
                    else:
                        compatible_metadata[f"{flat_key}_str"] = str(sub_value)
            elif isinstance(value, datetime):
                compatible_metadata[key] = value.isoformat()
            else:
                # Convert other types to string
                compatible_metadata[f"{key}_str"] = str(value)

        return compatible_metadata


class MetadataFilter:
    """Filter documents based on metadata criteria."""

    def __init__(self, filters: Dict[str, Any]):
        """
        Initialize metadata filter.

        Args:
            filters: Dictionary of filter criteria
        """
        self.filters = filters

    def filter_documents(self, documents: List[Document]) -> List[Document]:
        """
        Filter documents based on metadata.

        Args:
            documents: List of documents to filter

        Returns:
            Filtered list of documents
        """
        filtered_docs = []

        for doc in documents:
            if self._matches_filters(doc):
                filtered_docs.append(doc)

        return filtered_docs

    def _matches_filters(self, doc: Document) -> bool:
        """Check if document matches all filters."""
        if not doc.metadata:
            return False

        for field, criteria in self.filters.items():
            if not self._matches_criterion(doc.metadata.get(field), criteria):
                return False

        return True

    def _matches_criterion(self, value: Any, criterion: Any) -> bool:
        """Check if value matches criterion."""
        if isinstance(criterion, dict):
            # Handle operators
            for op, op_value in criterion.items():
                if op == "$eq":
                    if value != op_value:
                        return False
                elif op == "$ne":
                    if value == op_value:
                        return False
                elif op == "$gt":
                    if not (value > op_value):
                        return False
                elif op == "$gte":
                    if not (value >= op_value):
                        return False
                elif op == "$lt":
                    if not (value < op_value):
                        return False
                elif op == "$lte":
                    if not (value <= op_value):
                        return False
                elif op == "$in":
                    if value not in op_value:
                        return False
                elif op == "$nin":
                    if value in op_value:
                        return False
                elif op == "$contains":
                    if op_value not in str(value):
                        return False
                elif op == "$regex":
                    import re

                    if not re.search(op_value, str(value)):
                        return False
        else:
            # Direct equality check
            if value != criterion:
                return False

        return True
