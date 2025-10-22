"""DOCX parser using python-docx library."""

from pathlib import Path
from typing import Any, Optional

import docx

from components.parsers.base.base_parser import BaseParser, ParserConfig
from components.parsers.docx.docx_utils import (
    DocxBlobProcessor,
    DocxChunker,
    DocxDocumentFactory,
    DocxHeaderFooterExtractor,
    DocxMetadataExtractor,
    DocxTempFileHandler,
)
from core.base import ProcessingResult
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.docx.python_docx_parser")


class DocxParser_PythonDocx(BaseParser):
    """DOCX parser using python-docx library."""

    def __init__(
        self,
        name: str = "DocxParser_PythonDocx",
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(config or {})  # Call BaseParser init
        self.name = name
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.chunk_strategy = self.config.get("chunk_strategy", "paragraphs")
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.extract_tables = self.config.get("extract_tables", True)
        self.extract_headers = self.config.get("extract_headers", True)
        self.extract_footers = self.config.get("extract_footers", False)
        self.extract_comments = self.config.get("extract_comments", False)

    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata."""
        return ParserConfig(
            name="DocxParser_PythonDocx",
            display_name="Python-docx DOCX Parser",
            version="1.0.0",
            supported_extensions=[".docx"],
            mime_types=[
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ],
            capabilities=[
                "text_extraction",
                "metadata_extraction",
                "table_extraction",
                "header_footer_extraction",
            ],
            dependencies={
                "python-docx": ["python-docx>=0.8.11"],
            },
            default_config={
                "chunk_size": 1000,
                "chunk_strategy": "paragraphs",
                "extract_metadata": True,
                "extract_tables": True,
            },
        )

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        return file_path.lower().endswith(".docx")

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def parse(self, source: str, **kwargs):
        """Parse DOCX using python-docx."""
        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}],
            )

        try:
            doc = docx.Document(source)

            # Extract content using shared processor
            content_parts = DocxBlobProcessor.extract_content_from_doc(
                doc, self.extract_tables
            )

            # Extract headers and footers
            headers = DocxHeaderFooterExtractor.extract_headers(
                doc, self.extract_headers
            )
            footers = DocxHeaderFooterExtractor.extract_footers(
                doc, self.extract_footers
            )

            # Combine all content
            all_content = headers + content_parts + footers
            full_text = "\n\n".join(all_content)

            if not full_text.strip():
                return ProcessingResult(
                    documents=[],
                    errors=[
                        {"error": "No text extracted from document", "source": source}
                    ],
                )

            # Build metadata
            metadata = self._build_metadata(doc, path)

            # Create documents with chunking if needed
            documents = self._create_documents(full_text, metadata, str(path))

            return ProcessingResult(
                documents=documents,
                errors=[],
                metrics={
                    "total_documents": len(documents),
                    "parser_type": self.name,
                    "tool": "python-docx",
                },
            )

        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            return ProcessingResult(
                documents=[], errors=[{"error": str(e), "source": source}]
            )

    def _build_metadata(self, doc, path: Path) -> dict[str, Any]:
        """Build metadata dictionary for the document."""
        metadata = {
            "source": str(path),
            "file_name": path.name,
            "parser": self.name,
            "tool": "python-docx",
            "paragraphs": len(doc.paragraphs),
            "tables": len(doc.tables) if self.extract_tables else 0,
        }

        if self.extract_metadata:
            metadata = DocxMetadataExtractor.extract_document_properties(doc, metadata)

        return metadata

    def _create_documents(
        self, full_text: str, metadata: dict[str, Any], source_path: str
    ) -> list:
        """Create documents with optional chunking."""
        if not (self.chunk_size and self.chunk_size > 0):
            return [
                DocxDocumentFactory.create_single_document(
                    full_text, metadata, source_path
                )
            ]

        # Apply chunking strategy
        if self.chunk_strategy == "paragraphs":
            chunks = DocxChunker.chunk_by_paragraphs(
                full_text, self.chunk_size, self.chunk_overlap
            )
        elif self.chunk_strategy == "characters":
            chunks = DocxChunker.chunk_by_characters(full_text, self.chunk_size)
        else:
            # Default to paragraph chunking for unknown strategies
            chunks = DocxChunker.chunk_by_paragraphs(
                full_text, self.chunk_size, self.chunk_overlap
            )

        return DocxDocumentFactory.create_documents_from_chunks(
            chunks, metadata, source_path, self.chunk_strategy
        )

    def _iter_block_items(self, parent):
        """Iterate through document elements in order."""

        if isinstance(parent, docx.document.Document):
            parent_elm = parent.element.body
        else:
            raise ValueError("Unsupported parent type")

        for child in parent_elm.iterchildren():
            if isinstance(child, docx.oxml.text.paragraph.CT_P):
                yield docx.text.paragraph.Paragraph(child, parent)
            elif isinstance(child, docx.oxml.table.CT_Tbl):
                yield docx.table.Table(child, parent)

    def parse_blob(self, data: bytes, metadata: dict[str, Any] | None = None) -> list:
        """Parse DOCX from raw bytes using in-memory buffer."""
        try:
            # python-docx needs a file on disk, so write temporarily
            with DocxTempFileHandler(data) as tmp_path:
                doc = docx.Document(tmp_path)

                # Extract content using shared processor
                content_parts = DocxBlobProcessor.extract_content_from_doc(
                    doc, self.extract_tables
                )

                # Extract headers and footers
                headers = DocxHeaderFooterExtractor.extract_headers(
                    doc, self.extract_headers
                )
                footers = DocxHeaderFooterExtractor.extract_footers(
                    doc, self.extract_footers
                )

                # Combine all content
                all_content = headers + content_parts + footers
                full_text = "\n\n".join(all_content)

                if not full_text.strip():
                    logger.warning("No text extracted from DOCX blob")
                    return []

                # Build metadata for blob
                filename = (
                    metadata.get("filename", "document.docx")
                    if metadata
                    else "document.docx"
                )
                base_metadata = self._build_blob_metadata(doc, filename, metadata)

                # Create documents with chunking if needed
                return self._create_documents(full_text, base_metadata, filename)

        except Exception as e:
            logger.error(f"Failed to parse DOCX blob: {e}")
            return []

    def _build_blob_metadata(
        self, doc, filename: str, provided_metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Build metadata dictionary for blob parsing."""
        base_metadata = {
            "source": filename,
            "file_name": filename,
            "parser": self.name,
            "tool": "python-docx",
            "paragraphs": len(doc.paragraphs),
            "tables": len(doc.tables) if self.extract_tables else 0,
        }

        # Add provided metadata
        if provided_metadata:
            base_metadata |= provided_metadata

        if self.extract_metadata:
            base_metadata = DocxMetadataExtractor.extract_document_properties(
                doc, base_metadata
            )

        return base_metadata
