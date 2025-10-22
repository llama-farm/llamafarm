"""Shared utilities for DOCX parsers to eliminate code duplication."""

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from core.base import Document
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.docx.docx_utils")


class DocxMetadataExtractor:
    """Utility class for extracting metadata from DOCX documents."""

    @staticmethod
    def extract_document_properties(
        doc, base_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract document properties using python-docx."""
        try:
            metadata = base_metadata.copy()

            # Document properties
            if hasattr(doc.core_properties, "title") and doc.core_properties.title:
                metadata["title"] = doc.core_properties.title
            if hasattr(doc.core_properties, "author") and doc.core_properties.author:
                metadata["author"] = doc.core_properties.author
            if hasattr(doc.core_properties, "subject") and doc.core_properties.subject:
                metadata["subject"] = doc.core_properties.subject
            if (
                hasattr(doc.core_properties, "keywords")
                and doc.core_properties.keywords
            ):
                metadata["keywords"] = doc.core_properties.keywords
            if hasattr(doc.core_properties, "created") and doc.core_properties.created:
                metadata["created"] = str(doc.core_properties.created)
            if (
                hasattr(doc.core_properties, "modified")
                and doc.core_properties.modified
            ):
                metadata["modified"] = str(doc.core_properties.modified)
            if (
                hasattr(doc.core_properties, "revision")
                and doc.core_properties.revision
            ):
                metadata["revision"] = doc.core_properties.revision

            # Document statistics
            metadata["paragraph_count"] = len(doc.paragraphs)
            metadata["section_count"] = len(doc.sections)

            return metadata

        except Exception as e:
            logger.debug(f"Failed to extract enhanced metadata: {e}")
            return base_metadata


class DocxChunker:
    """Utility class for chunking DOCX content."""

    @staticmethod
    def chunk_by_paragraphs(
        text: str, chunk_size: int, chunk_overlap: int = 0
    ) -> list[str]:
        """Chunk text by paragraphs with size and overlap constraints."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk: list[str] = []
        current_size = 0

        for para in paragraphs:
            para_text = para.strip()
            if not para_text:
                continue

            para_size = len(para_text)

            if current_size + para_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append("\n\n".join(current_chunk))

                # Start new chunk with overlap
                if chunk_overlap > 0:
                    # Get the tail of the previous chunk for overlap
                    prev_chunk_text = "\n\n".join(current_chunk)
                    overlap_text = (
                        prev_chunk_text[-chunk_overlap:]
                        if len(prev_chunk_text) > chunk_overlap
                        else prev_chunk_text
                    )
                    current_chunk = [overlap_text, para_text]
                    current_size = len(overlap_text) + para_size
                else:
                    current_chunk = [para_text]
                    current_size = para_size
            else:
                current_chunk.append(para_text)
                current_size += para_size

        # Add last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    @staticmethod
    def chunk_by_characters(text: str, chunk_size: int) -> list[str]:
        """Simple character-based chunking."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks


class DocxDocumentFactory:
    """Factory for creating Document objects with consistent metadata."""

    @staticmethod
    def create_documents_from_chunks(
        chunks: list[str],
        base_metadata: dict[str, Any],
        source_path: str,
        chunk_strategy: str = "paragraphs",
    ) -> list[Document]:
        """Create Document objects from text chunks."""
        documents = []

        for i, chunk_content in enumerate(chunks):
            chunk_metadata = base_metadata | {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_strategy": chunk_strategy,
            }

            doc = Document(
                content=chunk_content,
                metadata=chunk_metadata,
                id=f"{Path(source_path).stem}_chunk_{i + 1}",
                source=source_path,
            )
            documents.append(doc)

        return documents

    @staticmethod
    def create_single_document(
        content: str, metadata: dict[str, Any], source_path: str
    ) -> Document:
        """Create a single Document object."""
        return Document(
            content=content,
            metadata=metadata,
            id=Path(source_path).stem,
            source=source_path,
        )


class DocxBlobProcessor:
    """Shared processor for DOCX blob parsing to eliminate duplication."""

    @staticmethod
    def extract_content_from_doc(doc, extract_tables: bool = True) -> list[str]:
        """Extract content parts from a python-docx Document object."""
        content_parts = []

        # Extract paragraphs and tables in order
        for element in DocxBlobProcessor._iter_block_items(doc):
            if hasattr(element, "text"):  # Paragraph
                text = element.text.strip()
                if text:
                    # Check if it's a heading
                    if (
                        hasattr(element, "style")
                        and element.style
                        and element.style.name
                        and "Heading" in element.style.name
                    ):
                        content_parts.append(f"\n## {text}\n")
                    else:
                        content_parts.append(text)

            elif hasattr(element, "rows") and extract_tables:  # Table
                if table_text := DocxTableExtractor.extract_table_as_text(element):
                    content_parts.append(f"\n{table_text}\n")

        return content_parts

    @staticmethod
    def _iter_block_items(parent):
        """Iterate through document elements in order."""
        try:
            import docx

            if isinstance(parent, docx.document.Document):
                parent_elm = parent.element.body
            else:
                raise ValueError("Unsupported parent type")

            for child in parent_elm.iterchildren():
                if hasattr(docx.oxml.text.paragraph, "CT_P") and isinstance(
                    child, docx.oxml.text.paragraph.CT_P
                ):
                    yield docx.text.paragraph.Paragraph(child, parent)
                elif hasattr(docx.oxml.table, "CT_Tbl") and isinstance(
                    child, docx.oxml.table.CT_Tbl
                ):
                    yield docx.table.Table(child, parent)
        except ImportError:
            # Fallback to simple paragraph iteration if docx not available
            yield from parent.paragraphs


class DocxTempFileHandler:
    """Context manager for handling temporary DOCX files."""

    def __init__(self, data: bytes):
        self.data = data
        self.tmp_path: Optional[str] = None

    def __enter__(self) -> str:
        """Create temporary file and return path."""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
            tmp_file.write(self.data)
            self.tmp_path = tmp_file.name
        return self.tmp_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary file."""
        if self.tmp_path and os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)


class DocxTableExtractor:
    """Utility for extracting tables from DOCX documents."""

    @staticmethod
    def extract_table_as_text(table) -> str:
        """Extract table as formatted text."""
        rows = []
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                cell_text = " ".join(p.text for p in cell.paragraphs).strip()
                row_text.append(cell_text)
            rows.append(" | ".join(row_text))

        if rows and len(rows) > 1:
            rows.insert(1, "-" * len(rows[0]))

        return "\n".join(rows)


class DocxHeaderFooterExtractor:
    """Utility for extracting headers and footers from DOCX documents."""

    @staticmethod
    def extract_headers(doc, extract_headers: bool = True) -> list[str]:
        """Extract headers from all sections."""
        if not extract_headers:
            return []

        headers = []
        for section in doc.sections:
            if header := section.header:
                if header_text := "\n".join(
                    p.text for p in header.paragraphs if p.text.strip()
                ):
                    headers.append(f"Header: {header_text}")
        return headers

    @staticmethod
    def extract_footers(doc, extract_footers: bool = True) -> list[str]:
        """Extract footers from all sections."""
        if not extract_footers:
            return []

        footers = []
        for section in doc.sections:
            if footer := section.footer:
                if footer_text := "\n".join(
                    p.text for p in footer.paragraphs if p.text.strip()
                ):
                    footers.append(f"Footer: {footer_text}")
        return footers
