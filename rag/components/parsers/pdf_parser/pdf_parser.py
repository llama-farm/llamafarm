"""PDF parser for extracting text, metadata, and structure from PDF documents."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import PyPDF2

    PYPDF2_AVAILABLE = True
except ImportError:
    PyPDF2 = None
    PYPDF2_AVAILABLE = False

# Type hints compatibility
if PYPDF2_AVAILABLE:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        PdfReaderType = PyPDF2.PdfReader
else:
    PdfReaderType = None

from core.base import Parser, Document, ProcessingResult

logger = logging.getLogger(__name__)


class PDFParser(Parser):
    """Parser for PDF files with text and metadata extraction."""

    def __init__(self, name: str = "PDFParser", config: Dict[str, Any] = None):
        super().__init__(name, config)
        config = config or {}

        if not PYPDF2_AVAILABLE:
            raise ImportError(
                "PyPDF2 is required for PDF parsing. Install it with: pip install PyPDF2"
            )

        # Configuration options
        self.extract_metadata = config.get("extract_metadata", True)
        self.extract_page_structure = config.get("extract_page_structure", True)
        self.combine_pages = config.get("combine_pages", True)
        self.page_separator = config.get("page_separator", "\n\n--- Page Break ---\n\n")
        self.min_text_length = config.get("min_text_length", 10)
        self.include_page_numbers = config.get("include_page_numbers", True)
        self.extract_outline = config.get("extract_outline", True)

    def validate_config(self) -> bool:
        """Validate parser configuration."""
        if self.min_text_length < 0:
            raise ValueError("min_text_length must be non-negative")
        return True

    def parse(self, source: str) -> ProcessingResult:
        """Parse PDF file into documents.

        Args:
            source: Path to PDF file

        Returns:
            ProcessingResult containing extracted documents
        """
        documents = []
        errors = []

        try:
            pdf_path = Path(source)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {source}")

            # if not pdf_path.suffix.lower() == '.pdf':
            #     raise ValueError(f"File is not a PDF: {source}")

            with open(pdf_path, "rb") as file:
                try:
                    reader = PyPDF2.PdfReader(file)

                    # Extract document-level metadata
                    doc_metadata = self._extract_document_metadata(reader, pdf_path)

                    # Extract outline/bookmarks if requested
                    outline_text = ""
                    if self.extract_outline and reader.outline:
                        outline_text = self._extract_outline_text(reader.outline)

                    # Extract text from pages
                    page_texts = []
                    page_errors = []

                    for page_num, page in enumerate(reader.pages, 1):
                        try:
                            page_text = self._extract_page_text(page, page_num)
                            if len(page_text.strip()) >= self.min_text_length:
                                page_texts.append(
                                    {
                                        "text": page_text,
                                        "page_num": page_num,
                                        "metadata": self._extract_page_metadata(
                                            page, page_num
                                        ),
                                    }
                                )
                        except Exception as e:
                            page_errors.append(
                                {"page": page_num, "error": str(e), "source": source}
                            )
                            logger.warning(
                                f"Error extracting text from page {page_num}: {e}"
                            )

                    # Create documents based on configuration
                    if self.combine_pages:
                        # Single document with all pages combined
                        combined_text = self._combine_page_texts(
                            page_texts, outline_text
                        )
                        if combined_text.strip():
                            doc = Document(
                                content=combined_text,
                                metadata={
                                    **doc_metadata,
                                    "total_pages": len(reader.pages),
                                    "extracted_pages": len(page_texts),
                                    "has_outline": bool(outline_text),
                                },
                                id=f"{pdf_path.stem}",
                                source=source,
                            )
                            documents.append(doc)
                    else:
                        # Separate document for each page
                        for page_data in page_texts:
                            page_content = page_data["text"]
                            if outline_text and page_data["page_num"] == 1:
                                # Include outline in first page
                                page_content = f"{outline_text}\n\n{page_content}"

                            doc = Document(
                                content=page_content,
                                metadata={
                                    **doc_metadata,
                                    **page_data["metadata"],
                                    "page_number": page_data["page_num"],
                                    "total_pages": len(reader.pages),
                                },
                                id=f"{pdf_path.stem}_page_{page_data['page_num']}",
                                source=source,
                            )
                            documents.append(doc)

                    # Add page-level errors to overall errors
                    errors.extend(page_errors)

                except Exception as e:
                    errors.append(
                        {
                            "error": f"Failed to read PDF structure: {str(e)}",
                            "source": source,
                        }
                    )

        except Exception as e:
            errors.append(
                {"error": f"Failed to open PDF file: {str(e)}", "source": source}
            )

        return ProcessingResult(
            documents=documents,
            errors=errors,
            metrics={
                "total_documents": len(documents),
                "total_errors": len(errors),
                "file_processed": source,
                "parser_type": self.name,
            },
        )

    def _extract_document_metadata(self, reader, pdf_path: Path) -> Dict[str, Any]:
        """Extract document-level metadata from PDF."""
        metadata = {
            "source_file": pdf_path.name,
            "file_size_bytes": pdf_path.stat().st_size,
            "parser_type": "PDFParser",
        }

        if not self.extract_metadata:
            return metadata

        try:
            # PDF metadata
            pdf_info = reader.metadata
            if pdf_info:
                # Standard PDF metadata fields
                metadata.update(
                    {
                        "title": pdf_info.get("/Title", "").strip()
                        if pdf_info.get("/Title")
                        else "",
                        "author": pdf_info.get("/Author", "").strip()
                        if pdf_info.get("/Author")
                        else "",
                        "subject": pdf_info.get("/Subject", "").strip()
                        if pdf_info.get("/Subject")
                        else "",
                        "creator": pdf_info.get("/Creator", "").strip()
                        if pdf_info.get("/Creator")
                        else "",
                        "producer": pdf_info.get("/Producer", "").strip()
                        if pdf_info.get("/Producer")
                        else "",
                    }
                )

                # Date fields
                for date_field, meta_key in [
                    ("/CreationDate", "creation_date"),
                    ("/ModDate", "modification_date"),
                ]:
                    if pdf_info.get(date_field):
                        try:
                            # PDF dates are in format like "D:20231201120000+00'00'"
                            date_str = str(pdf_info[date_field])
                            if date_str.startswith("D:"):
                                date_str = date_str[2:16]  # Extract YYYYMMDDHHMMSS
                                parsed_date = datetime.strptime(
                                    date_str, "%Y%m%d%H%M%S"
                                )
                                metadata[meta_key] = parsed_date.isoformat()
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Could not parse date {date_field}: {e}")

            # Document structure info
            metadata.update(
                {
                    "total_pages": len(reader.pages),
                    "is_encrypted": reader.is_encrypted,
                    "has_outline": bool(reader.outline),
                }
            )

        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")

        return metadata

    def _extract_page_metadata(self, page, page_num: int) -> Dict[str, Any]:
        """Extract metadata from a specific page."""
        metadata = {"page_number": page_num}

        if not self.extract_page_structure:
            return metadata

        try:
            # Page dimensions
            if hasattr(page, "mediabox"):
                mediabox = page.mediabox
                metadata.update(
                    {
                        "page_width": float(mediabox.width),
                        "page_height": float(mediabox.height),
                        "page_rotation": page.get("/Rotate", 0)
                        if hasattr(page, "get")
                        else 0,
                    }
                )
        except Exception as e:
            logger.debug(f"Could not extract page {page_num} metadata: {e}")

        return metadata

    def _extract_page_text(self, page, page_num: int) -> str:
        """Extract text from a PDF page."""
        try:
            text = page.extract_text()

            # Clean up the text
            text = text.strip()

            # Add page number if requested
            if self.include_page_numbers and text:
                text = f"[Page {page_num}]\n\n{text}"

            return text

        except Exception as e:
            logger.warning(f"Error extracting text from page {page_num}: {e}")
            return ""

    def _extract_outline_text(self, outline: List) -> str:
        """Extract text from PDF outline/bookmarks."""

        def extract_outline_recursive(items: List, level: int = 0) -> List[str]:
            outline_items = []
            for item in items:
                if isinstance(item, list):
                    # Nested outline
                    outline_items.extend(extract_outline_recursive(item, level + 1))
                else:
                    # Individual outline item
                    try:
                        title = str(item.title) if hasattr(item, "title") else str(item)
                        indent = "  " * level
                        outline_items.append(f"{indent}- {title}")
                    except Exception as e:
                        logger.debug(f"Could not extract outline item: {e}")

            return outline_items

        try:
            outline_items = extract_outline_recursive(outline)
            if outline_items:
                return "Document Outline:\n" + "\n".join(outline_items) + "\n"
        except Exception as e:
            logger.warning(f"Error extracting outline: {e}")

        return ""

    def _combine_page_texts(self, page_texts: List[Dict], outline_text: str) -> str:
        """Combine text from multiple pages into a single document."""
        combined_parts = []

        # Add outline at the beginning if available
        if outline_text:
            combined_parts.append(outline_text)

        # Add page texts
        for page_data in page_texts:
            combined_parts.append(page_data["text"])

        return self.page_separator.join(combined_parts)
