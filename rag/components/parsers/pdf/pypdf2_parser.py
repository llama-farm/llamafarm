"""PDF parser using PyPDF2 library."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from components.parsers.base.base_parser import BaseParser, ParserConfig

logger = logging.getLogger(__name__)


class PDFParser_PyPDF2(BaseParser):
    """PDF parser using PyPDF2 for text extraction with enhanced capabilities."""

    def __init__(
        self, name: str = "PDFParser_PyPDF2", config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)  # Call BaseParser init
        self.name = name
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.chunk_strategy = self.config.get("chunk_strategy", "paragraphs")
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.preserve_layout = self.config.get("preserve_layout", True)
        self.extract_page_info = self.config.get("extract_page_info", True)
        self.extract_page_structure = self.config.get("extract_page_structure", True)
        self.combine_pages = self.config.get("combine_pages", True)
        self.page_separator = self.config.get(
            "page_separator", "\n\n--- Page Break ---\n\n"
        )
        self.min_text_length = self.config.get("min_text_length", 10)
        self.include_page_numbers = self.config.get("include_page_numbers", True)
        self.extract_annotations = self.config.get("extract_annotations", False)
        self.extract_links = self.config.get("extract_links", False)
        self.extract_form_fields = self.config.get("extract_form_fields", False)
        self.extract_outlines = self.config.get("extract_outlines", False)
        self.extract_outline = self.config.get(
            "extract_outline", True
        )  # Different from extract_outlines
        self.extract_images = self.config.get("extract_images", False)
        self.extract_xmp_metadata = self.config.get("extract_xmp_metadata", False)
        self.clean_text = self.config.get("clean_text", True)

    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata."""
        return ParserConfig(
            name="PDFParser_PyPDF2",
            display_name="PyPDF2 PDF Parser",
            version="1.0.0",
            supported_extensions=[".pdf"],
            mime_types=["application/pdf"],
            capabilities=[
                "text_extraction",
                "metadata_extraction",
                "layout_preservation",
            ],
            dependencies={"PyPDF2": ["PyPDF2>=3.0.0"]},
            default_config={
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "chunk_strategy": "paragraphs",
                "extract_metadata": True,
                "preserve_layout": True,
            },
        )

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        return file_path.lower().endswith(".pdf")

    def validate_config(self) -> bool:
        """Validate configuration."""
        if self.min_text_length < 0:
            raise ValueError("min_text_length must be non-negative")
        return True

    def parse_blob(self, data: bytes, metadata: Dict[str, Any] = None) -> List:
        """Parse PDF from raw bytes."""
        from core.base import Document
        import io

        try:
            import PyPDF2
        except ImportError:
            print("PyPDF2 not installed. Install with: pip install PyPDF2")
            return []

        try:
            # Create a BytesIO object from the raw data
            pdf_file = io.BytesIO(data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            documents = []
            total_text = ""

            # Extract text from all pages
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > self.min_text_length:
                        total_text += page_text + "\n"
                except Exception as e:
                    print(f"Error extracting text from page {page_num}: {e}")
                    continue

            # Create document if we extracted any text
            if total_text.strip():
                filename = (
                    metadata.get("filename", "unknown.pdf")
                    if metadata
                    else "unknown.pdf"
                )
                doc = Document(
                    content=total_text,
                    metadata={
                        "source": filename,
                        "parser": "PDFParser_PyPDF2",
                        "page_count": len(pdf_reader.pages),
                        **(metadata or {}),
                    },
                    source=filename,
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"Error parsing PDF: {e}")
            return []

    def parse(self, source: str, **kwargs):
        """Parse PDF using PyPDF2."""
        from core.base import Document, ProcessingResult

        try:
            import PyPDF2
        except ImportError:
            return ProcessingResult(
                documents=[],
                errors=[
                    {
                        "error": "PyPDF2 not installed. Install with: pip install PyPDF2",
                        "source": source,
                    }
                ],
            )

        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}],
            )

        try:
            # First check if file looks like a PDF
            with open(source, "rb") as f:
                header = f.read(8)
                f.seek(0)

                if not header.startswith(b"%PDF"):
                    return ProcessingResult(
                        documents=[],
                        errors=[
                            {
                                "error": f"File is not a PDF (invalid header): {source}",
                                "source": source,
                            }
                        ],
                    )

                try:
                    pdf_reader = PyPDF2.PdfReader(f)
                except PyPDF2.errors.PdfReadError as e:
                    return ProcessingResult(
                        documents=[],
                        errors=[
                            {
                                "error": f"File is not a valid PDF: {str(e)}",
                                "source": source,
                            }
                        ],
                    )

                text = ""
                metadata = {}

                # Extract metadata using PyPDF2's built-in metadata
                if self.extract_metadata and pdf_reader.metadata:
                    metadata = {
                        "title": pdf_reader.metadata.get("/Title"),
                        "author": pdf_reader.metadata.get("/Author"),
                        "subject": pdf_reader.metadata.get("/Subject"),
                        "creator": pdf_reader.metadata.get("/Creator"),
                        "producer": pdf_reader.metadata.get("/Producer"),
                        "creation_date": str(pdf_reader.metadata.get("/CreationDate"))
                        if pdf_reader.metadata.get("/CreationDate")
                        else None,
                        "modification_date": str(pdf_reader.metadata.get("/ModDate"))
                        if pdf_reader.metadata.get("/ModDate")
                        else None,
                        "pages": len(pdf_reader.pages),
                        "encrypted": pdf_reader.is_encrypted,
                    }
                    # Remove None values
                    metadata = {k: v for k, v in metadata.items() if v is not None}

                # Extract XMP metadata if requested
                if self.extract_xmp_metadata:
                    try:
                        xmp = pdf_reader.xmp_metadata
                        if xmp:
                            metadata["xmp_metadata"] = {
                                "dc_title": xmp.dc_title,
                                "dc_creator": xmp.dc_creator,
                                "dc_description": xmp.dc_description,
                                "dc_subject": xmp.dc_subject,
                                "dc_date": xmp.dc_date,
                                "dc_language": xmp.dc_language,
                                "pdf_keywords": xmp.pdf_keywords,
                                "pdf_producer": xmp.pdf_producer,
                            }
                    except:
                        pass  # XMP metadata may not be available

                # Extract document-level outlines/bookmarks if requested
                if self.extract_outlines:
                    try:
                        outlines = pdf_reader.outline
                        if outlines:
                            metadata["outlines"] = self._extract_outline_structure(
                                outlines
                            )
                    except:
                        pass

                # Extract form fields if requested
                if self.extract_form_fields:
                    try:
                        fields = pdf_reader.get_form_text_fields()
                        if fields:
                            metadata["form_fields"] = fields
                    except:
                        pass

                page_texts = []
                page_metadata = []

                # Extract text from all pages using PyPDF2's extraction modes
                for page_num, page in enumerate(pdf_reader.pages):
                    page_info = {
                        "page_number": page_num + 1,
                        "rotation": page.rotation if hasattr(page, "rotation") else 0,
                    }

                    # Use PyPDF2's layout-preserving extraction (with backward compatibility)
                    try:
                        if self.preserve_layout:
                            page_text = page.extract_text(extraction_mode="layout")
                        else:
                            page_text = page.extract_text()
                    except TypeError:
                        # Fallback for older PyPDF2 versions that don't support extraction_mode
                        page_text = page.extract_text()

                    # Extract annotations if requested (PyPDF2 built-in)
                    if (
                        self.extract_annotations
                        and hasattr(page, "/Annots")
                        and page["/Annots"]
                    ):
                        try:
                            annotations = []
                            for annot in page["/Annots"]:
                                annot_obj = annot.get_object()
                                if "/Contents" in annot_obj:
                                    annotations.append(str(annot_obj["/Contents"]))
                            if annotations:
                                page_info["annotations"] = annotations
                        except:
                            pass  # Skip if annotations can't be extracted

                    # Extract images from page if requested
                    if self.extract_images:
                        try:
                            images = page.images
                            if images:
                                page_info["images"] = [
                                    {"name": img.name, "data": img.data}
                                    for img in images
                                ]
                        except:
                            pass  # Skip if images can't be extracted

                    if page_text:
                        if self.extract_page_info:
                            page_texts.append(
                                f"\n--- Page {page_num + 1} ---\n{page_text}"
                            )
                        else:
                            page_texts.append(page_text)
                        page_metadata.append(page_info)

                # Join all page texts with appropriate separator
                if self.combine_pages:
                    text = self.page_separator.join(page_texts) if page_texts else ""
                else:
                    text = "\n\n".join(page_texts) if page_texts else ""

            if not text.strip():
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No text extracted from PDF", "source": source}],
                )

            # Add parser info to metadata
            metadata.update(
                {
                    "source": str(path),
                    "source_file": str(path),
                    "file_name": path.name,
                    "file_size_bytes": path.stat().st_size,
                    "total_pages": len(page_metadata),
                    "parser": self.name,
                    "parser_type": "PDFParser",
                    "tool": "PyPDF2",
                    "pages_processed": len(page_metadata),
                    "extraction_mode": "layout" if self.preserve_layout else "standard",
                }
            )

            if page_metadata:
                metadata["page_info"] = page_metadata

            documents = []

            # Check if we should create separate documents for each page
            if not self.combine_pages and len(page_texts) > 1:
                # Create separate documents for each page
                for i, page_text in enumerate(page_texts):
                    if len(page_text.strip()) >= self.min_text_length:
                        page_meta = metadata.copy()
                        page_meta.update(
                            {
                                "page_number": i + 1,
                                "total_pages": len(page_texts),
                                "is_single_page": True,
                            }
                        )

                        # Add page-specific info if available
                        if i < len(page_metadata) and page_metadata:
                            page_meta.update(page_metadata[i])

                        doc = Document(
                            content=page_text,
                            metadata=page_meta,
                            id=f"{path.stem}_page_{i + 1}",
                            source=str(path),
                        )
                        documents.append(doc)
            elif self.combine_pages:
                # When combine_pages is True, create single document regardless of chunk_size
                doc = Document(
                    content=text, metadata=metadata, id=path.stem, source=str(path)
                )
                documents.append(doc)
            elif self.chunk_size and self.chunk_size > 0:
                # Apply chunking if configured (only when not combining pages)
                chunks = self._chunk_text(text)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update(
                        {"chunk_index": i, "total_chunks": len(chunks)}
                    )

                    doc = Document(
                        content=chunk,
                        metadata=chunk_metadata,
                        id=f"{path.stem}_chunk_{i + 1}",
                        source=str(path),
                    )
                    documents.append(doc)
            else:
                # Single document with all content (fallback)
                doc = Document(
                    content=text, metadata=metadata, id=path.stem, source=str(path)
                )
                documents.append(doc)

            return ProcessingResult(
                documents=documents,
                errors=[],
                metrics={
                    "total_documents": len(documents),
                    "parser_type": self.name,
                    "tool": "PyPDF2",
                },
            )

        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            return ProcessingResult(
                documents=[], errors=[{"error": str(e), "source": source}]
            )

    def _chunk_text(self, text: str) -> List[str]:
        """Text chunking using PyPDF2 extracted structure."""
        if self.chunk_strategy == "paragraphs":
            # Split by double newlines (paragraph breaks that PyPDF2 preserves)
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            chunks = []
            current_chunk = ""

            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks if chunks else [text]

        elif self.chunk_strategy == "sentences":
            # Use PyPDF2's natural line breaks for better sentence detection
            sentences = []
            for line in text.split("\n"):
                if line.strip():
                    # Split by sentence endings while preserving structure
                    line_sentences = []
                    current = ""
                    for char in line:
                        current += char
                        if char in ".!?" and (
                            len(current) > 10
                        ):  # Avoid splitting abbreviations
                            line_sentences.append(current.strip())
                            current = ""
                    if current.strip():
                        line_sentences.append(current.strip())
                    sentences.extend(line_sentences)

            # Combine sentences into chunks
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks if chunks else [text]

        else:
            # Character-based chunking with overlap
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i : i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
            return chunks

    def _extract_outline_structure(self, outlines, level=0):
        """Extract outline/bookmark structure from PyPDF2."""
        outline_data = []
        for item in outlines:
            if isinstance(item, list):
                # Nested outline
                outline_data.extend(self._extract_outline_structure(item, level + 1))
            else:
                # Individual outline item
                outline_info = {
                    "title": str(item.title) if hasattr(item, "title") else str(item),
                    "level": level,
                }
                if hasattr(item, "page") and item.page:
                    try:
                        # Get page number if available
                        if hasattr(item.page, "idnum"):
                            outline_info["page"] = item.page.idnum
                    except:
                        pass
                outline_data.append(outline_info)
        return outline_data
