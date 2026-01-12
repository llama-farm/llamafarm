"""
Docling Parser - Advanced document parsing using IBM's Docling library.

Features:
- AI-powered layout analysis (DocLayNet)
- Table structure recognition (TableFormer)
- Smart chunking with HybridChunker
- Comprehensive metadata extraction
- Multi-format support (PDF, DOCX, PPTX, XLSX, HTML, images)
"""

import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from core.base import Document, ProcessingResult
from core.logging import RAGStructLogger

from ..base.base_parser import BaseParser, ParserConfig

logger = RAGStructLogger("rag.components.parsers.docling.docling_parser")

# Lazy imports to handle missing dependencies gracefully
DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import ConversionResult

    DOCLING_AVAILABLE = True
except ImportError:
    logger.warning("Docling not available. Install with: pip install docling docling-core[chunking]")

# Chunking imports (separate try block as chunking is optional)
DOCLING_CHUNKING_AVAILABLE = False
try:
    from docling.chunking import HybridChunker, HierarchicalChunker
    DOCLING_CHUNKING_AVAILABLE = True
except ImportError:
    logger.debug("Docling chunking not available. Install with: pip install docling-core[chunking]")


class DoclingParser(BaseParser):
    """
    Document parser using IBM's Docling library.

    Provides AI-powered document understanding with:
    - Layout analysis for complex documents
    - Table structure recognition
    - Smart chunking that respects document structure
    - Comprehensive metadata extraction
    """

    def __init__(self, name: str = "DoclingParser", config: dict[str, Any] | None = None):
        """
        Initialize the Docling parser.

        Args:
            name: Parser name identifier
            config: Parser configuration options:
                - chunk_size: Target chunk size in tokens (default: 512)
                - chunk_overlap: Overlap between chunks (default: 50)
                - chunk_strategy: 'hybrid', 'hierarchical', or 'none' (default: 'hybrid')
                - extract_metadata: Extract document metadata (default: True)
                - extract_tables: Extract tables (default: True)
                - extract_headings: Extract headings (default: True)
                - output_format: 'markdown', 'text', or 'json' (default: 'markdown')
                - preserve_structure: Preserve document structure (default: True)
                - page_aware: Include page numbers in chunks (default: True)
        """
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "Docling is required. Install with: pip install docling docling-core[chunking]"
            )

        self.name = name
        self.config = config or {}

        # Configuration with defaults
        self.chunk_size = self.config.get("chunk_size", 512)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)
        self.chunk_strategy = self.config.get("chunk_strategy", "hybrid")
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.extract_tables = self.config.get("extract_tables", True)
        self.extract_headings = self.config.get("extract_headings", True)
        self.output_format = self.config.get("output_format", "markdown")
        self.preserve_structure = self.config.get("preserve_structure", True)
        self.page_aware = self.config.get("page_aware", True)

        # Initialize Docling converter
        self.converter = DocumentConverter()

        # Initialize chunker if available and strategy is not 'none'
        self.chunker = None
        if DOCLING_CHUNKING_AVAILABLE and self.chunk_strategy != "none":
            self._initialize_chunker()

    def _initialize_chunker(self) -> None:
        """Initialize the appropriate chunker based on strategy."""
        try:
            if self.chunk_strategy == "hybrid":
                # HybridChunker combines structure-aware and tokenization-aware chunking
                self.chunker = HybridChunker(
                    tokenizer="sentence-transformers/all-MiniLM-L6-v2",
                    max_tokens=self.chunk_size,
                )
            elif self.chunk_strategy == "hierarchical":
                # HierarchicalChunker creates one chunk per document element
                self.chunker = HierarchicalChunker(
                    merge_list_items=True,
                )
            logger.debug(f"Initialized {self.chunk_strategy} chunker with max_tokens={self.chunk_size}")
        except Exception as e:
            logger.warning(f"Failed to initialize chunker: {e}. Chunking will be disabled.")
            self.chunker = None

    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata from config."""
        return ParserConfig(
            name="DoclingParser",
            display_name="Docling Parser (IBM)",
            version="1.0.0",
            supported_extensions=[".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm",
                                   ".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
            mime_types=["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       "text/html", "image/png", "image/jpeg", "image/tiff", "image/bmp"],
            capabilities=["text_extraction", "layout_analysis", "table_extraction",
                         "heading_extraction", "smart_chunking", "page_aware_splitting",
                         "metadata_extraction", "structure_preservation", "ocr_support"],
            dependencies={"required": ["docling", "docling-core"], "optional": ["transformers", "torch"]},
            default_config={
                "chunk_size": 512,
                "chunk_overlap": 50,
                "chunk_strategy": "hybrid",
                "extract_metadata": True,
                "output_format": "markdown",
            },
        )

    def parse(self, source: str) -> ProcessingResult:
        """
        Parse a document file using Docling.

        Args:
            source: Path to the document file

        Returns:
            ProcessingResult with parsed documents
        """
        documents = []
        errors = []

        try:
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"File not found: {source}")

            # Convert document
            result = self.converter.convert(source)

            # Extract content and metadata
            documents = self._process_conversion_result(result, source_path)

        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            errors.append({
                "source": source,
                "error": str(e),
                "parser": self.name,
            })

        return ProcessingResult(
            documents=documents,
            errors=errors,
            metrics={
                "total_documents": len(documents),
                "total_errors": len(errors),
                "parser_type": self.name,
                "chunking_applied": self.chunker is not None,
                "chunk_strategy": self.chunk_strategy,
            },
        )

    def parse_blob(self, blob_data: bytes, metadata: dict[str, Any]) -> list[Document]:
        """
        Parse document from blob data.

        Args:
            blob_data: Raw bytes of the document
            metadata: Metadata dict containing:
                - filename: Original filename
                - content_type: MIME type
                - dataset_name: Name of the dataset (optional)
                - upload_timestamp: When file was uploaded (optional)
                - file_hash: Content hash (optional)

        Returns:
            List of Document objects with comprehensive metadata
        """
        filename = metadata.get("filename", "unknown")

        # Write blob to temporary file for Docling processing
        suffix = Path(filename).suffix or ".bin"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(blob_data)
            tmp_path = Path(tmp_file.name)

        try:
            # Convert document
            result = self.converter.convert(str(tmp_path))

            # Process with comprehensive metadata
            documents = self._process_conversion_result(
                result,
                tmp_path,
                original_filename=filename,
                additional_metadata=metadata,
            )

            return documents

        except Exception as e:
            logger.error(f"Failed to parse blob {filename}: {e}")
            # Return minimal document with error info
            return [
                Document(
                    content=f"[Parsing failed: {e}]",
                    metadata={
                        **self._build_base_metadata(metadata),
                        "parsing_error": str(e),
                        "parser_type": self.name,
                    },
                )
            ]
        finally:
            # Clean up temp file
            try:
                tmp_path.unlink()
            except Exception:
                pass

    def _process_conversion_result(
        self,
        result: "ConversionResult",
        source_path: Path,
        original_filename: str | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Process Docling conversion result into Document objects.

        Args:
            result: Docling ConversionResult
            source_path: Path to source file
            original_filename: Original filename (if different from source)
            additional_metadata: Additional metadata to include

        Returns:
            List of Document objects
        """
        documents = []
        additional_metadata = additional_metadata or {}

        # Get the document
        doc = result.document

        # Extract document-level metadata
        doc_metadata = self._extract_document_metadata(doc, source_path, original_filename, additional_metadata)

        # Export content based on format
        if self.output_format == "markdown":
            content = doc.export_to_markdown()
        elif self.output_format == "json":
            content = doc.export_to_json()
        else:
            # Text format - strip markdown
            content = doc.export_to_markdown()  # Docling doesn't have plain text export

        # Apply chunking if available
        if self.chunker and self.chunk_strategy != "none":
            documents = self._apply_chunking(doc, doc_metadata, content)
        else:
            # Return as single document
            documents = [
                Document(
                    content=content,
                    metadata={
                        **doc_metadata,
                        "chunk_index": 0,
                        "chunk_num": 1,
                        "total_chunks": 1,
                        "is_chunked": False,
                    },
                    id=self._generate_doc_id(content, doc_metadata),
                    source=original_filename or str(source_path),
                )
            ]

        return documents

    def _apply_chunking(
        self,
        doc: Any,
        doc_metadata: dict[str, Any],
        full_content: str,
    ) -> list[Document]:
        """
        Apply Docling's smart chunking to the document.

        Args:
            doc: Docling document object
            doc_metadata: Base document metadata
            full_content: Full document content (for fallback)

        Returns:
            List of chunked Document objects
        """
        documents = []

        try:
            chunks = list(self.chunker.chunk(dl_doc=doc))
            total_chunks = len(chunks)

            for i, chunk in enumerate(chunks):
                # Extract chunk text
                chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)

                # Extract chunk metadata from docling
                chunk_meta = {}
                if hasattr(chunk, "meta"):
                    meta = chunk.meta
                    # Extract heading hierarchy
                    if hasattr(meta, "headings") and meta.headings:
                        chunk_meta["headings"] = meta.headings
                    # Extract document location
                    if hasattr(meta, "doc_items") and meta.doc_items:
                        # Get page numbers from doc_items
                        pages = set()
                        for item in meta.doc_items:
                            if hasattr(item, "prov") and item.prov:
                                for prov in item.prov:
                                    if hasattr(prov, "page_no"):
                                        pages.add(prov.page_no)
                        if pages:
                            chunk_meta["page_numbers"] = sorted(pages)
                            chunk_meta["page_start"] = min(pages)
                            chunk_meta["page_end"] = max(pages)

                # Build chunk metadata
                chunk_metadata = {
                    **doc_metadata,
                    **chunk_meta,
                    "chunk_index": i,
                    "chunk_num": i + 1,
                    "total_chunks": total_chunks,
                    "chunk_strategy": self.chunk_strategy,
                    "chunk_size_config": self.chunk_size,
                    "chunk_char_count": len(chunk_text),
                    "is_chunked": True,
                    "has_overlap": i > 0 and self.chunk_overlap > 0,
                }

                doc_id = self._generate_doc_id(chunk_text, chunk_metadata, chunk_index=i)

                documents.append(
                    Document(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        id=doc_id,
                        source=doc_metadata.get("source_filename"),
                    )
                )

            logger.debug(f"Created {total_chunks} chunks using {self.chunk_strategy} strategy")

        except Exception as e:
            logger.warning(f"Chunking failed: {e}. Returning full document.")
            # Fallback to single document
            documents = [
                Document(
                    content=full_content,
                    metadata={
                        **doc_metadata,
                        "chunk_index": 0,
                        "chunk_num": 1,
                        "total_chunks": 1,
                        "is_chunked": False,
                        "chunking_error": str(e),
                    },
                    id=self._generate_doc_id(full_content, doc_metadata),
                    source=doc_metadata.get("source_filename"),
                )
            ]

        return documents

    def _extract_document_metadata(
        self,
        doc: Any,
        source_path: Path,
        original_filename: str | None,
        additional_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract comprehensive metadata from the document.

        Args:
            doc: Docling document object
            source_path: Path to source file
            original_filename: Original filename
            additional_metadata: Additional metadata from caller

        Returns:
            Comprehensive metadata dictionary
        """
        # Base metadata
        metadata = self._build_base_metadata(additional_metadata)

        # File-level metadata
        filename = original_filename or source_path.name
        metadata.update({
            "source_filename": filename,
            "source_path": str(source_path),
            "file_extension": source_path.suffix.lower(),
            "parser_type": self.name,
            "parser_version": "1.0.0",
            "output_format": self.output_format,
            "parsing_timestamp": datetime.now().isoformat(),
        })

        # Document structure metadata
        if self.extract_metadata:
            try:
                # Extract headings
                if self.extract_headings and hasattr(doc, "main_text"):
                    headings = []
                    for item in doc.main_text:
                        if hasattr(item, "label") and "heading" in str(item.label).lower():
                            heading_text = item.text if hasattr(item, "text") else str(item)
                            headings.append({
                                "text": heading_text[:200],  # Truncate long headings
                                "level": self._detect_heading_level(item),
                            })
                    if headings:
                        metadata["headings"] = headings[:20]  # Limit to 20 headings
                        metadata["heading_count"] = len(headings)

                # Extract page count
                if hasattr(doc, "pages"):
                    metadata["page_count"] = len(doc.pages) if doc.pages else 1

                # Extract tables count
                if self.extract_tables and hasattr(doc, "tables"):
                    metadata["table_count"] = len(doc.tables) if doc.tables else 0

                # Document properties if available
                if hasattr(doc, "origin") and doc.origin:
                    origin = doc.origin
                    if hasattr(origin, "filename"):
                        metadata["docling_origin_filename"] = origin.filename
                    if hasattr(origin, "mimetype"):
                        metadata["docling_mimetype"] = origin.mimetype

            except Exception as e:
                logger.debug(f"Error extracting document metadata: {e}")

        return metadata

    def _build_base_metadata(self, additional_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Build base metadata from additional metadata passed in.

        This preserves important context like dataset name, upload time, etc.
        """
        metadata = {}

        # Preserve key metadata fields from the upload/ingest context
        preserve_fields = [
            "dataset_name",
            "upload_timestamp",
            "file_hash",
            "content_type",
            "original_file_name",
            "resolved_file_name",
            "file_size",
            "mime_type",
            "namespace",
            "project",
        ]

        for field in preserve_fields:
            if field in additional_metadata:
                metadata[field] = additional_metadata[field]

        # Also preserve any custom metadata
        custom_metadata = additional_metadata.get("custom_metadata", {})
        if custom_metadata:
            metadata["custom_metadata"] = custom_metadata

        return metadata

    def _detect_heading_level(self, item: Any) -> int:
        """Detect heading level from a document item."""
        if hasattr(item, "label"):
            label = str(item.label).lower()
            for i in range(1, 7):
                if f"heading_{i}" in label or f"h{i}" in label:
                    return i
        return 1  # Default to level 1

    def _generate_doc_id(
        self,
        content: str,
        metadata: dict[str, Any],
        chunk_index: int | None = None,
    ) -> str:
        """Generate a unique document ID based on content and metadata."""
        # Create hash from content + key metadata
        hash_input = content[:1000]  # Use first 1000 chars
        hash_input += metadata.get("source_filename", "")
        if chunk_index is not None:
            hash_input += f"_chunk_{chunk_index}"

        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        filename = metadata.get("source_filename", "doc")
        base_name = Path(filename).stem[:20]  # Truncate long names

        if chunk_index is not None:
            return f"{base_name}_{content_hash}_c{chunk_index}"
        return f"{base_name}_{content_hash}"

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        path = Path(file_path)
        supported = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm",
                    ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
        return path.suffix.lower() in supported
