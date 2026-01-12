"""
MarkItDown Parser - Fast document-to-markdown conversion using Microsoft's MarkItDown.

Features:
- Fast, lightweight conversion
- Multi-format support (PDF, DOCX, XLSX, HTML, images, audio, etc.)
- Clean markdown output
- Structure preservation (headings, tables, lists)
- Optional LLM integration for image descriptions
"""

import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from core.base import Document, ProcessingResult
from core.logging import RAGStructLogger

from ..base.base_parser import BaseParser, ParserConfig

logger = RAGStructLogger("rag.components.parsers.markitdown.markitdown_parser")

# Lazy imports to handle missing dependencies gracefully
MARKITDOWN_AVAILABLE = False
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    logger.warning("MarkItDown not available. Install with: pip install 'markitdown[all]'")


class MarkItDownParser(BaseParser):
    """
    Document parser using Microsoft's MarkItDown library.

    Provides fast, lightweight document-to-markdown conversion with:
    - Multi-format support
    - Structure preservation
    - Clean markdown output
    - Optional LLM integration for images
    """

    def __init__(self, name: str = "MarkItDownParser", config: dict[str, Any] | None = None):
        """
        Initialize the MarkItDown parser.

        Args:
            name: Parser name identifier
            config: Parser configuration options:
                - output_format: 'markdown' or 'text' (default: 'markdown')
                - extract_metadata: Extract file metadata (default: True)
                - use_llm_for_images: Use LLM for image descriptions (default: False)
                - llm_client: OpenAI client for image descriptions (optional)
                - llm_model: Model to use for images (default: 'gpt-4o')
                - docintel_endpoint: Azure Document Intelligence endpoint (optional)
        """
        if not MARKITDOWN_AVAILABLE:
            raise ImportError(
                "MarkItDown is required. Install with: pip install 'markitdown[all]'"
            )

        self.name = name
        self.config = config or {}

        # Configuration with defaults
        self.output_format = self.config.get("output_format", "markdown")
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.use_llm_for_images = self.config.get("use_llm_for_images", False)

        # Initialize MarkItDown
        md_kwargs = {}

        # Add LLM client if configured for image descriptions
        if self.use_llm_for_images:
            llm_client = self.config.get("llm_client")
            llm_model = self.config.get("llm_model", "gpt-4o")
            if llm_client:
                md_kwargs["llm_client"] = llm_client
                md_kwargs["llm_model"] = llm_model

        # Add Azure Document Intelligence endpoint if configured
        docintel_endpoint = self.config.get("docintel_endpoint")
        if docintel_endpoint:
            md_kwargs["docintel_endpoint"] = docintel_endpoint

        self.converter = MarkItDown(**md_kwargs) if md_kwargs else MarkItDown()

    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata from config."""
        return ParserConfig(
            name="MarkItDownParser",
            display_name="MarkItDown Parser (Microsoft)",
            version="1.0.0",
            supported_extensions=[".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm",
                                   ".csv", ".json", ".xml", ".epub", ".png", ".jpg", ".jpeg",
                                   ".wav", ".mp3"],
            mime_types=["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       "application/vnd.ms-excel", "text/html", "text/csv", "application/json",
                       "application/xml", "application/epub+zip", "image/png", "image/jpeg",
                       "audio/wav", "audio/mpeg"],
            capabilities=["text_extraction", "markdown_conversion", "table_extraction",
                         "multi_format_support", "fast_processing", "structure_preservation"],
            dependencies={"required": ["markitdown"], "optional": ["openai", "azure-ai-documentintelligence"]},
            default_config={
                "output_format": "markdown",
                "extract_metadata": True,
                "use_llm_for_images": False,
            },
        )

    def parse(self, source: str) -> ProcessingResult:
        """
        Parse a document file using MarkItDown.

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
            result = self.converter.convert(str(source_path))

            # Build metadata
            metadata = self._build_file_metadata(source_path, {})

            # Create document
            content = result.text_content if hasattr(result, "text_content") else str(result)

            documents = [
                Document(
                    content=content,
                    metadata={
                        **metadata,
                        "chunk_index": 0,
                        "chunk_num": 1,
                        "total_chunks": 1,
                        "is_chunked": False,
                    },
                    id=self._generate_doc_id(content, metadata),
                    source=str(source_path),
                )
            ]

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
                "chunking_applied": False,
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

        # Write blob to temporary file for MarkItDown processing
        suffix = Path(filename).suffix or ".bin"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(blob_data)
            tmp_path = Path(tmp_file.name)

        try:
            # Convert document
            result = self.converter.convert(str(tmp_path))

            # Get content
            content = result.text_content if hasattr(result, "text_content") else str(result)

            # Build comprehensive metadata
            doc_metadata = self._build_file_metadata(tmp_path, metadata, original_filename=filename)

            # Add content statistics
            doc_metadata.update(self._extract_content_stats(content))

            # Create document
            doc = Document(
                content=content,
                metadata={
                    **doc_metadata,
                    "chunk_index": 0,
                    "chunk_num": 1,
                    "total_chunks": 1,
                    "is_chunked": False,
                },
                id=self._generate_doc_id(content, doc_metadata),
                source=filename,
            )

            return [doc]

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
                        "source_filename": filename,
                    },
                )
            ]
        finally:
            # Clean up temp file
            try:
                tmp_path.unlink()
            except Exception:
                pass

    def _build_file_metadata(
        self,
        source_path: Path,
        additional_metadata: dict[str, Any],
        original_filename: str | None = None,
    ) -> dict[str, Any]:
        """
        Build comprehensive file metadata.

        Args:
            source_path: Path to source file
            additional_metadata: Additional metadata from caller
            original_filename: Original filename if different from source

        Returns:
            Comprehensive metadata dictionary
        """
        # Start with base metadata
        metadata = self._build_base_metadata(additional_metadata)

        filename = original_filename or source_path.name

        # File-level metadata
        metadata.update({
            "source_filename": filename,
            "file_extension": source_path.suffix.lower(),
            "parser_type": self.name,
            "parser_version": "1.0.0",
            "output_format": self.output_format,
            "parsing_timestamp": datetime.now().isoformat(),
        })

        # File size if available
        try:
            if source_path.exists():
                metadata["file_size_bytes"] = source_path.stat().st_size
        except Exception:
            pass

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

    def _extract_content_stats(self, content: str) -> dict[str, Any]:
        """
        Extract basic statistics from the parsed content.

        Args:
            content: Parsed markdown content

        Returns:
            Dictionary with content statistics
        """
        stats = {
            "char_count": len(content),
            "word_count": len(content.split()),
            "line_count": content.count("\n") + 1,
        }

        # Count markdown elements
        lines = content.split("\n")

        # Count headings
        heading_count = sum(1 for line in lines if line.strip().startswith("#"))
        if heading_count > 0:
            stats["heading_count"] = heading_count

        # Count list items
        list_count = sum(1 for line in lines if line.strip().startswith(("-", "*", "1.", "2.", "3.")))
        if list_count > 0:
            stats["list_item_count"] = list_count

        # Check for tables (pipe characters)
        table_lines = sum(1 for line in lines if "|" in line)
        if table_lines >= 2:  # At least header + separator
            stats["has_tables"] = True
            stats["table_line_count"] = table_lines

        # Check for code blocks
        code_block_count = content.count("```")
        if code_block_count >= 2:
            stats["code_block_count"] = code_block_count // 2

        return stats

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
        supported = [".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm",
                    ".csv", ".json", ".xml", ".epub", ".png", ".jpg", ".jpeg",
                    ".wav", ".mp3"]
        return path.suffix.lower() in supported
