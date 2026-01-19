"""Universal Parser using MarkItDown + multiple chunking strategies for lightweight RAG.

This parser combines:
- MarkItDown (Microsoft): Structure-preserving document to markdown conversion
- Multiple chunking strategies: semantic, section, paragraph, sentence, page, fixed
- OCR fallback: Optional OCR via Universal Runtime for scanned documents

Chunking Strategies:
- semantic: Uses SemChunk for intelligent token-aware chunking (default)
- section: Splits on markdown headers (# ## ###)
- paragraph: Splits on double newlines
- sentence: Splits on sentence boundaries (. ! ?)
- page: Splits on page breaks (\\f or --- Page markers)
- fixed: Fixed-size character chunks with overlap
"""

import re
from pathlib import Path
from typing import Any

from components.parsers.base.base_parser import BaseParser, ParserConfig
from core.base import Document, ProcessingResult
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.universal.parser")

# Valid chunking strategies
CHUNK_STRATEGIES = ["semantic", "section", "paragraph", "sentence", "page", "fixed"]


class UniversalParser(BaseParser):
    """Universal parser using MarkItDown + multiple chunking strategies.

    Features:
    - Parses PDF, DOCX, XLSX, HTML, JSON, XML, images, and more
    - Converts to clean markdown preserving structure
    - Multiple chunking strategies: semantic, section, paragraph, sentence, page, fixed
    - Optional OCR fallback for scanned documents
    - Lightweight dependencies (no llama-index, no docling)
    """

    def __init__(
        self, name: str = "UniversalParser", config: dict[str, Any] | None = None
    ):
        """Initialize UniversalParser.

        Args:
            name: Parser name
            config: Configuration dictionary with options:
                - chunk_size: Target chunk size in tokens (default: 1024)
                - chunk_strategy: Chunking strategy (default: 'semantic')
                    - 'semantic': SemChunk token-aware chunking
                    - 'section': Split on markdown headers (# ## ###)
                    - 'paragraph': Split on double newlines
                    - 'sentence': Split on sentence boundaries
                    - 'page': Split on page breaks
                    - 'fixed': Fixed-size character chunks with overlap
                - chunk_overlap: Overlap for fixed chunking (default: 0.1 = 10%)
                - min_chunk_size: Minimum chunk size in chars (default: 100)
                - use_ocr: Enable OCR fallback (default: True)
                - ocr_min_text_threshold: Min chars before OCR triggers (default: 50)
                - model_for_tokens: Tokenizer model name (default: 'gpt-4')
                - universal_runtime_url: URL for OCR service (default: http://127.0.0.1:11540)
        """
        super().__init__(config)
        self.name = name

        # Configuration with defaults
        self.chunk_size = self.config.get("chunk_size", 1024)
        self.chunk_strategy = self.config.get("chunk_strategy", "semantic")
        self.chunk_overlap = self.config.get("chunk_overlap", 0.1)
        self.min_chunk_size = self.config.get("min_chunk_size", 100)
        self.use_ocr = self.config.get("use_ocr", True)
        self.ocr_min_text_threshold = self.config.get("ocr_min_text_threshold", 50)
        self.model_for_tokens = self.config.get("model_for_tokens", "gpt-4")
        self.universal_runtime_url = self.config.get(
            "universal_runtime_url", "http://127.0.0.1:11540"
        )

        # Validate chunk strategy
        if self.chunk_strategy not in CHUNK_STRATEGIES:
            logger.warning(
                f"Unknown chunk_strategy '{self.chunk_strategy}', defaulting to 'semantic'. "
                f"Valid options: {CHUNK_STRATEGIES}"
            )
            self.chunk_strategy = "semantic"

        # Initialize MarkItDown
        try:
            from markitdown import MarkItDown

            self._markitdown = MarkItDown()
            self._markitdown_available = True
        except ImportError:
            logger.warning(
                "MarkItDown not available. Install with: pip install markitdown[all]"
            )
            self._markitdown = None
            self._markitdown_available = False

        # Initialize SemChunk
        try:
            import semchunk
            import tiktoken

            self._semchunk = semchunk
            self._tiktoken = tiktoken
            self._chunker = semchunk.chunkerify(
                tiktoken.encoding_for_model(self.model_for_tokens), self.chunk_size
            )
            self._semchunk_available = True
        except ImportError:
            logger.warning(
                "SemChunk not available. Install with: pip install semchunk tiktoken"
            )
            self._semchunk = None
            self._chunker = None
            self._semchunk_available = False

    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata."""
        return ParserConfig(
            name="UniversalParser",
            display_name="Universal Parser (MarkItDown + SemChunk)",
            version="1.0.0",
            supported_extensions=[
                ".pdf",
                ".docx",
                ".xlsx",
                ".pptx",
                ".csv",
                ".json",
                ".xml",
                ".html",
                ".htm",
                ".png",
                ".jpg",
                ".jpeg",
                ".txt",
                ".md",
            ],
            mime_types=[
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "text/csv",
                "application/json",
                "application/xml",
                "text/xml",
                "text/html",
                "image/png",
                "image/jpeg",
                "text/plain",
                "text/markdown",
            ],
            capabilities=[
                "text_extraction",
                "markdown_conversion",
                "semantic_chunking",
                "section_chunking",
                "paragraph_chunking",
                "sentence_chunking",
                "page_chunking",
                "fixed_chunking",
                "ocr_fallback",
            ],
            dependencies={
                "required": ["markitdown>=0.1.4", "semchunk>=2.0.0", "tiktoken>=0.5.0"],
                "optional": ["surya-ocr>=0.4.0", "llamafarm-common"],
            },
            default_config={
                "chunk_size": 1024,
                "chunk_strategy": "semantic",
                "chunk_overlap": 0.1,
                "min_chunk_size": 100,
                "use_ocr": True,
                "ocr_min_text_threshold": 50,
            },
        )

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        ext = Path(file_path).suffix.lower()
        supported = [
            ".pdf",
            ".docx",
            ".xlsx",
            ".pptx",
            ".csv",
            ".json",
            ".xml",
            ".html",
            ".htm",
            ".png",
            ".jpg",
            ".jpeg",
            ".txt",
            ".md",
        ]
        return ext in supported

    def parse(self, source: str, **kwargs) -> ProcessingResult:
        """Parse document and return chunked documents.

        Args:
            source: Path to file
            **kwargs: Additional options

        Returns:
            ProcessingResult with chunked documents
        """
        path = Path(source)

        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}],
            )

        if not self._markitdown_available:
            return ProcessingResult(
                documents=[],
                errors=[
                    {
                        "error": "MarkItDown not installed. Install with: pip install markitdown[all]",
                        "source": source,
                    }
                ],
            )

        try:
            # Step 1: Convert to markdown using MarkItDown
            logger.info(f"Parsing {source} with MarkItDown")
            result = self._markitdown.convert(str(path))
            text = result.text_content

            # Step 2: Check if OCR fallback needed
            if self.use_ocr and self._needs_ocr(str(path), text):
                logger.info(
                    f"Low text detected ({len(text)} chars), triggering OCR fallback"
                )
                ocr_text = self._ocr_fallback(str(path))
                if ocr_text and len(ocr_text) > len(text):
                    text = ocr_text
                    logger.info(f"OCR returned {len(ocr_text)} chars")

            if not text.strip():
                return ProcessingResult(
                    documents=[],
                    errors=[
                        {"error": "No text extracted from document", "source": source}
                    ],
                )

            # Step 3: Chunk the text based on strategy
            chunks = self._chunk_text(text)

            # Step 4: Create documents
            documents = []
            base_metadata = {
                "source": str(path),
                "source_file": str(path),
                "file_name": path.name,
                "file_size_bytes": path.stat().st_size,
                "parser": self.name,
                "parser_type": "UniversalParser",
                "tool": "MarkItDown-Universal",
                "chunk_strategy": self.chunk_strategy,
                "total_chunks": len(chunks),
            }

            for i, chunk_text in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_index": i,
                        "chunk_label": f"chunk_{i + 1}_of_{len(chunks)}",
                        "length": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                    }
                )

                doc = Document(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    id=f"{path.stem}_chunk_{i + 1}",
                    source=str(path),
                )
                documents.append(doc)

            logger.info(
                f"Parsed {source}: {len(chunks)} chunks, {len(text)} chars total"
            )

            return ProcessingResult(
                documents=documents,
                errors=[],
                metrics={
                    "total_documents": len(documents),
                    "total_characters": len(text),
                    "parser_type": self.name,
                    "tool": "MarkItDown-Universal",
                    "chunk_strategy": self.chunk_strategy,
                },
            )

        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            return ProcessingResult(
                documents=[],
                errors=[{"error": str(e), "source": source}],
            )

    def _needs_ocr(self, file_path: str, text: str) -> bool:
        """Check if OCR fallback is needed.

        OCR is triggered if:
        1. File is an image (PNG, JPG, etc.)
        2. PDF with very little text extracted (< threshold chars)
        """
        ext = Path(file_path).suffix.lower()

        # Always trigger OCR for images
        if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            return True

        # Trigger OCR for PDFs with low text
        return ext == ".pdf" and len(text.strip()) < self.ocr_min_text_threshold

    def _ocr_fallback(self, file_path: str) -> str | None:
        """Call Universal Runtime OCR endpoint.

        Args:
            file_path: Path to file

        Returns:
            Extracted text or None if OCR fails
        """
        try:
            import requests

            ocr_endpoint = f"{self.universal_runtime_url}/v1/ocr"

            with open(file_path, "rb") as f:
                files = {"file": f}
                response = requests.post(ocr_endpoint, files=files, timeout=60)
                response.raise_for_status()
                return response.json().get("text", "")

        except Exception as e:
            logger.warning(f"OCR fallback failed: {e}")
            return None

    def _chunk_text(self, text: str) -> list[str]:
        """Chunk text based on configured strategy.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        strategy = self.chunk_strategy

        if strategy == "semantic":
            chunks = self._chunk_semantic(text)
        elif strategy == "section":
            chunks = self._chunk_by_section(text)
        elif strategy == "paragraph":
            chunks = self._chunk_by_paragraph(text)
        elif strategy == "sentence":
            chunks = self._chunk_by_sentence(text)
        elif strategy == "page":
            chunks = self._chunk_by_page(text)
        elif strategy == "fixed":
            chunks = self._chunk_fixed(text)
        else:
            # Default to semantic
            chunks = self._chunk_semantic(text)

        # Filter out chunks smaller than minimum size
        chunks = [c for c in chunks if len(c) >= self.min_chunk_size]

        # If filtering removed everything, return original text as single chunk
        if not chunks:
            return [text.strip()] if text.strip() else []

        return chunks

    def _chunk_semantic(self, text: str) -> list[str]:
        """Chunk text semantically using SemChunk.

        SemChunk respects sentence and paragraph boundaries,
        producing cleaner chunks than character-based splitting.
        This is the recommended strategy for most use cases.
        """
        if not self._semchunk_available or not self._chunker:
            logger.warning("SemChunk not available, falling back to paragraph chunking")
            return self._chunk_by_paragraph(text)

        try:
            chunks = self._chunker(text)
            # Filter empty chunks
            return [c.strip() for c in chunks if c.strip()]
        except Exception as e:
            logger.warning(f"SemChunk failed, falling back to paragraph chunking: {e}")
            return self._chunk_by_paragraph(text)

    def _chunk_by_section(self, text: str) -> list[str]:
        """Chunk text by markdown headers (sections).

        Splits on # ## ### etc. headers, keeping each section together.
        Good for structured documents with clear headings.
        """
        # Split on markdown headers (# ## ### etc.)
        # Pattern matches newline followed by one or more # and space
        sections = re.split(r"\n(?=#+\s)", text)

        # Filter empty sections and strip whitespace
        sections = [s.strip() for s in sections if s.strip()]

        # If no sections found, fall back to paragraph chunking
        if len(sections) <= 1:
            logger.debug("No section headers found, falling back to paragraph chunking")
            return self._chunk_by_paragraph(text)

        return sections

    def _chunk_by_paragraph(self, text: str) -> list[str]:
        """Chunk text by paragraphs (double newlines).

        Simple and effective for most text. Preserves natural
        paragraph boundaries in the source document.
        """
        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n", text)

        # Filter empty paragraphs and strip whitespace
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # If paragraphs are very long, sub-chunk them
        result = []
        max_chars = self.chunk_size * 4  # Approximate chars per token

        for para in paragraphs:
            if len(para) <= max_chars:
                result.append(para)
            else:
                # Sub-chunk long paragraphs by sentences
                sub_chunks = self._chunk_by_sentence(para)
                result.extend(sub_chunks)

        return result if result else [text.strip()]

    def _chunk_by_sentence(self, text: str) -> list[str]:
        """Chunk text by sentences.

        Splits on sentence boundaries (. ! ?) while respecting
        common abbreviations. Good for fine-grained chunking.
        """
        # Split on sentence endings, but not on common abbreviations
        # This is a simplified approach - more sophisticated NLP could be used
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)

        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        # Group sentences to meet minimum chunk size
        result = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            current_chunk.append(sentence)
            current_size += len(sentence)

            # If we've accumulated enough text, start a new chunk
            if current_size >= self.min_chunk_size:
                result.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        # Don't forget the last chunk
        if current_chunk:
            result.append(" ".join(current_chunk))

        return result if result else [text.strip()]

    def _chunk_by_page(self, text: str) -> list[str]:
        """Chunk text by page breaks.

        Splits on form feed characters (\\f) or page markers.
        Best for PDFs and documents with clear page boundaries.
        """
        # Split on form feed (page break)
        pages = text.split("\f")

        # Filter empty pages and strip whitespace
        pages = [p.strip() for p in pages if p.strip()]

        # If no page breaks found, try splitting on "--- Page" markers
        if len(pages) <= 1:
            pages = re.split(r"\n---\s*Page\s*\d+\s*---\n", text)
            pages = [p.strip() for p in pages if p.strip()]

        # If still single chunk, fall back to section chunking
        if len(pages) <= 1:
            logger.debug("No page breaks found, falling back to section chunking")
            return self._chunk_by_section(text)

        return pages

    def _chunk_fixed(self, text: str) -> list[str]:
        """Chunk text into fixed-size pieces with overlap.

        Simple character-based chunking. Use when other strategies
        don't work well for your content.
        """
        chunk_size = self.chunk_size * 4  # Approximate chars per token
        overlap = int(chunk_size * self.chunk_overlap)

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary for cleaner chunks
            if end < len(text):
                # Look for sentence ending near the boundary
                for char in ".!?\n":
                    pos = text.rfind(char, start + (chunk_size // 2), end)
                    if pos > start + (chunk_size // 2):
                        end = pos + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position, accounting for overlap
            start = end - overlap if end < len(text) else end

        return chunks if chunks else [text.strip()]

    def parse_blob(
        self, data: bytes, metadata: dict[str, Any] | None = None
    ) -> list[Document]:
        """Parse document from raw bytes.

        Args:
            data: Raw bytes of the document
            metadata: Document metadata (filename, content_type, etc.)

        Returns:
            List of Document objects
        """
        import os
        import tempfile

        metadata = metadata or {}
        filename = metadata.get("filename", "temp")
        suffix = Path(filename).suffix if filename else ".txt"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            result = self.parse(tmp_path)

            # Add metadata to each document
            for doc in result.documents:
                doc.metadata.update(metadata)

            return result.documents
        finally:
            os.unlink(tmp_path)
