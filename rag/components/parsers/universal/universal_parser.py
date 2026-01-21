"""Universal Parser using Microsoft MarkItDown for document extraction.

This parser provides a unified interface for parsing many document types with
configurable chunking strategies:
- semantic: AI-based semantic boundary detection (SemChunk)
- sections: Split on markdown headers (h1-h6)
- paragraphs: Split on double newlines
- sentences: Split on sentence boundaries
- characters: Fixed character count (fallback)
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from components.parsers.base.base_parser import BaseParser, ParserConfig
from core.base import Document, ProcessingResult
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.universal")

# Chunking strategy type
ChunkStrategy = Literal["semantic", "sections", "paragraphs", "sentences", "characters"]

# Try to import optional dependencies
try:
    from markitdown import MarkItDown

    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    MarkItDown = None  # type: ignore

try:
    import semchunk

    SEMCHUNK_AVAILABLE = True
except ImportError:
    SEMCHUNK_AVAILABLE = False
    semchunk = None  # type: ignore

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None  # type: ignore


class UniversalParser(BaseParser):
    """Universal document parser using MarkItDown with configurable chunking.

    Features:
    - Parses 30+ file formats via MarkItDown
    - Multiple chunking strategies (semantic, sections, paragraphs, sentences, characters)
    - Rich chunk metadata (index, label, position, counts)
    - OCR fallback for scanned documents
    - Graceful degradation when dependencies unavailable
    """

    def __init__(
        self, name: str | None = None, config: dict[str, Any] | None = None
    ):
        """Initialize the UniversalParser.

        Args:
            name: Optional parser name (for BlobProcessor compatibility, ignored)
            config: Parser configuration with optional keys:
                - chunk_size: Target chunk size in characters (default: 1024)
                - chunk_overlap: Overlap between chunks (default: 100)
                - chunk_strategy: Chunking strategy (default: semantic)
                - use_ocr: Enable OCR fallback (default: True)
                - ocr_endpoint: OCR service URL
                - extract_metadata: Extract document metadata (default: True)
                - min_chunk_size: Minimum chunk size (default: 50)
                - max_chunk_size: Maximum chunk size (default: 8000)
        """
        # name parameter is for BlobProcessor compatibility - ignored
        super().__init__(config)

        # Extract config with defaults
        self.chunk_size = self.config.get("chunk_size", 1024)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.chunk_strategy: ChunkStrategy = self.config.get(
            "chunk_strategy", "semantic"
        )
        self.use_ocr = self.config.get("use_ocr", True)
        self.ocr_endpoint = self.config.get(
            "ocr_endpoint", "http://127.0.0.1:8000/v1/vision/ocr"
        )
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.min_chunk_size = self.config.get("min_chunk_size", 50)
        self.max_chunk_size = self.config.get("max_chunk_size", 8000)

        # Initialize MarkItDown if available
        self._markitdown = MarkItDown() if MARKITDOWN_AVAILABLE else None

        # Initialize tiktoken encoder for semantic chunking
        self._encoder = None
        if TIKTOKEN_AVAILABLE and SEMCHUNK_AVAILABLE:
            try:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                self.logger.warning(f"Failed to load tiktoken encoder: {e}")

        self.logger.info(
            "UniversalParser initialized",
            chunk_strategy=self.chunk_strategy,
            chunk_size=self.chunk_size,
            markitdown_available=MARKITDOWN_AVAILABLE,
            semchunk_available=SEMCHUNK_AVAILABLE,
        )

    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata from config.yaml."""
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                raw_data = yaml.safe_load(f)

                # Handle new parsers array structure
                if "parsers" in raw_data and raw_data["parsers"]:
                    data = raw_data["parsers"][0]  # Get first parser config
                else:
                    data = raw_data  # Legacy format

                return ParserConfig(
                    name=data.get("name", "UniversalParser"),
                    display_name=data.get("display_name", "Universal Parser"),
                    version=data.get("version", "1.0.0"),
                    supported_extensions=data.get("supported_extensions", []),
                    mime_types=data.get("mime_types", []),
                    capabilities=data.get("capabilities", []),
                    dependencies=data.get("dependencies", {}),
                    default_config=data.get("default_config", {}),
                )
        # Fallback metadata
        return ParserConfig(
            name="UniversalParser",
            display_name="Universal Parser",
            version="1.0.0",
            supported_extensions=[
                ".pdf",
                ".docx",
                ".txt",
                ".md",
                ".html",
                ".csv",
                ".json",
            ],
            mime_types=[
                "application/pdf",
                "text/plain",
                "text/html",
                "text/markdown",
            ],
            capabilities=["text_extraction", "chunking"],
            dependencies={"required": ["markitdown"], "optional": ["semchunk"]},
            default_config={"chunk_size": 1024, "chunk_strategy": "semantic"},
        )

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file.

        Args:
            file_path: Path to the file

        Returns:
            True if the file extension is supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in self.metadata.supported_extensions

    def parse(self, source: str) -> ProcessingResult:
        """Parse a file and return chunked documents.

        Args:
            source: Path to file

        Returns:
            ProcessingResult with documents and any errors
        """
        file_path = Path(source)
        errors: list[dict[str, Any]] = []
        documents: list[Document] = []

        if not file_path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "file": source}],
            )

        self.logger.info("Parsing file", file_path=str(file_path))

        # Extract text using MarkItDown
        text, doc_metadata = self._extract_text(file_path)

        # Try OCR if text is too short and OCR is enabled
        if (
            (not text or len(text.strip()) < self.min_chunk_size)
            and self.use_ocr
            and self._needs_ocr(file_path, text)
        ):
            ocr_text = self._run_remote_ocr(str(file_path))
            if ocr_text and len(ocr_text) > len(text or ""):
                text = ocr_text
                doc_metadata["ocr_applied"] = True
                self.logger.info("OCR applied", file_path=str(file_path))

        if not text or len(text.strip()) < self.min_chunk_size:
            return ProcessingResult(
                documents=[],
                errors=[
                    {"error": f"No extractable text found: {source}", "file": source}
                ],
            )

        # Chunk the text
        chunks = self._chunk_text(text)

        # Build document metadata
        file_stat = file_path.stat()
        total_chunks = len(chunks)

        for i, chunk_text in enumerate(chunks):
            # Build chunk-level metadata
            chunk_metadata = {
                # Document-level
                "document_name": file_path.name,
                "document_path": str(file_path.absolute()),
                "document_type": file_path.suffix.lower(),
                "document_size": file_stat.st_size,
                "processed_at": datetime.now(UTC).isoformat(),
                # Chunk-level
                "chunk_index": i,
                "chunk_label": f"{i + 1}/{total_chunks}",
                "total_chunks": total_chunks,
                "chunk_position": self._get_chunk_position(i, total_chunks),
                "character_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "sentence_count": self._count_sentences(chunk_text),
                # Parser info
                "parser": "UniversalParser",
                "chunk_strategy": self.chunk_strategy,
            }

            # Add document metadata from extraction
            chunk_metadata.update(doc_metadata)

            doc = self.create_document(
                content=chunk_text,
                metadata=chunk_metadata,
                source=str(file_path),
            )
            documents.append(doc)

        self.logger.info(
            "Parsing complete",
            file_path=str(file_path),
            chunks_created=len(documents),
            chunk_strategy=self.chunk_strategy,
        )

        return ProcessingResult(
            documents=documents,
            errors=errors,
            metrics={
                "total_chunks": total_chunks,
                "total_characters": sum(len(d.content) for d in documents),
                "chunk_strategy": self.chunk_strategy,
            },
        )

    def _extract_text(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Extract text and metadata from file using MarkItDown.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        metadata: dict[str, Any] = {}

        if not MARKITDOWN_AVAILABLE or self._markitdown is None:
            # Fallback: read as plain text
            self.logger.warning(
                "MarkItDown not available, falling back to plain text",
                file_path=str(file_path),
            )
            try:
                encoding = self.detect_encoding(str(file_path))
                with open(file_path, encoding=encoding, errors="ignore") as f:
                    return f.read(), metadata
            except Exception as e:
                self.logger.error(
                    f"Failed to read file: {e}", file_path=str(file_path)
                )
                return "", metadata

        try:
            result = self._markitdown.convert(str(file_path))
            text = result.text_content if hasattr(result, "text_content") else ""

            # Extract any metadata from MarkItDown result
            if (
                hasattr(result, "metadata")
                and result.metadata
                and isinstance(result.metadata, dict)
            ):
                metadata.update(result.metadata)

            return text, metadata

        except Exception as e:
            self.logger.error(
                f"MarkItDown extraction failed: {e}", file_path=str(file_path)
            )
            # Fallback to plain text
            try:
                encoding = self.detect_encoding(str(file_path))
                with open(file_path, encoding=encoding, errors="ignore") as f:
                    return f.read(), metadata
            except Exception:
                return "", metadata

    def _chunk_text(self, text: str) -> list[str]:
        """Chunk text using the configured strategy.

        Args:
            text: Full text to chunk

        Returns:
            List of text chunks
        """
        strategy = self.chunk_strategy

        # Try semantic chunking first if available
        if strategy == "semantic":
            chunks = self._chunk_semantic(text)
            if chunks:
                return chunks
            # Fall back to paragraphs if semantic fails
            self.logger.warning(
                "Semantic chunking unavailable, falling back to paragraphs"
            )
            strategy = "paragraphs"

        if strategy == "sections":
            return self._chunk_sections(text)
        elif strategy == "paragraphs":
            return self._chunk_paragraphs(text)
        elif strategy == "sentences":
            return self._chunk_sentences(text)
        else:  # characters (fallback)
            return self._chunk_characters(text)

    def _chunk_semantic(self, text: str) -> list[str]:
        """Chunk text using SemChunk's semantic boundary detection.

        Args:
            text: Text to chunk

        Returns:
            List of semantically-bounded chunks, or empty list if unavailable
        """
        if not SEMCHUNK_AVAILABLE or not self._encoder:
            return []

        try:
            # SemChunk uses token count, convert character size to approx tokens
            # Average ~4 chars per token for English
            token_chunk_size = max(self.chunk_size // 4, 100)

            chunker = semchunk.chunkerify(self._encoder, chunk_size=token_chunk_size)
            chunks = chunker(text)

            # Filter out empty chunks and ensure minimum size
            return [c for c in chunks if c and len(c.strip()) >= self.min_chunk_size]

        except Exception as e:
            self.logger.warning(f"Semantic chunking failed: {e}")
            return []

    def _chunk_sections(self, text: str) -> list[str]:
        """Chunk text by markdown headers (h1-h6).

        Each header is grouped with its following content until the next header.

        Args:
            text: Text to chunk

        Returns:
            List of section-based chunks
        """
        # Pattern for markdown headers - captures the full header line
        header_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)

        # Split text by headers, keeping the headers
        parts = header_pattern.split(text)
        chunks: list[str] = []

        # First element is text before any headers
        if parts[0].strip():
            preamble = parts[0].strip()
            if len(preamble) >= self.min_chunk_size:
                chunks.extend(self._split_if_too_large(preamble))

        # Group headers with their content (header at odd index, content at even)
        for i in range(1, len(parts), 2):
            header = parts[i]
            content = parts[i + 1] if (i + 1) < len(parts) else ""
            section = (header + content).strip()
            if section and len(section) >= self.min_chunk_size:
                chunks.extend(self._split_if_too_large(section))

        # If no sections found or only preamble, fall back to paragraphs
        if not chunks:
            return self._chunk_paragraphs(text)

        return chunks

    def _chunk_paragraphs(self, text: str) -> list[str]:
        """Chunk text by paragraphs (double newlines).

        Args:
            text: Text to chunk

        Returns:
            List of paragraph-based chunks
        """
        # Split on double newlines
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if adding this paragraph exceeds chunk size
            if (
                len(current_chunk) + len(para) + 2 > self.chunk_size
                and current_chunk
            ):
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)

        # Handle case where paragraphs are too large
        result = []
        for chunk in chunks:
            result.extend(self._split_if_too_large(chunk))

        return result if result else [text]

    def _chunk_sentences(self, text: str) -> list[str]:
        """Chunk text by sentence boundaries.

        Args:
            text: Text to chunk

        Returns:
            List of sentence-based chunks
        """
        # Simple sentence boundary detection
        sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        sentences = sentence_endings.split(text)

        chunks: list[str] = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if adding this sentence exceeds chunk size
            if (
                len(current_chunk) + len(sentence) + 1 > self.chunk_size
                and current_chunk
            ):
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def _chunk_characters(self, text: str) -> list[str]:
        """Chunk text by fixed character count with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of character-based chunks
        """
        chunks: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Try to break at word boundary
            if end < text_len:
                space_idx = text.rfind(" ", start, end)
                if space_idx > start + self.min_chunk_size:
                    end = space_idx

            chunk = text[start:end].strip()
            if chunk and len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)

            # Move start forward, accounting for overlap
            next_start = end - self.chunk_overlap

            # Ensure we always make progress
            if next_start <= start:
                next_start = end

            # Stop if we've reached the end
            if end >= text_len:
                break

            start = next_start

        return chunks if chunks else [text]

    def _split_if_too_large(self, text: str) -> list[str]:
        """Split text if it exceeds max chunk size.

        Args:
            text: Text to potentially split

        Returns:
            List of chunks within size limits
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        # Use character chunking to split large text
        return self._chunk_characters(text)

    def _get_chunk_position(self, index: int, total: int) -> str:
        """Get human-readable chunk position.

        Args:
            index: Zero-based chunk index
            total: Total number of chunks

        Returns:
            Position label (start, middle, end, only)
        """
        if total == 1:
            return "only"
        if index == 0:
            return "start"
        if index == total - 1:
            return "end"
        return "middle"

    def _count_sentences(self, text: str) -> int:
        """Count approximate number of sentences in text.

        Args:
            text: Text to analyze

        Returns:
            Sentence count
        """
        # Simple heuristic: count sentence-ending punctuation
        endings = len(re.findall(r"[.!?]+", text))
        return max(endings, 1)

    def _needs_ocr(self, file_path: Path, text: str | None) -> bool:
        """Check if OCR should be attempted.

        Args:
            file_path: Path to the file
            text: Extracted text (if any)

        Returns:
            True if OCR should be attempted
        """
        # Always try OCR for image files
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
        if file_path.suffix.lower() in image_extensions:
            return True

        # Try OCR if extracted text is very short or empty (likely scanned PDF)
        text_len = len((text or "").strip())
        return text_len < 50

    def _run_remote_ocr(self, file_path: str) -> str | None:
        """Run OCR via Universal Runtime endpoint.

        Args:
            file_path: Path to the file

        Returns:
            OCR text or None if failed
        """
        try:
            import base64

            import requests

            # Read file and encode as base64
            with open(file_path, "rb") as f:
                file_data = base64.b64encode(f.read()).decode("utf-8")

            response = requests.post(
                self.ocr_endpoint,
                json={
                    "image": file_data,
                    "filename": Path(file_path).name,
                },
                timeout=60,
            )

            if response.ok:
                result = response.json()
                return result.get("text", "")

        except Exception as e:
            self.logger.warning(f"OCR request failed: {e}")

        return None

    def parse_blob(self, blob_data: bytes, metadata: dict[str, Any]) -> list[Document]:
        """Parse raw blob data.

        Args:
            blob_data: Raw bytes of the document
            metadata: Metadata about the blob

        Returns:
            List of Document objects
        """
        import tempfile

        filename = metadata.get("filename", "temp.txt")
        suffix = Path(filename).suffix or ".txt"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(blob_data)
            tmp_path = tmp.name

        try:
            result = self.parse(tmp_path)
            # Add provided metadata to each document
            for doc in result.documents:
                doc.metadata.update(metadata)
            return result.documents
        finally:
            Path(tmp_path).unlink(missing_ok=True)
