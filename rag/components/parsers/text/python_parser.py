"""Text parser using native Python (no external dependencies)."""

from pathlib import Path
from typing import Dict, Any, List, Optional

from core.logging import RAGStructLogger
logger = RAGStructLogger("rag.components.parsers.textthon_parser")


class TextParser_Python:
    """Text parser using pure Python for maximum compatibility."""

    def __init__(
        self, name: str = "TextParser_Python", config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.chunk_strategy = self.config.get("chunk_strategy", "sentences")
        self.encoding = self.config.get("encoding", "utf-8")
        self.clean_text = self.config.get("clean_text", True)

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def parse_blob(
        self, blob_data: bytes, metadata: Dict[str, Any]
    ) -> List["Document"]:
        """Parse raw blob data by writing to temp file and calling parse."""
        import tempfile
        from core.base import Document

        # Write blob to temp file
        suffix = f".{metadata.get('filename', 'temp.txt').split('.')[-1]}"
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as tmp:
            tmp.write(blob_data)
            tmp_path = tmp.name

        try:
            # Parse the temp file
            result = self.parse(tmp_path)
            if result and result.documents:
                # Update metadata for all chunks
                for doc in result.documents:
                    doc.metadata.update(metadata)
                    doc.metadata["parser"] = self.name
                return result.documents
            return []
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    def parse(self, source: str, **kwargs):
        """Parse text file using native Python."""
        from core.base import Document, ProcessingResult

        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}],
            )

        try:
            # Try different encodings if needed
            encodings = [self.encoding, "utf-8", "latin-1", "cp1252", "iso-8859-1"]
            text = None
            used_encoding = None

            for encoding in encodings:
                try:
                    with open(path, "r", encoding=encoding) as f:
                        text = f.read()
                        used_encoding = encoding
                        break
                except UnicodeDecodeError:
                    continue

            if text is None:
                # Last resort: read with error replacement
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                    used_encoding = "utf-8 (with replacements)"

            # Clean text if configured
            if self.clean_text:
                # Remove excessive whitespace
                lines = text.split("\n")
                lines = [line.strip() for line in lines]
                text = "\n".join(line for line in lines if line)

            metadata = {
                "source": str(path),
                "file_name": path.name,
                "parser": self.name,
                "tool": "Python",
                "encoding": used_encoding,
                "file_size": path.stat().st_size,
                "line_count": text.count("\n") + 1,
                "word_count": len(text.split()),
                "char_count": len(text),
            }

            documents = []

            # Apply chunking if configured
            if self.chunk_size and self.chunk_size > 0:
                chunks = self._chunk_text(text)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "chunk_strategy": self.chunk_strategy,
                        }
                    )

                    doc = Document(
                        content=chunk,
                        metadata=chunk_metadata,
                        id=f"{path.stem}_chunk_{i + 1}",
                        source=str(path),
                    )
                    documents.append(doc)
            else:
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
                    "tool": "Python",
                },
            )

        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            return ProcessingResult(
                documents=[], errors=[{"error": str(e), "source": source}]
            )

    def _chunk_text(self, text: str) -> List[str]:
        """Text chunking implementation."""
        if self.chunk_strategy == "sentences":
            return self._chunk_by_sentences(text)
        elif self.chunk_strategy == "paragraphs":
            return self._chunk_by_paragraphs(text)
        else:  # characters
            return self._chunk_by_characters(text)

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk by sentences."""
        # Simple sentence detection
        sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        sentences = []
        current = ""

        for char in text:
            current += char
            if any(current.endswith(end) for end in sentence_endings):
                sentences.append(current.strip())
                current = ""

        if current.strip():
            sentences.append(current.strip())

        # Combine sentences into chunks
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """Chunk by paragraphs."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _chunk_by_characters(self, text: str) -> List[str]:
        """Simple character-based chunking with overlap."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
