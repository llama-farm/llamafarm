"""Abstract base parser class for all RAG parsers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Import from rag module
from core.base import Document, ProcessingResult
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.base.base_parser")


@dataclass
class ParserConfig:
    """Configuration for a parser."""

    name: str
    display_name: str
    version: str
    supported_extensions: list[str]
    mime_types: list[str]
    capabilities: list[str]
    dependencies: dict[str, list[str]]
    default_config: dict[str, Any]


class BaseParser(ABC):
    """Abstract base class for all parsers."""

    def __init__(self, config: dict[str, Any] = None):
        """Initialize parser with configuration.

        Args:
            config: Parser configuration dictionary
        """
        self.config = config or {}
        self.logger = logger.bind(name=self.__class__.__name__)

        # Load parser metadata
        self.metadata = self._load_metadata()

        # Validate configuration against schema
        self._validate_config()

    @abstractmethod
    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata from config.yaml.

        Returns:
            ParserConfig object with metadata
        """
        pass

    @abstractmethod
    def parse(self, source: str) -> ProcessingResult:
        """Parse a file or directory.

        Args:
            source: Path to file or directory

        Returns:
            ProcessingResult with documents and any errors
        """
        pass

    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file.

        Args:
            file_path: Path to file

        Returns:
            True if parser can handle the file
        """
        pass

    def parse_blob(self, blob_data: bytes, metadata: dict[str, Any]) -> list[Document]:
        """Parse raw blob data.

        Args:
            blob_data: Raw bytes of the document
            metadata: Metadata about the blob (filename, content_type, etc.)

        Returns:
            List of Document objects
        """
        # Default implementation writes to temp file and calls parse
        import os
        import tempfile

        filename = metadata.get("filename", "temp")
        suffix = Path(filename).suffix if filename else ".txt"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(blob_data)
            tmp_path = tmp.name

        try:
            result = self.parse(tmp_path)
            # Convert ProcessingResult to list of Documents
            documents = result.documents if hasattr(result, "documents") else []
            # Add metadata to each document
            for doc in documents:
                doc.metadata.update(metadata)
                doc.metadata["parser"] = self.__class__.__name__
            return documents
        finally:
            os.unlink(tmp_path)

    def _validate_config(self) -> None:
        """Validate configuration against parser schema.

        Override in subclasses to implement custom validation.
        Default implementation does nothing.
        """
        # Subclasses can override to validate against their schema.json
        return None

    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding.

        Args:
            file_path: Path to file

        Returns:
            Detected encoding
        """
        try:
            import chardet

            with open(file_path, "rb") as f:
                result = chardet.detect(f.read(10000))
                return result["encoding"] or "utf-8"
        except Exception:
            return "utf-8"

    def create_document(
        self,
        content: str,
        metadata: dict[str, Any] = None,
        doc_id: str = None,
        source: str = None,
    ) -> Document:
        """Create a Document object.

        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Document ID
            source: Source file path

        Returns:
            Document object
        """
        return Document(
            content=content, metadata=metadata or {}, id=doc_id, source=source
        )
