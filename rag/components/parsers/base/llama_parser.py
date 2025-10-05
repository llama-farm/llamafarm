"""LlamaIndex-based parser wrapper with unified chunking."""

from pathlib import Path
from typing import Dict, Any, List, Optional

from .base_parser import BaseParser, ParserConfig
import sys

from core.logging import RAGStructLogger
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.base import Document, ProcessingResult

logger = RAGStructLogger("rag.components.parsers.base.llama_parser")

# Lazy imports to avoid missing dependencies
LLAMA_INDEX_AVAILABLE = False
try:
    from llama_index.core import SimpleDirectoryReader, Document as LlamaDocument
    from llama_index.core.node_parser import (
        SentenceSplitter,
        TokenTextSplitter,
        MarkdownNodeParser,
        SemanticSplitterNodeParser,
    )

    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    logger.warning("LlamaIndex not available. Install with: pip install llama-index")


class LlamaIndexParser(BaseParser):
    """Base parser using LlamaIndex for document loading and chunking."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize LlamaIndex parser.

        Args:
            config: Parser configuration
        """
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is required. Install with: pip install llama-index"
            )

        super().__init__(config)

        # Initialize reader (to be set by subclass)
        self.reader = None

        # Initialize text splitter based on config
        self.text_splitter = self._create_text_splitter()

    def _create_text_splitter(self):
        """Create appropriate text splitter based on configuration.

        Returns:
            LlamaIndex text splitter
        """
        chunk_size = self.config.get("chunk_size", None)

        # No chunking if chunk_size is None
        if chunk_size is None:
            return None

        chunk_overlap = self.config.get("chunk_overlap", 0)
        chunk_strategy = self.config.get("chunk_strategy", "characters")

        # Map schema.yaml strategies to LlamaIndex splitters
        if chunk_strategy == "sentences":
            return SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=" ",
                paragraph_separator="\n\n",
                secondary_chunking_regex="[^.!?]+[.!?]",
            )
        elif chunk_strategy == "paragraphs":
            # Use markdown parser for paragraph-aware splitting
            return MarkdownNodeParser()
        elif chunk_strategy == "semantic":
            # Advanced semantic chunking (requires embedding model)
            try:
                from llama_index.core.embeddings import OpenAIEmbedding

                return SemanticSplitterNodeParser(
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                    embed_model=OpenAIEmbedding(),  # Can be configured
                )
            except:
                logger.warning(
                    "Semantic chunking unavailable, falling back to sentence splitting"
                )
                return SentenceSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
        else:  # characters or default
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=" ",
                backup_separators=["\n", ".", "!", "?", ",", ";", ":", " ", ""],
            )

    def _llama_to_rag_documents(self, llama_docs: List) -> List[Document]:
        """Convert LlamaIndex documents to RAG documents.

        Args:
            llama_docs: List of LlamaIndex documents

        Returns:
            List of RAG Document objects
        """
        rag_docs = []

        for llama_doc in llama_docs:
            # Extract content
            content = llama_doc.text if hasattr(llama_doc, "text") else str(llama_doc)

            # Extract and enhance metadata
            metadata = llama_doc.metadata if hasattr(llama_doc, "metadata") else {}

            # Add parser-specific metadata
            metadata["parser_type"] = self.__class__.__name__
            metadata["parser_version"] = (
                self.metadata.version if hasattr(self, "metadata") else "1.0.0"
            )

            # Create RAG document
            doc = self.create_document(
                content=content,
                metadata=metadata,
                doc_id=llama_doc.id_ if hasattr(llama_doc, "id_") else None,
                source=metadata.get("file_path", metadata.get("source", None)),
            )
            rag_docs.append(doc)

        return rag_docs

    def _apply_chunking(self, documents: List[Document]) -> List[Document]:
        """Apply chunking to documents if configured.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        if not self.text_splitter:
            return documents

        chunked_docs = []

        for doc in documents:
            # Convert to LlamaIndex document for chunking
            llama_doc = LlamaDocument(
                text=doc.content, metadata=doc.metadata, id_=doc.id
            )

            # Apply chunking
            try:
                if hasattr(self.text_splitter, "split"):
                    chunks = self.text_splitter.split([llama_doc])
                else:
                    chunks = self.text_splitter.get_nodes_from_documents([llama_doc])
            except Exception as e:
                logger.warning(f"Chunking failed: {e}, returning original document")
                chunked_docs.append(doc)
                continue

            # Convert chunks back to documents
            for i, chunk in enumerate(chunks):
                chunk_content = chunk.text if hasattr(chunk, "text") else str(chunk)
                chunk_metadata = doc.metadata.copy()

                # Add chunk-specific metadata
                chunk_metadata.update(
                    {
                        "chunk_num": i + 1,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_strategy": self.config.get(
                            "chunk_strategy", "characters"
                        ),
                        "chunk_size": self.config.get("chunk_size"),
                        "has_overlap": self.config.get("chunk_overlap", 0) > 0
                        and i > 0,
                    }
                )

                chunked_doc = self.create_document(
                    content=chunk_content,
                    metadata=chunk_metadata,
                    doc_id=f"{doc.id}_chunk_{i + 1}" if doc.id else f"chunk_{i + 1}",
                    source=doc.source,
                )
                chunked_docs.append(chunked_doc)

        return chunked_docs

    def parse(self, source: str) -> ProcessingResult:
        """Parse documents using LlamaIndex reader.

        Args:
            source: Path to file or directory

        Returns:
            ProcessingResult with documents
        """
        if not self.reader:
            raise NotImplementedError("Subclass must set self.reader")

        documents = []
        errors = []

        try:
            # Load documents using LlamaIndex reader
            llama_docs = self.reader.load_data(source)

            # Convert to RAG documents
            rag_docs = self._llama_to_rag_documents(llama_docs)

            # Apply chunking if configured
            documents = self._apply_chunking(rag_docs)

        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            errors.append(
                {"source": source, "error": str(e), "parser": self.__class__.__name__}
            )

        return ProcessingResult(
            documents=documents,
            errors=errors,
            metrics={
                "total_documents": len(documents),
                "total_errors": len(errors),
                "parser_type": self.__class__.__name__,
                "chunking_applied": self.text_splitter is not None,
            },
        )

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the file.

        Default implementation checks extensions and mime types.
        Subclasses can override for content-based detection.
        """
        path = Path(file_path)

        # Check extension
        if hasattr(self, "metadata") and self.metadata:
            if path.suffix.lower() in self.metadata.supported_extensions:
                return True

        # Check mime type
        try:
            import magic

            mime = magic.from_file(str(file_path), mime=True)
            if hasattr(self, "metadata") and self.metadata:
                if mime in self.metadata.mime_types:
                    return True
        except:
            pass

        return False

    def _load_metadata(self) -> ParserConfig:
        """Default metadata loader. Subclasses should override."""
        return ParserConfig(
            name="llama_parser",
            display_name="LlamaIndex Parser",
            version="1.0.0",
            supported_extensions=[],
            mime_types=[],
            capabilities=["text_extraction", "chunking"],
            dependencies={"required": ["llama-index"], "optional": []},
            default_config={},
        )
