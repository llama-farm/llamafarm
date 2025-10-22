"""DOCX parser using LlamaIndex."""

from pathlib import Path
from typing import Any, Optional

import docx
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.readers.file import DocxReader

from components.parsers.base.base_parser import BaseParser, ParserConfig
from components.parsers.docx.docx_utils import (
    DocxChunker,
    DocxDocumentFactory,
    DocxMetadataExtractor,
    DocxTempFileHandler,
)
from core.base import ProcessingResult
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.docx.llamaindex_parser")


class DocxParser_LlamaIndex(BaseParser):
    """DOCX parser using LlamaIndex with python-docx backend."""

    def __init__(
        self,
        name: str = "DocxParser_LlamaIndex",
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(config or {})  # Call BaseParser init
        self.name = name

        # Chunking configuration
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.chunk_strategy = self.config.get("chunk_strategy", "paragraphs")

        # Feature flags
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.extract_tables = self.config.get("extract_tables", True)
        self.extract_headers_footers = self.config.get("extract_headers_footers", False)
        self.extract_comments = self.config.get("extract_comments", False)
        self.extract_images = self.config.get("extract_images", False)
        self.preserve_formatting = self.config.get("preserve_formatting", False)

    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata."""
        return ParserConfig(
            name="DocxParser_LlamaIndex",
            display_name="LlamaIndex DOCX Parser",
            version="1.0.0",
            supported_extensions=[".docx"],
            mime_types=[
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ],
            capabilities=[
                "text_extraction",
                "metadata_extraction",
                "table_extraction",
                "header_footer_extraction",
            ],
            dependencies={
                "llama-index": ["llama-index>=0.9.0"],
                "python-docx": ["python-docx>=0.8.11"],
            },
            default_config={
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "chunk_strategy": "paragraphs",
                "extract_metadata": True,
                "extract_tables": True,
            },
        )

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        return file_path.lower().endswith(".docx")

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def parse(self, source: str, **kwargs) -> ProcessingResult:
        """Parse DOCX/DOC using LlamaIndex."""
        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}],
            )

        try:
            # Use LlamaIndex DocxReader
            reader = DocxReader()
            llama_docs = reader.load_data(file=path)

            documents = []

            for llama_doc in llama_docs:
                content = (
                    llama_doc.text if hasattr(llama_doc, "text") else str(llama_doc)
                )

                metadata = {
                    "source": str(path),
                    "file_name": path.name,
                    "parser": "DocxParser_LlamaIndex",
                    "tool": "LlamaIndex",
                    "file_size": path.stat().st_size,
                }

                # Add LlamaIndex metadata
                if hasattr(llama_doc, "metadata"):
                    metadata.update(llama_doc.metadata)

                # Extract additional metadata if python-docx is available
                if self.extract_metadata:
                    try:
                        doc = docx.Document(str(path))
                        metadata = DocxMetadataExtractor.extract_document_properties(
                            doc, metadata
                        )

                        if self.extract_tables:
                            metadata["table_count"] = len(doc.tables)

                    except ImportError:
                        logger.debug(
                            "python-docx not available for enhanced metadata extraction"
                        )
                    except Exception as e:
                        logger.debug(f"Failed to extract enhanced metadata: {e}")

                # Apply chunking if needed
                if self.chunk_size:
                    if self.chunk_strategy == "sentences":
                        splitter = SentenceSplitter(
                            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                        )
                    elif self.chunk_strategy == "paragraphs":
                        # For paragraph-based chunking, try to use the document structure
                        try:
                            doc = docx.Document(str(path))

                            # Extract full text first
                            paragraphs = [
                                para.text.strip()
                                for para in doc.paragraphs
                                if para.text.strip()
                            ]
                            full_text = "\n\n".join(paragraphs)

                            # Use shared chunking utility
                            chunks = DocxChunker.chunk_by_paragraphs(
                                full_text, self.chunk_size, self.chunk_overlap
                            )

                            # Create documents from chunks using shared factory
                            documents = (
                                DocxDocumentFactory.create_documents_from_chunks(
                                    chunks, metadata, str(path), "paragraphs"
                                )
                            )

                            return ProcessingResult(
                                documents=documents, errors=[]
                            )  # Wrap in ProcessingResult

                        except ImportError:
                            # Fall back to token-based chunking
                            splitter = TokenTextSplitter(
                                chunk_size=self.chunk_size,
                                chunk_overlap=self.chunk_overlap,
                            )
                    else:  # characters or tokens
                        splitter = TokenTextSplitter(
                            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                        )

                    # Apply splitter if we didn't do paragraph chunking
                    if not documents:
                        nodes = splitter.get_nodes_from_documents([llama_doc])

                        chunks = [
                            node.text if hasattr(node, "text") else str(node)
                            for node in nodes
                        ]
                        documents = DocxDocumentFactory.create_documents_from_chunks(
                            chunks, metadata, str(path), self.chunk_strategy
                        )
                else:
                    # Single document
                    documents.append(
                        DocxDocumentFactory.create_single_document(
                            content, metadata, str(path)
                        )
                    )

            return ProcessingResult(documents=documents, errors=[])

        except Exception as e:
            logger.error(f"Failed to parse DOCX file {source}: {e}")
            return ProcessingResult(
                documents=[], errors=[{"error": str(e), "source": source}]
            )

    def parse_blob(self, data: bytes, metadata: dict[str, Any] | None = None) -> list:
        """Parse DOCX from raw bytes using in-memory buffer."""

        try:
            # LlamaIndex DocxReader needs a file on disk, so write temporarily
            with DocxTempFileHandler(data) as tmp_path:
                reader = DocxReader()
                llama_docs = reader.load_data(file=Path(tmp_path))

                filename = (
                    metadata.get("filename", "document.docx")
                    if metadata
                    else "document.docx"
                )
                documents = []

                for llama_doc in llama_docs:
                    content = (
                        llama_doc.text if hasattr(llama_doc, "text") else str(llama_doc)
                    )

                    # Build base metadata
                    base_metadata = self._build_llama_blob_metadata(
                        llama_doc, filename, metadata, tmp_path
                    )

                    # Apply chunking if needed
                    if self.chunk_size:
                        documents.extend(
                            self._chunk_llama_content(
                                content, llama_doc, base_metadata, filename, tmp_path
                            )
                        )
                    else:
                        documents.append(
                            DocxDocumentFactory.create_single_document(
                                content, base_metadata, filename
                            )
                        )

                return documents

        except Exception as e:
            logger.error(f"Failed to parse DOCX blob: {e}")
            return []

    def _build_llama_blob_metadata(
        self,
        llama_doc,
        filename: str,
        provided_metadata: dict[str, Any] | None,
        tmp_path: str,
    ) -> dict[str, Any]:
        """Build metadata for LlamaIndex blob parsing."""
        base_metadata = {
            "source": filename,
            "file_name": filename,
            "parser": "DocxParser_LlamaIndex",
            "tool": "LlamaIndex",
        }

        # Add provided metadata
        if provided_metadata:
            base_metadata |= provided_metadata

        # Add LlamaIndex metadata
        if hasattr(llama_doc, "metadata"):
            base_metadata.update(llama_doc.metadata)

        # Extract additional metadata if python-docx is available
        if self.extract_metadata:
            try:
                doc = docx.Document(tmp_path)
                base_metadata = DocxMetadataExtractor.extract_document_properties(
                    doc, base_metadata
                )
                if self.extract_tables:
                    base_metadata["table_count"] = len(doc.tables)
            except ImportError:
                logger.debug(
                    "python-docx not available for enhanced metadata extraction"
                )
            except Exception as e:
                logger.debug(f"Failed to extract enhanced metadata: {e}")

        return base_metadata

    def _chunk_llama_content(
        self,
        content: str,
        llama_doc,
        base_metadata: dict[str, Any],
        filename: str,
        tmp_path: str,
    ) -> list:
        """Apply chunking to LlamaIndex content."""
        if self.chunk_strategy == "sentences":
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif self.chunk_strategy == "paragraphs":
            # For paragraph-based chunking, try to use the document structure
            try:
                doc = docx.Document(tmp_path)
                paragraphs = [
                    para.text.strip() for para in doc.paragraphs if para.text.strip()
                ]
                full_text = "\n\n".join(paragraphs)
                chunks = DocxChunker.chunk_by_paragraphs(
                    full_text, self.chunk_size, self.chunk_overlap
                )
                return DocxDocumentFactory.create_documents_from_chunks(
                    chunks, base_metadata, filename, "paragraphs"
                )
            except ImportError:
                # Fall back to token-based chunking
                splitter = TokenTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
        else:  # characters or tokens
            splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        # Apply splitter
        nodes = splitter.get_nodes_from_documents([llama_doc])
        chunks = [node.text if hasattr(node, "text") else str(node) for node in nodes]
        return DocxDocumentFactory.create_documents_from_chunks(
            chunks, base_metadata, filename, self.chunk_strategy
        )
