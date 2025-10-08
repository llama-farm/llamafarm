"""DOCX parser using LlamaIndex."""

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import docx
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.readers.file import DocxReader

from components.parsers.base.base_parser import BaseParser, ParserConfig
from core.base import Document, ProcessingResult
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
        return file_path.lower().endswith(".docx") or file_path.lower().endswith(".doc")

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def parse(self, source: str, **kwargs):
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

                        # Document properties
                        if (
                            hasattr(doc.core_properties, "title")
                            and doc.core_properties.title
                        ):
                            metadata["title"] = doc.core_properties.title
                        if (
                            hasattr(doc.core_properties, "author")
                            and doc.core_properties.author
                        ):
                            metadata["author"] = doc.core_properties.author
                        if (
                            hasattr(doc.core_properties, "created")
                            and doc.core_properties.created
                        ):
                            metadata["created"] = str(doc.core_properties.created)
                        if (
                            hasattr(doc.core_properties, "modified")
                            and doc.core_properties.modified
                        ):
                            metadata["modified"] = str(doc.core_properties.modified)

                        # Document statistics
                        metadata["paragraph_count"] = len(doc.paragraphs)
                        metadata["section_count"] = len(doc.sections)

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

                        # Group paragraphs into chunks
                        chunks = []
                        current_chunk: list[str] = []
                        current_size = 0

                        for para in doc.paragraphs:
                            para_text = para.text.strip()
                            if not para_text:
                                continue

                            para_size = len(para_text)

                            if (
                                current_size + para_size > self.chunk_size
                                and current_chunk
                            ):
                                # Save current chunk
                                chunks.append("\n\n".join(current_chunk))
                                # Start new chunk with overlap
                                if self.chunk_overlap > 0 and current_chunk:
                                    overlap_text = current_chunk[-1][
                                        : self.chunk_overlap
                                    ]
                                    current_chunk = [overlap_text, para_text]
                                    current_size = len(overlap_text) + para_size
                                else:
                                    current_chunk = [para_text]
                                    current_size = para_size
                            else:
                                current_chunk.append(para_text)
                                current_size += para_size

                        # Add last chunk
                        if current_chunk:
                            chunks.append("\n\n".join(current_chunk))

                        # Create documents from chunks
                        for i, chunk_content in enumerate(chunks):
                            chunk_metadata = metadata | {
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "chunk_strategy": "paragraphs",
                            }

                            doc = Document(
                                content=chunk_content,
                                metadata=chunk_metadata,
                                id=f"{path.stem}_chunk_{i + 1}",
                                source=str(path),
                            )
                            documents.append(doc)

                        return documents  # Return early if paragraph chunking succeeded

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

                    for i, node in enumerate(nodes):
                        chunk_metadata = metadata | {
                            "chunk_index": i,
                            "total_chunks": len(nodes),
                            "chunk_strategy": self.chunk_strategy,
                        }

                        doc = Document(
                            content=node.text if hasattr(node, "text") else str(node),
                            metadata=chunk_metadata,
                            id=f"{path.stem}_chunk_{i + 1}",
                            source=str(path),
                        )
                        documents.append(doc)
                else:
                    # Single document
                    doc = Document(
                        content=content,
                        metadata=metadata,
                        id=path.stem,
                        source=str(path),
                    )
                    documents.append(doc)

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
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
                tmp_file.write(data)
                tmp_path = tmp_file.name

            try:
                # Use LlamaIndex DocxReader
                reader = DocxReader()
                llama_docs = reader.load_data(file=Path(tmp_path))

                documents = []
                filename = (
                    metadata.get("filename", "document.docx")
                    if metadata
                    else "document.docx"
                )

                for llama_doc in llama_docs:
                    content = (
                        llama_doc.text if hasattr(llama_doc, "text") else str(llama_doc)
                    )

                    base_metadata = {
                        "source": filename,
                        "file_name": filename,
                        "parser": "DocxParser_LlamaIndex",
                        "tool": "LlamaIndex",
                    }

                    # Add provided metadata
                    if metadata:
                        base_metadata |= metadata

                    # Add LlamaIndex metadata
                    if hasattr(llama_doc, "metadata"):
                        base_metadata.update(llama_doc.metadata)

                    # Extract additional metadata if python-docx is available
                    if self.extract_metadata:
                        try:
                            doc = docx.Document(tmp_path)

                            # Document properties
                            if (
                                hasattr(doc.core_properties, "title")
                                and doc.core_properties.title
                            ):
                                base_metadata["title"] = doc.core_properties.title
                            if (
                                hasattr(doc.core_properties, "author")
                                and doc.core_properties.author
                            ):
                                base_metadata["author"] = doc.core_properties.author
                            if (
                                hasattr(doc.core_properties, "created")
                                and doc.core_properties.created
                            ):
                                base_metadata["created"] = str(
                                    doc.core_properties.created
                                )
                            if (
                                hasattr(doc.core_properties, "modified")
                                and doc.core_properties.modified
                            ):
                                base_metadata["modified"] = str(
                                    doc.core_properties.modified
                                )

                            # Document statistics
                            base_metadata["paragraph_count"] = len(doc.paragraphs)
                            base_metadata["section_count"] = len(doc.sections)

                            if self.extract_tables:
                                base_metadata["table_count"] = len(doc.tables)

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
                                chunk_size=self.chunk_size,
                                chunk_overlap=self.chunk_overlap,
                            )
                        elif self.chunk_strategy == "paragraphs":
                            # For paragraph-based chunking, try to use the document structure
                            try:
                                doc = docx.Document(tmp_path)

                                # Group paragraphs into chunks
                                chunks = []
                                current_chunk: list[str] = []
                                current_size = 0

                                for para in doc.paragraphs:
                                    para_text = para.text.strip()
                                    if not para_text:
                                        continue

                                    para_size = len(para_text)

                                    if (
                                        current_size + para_size > self.chunk_size
                                        and current_chunk
                                    ):
                                        # Save current chunk
                                        chunks.append("\n\n".join(current_chunk))
                                        # Start new chunk with overlap
                                        if self.chunk_overlap > 0 and current_chunk:
                                            overlap_text = current_chunk[-1][
                                                : self.chunk_overlap
                                            ]
                                            current_chunk = [overlap_text, para_text]
                                            current_size = len(overlap_text) + para_size
                                        else:
                                            current_chunk = [para_text]
                                            current_size = para_size
                                    else:
                                        current_chunk.append(para_text)
                                        current_size += para_size

                                # Add last chunk
                                if current_chunk:
                                    chunks.append("\n\n".join(current_chunk))

                                # Create documents from chunks
                                for i, chunk_content in enumerate(chunks):
                                    chunk_metadata = base_metadata | {
                                        "chunk_index": i,
                                        "total_chunks": len(chunks),
                                        "chunk_strategy": "paragraphs",
                                    }

                                    doc = Document(
                                        content=chunk_content,
                                        metadata=chunk_metadata,
                                        id=f"{Path(filename).stem}_chunk_{i + 1}",
                                        source=filename,
                                    )
                                    documents.append(doc)

                                return documents  # Return early if paragraph chunking succeeded

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

                        # Apply splitter if we didn't do paragraph chunking
                        nodes = splitter.get_nodes_from_documents([llama_doc])

                        for i, node in enumerate(nodes):
                            chunk_metadata = base_metadata | {
                                "chunk_index": i,
                                "total_chunks": len(nodes),
                                "chunk_strategy": self.chunk_strategy,
                            }

                            doc = Document(
                                content=node.text
                                if hasattr(node, "text")
                                else str(node),
                                metadata=chunk_metadata,
                                id=f"{Path(filename).stem}_chunk_{i + 1}",
                                source=filename,
                            )
                            documents.append(doc)
                    else:
                        # Single document
                        doc = Document(
                            content=content,
                            metadata=base_metadata,
                            id=Path(filename).stem,
                            source=filename,
                        )
                        documents.append(doc)

                return documents

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Failed to parse DOCX blob: {e}")
            return []
