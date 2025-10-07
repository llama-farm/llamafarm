"""DOCX parser using LlamaIndex."""

from pathlib import Path
from typing import Dict, Any, Optional

from core.logging import RAGStructLogger
logger = RAGStructLogger("rag.components.parsers.docx.llamaindex_parser")


class DocxParser_LlamaIndex:
    """DOCX parser using LlamaIndex with python-docx backend."""

    def __init__(
        self,
        name: str = "DocxParser_LlamaIndex",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.config = config or {}

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

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def parse(self, source: str, **kwargs):
        """Parse DOCX using LlamaIndex."""
        from core.base import Document, ProcessingResult

        try:
            from llama_index.readers.file import DocxReader
            from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
        except ImportError:
            return ProcessingResult(
                documents=[],
                errors=[
                    {
                        "error": "LlamaIndex not installed. Install with: pip install llama-index llama-index-readers-file",
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
                        import docx

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
                            import docx

                            doc = docx.Document(str(path))

                            # Group paragraphs into chunks
                            chunks = []
                            current_chunk = []
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
                                chunk_metadata = metadata.copy()
                                chunk_metadata.update(
                                    {
                                        "chunk_index": i,
                                        "total_chunks": len(chunks),
                                        "chunk_strategy": "paragraphs",
                                    }
                                )

                                doc = Document(
                                    content=chunk_content,
                                    metadata=chunk_metadata,
                                    id=f"{path.stem}_chunk_{i + 1}",
                                    source=str(path),
                                )
                                documents.append(doc)

                            continue  # Skip the default chunking

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
                            chunk_metadata = metadata.copy()
                            chunk_metadata.update(
                                {
                                    "chunk_index": i,
                                    "total_chunks": len(nodes),
                                    "chunk_strategy": self.chunk_strategy,
                                }
                            )

                            doc = Document(
                                content=node.text
                                if hasattr(node, "text")
                                else str(node),
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
