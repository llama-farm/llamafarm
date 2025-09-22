"""Markdown parser using LlamaIndex."""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MarkdownParser_LlamaIndex:
    """Markdown parser using LlamaIndex with advanced features."""

    def __init__(
        self,
        name: str = "MarkdownParser_LlamaIndex",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.config = config or {}

        # Chunking configuration
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.chunk_strategy = self.config.get("chunk_strategy", "headings")

        # Feature flags
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.extract_headings = self.config.get("extract_headings", True)
        self.extract_links = self.config.get("extract_links", True)
        self.extract_code_blocks = self.config.get("extract_code_blocks", True)
        self.preserve_formatting = self.config.get("preserve_formatting", False)

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def parse(self, source: str, **kwargs):
        """Parse markdown using LlamaIndex."""
        from core.base import Document, ProcessingResult

        try:
            from llama_index.core import SimpleDirectoryReader
            from llama_index.readers.file import MarkdownReader
            from llama_index.core.node_parser import MarkdownNodeParser
        except ImportError:
            return ProcessingResult(
                documents=[],
                errors=[
                    {
                        "error": "LlamaIndex not installed. Install with: pip install llama-index",
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
            # Use LlamaIndex MarkdownReader
            reader = MarkdownReader()
            llama_docs = reader.load_data(file=path)

            documents = []

            # Process with MarkdownNodeParser if chunking is needed
            if self.chunk_size and self.chunk_strategy == "headings":
                parser = MarkdownNodeParser()
                nodes = parser.get_nodes_from_documents(llama_docs)

                for i, node in enumerate(nodes):
                    metadata = {
                        "source": str(path),
                        "file_name": path.name,
                        "parser": "MarkdownParser_LlamaIndex",
                        "tool": "LlamaIndex",
                        "chunk_index": i,
                        "total_chunks": len(nodes),
                        "chunk_strategy": self.chunk_strategy,
                    }

                    # Extract node metadata
                    if hasattr(node, "metadata"):
                        metadata.update(node.metadata)

                    # Add heading info if available
                    if self.extract_headings and hasattr(node, "metadata"):
                        if "Header_1" in node.metadata:
                            metadata["section"] = node.metadata.get("Header_1")
                        if "Header_2" in node.metadata:
                            metadata["subsection"] = node.metadata.get("Header_2")

                    doc = Document(
                        content=node.text if hasattr(node, "text") else str(node),
                        metadata=metadata,
                        id=f"{path.stem}_chunk_{i + 1}",
                        source=str(path),
                    )
                    documents.append(doc)
            else:
                # Process as single document or with different chunking
                for llama_doc in llama_docs:
                    content = (
                        llama_doc.text if hasattr(llama_doc, "text") else str(llama_doc)
                    )

                    metadata = {
                        "source": str(path),
                        "file_name": path.name,
                        "parser": "MarkdownParser_LlamaIndex",
                        "tool": "LlamaIndex",
                        "file_size": path.stat().st_size,
                    }

                    # Add LlamaIndex metadata
                    if hasattr(llama_doc, "metadata"):
                        metadata.update(llama_doc.metadata)

                    # Apply chunking if needed
                    if self.chunk_size and self.chunk_strategy != "headings":
                        from llama_index.core.node_parser import (
                            SentenceSplitter,
                            TokenTextSplitter,
                        )

                        if self.chunk_strategy == "sentences":
                            splitter = SentenceSplitter(
                                chunk_size=self.chunk_size,
                                chunk_overlap=self.chunk_overlap,
                            )
                        else:  # characters or tokens
                            splitter = TokenTextSplitter(
                                chunk_size=self.chunk_size,
                                chunk_overlap=self.chunk_overlap,
                            )

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
            logger.error(f"Failed to parse markdown file {source}: {e}")
            return ProcessingResult(
                documents=[], errors=[{"error": str(e), "source": source}]
            )
