"""PDF parser using LlamaIndex."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from components.parsers.base.base_parser import BaseParser, ParserConfig

logger = logging.getLogger(__name__)


class PDFParser_LlamaIndex(BaseParser):
    """PDF parser using LlamaIndex with multiple backend options."""

    def __init__(
        self,
        name: str = "PDFParser_LlamaIndex",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)  # Call BaseParser init
        self.name = name

        # Chunking configuration
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.chunk_strategy = self.config.get("chunk_strategy", "characters")

        # Feature flags
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.extract_images = self.config.get("extract_images", False)
        self.extract_tables = self.config.get("extract_tables", False)
        self.preserve_layout = self.config.get("preserve_layout", False)

        # Fallback strategies for different PDF types
        self.fallback_strategies = self.config.get(
            "fallback_strategies",
            [
                "llama_pdf_reader",
                "llama_pymupdf_reader",
                "direct_pymupdf",
                "pypdf2_fallback",
            ],
        )

    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata."""
        return ParserConfig(
            name="PDFParser_LlamaIndex",
            display_name="LlamaIndex PDF Parser",
            version="1.0.0",
            supported_extensions=[".pdf"],
            mime_types=["application/pdf"],
            capabilities=[
                "text_extraction",
                "metadata_extraction",
                "table_extraction",
                "image_extraction",
            ],
            dependencies={
                "llama-index": ["llama-index>=0.9.0"],
                "PyMuPDF": ["PyMuPDF>=1.23.0"],
            },
            default_config={
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "chunk_strategy": "characters",
                "extract_metadata": True,
                "extract_images": False,
                "extract_tables": False,
            },
        )

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        return file_path.lower().endswith(".pdf")

    def parse_blob(self, data: bytes, metadata: Dict[str, Any] = None) -> List:
        """Parse PDF from raw bytes."""
        import os
        import tempfile

        from core.base import Document

        try:
            from llama_index.core import SimpleDirectoryReader
        except ImportError:
            # Fall back to PyPDF2 parser
            from components.parsers.pdf.pypdf2_parser import PDFParser_PyPDF2

            fallback_parser = PDFParser_PyPDF2(config=self.config)
            return fallback_parser.parse_blob(data, metadata)

        try:
            # LlamaIndex needs a file on disk, so write temporarily
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(data)
                tmp_path = tmp_file.name

            try:
                # Use LlamaIndex to read the PDF
                reader = SimpleDirectoryReader(input_files=[tmp_path])
                llama_docs = reader.load_data()

                documents = []
                filename = (
                    metadata.get("filename", "unknown.pdf")
                    if metadata
                    else "unknown.pdf"
                )

                for llama_doc in llama_docs:
                    doc = Document(
                        content=llama_doc.text,
                        metadata={
                            "source": filename,
                            "parser": "PDFParser_LlamaIndex",
                            **(metadata or {}),
                        },
                        source=filename,
                    )
                    documents.append(doc)

                return documents

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            print(f"Error parsing PDF with LlamaIndex, falling back to PyPDF2: {e}")
            # Fall back to PyPDF2 parser
            from components.parsers.pdf.pypdf2_parser import PDFParser_PyPDF2

            fallback_parser = PDFParser_PyPDF2(config=self.config)
            return fallback_parser.parse_blob(data, metadata)

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def parse(self, source: str, **kwargs):
        """Parse PDF using LlamaIndex with fallback strategies."""
        from core.base import Document, ProcessingResult

        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}],
            )

        # Try different parsing strategies
        for strategy in self.fallback_strategies:
            result = self._try_strategy(strategy, path)
            if result and result.documents:
                return result

        # All strategies failed
        return ProcessingResult(
            documents=[],
            errors=[{"error": "All PDF parsing strategies failed", "source": source}],
        )

    def _try_strategy(self, strategy: str, path: Path):
        """Try a specific parsing strategy."""
        from core.base import Document, ProcessingResult

        try:
            if strategy == "llama_pdf_reader":
                return self._parse_with_llama_pdf(path)
            elif strategy == "llama_pymupdf_reader":
                return self._parse_with_llama_pymupdf(path)
            elif strategy == "direct_pymupdf":
                return self._parse_with_direct_pymupdf(path)
            elif strategy == "pypdf2_fallback":
                return self._parse_with_pypdf2(path)
        except Exception as e:
            logger.debug(f"Strategy {strategy} failed: {e}")
            return None

    def _parse_with_llama_pdf(self, path: Path):
        """Parse using LlamaIndex PDFReader."""
        from core.base import Document, ProcessingResult

        try:
            from llama_index.readers.file import PDFReader
            from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
        except ImportError:
            return None

        try:
            reader = PDFReader()
            llama_docs = reader.load_data(file=path)

            documents = []
            global_chunk_index = 0  # Track chunks across all pages

            for page_num, llama_doc in enumerate(llama_docs):
                content = (
                    llama_doc.text if hasattr(llama_doc, "text") else str(llama_doc)
                )

                metadata = {
                    "source": str(path),
                    "file_name": path.name,
                    "parser": "PDFParser_LlamaIndex",
                    "tool": "LlamaIndex",
                    "strategy": "llama_pdf_reader",
                    "file_size": path.stat().st_size,
                    "page_number": page_num + 1,  # Add page number
                }

                # Add LlamaIndex metadata
                if hasattr(llama_doc, "metadata"):
                    metadata.update(llama_doc.metadata)

                # Apply chunking if needed
                if self.chunk_size:
                    if self.chunk_strategy == "sentences":
                        splitter = SentenceSplitter(
                            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                        )
                    else:  # characters or tokens
                        splitter = TokenTextSplitter(
                            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                        )

                    nodes = splitter.get_nodes_from_documents([llama_doc])

                    for i, node in enumerate(nodes):
                        global_chunk_index += 1  # Increment global counter
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update(
                            {
                                "chunk_index": global_chunk_index - 1,
                                "page_chunk_index": i,  # Chunk index within page
                                "total_page_chunks": len(nodes),
                                "total_chunks": len(nodes),
                                "chunk_strategy": self.chunk_strategy,
                            }
                        )

                        if hasattr(node, "metadata") and "page_label" in node.metadata:
                            chunk_metadata["page_label"] = node.metadata["page_label"]

                        doc = Document(
                            content=node.text if hasattr(node, "text") else str(node),
                            metadata=chunk_metadata,
                            id=f"{path.stem}_chunk_{global_chunk_index}",
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
            logger.debug(f"LlamaIndex PDF reader failed: {e}")
            return None

    def _parse_with_llama_pymupdf(self, path: Path):
        """Parse using LlamaIndex PyMuPDFReader."""
        from core.base import Document, ProcessingResult

        try:
            from llama_index.readers.file import PyMuPDFReader
            from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
        except ImportError:
            return None

        try:
            reader = PyMuPDFReader()
            llama_docs = reader.load(file_path=str(path))

            documents = []
            global_chunk_index = 0  # Track chunks across all pages

            for page_num, llama_doc in enumerate(llama_docs):
                content = (
                    llama_doc.text if hasattr(llama_doc, "text") else str(llama_doc)
                )

                metadata = {
                    "source": str(path),
                    "file_name": path.name,
                    "parser": "PDFParser_LlamaIndex",
                    "tool": "LlamaIndex-PyMuPDF",
                    "strategy": "llama_pymupdf_reader",
                    "file_size": path.stat().st_size,
                    "page_number": page_num + 1,  # Add page number
                }

                # Add LlamaIndex metadata including page info
                if hasattr(llama_doc, "metadata"):
                    metadata.update(llama_doc.metadata)

                # Apply chunking if needed
                if self.chunk_size:
                    if self.chunk_strategy == "sentences":
                        splitter = SentenceSplitter(
                            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                        )
                    else:
                        splitter = TokenTextSplitter(
                            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                        )

                    nodes = splitter.get_nodes_from_documents([llama_doc])

                    for i, node in enumerate(nodes):
                        global_chunk_index += 1  # Increment global counter
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update(
                            {
                                "chunk_index": global_chunk_index - 1,
                                "page_chunk_index": i,  # Chunk index within page
                                "total_page_chunks": len(nodes),
                                "total_chunks": len(nodes),
                                "chunk_strategy": self.chunk_strategy,
                            }
                        )

                        doc = Document(
                            content=node.text if hasattr(node, "text") else str(node),
                            metadata=chunk_metadata,
                            id=f"{path.stem}_chunk_{global_chunk_index}",
                            source=str(path),
                        )
                        documents.append(doc)
                else:
                    doc = Document(
                        content=content,
                        metadata=metadata,
                        id=path.stem,
                        source=str(path),
                    )
                    documents.append(doc)

            return ProcessingResult(documents=documents, errors=[])

        except Exception as e:
            logger.debug(f"LlamaIndex PyMuPDF reader failed: {e}")
            return None

    def _parse_with_direct_pymupdf(self, path: Path):
        """Parse directly using PyMuPDF."""
        from core.base import Document, ProcessingResult

        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None

        try:
            pdf = fitz.open(str(path))
            text_parts = []

            for page_num, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

            pdf.close()

            if not text_parts:
                return None

            full_text = "\n\n".join(text_parts)

            metadata = {
                "source": str(path),
                "file_name": path.name,
                "parser": "PDFParser_LlamaIndex",
                "tool": "PyMuPDF-Direct",
                "strategy": "direct_pymupdf",
                "file_size": path.stat().st_size,
                "total_pages": len(text_parts),
            }

            # Apply chunking if needed
            if self.chunk_size:
                from llama_index.core import Document as LlamaDoc
                from llama_index.core.node_parser import (
                    SentenceSplitter,
                    TokenTextSplitter,
                )

                llama_doc = LlamaDoc(text=full_text, metadata=metadata)

                if self.chunk_strategy == "sentences":
                    splitter = SentenceSplitter(
                        chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                    )
                else:
                    splitter = TokenTextSplitter(
                        chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                    )

                nodes = splitter.get_nodes_from_documents([llama_doc])
                documents = []

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
                        content=node.text,
                        metadata=chunk_metadata,
                        id=f"{path.stem}_chunk_{i + 1}",
                        source=str(path),
                    )
                    documents.append(doc)
            else:
                doc = Document(
                    content=full_text, metadata=metadata, id=path.stem, source=str(path)
                )
                documents = [doc]

            return ProcessingResult(documents=documents, errors=[])

        except Exception as e:
            logger.debug(f"Direct PyMuPDF failed: {e}")
            return None

    def _parse_with_pypdf2(self, path: Path):
        """Fallback to PyPDF2."""
        from core.base import Document, ProcessingResult

        try:
            import PyPDF2
        except ImportError:
            return None

        try:
            with open(path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text_parts = []

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

            if not text_parts:
                return None

            full_text = "\n\n".join(text_parts)

            metadata = {
                "source": str(path),
                "file_name": path.name,
                "parser": "PDFParser_LlamaIndex",
                "tool": "PyPDF2-Fallback",
                "strategy": "pypdf2_fallback",
                "file_size": path.stat().st_size,
                "total_pages": len(text_parts),
            }

            doc = Document(
                content=full_text, metadata=metadata, id=path.stem, source=str(path)
            )

            return ProcessingResult(documents=[doc], errors=[])

        except Exception as e:
            logger.debug(f"PyPDF2 fallback failed: {e}")
            return None
