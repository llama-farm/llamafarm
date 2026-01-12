"""RAG Parse-Only endpoint for document parsing without database storage.

This endpoint allows users to parse documents and get the content back
without storing anything to the RAG database. Useful for:
- Testing parser configurations
- Previewing document content before ingestion
- Extracting content for external processing
"""

import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from fastapi import File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class OutputFormat(str, Enum):
    """Output format for parsed content."""

    MARKDOWN = "markdown"
    TEXT = "text"
    JSON = "json"


class ChunkStrategy(str, Enum):
    """Chunking strategy options."""

    NONE = "none"
    CHARACTERS = "characters"
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"
    SECTIONS = "sections"
    HYBRID = "hybrid"


class ParsedChunk(BaseModel):
    """A single parsed chunk."""

    content: str = Field(..., description="The chunk content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Chunk metadata"
    )


class ParseResponse(BaseModel):
    """Response model for parse endpoint."""

    content: str = Field(..., description="Full parsed content")
    format: str = Field(..., description="Output format (markdown, text, json)")
    chunks: list[ParsedChunk] = Field(
        default_factory=list, description="Chunked content (if chunking enabled)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    parser_used: str = Field(default="unknown", description="Parser that was used")
    filename: str = Field(..., description="Original filename")


# Supported file extensions and their MIME types
SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".html": "text/html",
    ".htm": "text/html",
    ".msg": "application/vnd.ms-outlook",
}


def get_file_extension(filename: str) -> str:
    """Get lowercase file extension from filename."""
    return Path(filename).suffix.lower()


def is_supported_file(filename: str) -> bool:
    """Check if file type is supported for parsing."""
    ext = get_file_extension(filename)
    return ext in SUPPORTED_EXTENSIONS


async def parse_file_content(
    file_path: str,
    filename: str,
    output_format: OutputFormat = OutputFormat.MARKDOWN,
    chunk_strategy: ChunkStrategy = ChunkStrategy.NONE,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
) -> dict[str, Any]:
    """
    Parse a file and return its content.

    This function uses the RAG parsers to extract content from documents
    without storing anything to a database.

    Args:
        file_path: Path to the file to parse
        filename: Original filename (for extension detection)
        output_format: Desired output format
        chunk_strategy: How to chunk the content
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        Dictionary with content, chunks, and metadata
    """
    ext = get_file_extension(filename)
    parser_used = "unknown"
    content = ""
    chunks = []
    metadata = {
        "filename": filename,
        "file_extension": ext,
    }

    try:
        # Try Docling for PDFs first
        if ext == ".pdf":
            try:
                from docling.document_converter import DocumentConverter

                converter = DocumentConverter()
                result = converter.convert(file_path)

                if output_format == OutputFormat.MARKDOWN:
                    content = result.document.export_to_markdown()
                elif output_format == OutputFormat.TEXT:
                    content = result.document.export_to_text()
                else:
                    content = result.document.export_to_markdown()

                parser_used = "DoclingParser"

                # Extract metadata
                metadata["page_count"] = len(result.document.pages) if hasattr(result.document, "pages") else 0

                # Handle chunking if requested
                if chunk_strategy != ChunkStrategy.NONE and chunk_strategy != ChunkStrategy.HYBRID:
                    chunks = _apply_simple_chunking(
                        content, chunk_strategy, chunk_size, chunk_overlap
                    )
                elif chunk_strategy == ChunkStrategy.HYBRID:
                    # Use Docling's HybridChunker
                    try:
                        from docling.chunking import HybridChunker

                        chunker = HybridChunker(max_tokens=chunk_size)
                        doc_chunks = list(chunker.chunk(result.document))
                        chunks = [
                            {
                                "content": chunk.text,
                                "metadata": {
                                    "chunk_index": i,
                                    "total_chunks": len(doc_chunks),
                                    "chunking_strategy": "hybrid",
                                },
                            }
                            for i, chunk in enumerate(doc_chunks)
                        ]
                    except ImportError:
                        # Fall back to simple chunking
                        chunks = _apply_simple_chunking(
                            content, ChunkStrategy.PARAGRAPHS, chunk_size, chunk_overlap
                        )

            except ImportError:
                # Docling not available, try PyMuPDF fallback
                content, parser_used = _parse_pdf_with_pymupdf(file_path, metadata)
                if chunk_strategy != ChunkStrategy.NONE:
                    chunks = _apply_simple_chunking(
                        content, chunk_strategy, chunk_size, chunk_overlap
                    )

        # Try MarkItDown for Office documents
        elif ext in [".docx", ".doc", ".pptx", ".xlsx", ".xls", ".html", ".htm"]:
            content, parser_used = _parse_with_markitdown(file_path, filename, metadata)
            if chunk_strategy != ChunkStrategy.NONE:
                chunks = _apply_simple_chunking(
                    content, chunk_strategy, chunk_size, chunk_overlap
                )

        # Handle text files directly
        elif ext in [".txt", ".md", ".csv"]:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            parser_used = "TextParser"
            metadata["char_count"] = len(content)
            metadata["line_count"] = content.count("\n") + 1

            if chunk_strategy != ChunkStrategy.NONE:
                chunks = _apply_simple_chunking(
                    content, chunk_strategy, chunk_size, chunk_overlap
                )

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Convert output format if needed
        if output_format == OutputFormat.TEXT and parser_used != "TextParser":
            # Strip markdown formatting for plain text
            content = _strip_markdown(content)

    except Exception as e:
        logger.error(f"Error parsing file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to parse file: {str(e)}"
        ) from e

    return {
        "content": content,
        "format": output_format.value,
        "chunks": chunks,
        "metadata": metadata,
        "parser_used": parser_used,
    }


def _parse_pdf_with_pymupdf(file_path: str, metadata: dict) -> tuple[str, str]:
    """Parse PDF using PyMuPDF (fallback when Docling not available)."""
    try:
        import pymupdf

        doc = pymupdf.open(file_path)
        text_parts = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"## Page {page_num + 1}\n\n{text}")

        content = "\n\n".join(text_parts)
        metadata["page_count"] = len(doc)
        doc.close()
        return content, "PyMuPDFParser"
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="No PDF parser available. Install docling or pymupdf.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"PyMuPDF parsing failed: {str(e)}"
        ) from e


def _parse_with_markitdown(
    file_path: str, filename: str, metadata: dict
) -> tuple[str, str]:
    """Parse file using MarkItDown."""
    try:
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert(file_path)
        content = result.text_content
        return content, "MarkItDownParser"
    except ImportError:
        # MarkItDown not available - provide helpful error
        ext = Path(filename).suffix.lower()
        raise HTTPException(
            status_code=415,
            detail=f"Cannot parse {ext} files. MarkItDown parser not installed. Supported without MarkItDown: PDF, TXT, MD, CSV",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"MarkItDown parsing failed: {str(e)}"
        ) from e


def _apply_simple_chunking(
    content: str,
    strategy: ChunkStrategy,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Apply simple chunking to content."""
    try:
        # Try to import SimpleChunker from RAG module
        import sys
        from pathlib import Path

        # Add RAG path if not present
        rag_path = Path(__file__).parent.parent.parent.parent.parent / "rag"
        if str(rag_path) not in sys.path:
            sys.path.insert(0, str(rag_path))

        from components.chunkers import SimpleChunker, ChunkingStrategy

        # Map strategy names
        strategy_map = {
            ChunkStrategy.CHARACTERS: ChunkingStrategy.CHARACTERS,
            ChunkStrategy.SENTENCES: ChunkingStrategy.SENTENCES,
            ChunkStrategy.PARAGRAPHS: ChunkingStrategy.PARAGRAPHS,
            ChunkStrategy.SECTIONS: ChunkingStrategy.SECTIONS,
        }

        chunker_strategy = strategy_map.get(strategy, ChunkingStrategy.SENTENCES)
        chunker = SimpleChunker(
            strategy=chunker_strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        chunk_results = chunker.chunk(content)
        return [
            {
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk in chunk_results
        ]

    except ImportError:
        # SimpleChunker not available, use basic chunking
        return _basic_chunking(content, chunk_size, overlap)


def _basic_chunking(content: str, chunk_size: int, overlap: int) -> list[dict]:
    """Basic character-based chunking fallback."""
    chunks = []
    start = 0
    total_len = len(content)

    while start < total_len:
        end = min(start + chunk_size, total_len)
        chunk_content = content[start:end]
        chunks.append({
            "content": chunk_content,
            "metadata": {
                "chunk_index": len(chunks),
                "start_char": start,
                "end_char": end,
                "chunking_strategy": "characters",
            },
        })
        start = end - overlap if overlap > 0 else end

    # Update total_chunks in metadata
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)

    return chunks


def _strip_markdown(content: str) -> str:
    """Strip basic markdown formatting from content."""
    import re

    # Remove headers
    content = re.sub(r"^#+\s*", "", content, flags=re.MULTILINE)
    # Remove bold/italic
    content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
    content = re.sub(r"\*([^*]+)\*", r"\1", content)
    content = re.sub(r"__([^_]+)__", r"\1", content)
    content = re.sub(r"_([^_]+)_", r"\1", content)
    # Remove links but keep text
    content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
    # Remove code blocks
    content = re.sub(r"```[^`]*```", "", content, flags=re.DOTALL)
    content = re.sub(r"`([^`]+)`", r"\1", content)

    return content.strip()
