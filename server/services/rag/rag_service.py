"""Lightweight RAG service for server using shared library."""

import logging
import re
import uuid
from pathlib import Path
from typing import Any

from llamafarm_rag_common import (
    BasicSimilarityStrategy,
    ChromaStore,
    Document,
    UniversalEmbedder,
)

from .parsers.markitdown_parser import MarkItDownParser

logger = logging.getLogger(__name__)

# Supported chunking strategies
CHUNK_STRATEGIES = ["semantic", "section", "paragraph", "sentence", "page", "fixed"]


class LightweightRAGService:
    """Lightweight RAG service using shared components."""

    def __init__(
        self,
        project_dir: Path | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        universal_runtime_url: str = "http://127.0.0.1:11540",
    ):
        """Initialize RAG service.

        Args:
            project_dir: Project directory for storing data
            embedding_model: Model to use for embeddings
            universal_runtime_url: URL for Universal Runtime
        """
        self.project_dir = project_dir or Path.home() / ".llamafarm" / "server_rag"
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Initialize parser
        logger.info("Initializing MarkItDown parser...")
        self.parser = MarkItDownParser(universal_runtime_url=universal_runtime_url)

        # Initialize embedder
        logger.info(f"Initializing embedder with model: {embedding_model}")
        self.embedder = UniversalEmbedder(
            config={
                "model": embedding_model,
                "api_base": universal_runtime_url,
            }
        )

        # Initialize vector store
        logger.info("Initializing ChromaDB vector store...")
        self.store = ChromaStore(
            config={
                "collection_name": "server_documents",
                "embedding_dimension": self.embedder.get_embedding_dimension(),
                "distance_metric": "cosine",
            },
            project_dir=self.project_dir,
        )

        # Initialize retrieval strategy
        self.retriever = BasicSimilarityStrategy(
            config={
                "similarity_threshold": 0.0,
                "max_results": 100,
            }
        )

        logger.info("RAG service initialized successfully")

    async def ingest_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
        chunk_strategy: str = "semantic",
        chunk_size: int = 1000,
    ) -> dict[str, Any]:
        """Ingest a file into the RAG system.

        Args:
            file_path: Path to file to ingest
            metadata: Optional metadata for the document
            chunk_strategy: Chunking strategy (semantic, section, paragraph, sentence, page, fixed)
            chunk_size: Target chunk size for applicable strategies

        Returns:
            Dict with ingestion results
        """
        metadata = metadata or {}

        # Validate chunk strategy
        if chunk_strategy not in CHUNK_STRATEGIES:
            logger.warning(f"Unknown chunk strategy '{chunk_strategy}', using 'semantic'")
            chunk_strategy = "semantic"

        logger.info(f"Ingesting file: {file_path} with {chunk_strategy} chunking")

        try:
            # Parse document
            text = await self.parser.parse(file_path, metadata)
            logger.info(f"Parsed {len(text)} characters from {file_path}")

            # Chunk with specified strategy
            chunks = self._chunk_text(text, strategy=chunk_strategy, chunk_size=chunk_size)
            logger.info(f"Split into {len(chunks)} chunks using {chunk_strategy} strategy")

            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    id=str(uuid.uuid4()),
                    content=chunk,
                    metadata={
                        **metadata,
                        "source": file_path,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
                documents.append(doc)

            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedder.embed([doc.content for doc in documents])
            for doc, embedding in zip(documents, embeddings):
                doc.embeddings = embedding

            # Add to vector store
            logger.info("Adding to vector store...")
            self.store.add_documents(documents)

            return {
                "success": True,
                "file": file_path,
                "chunks": len(chunks),
                "characters": len(text),
                "document_ids": [doc.id for doc in documents],
            }

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            return {
                "success": False,
                "file": file_path,
                "error": str(e),
            }

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search for documents relevant to query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Dict with search results
        """
        logger.info(f"Searching for: {query}")

        try:
            # Generate query embedding
            query_embedding = self.embedder.embed([query])[0]

            # Retrieve documents
            results = self.retriever.retrieve(
                query_embedding=query_embedding,
                vector_store=self.store,
                top_k=top_k,
            )

            # Format results
            formatted_results = []
            for doc, score in zip(results.documents, results.scores):
                formatted_results.append({
                    "content": doc.content,
                    "score": float(score),
                    "metadata": doc.metadata,
                    "id": doc.id,
                })

            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results),
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
            }

    def get_stats(self) -> dict[str, Any]:
        """Get RAG system statistics."""
        return {
            "document_count": self.store.get_count(),
            "collection": self.store.collection_name,
            "embedding_dimension": self.embedder.get_embedding_dimension(),
            "embedding_model": self.embedder.model,
        }

    def _chunk_text(
        self, text: str, strategy: str = "semantic", chunk_size: int = 1000
    ) -> list[str]:
        """Chunk text using specified strategy.

        Args:
            text: Text to chunk
            strategy: Chunking strategy
            chunk_size: Target chunk size for applicable strategies

        Returns:
            List of text chunks
        """
        if strategy == "semantic":
            return self._chunk_semantic(text, chunk_size)
        elif strategy == "section":
            return self._chunk_by_section(text)
        elif strategy == "paragraph":
            return self._chunk_by_paragraph(text, chunk_size)
        elif strategy == "sentence":
            return self._chunk_by_sentence(text, min_chunk_size=100)
        elif strategy == "page":
            return self._chunk_by_page(text)
        elif strategy == "fixed":
            return self._chunk_fixed(text, chunk_size)
        else:
            return self._chunk_semantic(text, chunk_size)

    def _chunk_semantic(self, text: str, chunk_size: int = 1000) -> list[str]:
        """Semantic chunking using SemChunk if available, else paragraph-based."""
        try:
            import semchunk

            return semchunk.chunk(
                text,
                chunk_size=chunk_size,
                memoize=True,
            )
        except ImportError:
            logger.warning("SemChunk not available, falling back to paragraph chunking")
            return self._chunk_by_paragraph(text, chunk_size)

    def _chunk_by_section(self, text: str) -> list[str]:
        """Chunk text by markdown headers (sections)."""
        sections = re.split(r"\n(?=#+\s)", text)
        sections = [s.strip() for s in sections if s.strip()]

        # If no sections found, fall back to paragraph
        if len(sections) <= 1:
            return self._chunk_by_paragraph(text, 1000)

        return sections

    def _chunk_by_paragraph(self, text: str, chunk_size: int = 1000) -> list[str]:
        """Chunk text by paragraphs (double newlines)."""
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Merge small paragraphs, split large ones
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If single paragraph is too large, split by sentence
                if len(para) > chunk_size * 2:
                    sub_chunks = self._chunk_by_sentence(para, min_chunk_size=100)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text.strip()]

    def _chunk_by_sentence(self, text: str, min_chunk_size: int = 100) -> list[str]:
        """Chunk text by sentences."""
        # Split on sentence boundaries
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)

        # Group sentences to meet minimum chunk size
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            current_chunk.append(sentence)
            current_size += len(sentence)

            if current_size >= min_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text.strip()]

    def _chunk_by_page(self, text: str) -> list[str]:
        """Chunk text by page breaks (form feed character)."""
        pages = text.split("\f")
        pages = [p.strip() for p in pages if p.strip()]

        # If no page breaks, fall back to paragraph
        if len(pages) <= 1:
            return self._chunk_by_paragraph(text, 1000)

        return pages

    def _chunk_fixed(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
        """Fixed-size chunking with overlap."""
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if end < len(text) else len(text)

        return chunks if chunks else [text.strip()]
