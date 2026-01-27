"""Preview handler for document chunking preview."""

from typing import Any

from core.base import Document, PreviewChunk, PreviewResult
from core.blob_processor import BlobProcessor
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.core.preview_handler")


class PreviewHandler:
    """Generates document preview using same processing as ingestion."""

    def __init__(self, blob_processor: BlobProcessor):
        self.blob_processor = blob_processor

    def generate_preview(
        self,
        file_data: bytes,
        metadata: dict[str, Any],
        chunk_size_override: int | None = None,
        chunk_overlap_override: int | None = None,
        chunk_strategy_override: str | None = None,
    ) -> PreviewResult:
        """Generate preview using the EXACT same processing as ingestion."""
        # Apply overrides to metadata
        if chunk_size_override is not None:
            metadata["chunk_size"] = chunk_size_override
        if chunk_overlap_override is not None:
            metadata["chunk_overlap"] = chunk_overlap_override
        if chunk_strategy_override is not None:
            metadata["chunk_strategy"] = chunk_strategy_override

        # Use SAME processing code as ingestion
        documents = self.blob_processor.process_blob(file_data, metadata)

        if not documents:
            raise ValueError("No chunks produced from document")

        # Extract full text for position mapping
        full_text = self._get_full_text(file_data, metadata, documents)

        # Compute positions (post-processing, doesn't affect chunks)
        preview_chunks = self._compute_positions(full_text, documents)

        # Extract parser from first document
        parser_used = (
            documents[0].metadata.get("parser", "unknown") if documents else "unknown"
        )

        # Get chunk settings from metadata or documents
        chunk_size = metadata.get(
            "chunk_size", documents[0].metadata.get("chunk_size", 500)
        )
        chunk_overlap = metadata.get(
            "chunk_overlap", documents[0].metadata.get("chunk_overlap", 50)
        )
        chunk_strategy = metadata.get("chunk_strategy", "characters")

        return PreviewResult(
            original_text=full_text,
            chunks=preview_chunks,
            file_info={
                "filename": metadata.get("filename", "unknown"),
                "size": len(file_data),
                "content_type": metadata.get("content_type"),
            },
            parser_used=parser_used,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            total_chunks=len(preview_chunks),
        )

    def _get_full_text(
        self, file_data: bytes, metadata: dict[str, Any], documents: list[Document]
    ) -> str:
        """Extract full text from file or from parsed documents."""
        filename = metadata.get("filename", "").lower()

        # For text-based files, decode directly
        text_extensions = {
            ".txt",
            ".md",
            ".markdown",
            ".rst",
            ".csv",
            ".json",
            ".yaml",
            ".yml",
        }
        if any(filename.endswith(ext) for ext in text_extensions):
            try:
                return file_data.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    return file_data.decode("latin-1")
                except UnicodeDecodeError:
                    pass

        # For binary files (PDF, DOCX, etc.), concatenate parsed chunks
        # This gives the best approximation of the full text
        return " ".join(doc.content for doc in documents)

    def _compute_positions(
        self, full_text: str, documents: list[Document]
    ) -> list[PreviewChunk]:
        """Map chunk content back to character positions in full text."""
        preview_chunks = []
        search_start = 0

        for i, doc in enumerate(documents):
            pos = full_text.find(doc.content, search_start)

            if pos >= 0:
                preview_chunks.append(
                    PreviewChunk(
                        chunk_index=i,
                        content=doc.content,
                        start_position=pos,
                        end_position=pos + len(doc.content),
                        metadata=doc.metadata,
                    )
                )
                # Account for overlap when searching for next chunk
                overlap = doc.metadata.get("chunk_overlap", 0)
                search_start = max(search_start, pos + len(doc.content) - overlap - 10)
            else:
                # Chunk not found in full text (rare edge case)
                preview_chunks.append(
                    PreviewChunk(
                        chunk_index=i,
                        content=doc.content,
                        start_position=-1,
                        end_position=-1,
                        metadata=doc.metadata,
                    )
                )

        return preview_chunks
