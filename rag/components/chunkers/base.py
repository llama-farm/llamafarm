"""Base classes for text chunking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    CHARACTERS = "characters"      # Fixed character count
    SENTENCES = "sentences"        # Split on sentence boundaries
    PARAGRAPHS = "paragraphs"      # Split on paragraph boundaries
    SECTIONS = "sections"          # Split on markdown headers
    PAGES = "pages"               # Split on page boundaries (from metadata)
    TOKENS = "tokens"             # Token-based splitting (requires tokenizer)


@dataclass
class ChunkResult:
    """Result of chunking operation."""

    content: str
    metadata: dict = field(default_factory=dict)

    # Position info
    chunk_index: int = 0
    total_chunks: int = 1

    # Source position (for reconstruction)
    start_offset: int = 0
    end_offset: int = 0

    # Strategy used
    strategy: str = ""

    # Overlap info
    overlap_before: int = 0
    overlap_after: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "strategy": self.strategy,
            "overlap_before": self.overlap_before,
            "overlap_after": self.overlap_after,
        }


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 0,
        strategy: ChunkingStrategy = ChunkingStrategy.CHARACTERS,
        config: dict | None = None,
    ):
        """Initialize chunker.

        Args:
            chunk_size: Target size for chunks (meaning depends on strategy)
            overlap: Number of units to overlap between chunks
            strategy: Chunking strategy to use
            config: Additional configuration options
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self.config = config or {}

    @property
    def name(self) -> str:
        """Return chunker name."""
        return self.__class__.__name__

    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[ChunkResult]:
        """Split text into chunks.

        Args:
            text: Text content to chunk
            metadata: Optional metadata to include in chunks

        Returns:
            List of ChunkResult objects
        """
        pass

    def _build_chunk_metadata(
        self,
        base_metadata: dict | None,
        chunk_index: int,
        total_chunks: int,
    ) -> dict:
        """Build metadata for a chunk.

        Args:
            base_metadata: Input metadata to preserve
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks

        Returns:
            Metadata dictionary for the chunk
        """
        metadata = dict(base_metadata) if base_metadata else {}

        # Add chunking info
        metadata.update({
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunker_name": self.name,
            "chunking_strategy": self.strategy.value if isinstance(self.strategy, ChunkingStrategy) else str(self.strategy),
            "chunk_size_target": self.chunk_size,
            "chunk_overlap": self.overlap,
            "is_chunked": total_chunks > 1,
        })

        return metadata

    def __repr__(self) -> str:
        return f"{self.name}(chunk_size={self.chunk_size}, overlap={self.overlap}, strategy={self.strategy})"
