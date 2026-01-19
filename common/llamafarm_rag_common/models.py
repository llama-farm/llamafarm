"""Shared data models for RAG components."""

from dataclasses import dataclass, field
from typing import Any

# Type alias for embedding vectors
EmbeddingVector = list[float]


@dataclass
class Document:
    """RAG document model.

    Represents a document chunk with content, metadata, and optional embeddings.
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embeddings: EmbeddingVector | None = None

    def __post_init__(self):
        """Ensure metadata is a dict."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalResult:
    """Result from a retrieval strategy.

    Contains documents, scores, and metadata about the retrieval process.
    """

    documents: list[Document]
    scores: list[float]
    strategy_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate that documents and scores have same length."""
        if len(self.documents) != len(self.scores):
            raise ValueError(
                f"Documents and scores must have same length: "
                f"{len(self.documents)} != {len(self.scores)}"
            )
