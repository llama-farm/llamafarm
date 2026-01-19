"""Base retrieval strategy for shared RAG library."""

from abc import ABC, abstractmethod
from typing import Any

from ..models import Document, RetrievalResult


class RetrievalStrategy(ABC):
    """Base class for retrieval strategies."""

    def __init__(
        self,
        name: str,
        config: dict[str, Any] | None = None,
    ):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def retrieve(
        self,
        query_embedding: list[float],
        vector_store: Any,
        top_k: int = 5,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve documents using this strategy.

        Args:
            query_embedding: The embedded query vector
            vector_store: The vector store to search
            top_k: Maximum number of documents to return
            **kwargs: Additional strategy-specific parameters

        Returns:
            RetrievalResult with documents, scores, and metadata
        """
        pass

    @abstractmethod
    def supports_vector_store(self, vector_store_type: str) -> bool:
        """Check if this strategy supports the given vector store type.

        Args:
            vector_store_type: Type of vector store (e.g., "ChromaStore")

        Returns:
            True if supported, False otherwise
        """
        pass
