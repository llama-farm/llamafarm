"""Basic similarity retrieval strategy."""

import logging
import math
from typing import Any

from ..models import Document, RetrievalResult
from .base import RetrievalStrategy

logger = logging.getLogger(__name__)


class BasicSimilarityStrategy(RetrievalStrategy):
    """Basic retrieval using simple vector similarity search."""

    def __init__(
        self,
        name: str = "BasicSimilarityStrategy",
        config: dict[str, Any] | None = None,
    ):
        super().__init__(name, config)
        config = config or {}

        # Configuration options
        self.similarity_threshold = config.get("similarity_threshold", 0.0)
        self.include_metadata = config.get("include_metadata", True)
        self.max_results = config.get("max_results", 100)

    def retrieve(
        self,
        query_embedding: list[float],
        vector_store: Any,
        top_k: int = 5,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve documents using basic similarity search.

        Args:
            query_embedding: The embedded query vector
            vector_store: The vector store to search in
            top_k: Maximum number of documents to return
            **kwargs: Additional parameters

        Returns:
            RetrievalResult with retrieved documents and scores
        """
        # Limit top_k to configured maximum
        effective_top_k = min(top_k, self.max_results)

        try:
            # Search vector store
            documents = vector_store.search(
                query="",  # Empty query since we provide embedding
                query_embedding=query_embedding,
                top_k=effective_top_k,
            )

            # Extract scores from metadata
            scores = []
            filtered_docs = []

            for doc in documents:
                # Try to get similarity score
                similarity_score = None
                if doc.metadata:
                    similarity_score = doc.metadata.get("similarity_score")

                # If no direct similarity score, convert from distance
                if similarity_score is None:
                    distance = (
                        doc.metadata.get("_score", float("inf"))
                        if doc.metadata
                        else float("inf")
                    )

                    # Convert distance to similarity score
                    scale_factor = 100.0
                    similarity_score = math.exp(-distance / scale_factor)

                    # Update metadata
                    if doc.metadata:
                        doc.metadata["_similarity_score"] = similarity_score
                        doc.metadata["_distance"] = distance

                # Apply similarity threshold
                if similarity_score >= self.similarity_threshold:
                    scores.append(similarity_score)

                    # Clean up metadata if requested
                    if not self.include_metadata and doc.metadata:
                        cleaned_metadata = {
                            k: v
                            for k, v in doc.metadata.items()
                            if not k.startswith("_")
                        }
                        doc.metadata = cleaned_metadata

                    filtered_docs.append(doc)

            # Create result
            result = RetrievalResult(
                documents=filtered_docs,
                scores=scores,
                strategy_metadata={
                    "strategy": "BasicSimilarityStrategy",
                    "version": "1.0.0",
                    "query_embedding_dim": len(query_embedding),
                    "similarity_threshold": self.similarity_threshold,
                    "requested_k": top_k,
                    "returned_count": len(filtered_docs),
                },
            )

            logger.debug(f"Retrieved {len(filtered_docs)} documents")
            return result

        except Exception as e:
            logger.error(f"Error in basic similarity retrieval: {e}")
            raise

    def supports_vector_store(self, vector_store_type: str) -> bool:
        """Check if this strategy supports the given vector store type."""
        # Basic similarity works with any vector store that supports search
        return True
