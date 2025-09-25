"""Base classes for retrieval strategies."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.base import Document, Component


@dataclass
class RetrievalResult:
    """Result from a retrieval strategy."""

    documents: List[Document]
    scores: List[float]
    strategy_metadata: Dict[str, Any]


class RetrievalStrategy(Component, ABC):
    """Base class for all retrieval strategies.

    Retrieval strategies define how to search and rank documents
    from a vector store given a query embedding.
    """

    def __init__(
        self, name: str, config: Dict[str, Any] = None, project_dir: Path | None = None
    ):
        super().__init__(name, config, project_dir)
        self.config = config or {}

    def process(self, data: Any) -> Any:
        """Process method required by Component base class."""
        # Retrieval strategies don't use the standard process method
        # They use the retrieve method instead
        raise NotImplementedError(
            "Use retrieve() method instead of process() for retrieval strategies"
        )

    @abstractmethod
    def retrieve(
        self, query_embedding: List[float], vector_store, top_k: int = 5, **kwargs
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


class HybridRetrievalStrategy(RetrievalStrategy):
    """Base class for hybrid retrieval strategies that combine multiple approaches."""

    def __init__(
        self, name: str, config: Dict[str, Any] = None, project_dir: Path | None = None
    ):
        super().__init__(name, config, project_dir)
        config = config or {}
        self.strategies = []
        self.weights = config.get("weights", [])

    def process(self, data: Any) -> Any:
        """Process method required by Component base class."""
        # Hybrid strategies don't use the standard process method
        raise NotImplementedError(
            "Use retrieve() method instead of process() for retrieval strategies"
        )

    def add_strategy(self, strategy: RetrievalStrategy, weight: float = 1.0):
        """Add a sub-strategy with optional weighting."""
        self.strategies.append(strategy)
        if len(self.weights) < len(self.strategies):
            self.weights.append(weight)

    def combine_results(
        self, results: List[RetrievalResult], top_k: int
    ) -> RetrievalResult:
        """Combine results from multiple strategies."""
        # Simple score combination - override in subclasses for more sophisticated merging
        doc_scores = {}
        all_docs = {}

        for i, result in enumerate(results):
            weight = self.weights[i] if i < len(self.weights) else 1.0

            for doc, score in zip(result.documents, result.scores):
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = 0
                    all_docs[doc.id] = doc

                doc_scores[doc.id] += weight * score

        # Sort by combined score
        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        return RetrievalResult(
            documents=[all_docs[doc_id] for doc_id, _ in sorted_items],
            scores=[score for _, score in sorted_items],
            strategy_metadata={
                "strategy": self.name,
                "num_sub_strategies": len(self.strategies),
                "weights": self.weights,
            },
        )
