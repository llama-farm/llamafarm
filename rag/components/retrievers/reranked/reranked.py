"""Reranked strategy - sophisticated multi-factor relevance scoring."""

from pathlib import Path
from typing import List, Dict, Any
from components.retrievers.base import RetrievalStrategy, RetrievalResult
from core.base import Document


class RerankedStrategy(RetrievalStrategy):
    """Universal re-ranking strategy with configurable factors.

    This strategy performs an initial broad search and then re-ranks results
    using multiple factors like recency, content length, and metadata-based
    boosts. This provides more sophisticated relevance scoring than simple
    vector similarity alone.

    Use Cases:
    - Production systems requiring nuanced ranking beyond similarity
    - Time-sensitive content where recency matters
    - Multi-factor relevance scenarios (priority, type, etc.)
    - Content where metadata provides additional ranking signals

    Performance: Slower (due to initial broad search + reranking)
    Complexity: High
    """

    def __init__(
        self,
        name: str = "RerankedStrategy",
        config: Dict[str, Any] = None,
        project_dir: Path | None = None,
    ):
        super().__init__(name, config, project_dir)
        config = config or {}
        self.initial_k = config.get("initial_k", 20)  # Retrieve more, then re-rank
        self.rerank_factors = config.get(
            "rerank_factors",
            {
                "recency": 0.1,  # Boost recent documents
                "length": 0.05,  # Slight preference for longer documents
                "metadata_boost": 0.2,  # Boost based on metadata signals
            },
        )
        self.distance_metric = config.get("distance_metric", "cosine")
        self.length_normalization = config.get(
            "length_normalization", 1000
        )  # Characters to normalize by

    def retrieve(
        self, query_embedding: List[float], vector_store, top_k: int = 5, **kwargs
    ) -> RetrievalResult:
        """Search with re-ranking based on multiple factors.

        Args:
            query_embedding: The embedded query vector
            vector_store: The vector store to search
            top_k: Number of final results to return
            **kwargs: Additional arguments

        Returns:
            RetrievalResult with re-ranked documents and scores
        """
        # Get initial results (more than needed for effective re-ranking)
        initial_k = max(self.initial_k, top_k * 2)

        documents = vector_store.search(
            query_embedding=query_embedding, top_k=initial_k
        )

        candidates = []

        for doc in documents:
            # Base similarity score from vector search
            base_score = doc.metadata.get("similarity_score", 0.0)

            # Apply re-ranking factors to get final score
            final_score = self._rerank_score(base_score, doc.content, doc.metadata)

            # Store both scores in metadata for transparency
            doc.metadata["similarity_score"] = final_score
            doc.metadata["base_similarity_score"] = base_score
            doc.metadata["rerank_boost"] = final_score - base_score

            candidates.append((doc, final_score))

        # Sort by re-ranked score and take top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_k]

        documents = [doc for doc, _ in top_candidates]
        scores = [score for _, score in top_candidates]

        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={
                "strategy": self.name,
                "version": "1.0.0",
                "initial_k": initial_k,
                "rerank_factors": self.rerank_factors,
                "candidates_considered": len(candidates),
                "average_boost": sum(doc.metadata["rerank_boost"] for doc in documents)
                / len(documents)
                if documents
                else 0,
            },
        )

    def _rerank_score(
        self, base_score: float, content: str, metadata: Dict[str, Any]
    ) -> float:
        """Apply re-ranking factors to base similarity score.

        Args:
            base_score: Original similarity score from vector search
            content: Document content for length-based factors
            metadata: Document metadata for various boosts

        Returns:
            Re-ranked final score
        """
        final_score = base_score

        # Recency boost (if timestamp available)
        if "timestamp" in metadata and self.rerank_factors.get("recency", 0) > 0:
            recency_boost = self._calculate_recency_boost(metadata["timestamp"])
            final_score += self.rerank_factors["recency"] * recency_boost

        # Length factor (configurable preference for content length)
        if self.rerank_factors.get("length", 0) > 0:
            length_factor = min(len(content) / self.length_normalization, 1.0)
            length_boost = self.rerank_factors["length"] * length_factor
            final_score += length_boost

        # Metadata-based boosts
        if self.rerank_factors.get("metadata_boost", 0) > 0:
            metadata_boost = self._calculate_metadata_boost(metadata)
            final_score += self.rerank_factors["metadata_boost"] * metadata_boost

        return final_score

    def _calculate_recency_boost(self, timestamp: str) -> float:
        """Calculate recency boost based on document timestamp.

        Args:
            timestamp: Document timestamp (ISO format expected)

        Returns:
            Recency boost factor (0.0 to 1.0)
        """
        try:
            from datetime import datetime, timezone

            # Try to import dateutil, fall back gracefully if not available
            try:
                import dateutil.parser
            except ImportError:
                return 0.0  # No boost if dateutil not available

            # Parse the timestamp
            doc_time = dateutil.parser.parse(timestamp)
            current_time = datetime.now(timezone.utc)

            # Calculate age in days
            age_days = (current_time - doc_time).days

            # Apply decay function (newer = higher boost)
            if age_days <= 1:
                return 1.0  # Maximum boost for very recent
            elif age_days <= 7:
                return 0.8  # High boost for past week
            elif age_days <= 30:
                return 0.5  # Medium boost for past month
            elif age_days <= 90:
                return 0.2  # Low boost for past quarter
            else:
                return 0.0  # No boost for older content

        except (ImportError, ValueError, TypeError):
            # If timestamp parsing fails, no boost
            return 0.0

    def _calculate_metadata_boost(self, metadata: Dict[str, Any]) -> float:
        """Calculate boost based on metadata signals.

        Args:
            metadata: Document metadata dictionary

        Returns:
            Metadata boost factor
        """
        boost = 0.0

        # Priority-based boosts
        priority = metadata.get("priority", "").lower()
        if priority == "high":
            boost += 0.3
        elif priority == "medium":
            boost += 0.1
        elif priority == "critical":
            boost += 0.5

        # Type-based boosts
        doc_type = metadata.get("type", "").lower()
        if doc_type in ["documentation", "tutorial", "guide"]:
            boost += 0.2
        elif doc_type in ["faq", "troubleshooting"]:
            boost += 0.3
        elif doc_type in ["policy", "procedure"]:
            boost += 0.1

        # Category-based boosts
        category = metadata.get("category", "").lower()
        if category in ["important", "featured", "recommended"]:
            boost += 0.15

        # Quality indicators
        if metadata.get("verified", False):
            boost += 0.1
        if metadata.get("expert_reviewed", False):
            boost += 0.1

        # Usage statistics
        view_count = metadata.get("view_count", 0)
        if isinstance(view_count, (int, float)) and view_count > 0:
            # Normalize view count boost (log scale to prevent domination)
            import math

            view_boost = min(math.log10(view_count + 1) / 4, 0.2)  # Max 0.2 boost
            boost += view_boost

        return min(boost, 1.0)  # Cap total metadata boost at 1.0

    def supports_vector_store(self, vector_store_type: str) -> bool:
        """This is universal - supports all vector stores."""
        return True

    def validate_config(self) -> bool:
        """Validate strategy configuration."""
        if self.initial_k < 1:
            return False
        if self.length_normalization <= 0:
            return False

        # Validate rerank factors
        for factor, weight in self.rerank_factors.items():
            if not isinstance(weight, (int, float)) or weight < 0:
                return False

        return True

    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this strategy."""
        return {
            "type": "object",
            "properties": {
                "initial_k": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 20,
                    "description": "Number of initial candidates to retrieve before reranking",
                },
                "distance_metric": {
                    "type": "string",
                    "enum": ["cosine", "euclidean", "manhattan", "dot"],
                    "default": "cosine",
                },
                "length_normalization": {
                    "type": "number",
                    "minimum": 1,
                    "default": 1000,
                    "description": "Character count to normalize length factor by",
                },
                "rerank_factors": {
                    "type": "object",
                    "properties": {
                        "recency": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.1,
                            "description": "Weight for recency-based boosting",
                        },
                        "length": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.05,
                            "description": "Weight for content length factor",
                        },
                        "metadata_boost": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.2,
                            "description": "Weight for metadata-based boosts",
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        }

    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance characteristics of this strategy."""
        return {
            "speed": "slower",
            "memory_usage": "medium",
            "complexity": "high",
            "accuracy": "very_high",
            "best_for": [
                "production_systems",
                "time_sensitive_content",
                "multi_factor_ranking",
            ],
            "notes": f"Retrieves {self.initial_k} candidates then reranks - slower but more sophisticated",
        }
