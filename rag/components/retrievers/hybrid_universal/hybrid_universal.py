"""Hybrid universal strategy - combines multiple strategies with configurable weights."""

from pathlib import Path
from typing import Any

from components.retrievers.base import (
    HybridRetrievalStrategy,
    RetrievalResult,
)
from core.base import Document


class HybridUniversalStrategy(HybridRetrievalStrategy):
    """Universal hybrid strategy combining multiple approaches.

    This strategy combines multiple retrieval strategies with configurable weights
    to balance different aspects of search quality. It can mix basic similarity,
    metadata filtering, multi-query, and reranking approaches to achieve optimal
    results for complex use cases.

    Use Cases:
    - Production systems requiring balanced precision and recall
    - Complex requirements that benefit from multiple search approaches
    - Scenarios where different strategies excel at different query types
    - When you want to hedge against the weaknesses of any single strategy

    Performance: Variable (depends on sub-strategies used)
    Complexity: High
    """

    def __init__(
        self,
        name: str = "HybridUniversalStrategy",
        config: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ):
        super().__init__(name, config, project_dir)
        config = config or {}

        # Initialize sub-strategies based on config
        strategies_config = config.get(
            "strategies",
            [
                {"type": "BasicSimilarityStrategy", "weight": 0.6},
                {"type": "MetadataFilteredStrategy", "weight": 0.4},
            ],
        )

        self.combination_method = config.get(
            "combination_method", "weighted_average"
        )  # weighted_average, rank_fusion
        self.normalize_scores = config.get("normalize_scores", True)
        self.diversity_boost = config.get(
            "diversity_boost", 0.0
        )  # Boost for result diversity

        # Create strategy instances
        self._initialize_strategies(strategies_config)

    def _initialize_strategies(self, strategies_config: list[dict[str, Any]]) -> None:
        """Initialize sub-strategies from configuration.

        Args:
            strategies_config: List of strategy configurations with types and weights
        """
        # Import strategies dynamically to avoid circular imports
        from components.retrievers.basic_similarity.basic_similarity import (
            BasicSimilarityStrategy,
        )
        from components.retrievers.metadata_filtered.metadata_filtered import (
            MetadataFilteredStrategy,
        )
        from components.retrievers.multi_query.multi_query import MultiQueryStrategy
        from components.retrievers.reranked.reranked import RerankedStrategy

        # Map strategy types to classes
        strategy_classes = {
            "BasicSimilarityStrategy": BasicSimilarityStrategy,
            "MetadataFilteredStrategy": MetadataFilteredStrategy,
            "MultiQueryStrategy": MultiQueryStrategy,
            "RerankedStrategy": RerankedStrategy,
            # Aliases for convenience
            "basic": BasicSimilarityStrategy,
            "filtered": MetadataFilteredStrategy,
            "multi_query": MultiQueryStrategy,
            "reranked": RerankedStrategy,
        }

        for strategy_config in strategies_config:
            strategy_type = strategy_config["type"]
            weight = strategy_config.get("weight", 1.0)
            strategy_params = strategy_config.get("config", {})

            # Create strategy instance
            if strategy_type in strategy_classes:
                strategy = strategy_classes[strategy_type](config=strategy_params)
                self.add_strategy(strategy, weight)
            else:
                print(
                    f"Warning: Unknown strategy type '{strategy_type}' in hybrid configuration"
                )

    def retrieve(
        self, query_embedding: list[float], vector_store, top_k: int = 5, **kwargs
    ) -> RetrievalResult:
        """Combine results from multiple strategies.

        Args:
            query_embedding: The embedded query vector
            vector_store: The vector store to search
            top_k: Number of final results to return
            **kwargs: Additional arguments passed to sub-strategies

        Returns:
            RetrievalResult with combined and weighted documents
        """
        if not self.strategies:
            # Fallback to basic strategy if no strategies configured
            from components.retrievers.basic_similarity.basic_similarity import (
                BasicSimilarityStrategy,
            )

            basic_strategy = BasicSimilarityStrategy()
            return basic_strategy.retrieve(
                query_embedding, vector_store, top_k, **kwargs
            )

        # Get results from all strategies
        results = []
        strategy_performances: dict[str, dict[str, Any]] = {}

        for strategy in self.strategies:
            try:
                # Get more results from each strategy for better combination
                result = strategy.retrieve(
                    query_embedding, vector_store, top_k * 2, **kwargs
                )
                results.append(result)

                # Track strategy performance
                strategy_performances[strategy.name] = {
                    "documents_found": len(result.documents),
                    "average_score": sum(result.scores) / len(result.scores)
                    if result.scores
                    else 0.0,
                    "success": True,
                }

            except Exception as e:
                # Log error and continue with other strategies
                self.logger.error(f"Strategy {strategy.name} failed: {e}")
                strategy_performances[strategy.name] = {
                    "success": False,
                    "error": str(e),
                }
                continue

        if not results:
            # No strategies succeeded
            return RetrievalResult(
                documents=[],
                scores=[],
                strategy_metadata={
                    "strategy": self.name,
                    "version": "1.0.0",
                    "error": "All sub-strategies failed",
                    "strategy_performances": strategy_performances,
                },
            )

        # Combine results using the specified method
        if self.combination_method == "rank_fusion":
            combined_result = self._rank_fusion_combine(results, top_k)
        else:
            # Default to weighted average
            combined_result = self.combine_results(results, top_k)

        # Add hybrid-specific metadata
        combined_result.strategy_metadata.update(
            {
                "version": "1.0.0",
                "combination_method": self.combination_method,
                "num_strategies": len(self.strategies),
                "strategy_performances": strategy_performances,
                "normalize_scores": self.normalize_scores,
                "diversity_boost": self.diversity_boost,
            }
        )

        return combined_result

    def _rank_fusion_combine(
        self, results: list[RetrievalResult], top_k: int
    ) -> RetrievalResult:
        """Combine results using reciprocal rank fusion.

        This method gives equal weight to the ranking position from each strategy,
        which can be more robust than score-based combination when strategies
        have different score distributions.

        Args:
            results: List of RetrievalResult objects from different strategies
            top_k: Number of final results to return

        Returns:
            Combined RetrievalResult
        """
        from collections import defaultdict

        doc_fusion_scores: defaultdict[str, float] = defaultdict(float)
        doc_objects: dict[str, Document] = {}

        for i, result in enumerate(results):
            strategy_weight = self.weights[i] if i < len(self.weights) else 1.0

            for rank, doc in enumerate(result.documents):
                doc_id = doc.id or f"doc_{hash(doc.content[:100])}"

                # Reciprocal rank fusion formula: weight / (rank + k)
                # Using k=60 as commonly recommended
                fusion_score = strategy_weight / (rank + 60)
                doc_fusion_scores[doc_id] += fusion_score
                doc_objects[doc_id] = doc

        # Sort by fusion score and get top results
        sorted_docs = sorted(
            doc_fusion_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        documents = []
        scores = []

        for doc_id, fusion_score in sorted_docs:
            doc = doc_objects[doc_id]
            doc.metadata["fusion_score"] = fusion_score
            documents.append(doc)
            scores.append(fusion_score)

        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={
                "strategy": self.name,
                "version": "1.0.0",
                "combination_method": "rank_fusion",
                "fusion_k": 60,
            },
        )

    def _apply_diversity_boost(
        self, documents: list[Document], scores: list[float]
    ) -> tuple[list[Document], list[float]]:
        """Apply diversity boost to reduce redundancy in results.

        Args:
            documents: List of documents
            scores: Corresponding scores

        Returns:
            Tuple of (documents, scores) with diversity adjustments
        """
        if self.diversity_boost <= 0 or len(documents) <= 1:
            return documents, scores

        # Simple diversity heuristic: reduce scores for very similar content
        adjusted_docs = []
        adjusted_scores = []
        seen_content_hashes = set()

        for doc, score in zip(documents, scores):
            # Create a simple content hash for similarity detection
            content_hash = hash(doc.content[:200])  # Use first 200 chars

            if content_hash in seen_content_hashes:
                # Apply diversity penalty
                adjusted_score = score * (1 - self.diversity_boost)
            else:
                adjusted_score = score
                seen_content_hashes.add(content_hash)

            adjusted_docs.append(doc)
            adjusted_scores.append(adjusted_score)

        # Re-sort by adjusted scores
        combined = list(zip(adjusted_docs, adjusted_scores))
        combined.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in combined], [score for _, score in combined]

    def supports_vector_store(self, vector_store_type: str) -> bool:
        """Check if all sub-strategies support the vector store."""
        if not self.strategies:
            return True  # Empty hybrid strategy is universal

        return all(
            strategy.supports_vector_store(vector_store_type)
            for strategy in self.strategies
        )

    def validate_config(self) -> bool:
        """Validate hybrid strategy configuration."""
        # Check that we have at least one strategy
        if not self.strategies:
            return False

        # Check that all weights are positive
        for weight in self.weights:
            if weight <= 0:
                return False

        # Check combination method
        if self.combination_method not in ["weighted_average", "rank_fusion"]:
            return False

        # Check diversity boost range
        if not (0 <= self.diversity_boost <= 1):
            return False

        # Validate all sub-strategies
        return all(strategy.validate_config() for strategy in self.strategies)

    def get_config_schema(self) -> dict[str, Any]:
        """Get configuration schema for this strategy."""
        return {
            "type": "object",
            "properties": {
                "combination_method": {
                    "type": "string",
                    "enum": ["weighted_average", "rank_fusion"],
                    "default": "weighted_average",
                    "description": "Method for combining results from multiple strategies",
                },
                "normalize_scores": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to normalize scores before combination",
                },
                "diversity_boost": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.0,
                    "description": "Boost factor for result diversity (reduces redundancy)",
                },
                "strategies": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "BasicSimilarityStrategy",
                                    "MetadataFilteredStrategy",
                                    "MultiQueryStrategy",
                                    "RerankedStrategy",
                                    "basic",
                                    "filtered",
                                    "multi_query",
                                    "reranked",
                                ],
                            },
                            "weight": {
                                "type": "number",
                                "minimum": 0,
                                "exclusiveMinimum": True,
                            },
                            "config": {
                                "type": "object",
                                "description": "Configuration for the individual strategy",
                            },
                        },
                        "required": ["type", "weight"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["strategies"],
            "additionalProperties": False,
        }

    def get_performance_info(self) -> dict[str, Any]:
        """Get performance characteristics of this strategy."""
        if not self.strategies:
            return {
                "speed": "unknown",
                "memory_usage": "unknown",
                "complexity": "high",
                "accuracy": "unknown",
                "best_for": ["production", "balanced_results", "complex_requirements"],
                "notes": "No strategies configured",
            }

        # Aggregate performance info from sub-strategies
        speeds = []
        complexities = []

        for strategy in self.strategies:
            perf_info = strategy.get_performance_info()
            speeds.append(perf_info.get("speed", "medium"))
            complexities.append(perf_info.get("complexity", "medium"))

        # Determine overall characteristics
        if "slower" in speeds or len(self.strategies) > 2:
            overall_speed = "slower"
        elif "medium" in speeds:
            overall_speed = "medium"
        else:
            overall_speed = "fast"

        return {
            "speed": overall_speed,
            "memory_usage": "high",  # Multiple strategies = more memory
            "complexity": "high",
            "accuracy": "very_high",
            "best_for": ["production", "balanced_results", "complex_requirements"],
            "notes": f"Combines {len(self.strategies)} strategies using {self.combination_method}",
        }
