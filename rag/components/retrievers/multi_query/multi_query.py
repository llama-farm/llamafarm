"""Multi-query strategy - enhanced recall through query variations."""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
from rag.components.retrievers.base import RetrievalStrategy, RetrievalResult
from rag.core.base import Document


class MultiQueryStrategy(RetrievalStrategy):
    """Universal multi-query strategy using query variations.
    
    This strategy improves recall by using multiple query variations to search
    the vector database. It aggregates results from all queries using configurable
    methods like max, mean, or weighted averaging.
    
    Use Cases:
    - Ambiguous queries that might have multiple interpretations
    - Query expansion scenarios to capture more relevant documents
    - Complex questions that benefit from multiple search approaches
    - Improving recall when initial queries miss relevant content
    
    Performance: Medium (multiple searches required)
    Complexity: Medium
    """
    
    def __init__(self, name: str = "MultiQueryStrategy", config: Dict[str, Any] = None):
        super().__init__(name, config)
        config = config or {}
        self.num_queries = config.get("num_queries", 3)
        self.aggregation_method = config.get("aggregation_method", "max")  # max, mean, weighted
        self.distance_metric = config.get("distance_metric", "cosine")
        self.search_multiplier = config.get("search_multiplier", 2)  # How many extra results to get per query
    
    def retrieve(
        self,
        query_embedding: List[float],
        vector_store,
        top_k: int = 5,
        query_variations: Optional[List[List[float]]] = None,
        **kwargs
    ) -> RetrievalResult:
        """Search using multiple query variations to improve recall.
        
        Args:
            query_embedding: The primary embedded query vector
            vector_store: The vector store to search
            top_k: Number of final results to return
            query_variations: Optional list of pre-computed query variations
            **kwargs: Additional arguments
            
        Returns:
            RetrievalResult with aggregated documents and scores
        """
        # If no variations provided, use the original query multiple times
        # In practice, you'd generate variations using query expansion techniques
        if query_variations is None:
            query_variations = [query_embedding] * min(self.num_queries, 3)
        else:
            query_variations = [query_embedding] + query_variations[:self.num_queries-1]
        
        # Collect results from all query variations
        doc_scores = defaultdict(list)
        doc_objects = {}
        total_searches = 0
        
        for i, query_var in enumerate(query_variations):
            documents = vector_store.search(
                query_embedding=query_var,
                top_k=top_k * self.search_multiplier  # Get more results for aggregation
            )
            total_searches += 1
            
            for doc in documents:
                doc_id = doc.id or f"doc_{hash(doc.content[:100])}"  # Create stable ID
                score = doc.metadata.get("similarity_score", 0.0)
                
                doc_scores[doc_id].append(score)
                doc_objects[doc_id] = doc
        
        # Aggregate scores using the specified method
        aggregated_scores = {}
        for doc_id, scores in doc_scores.items():
            aggregated_scores[doc_id] = self._aggregate_scores(scores)
        
        # Sort and get top results
        sorted_docs = sorted(
            aggregated_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        documents = []
        scores = []
        
        for doc_id, score in sorted_docs:
            doc = doc_objects[doc_id]
            # Update the similarity score in metadata
            doc.metadata["similarity_score"] = score
            doc.metadata["query_frequency"] = len(doc_scores[doc_id])  # How many queries found this doc
            documents.append(doc)
            scores.append(score)
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={
                "strategy": self.name,
                "version": "1.0.0",
                "num_query_variations": len(query_variations),
                "aggregation_method": self.aggregation_method,
                "total_unique_docs": len(doc_scores),
                "total_searches": total_searches,
                "search_multiplier": self.search_multiplier
            }
        )
    
    def _aggregate_scores(self, scores: List[float]) -> float:
        """Aggregate multiple scores using the configured method.
        
        Args:
            scores: List of scores from different query variations
            
        Returns:
            Single aggregated score
        """
        if not scores:
            return 0.0
            
        if self.aggregation_method == "max":
            return max(scores)
        elif self.aggregation_method == "mean":
            return float(np.mean(scores))
        elif self.aggregation_method == "weighted":
            # Weight later queries less (assuming first query is most important)
            weights = [1.0 / (i + 1) for i in range(len(scores))]
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            weight_sum = sum(weights)
            return weighted_sum / weight_sum if weight_sum > 0 else 0.0
        else:
            # Default to mean if unknown method
            return float(np.mean(scores))
    
    def supports_vector_store(self, vector_store_type: str) -> bool:
        """This is universal - supports all vector stores."""
        return True
    
    def validate_config(self) -> bool:
        """Validate strategy configuration."""
        if self.num_queries < 1:
            return False
        if self.aggregation_method not in ["max", "mean", "weighted"]:
            return False
        if self.search_multiplier < 1:
            return False
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this strategy."""
        return {
            "type": "object",
            "properties": {
                "num_queries": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                    "description": "Number of query variations to use"
                },
                "aggregation_method": {
                    "type": "string",
                    "enum": ["max", "mean", "weighted"],
                    "default": "max",
                    "description": "How to combine scores from multiple queries"
                },
                "distance_metric": {
                    "type": "string",
                    "enum": ["cosine", "euclidean", "manhattan", "dot"],
                    "default": "cosine"
                },
                "search_multiplier": {
                    "type": "number",
                    "minimum": 1,
                    "default": 2,
                    "description": "Multiplier for how many results to fetch per query"
                }
            },
            "additionalProperties": False
        }
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance characteristics of this strategy."""
        return {
            "speed": "medium",
            "memory_usage": "medium",
            "complexity": "medium",
            "accuracy": "high",
            "best_for": ["ambiguous_queries", "recall_optimization", "complex_questions"],
            "notes": f"Performs {self.num_queries} searches, good for improving recall"
        }