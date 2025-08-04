"""Basic similarity strategy - universal vector similarity search."""

from typing import List, Dict, Any
from ...base import RetrievalStrategy, RetrievalResult
from core.base import Document


class BasicSimilarityStrategy(RetrievalStrategy):
    """Universal basic vector similarity search strategy.
    
    This is the simplest and most universal retrieval strategy. It performs
    pure vector similarity search using the vector store's native search method.
    Works with any vector database and provides a reliable baseline.
    
    Use Cases:
    - Getting started with RAG
    - Simple semantic search  
    - Baseline performance testing
    - When you need fast, straightforward results
    
    Performance: Fast
    Complexity: Low
    """
    
    def __init__(self, name: str = "BasicSimilarityStrategy", config: Dict[str, Any] = None):
        super().__init__(name, config)
        config = config or {}
        self.distance_metric = config.get("distance_metric", "cosine")
    
    def retrieve(
        self,
        query_embedding: List[float],
        vector_store,
        top_k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        """Basic vector similarity search - delegates to vector store.
        
        Args:
            query_embedding: The embedded query vector
            vector_store: The vector store to search
            top_k: Number of results to return
            **kwargs: Additional arguments (ignored for basic strategy)
            
        Returns:
            RetrievalResult with documents and similarity scores
        """
        # Use the vector store's native search method
        documents = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Extract scores from metadata
        scores = []
        for doc in documents:
            score = doc.metadata.get("similarity_score", 0.0)
            scores.append(score)
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={
                "strategy": self.name,
                "version": "1.0.0",
                "distance_metric": self.distance_metric,
                "total_results": len(documents),
                "method": "vector_similarity"
            }
        )
    
    def supports_vector_store(self, vector_store_type: str) -> bool:
        """This is universal - supports all vector stores."""
        return True
    
    def validate_config(self) -> bool:
        """Validate strategy configuration."""
        # Basic strategy has minimal requirements
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this strategy."""
        return {
            "type": "object",
            "properties": {
                "distance_metric": {
                    "type": "string",
                    "enum": ["cosine", "euclidean", "manhattan", "dot"],
                    "default": "cosine",
                    "description": "Distance metric for similarity calculation"
                }
            },
            "additionalProperties": False
        }
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance characteristics of this strategy."""
        return {
            "speed": "fast",
            "memory_usage": "low", 
            "complexity": "low",
            "accuracy": "good",
            "best_for": ["simple_queries", "baseline_testing", "getting_started"]
        }