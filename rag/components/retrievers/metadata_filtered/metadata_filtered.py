"""Metadata filtered strategy - vector search with intelligent filtering."""

from typing import List, Dict, Any, Optional
from rag.components.retrievers.base import RetrievalStrategy, RetrievalResult
from rag.core.base import Document


class MetadataFilteredStrategy(RetrievalStrategy):
    """Universal metadata filtering strategy.
    
    Performs vector search with intelligent metadata filtering. Automatically
    detects if the vector store supports native filtering and falls back to
    post-search filtering when necessary.
    
    Use Cases:
    - Domain-specific searches (e.g., only documentation)
    - Multi-tenant applications with user-specific filtering
    - Content categorization and filtering
    - Restricting search scope by metadata attributes
    
    Performance: Medium (depends on filtering efficiency)
    Complexity: Medium
    """
    
    def __init__(self, name: str = "MetadataFilteredStrategy", config: Dict[str, Any] = None):
        super().__init__(name, config)
        config = config or {}
        self.default_filters = config.get("default_filters", {})
        self.distance_metric = config.get("distance_metric", "cosine")
        self.fallback_multiplier = config.get("fallback_multiplier", 3)  # Get 3x more docs when filtering
    
    def retrieve(
        self,
        query_embedding: List[float],
        vector_store,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """Search with metadata filtering.
        
        Args:
            query_embedding: The embedded query vector
            vector_store: The vector store to search
            top_k: Number of results to return
            metadata_filter: Additional filters to apply
            **kwargs: Additional arguments
            
        Returns:
            RetrievalResult with filtered documents and scores
        """
        # Combine default and provided filters
        filters = {**self.default_filters}
        if metadata_filter:
            filters.update(metadata_filter)
        
        # Check if vector store supports native filtering
        if hasattr(vector_store, 'search_with_filter') and filters:
            # Use native filtering
            documents = vector_store.search_with_filter(
                query_embedding=query_embedding,
                top_k=top_k,
                metadata_filter=filters
            )
            filtering_method = "native"
        else:
            # Fallback: search then filter
            search_k = top_k * self.fallback_multiplier if filters else top_k
            documents = vector_store.search(
                query_embedding=query_embedding,
                top_k=search_k
            )
            
            if filters:
                documents = self._filter_documents(documents, filters)[:top_k]
                filtering_method = "post_search"
            else:
                documents = documents[:top_k]
                filtering_method = "none"
        
        scores = [doc.metadata.get("similarity_score", 0.0) for doc in documents]
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={
                "strategy": self.name,
                "version": "1.0.0",
                "filters_applied": filters,
                "filtering_method": filtering_method,
                "total_results": len(documents),
                "fallback_multiplier": self.fallback_multiplier
            }
        )
    
    def _filter_documents(self, documents: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """Filter documents by metadata - universal implementation.
        
        Args:
            documents: List of documents to filter
            filters: Dictionary of filter criteria
            
        Returns:
            Filtered list of documents
        """
        filtered = []
        for doc in documents:
            if self._matches_filters(doc, filters):
                filtered.append(doc)
        return filtered
    
    def _matches_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """Check if document matches all filter criteria.
        
        Args:
            doc: Document to check
            filters: Filter criteria
            
        Returns:
            True if document matches all filters
        """
        for key, value in filters.items():
            if key not in doc.metadata:
                return False
            
            doc_value = doc.metadata[key]
            
            if isinstance(value, list):
                # List means "in" operation
                if doc_value not in value:
                    return False
            elif isinstance(value, dict):
                # Handle operators like {"$ne": "value", "$in": ["a", "b"]}
                if "$ne" in value and doc_value == value["$ne"]:
                    return False
                if "$in" in value and doc_value not in value["$in"]:
                    return False
                if "$nin" in value and doc_value in value["$nin"]:
                    return False
                if "$gt" in value and not (doc_value > value["$gt"]):
                    return False
                if "$gte" in value and not (doc_value >= value["$gte"]):
                    return False
                if "$lt" in value and not (doc_value < value["$lt"]):
                    return False
                if "$lte" in value and not (doc_value <= value["$lte"]):
                    return False
            else:
                # Direct equality
                if doc_value != value:
                    return False
        
        return True
    
    def supports_vector_store(self, vector_store_type: str) -> bool:
        """This is universal - supports all vector stores."""
        return True
    
    def validate_config(self) -> bool:
        """Validate strategy configuration."""
        # Check that filters are properly formatted
        try:
            self._validate_filters(self.default_filters)
            return True
        except ValueError:
            return False
    
    def _validate_filters(self, filters: Dict[str, Any]) -> None:
        """Validate filter format."""
        for key, value in filters.items():
            if not isinstance(key, str):
                raise ValueError(f"Filter key must be string, got {type(key)}")
            
            if isinstance(value, dict):
                # Validate operators
                valid_ops = {"$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"}
                for op in value.keys():
                    if op not in valid_ops:
                        raise ValueError(f"Invalid filter operator: {op}")
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this strategy."""
        return {
            "type": "object",
            "properties": {
                "distance_metric": {
                    "type": "string",
                    "enum": ["cosine", "euclidean", "manhattan", "dot"],
                    "default": "cosine"
                },
                "default_filters": {
                    "type": "object",
                    "description": "Default filters applied to all searches",
                    "default": {}
                },
                "fallback_multiplier": {
                    "type": "number",
                    "minimum": 1,
                    "default": 3,
                    "description": "Multiplier for search results when post-filtering is needed"
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
            "accuracy": "good",
            "best_for": ["filtered_search", "multi_tenant", "categorized_content"],
            "notes": "Performance depends on filtering efficiency and native database support"
        }