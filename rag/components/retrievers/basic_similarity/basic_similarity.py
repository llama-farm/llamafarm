"""Basic similarity retrieval strategy using cosine similarity."""

import logging
from typing import List, Dict, Any, Optional

from components.retrievers.base import RetrievalStrategy, RetrievalResult
from core.base import Document

logger = logging.getLogger(__name__)


class BasicSimilarityStrategy(RetrievalStrategy):
    """
    Basic retrieval strategy using simple vector similarity search.
    
    This strategy performs direct similarity search without any additional
    ranking or filtering.
    """
    
    def __init__(self, name: str = "BasicSimilarityStrategy", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        config = config or {}
        
        # Configuration options
        self.similarity_threshold = config.get("similarity_threshold", 0.0)
        self.include_metadata = config.get("include_metadata", True)
        self.max_results = config.get("max_results", 100)
    
    def retrieve(
        self,
        query_embedding: List[float],
        vector_store,
        top_k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve documents using basic similarity search.
        
        Args:
            query_embedding: The embedded query vector
            vector_store: The vector store to search in
            top_k: Maximum number of documents to return
            **kwargs: Additional parameters
            
        Returns:
            RetrievalResult with retrieved documents and scores
        """
        try:
            # Limit top_k to configured maximum
            effective_top_k = min(top_k, self.max_results)
            
            # Search vector store
            documents = vector_store.search(
                query="",  # Empty query since we provide embedding
                query_embedding=query_embedding,
                top_k=effective_top_k
            )
            
            # Extract scores from metadata if available
            scores = []
            filtered_docs = []
            
            for doc in documents:
                distance = doc.metadata.get('_score', float('inf')) if doc.metadata else float('inf')
                
                # Convert distance to similarity score (ChromaDB returns distances, lower is better)
                import math
                scale_factor = 100.0
                similarity_score = math.exp(-distance / scale_factor)
                
                # Apply similarity threshold (now comparing similarity scores)
                if similarity_score >= self.similarity_threshold:
                    scores.append(similarity_score)
                    
                    # Update metadata with both distance and similarity score
                    if doc.metadata:
                        doc.metadata['_similarity_score'] = similarity_score
                        # Keep original distance for reference
                        doc.metadata['_distance'] = distance
                    
                    # Clean up metadata if requested
                    if not self.include_metadata and doc.metadata:
                        # Remove internal scoring metadata but keep original metadata
                        cleaned_metadata = {k: v for k, v in doc.metadata.items() 
                                          if not k.startswith('_')}
                        doc.metadata = cleaned_metadata
                    
                    filtered_docs.append(doc)
            
            # Create result
            result = RetrievalResult(
                documents=filtered_docs,
                scores=scores,
                strategy_metadata={
                    "strategy": "basic_similarity",
                    "query_embedding_dim": len(query_embedding),
                    "similarity_threshold": self.similarity_threshold,
                    "requested_k": top_k,
                    "returned_count": len(filtered_docs)
                }
            )
            
            logger.debug(f"Retrieved {len(filtered_docs)} documents using basic similarity")
            return result
            
        except Exception as e:
            logger.error(f"Error in basic similarity retrieval: {e}")
            return RetrievalResult(
                documents=[],
                scores=[],
                strategy_metadata={
                    "strategy": "basic_similarity",
                    "error": str(e)
                }
            )
    
    def supports_vector_store(self, vector_store_type: str) -> bool:
        """Check if this strategy supports the given vector store type."""
        # Basic similarity works with any vector store that supports search
        supported_stores = [
            "ChromaStore", 
            "FAISSStore", 
            "PineconeStore", 
            "QdrantStore"
        ]
        return vector_store_type in supported_stores
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this strategy."""
        return {
            "name": self.name,
            "type": "basic_similarity",
            "description": "Simple vector similarity search without additional ranking",
            "supported_stores": [
                "ChromaStore", "FAISSStore", "PineconeStore", "QdrantStore"
            ],
            "parameters": {
                "similarity_threshold": {
                    "type": "float",
                    "default": 0.0,
                    "description": "Minimum similarity score for results"
                },
                "include_metadata": {
                    "type": "bool", 
                    "default": True,
                    "description": "Include document metadata in results"
                },
                "max_results": {
                    "type": "int",
                    "default": 100,
                    "description": "Maximum number of results to return"
                }
            }
        }