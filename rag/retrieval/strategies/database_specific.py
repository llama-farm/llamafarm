"""ChromaDB-specific retrieval strategies."""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .base import RetrievalStrategy, RetrievalResult, HybridRetrievalStrategy
from core.base import Document


class ChromaBasicStrategy(RetrievalStrategy):
    """Basic vector similarity search for ChromaDB."""
    
    def __init__(self, name: str = "ChromaBasicStrategy", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.distance_metric = config.get("distance_metric", "cosine")
    
    def retrieve(
        self,
        query_embedding: List[float],
        vector_store,
        top_k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        """Basic vector similarity search."""
        # Use ChromaDB's native search
        results = vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = []
        scores = []
        
        if results["documents"] and results["documents"][0]:
            for i, (content, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0] or [{}] * len(results["documents"][0]),
                results["distances"][0]
            )):
                # Convert distance to similarity score (higher is better)
                if self.distance_metric == "cosine":
                    score = 1.0 - distance  # Cosine distance to similarity
                else:
                    score = -distance  # Negative distance for ranking
                
                # Create document
                doc = Document(
                    id=metadata.get("id", f"doc_{i}"),
                    content=content,
                    metadata={**metadata, "similarity_score": score},
                    source=metadata.get("source")
                )
                
                documents.append(doc)
                scores.append(score)
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={
                "strategy": self.name,
                "distance_metric": self.distance_metric,
                "total_results": len(documents)
            }
        )
    
    def supports_vector_store(self, vector_store_type: str) -> bool:
        return vector_store_type == "ChromaStore"


class ChromaMetadataFilterStrategy(RetrievalStrategy):
    """ChromaDB strategy with metadata filtering."""
    
    def __init__(self, name: str = "ChromaMetadataFilterStrategy", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.default_filters = config.get("default_filters", {})
        self.distance_metric = config.get("distance_metric", "cosine")
    
    def retrieve(
        self,
        query_embedding: List[float],
        vector_store,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """Search with metadata filtering."""
        # Combine default and provided filters
        filters = {**self.default_filters}
        if metadata_filter:
            filters.update(metadata_filter)
        
        # Build Chroma where clause
        where_clause = None
        if filters:
            where_clause = self._build_where_clause(filters)
        
        # Search with filtering
        results = vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = []
        scores = []
        
        if results["documents"] and results["documents"][0]:
            for i, (content, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0] or [{}] * len(results["documents"][0]),
                results["distances"][0]
            )):
                # Convert distance to similarity score
                if self.distance_metric == "cosine":
                    score = 1.0 - distance
                else:
                    score = -distance
                
                doc = Document(
                    id=metadata.get("id", f"doc_{i}"),
                    content=content,
                    metadata={**metadata, "similarity_score": score},
                    source=metadata.get("source")
                )
                
                documents.append(doc)
                scores.append(score)
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={
                "strategy": self.name,
                "filters_applied": filters,
                "total_results": len(documents)
            }
        )
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                # IN operator for lists
                conditions.append({"key": key, "in": value})
            elif isinstance(value, dict) and "$in" in value:
                # Explicit $in operator
                conditions.append({"key": key, "in": value["$in"]})
            elif isinstance(value, dict) and "$ne" in value:
                # Not equal
                conditions.append({"key": key, "ne": value["$ne"]})
            else:
                # Direct equality
                conditions.append({"key": key, "match": {"value": value}})
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        else:
            return {}
    
    def supports_vector_store(self, vector_store_type: str) -> bool:
        return vector_store_type == "ChromaStore"


class ChromaMultiQueryStrategy(RetrievalStrategy):
    """ChromaDB strategy that uses multiple query variations."""
    
    def __init__(self, name: str = "ChromaMultiQueryStrategy", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.num_queries = config.get("num_queries", 3)
        self.aggregation_method = config.get("aggregation_method", "max")  # max, mean, weighted
        self.distance_metric = config.get("distance_metric", "cosine")
    
    def retrieve(
        self,
        query_embedding: List[float],
        vector_store,
        top_k: int = 5,
        query_variations: Optional[List[List[float]]] = None,
        **kwargs
    ) -> RetrievalResult:
        """Search using multiple query variations."""
        # If no variations provided, use the original query multiple times
        # In practice, you'd generate variations using query expansion
        if query_variations is None:
            query_variations = [query_embedding]
        else:
            query_variations = [query_embedding] + query_variations
        
        # Collect results from all query variations
        all_results = []
        for i, query_var in enumerate(query_variations[:self.num_queries]):
            results = vector_store.collection.query(
                query_embeddings=[query_var],
                n_results=top_k * 2,  # Get more results for aggregation
                include=["documents", "metadatas", "distances"]
            )
            all_results.append(results)
        
        # Aggregate results
        doc_scores = defaultdict(list)
        doc_metadata = {}
        doc_content = {}
        
        for results in all_results:
            if results["documents"] and results["documents"][0]:
                for content, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0] or [{}] * len(results["documents"][0]),
                    results["distances"][0]
                ):
                    doc_id = metadata.get("id", content[:50])  # Use content prefix as fallback ID
                    
                    # Convert distance to similarity score
                    if self.distance_metric == "cosine":
                        score = 1.0 - distance
                    else:
                        score = -distance
                    
                    doc_scores[doc_id].append(score)
                    doc_metadata[doc_id] = metadata
                    doc_content[doc_id] = content
        
        # Aggregate scores
        aggregated_scores = {}
        for doc_id, scores in doc_scores.items():
            if self.aggregation_method == "max":
                aggregated_scores[doc_id] = max(scores)
            elif self.aggregation_method == "mean":
                aggregated_scores[doc_id] = np.mean(scores)
            elif self.aggregation_method == "weighted":
                # Weight later queries less
                weights = [1.0 / (i + 1) for i in range(len(scores))]
                weighted_sum = sum(s * w for s, w in zip(scores, weights))
                weight_sum = sum(weights)
                aggregated_scores[doc_id] = weighted_sum / weight_sum if weight_sum > 0 else 0
        
        # Sort and get top results
        sorted_docs = sorted(
            aggregated_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        documents = []
        scores = []
        
        for doc_id, score in sorted_docs:
            doc = Document(
                id=doc_id,
                content=doc_content[doc_id],
                metadata={**doc_metadata[doc_id], "similarity_score": score},
                source=doc_metadata[doc_id].get("source")
            )
            documents.append(doc)
            scores.append(score)
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={
                "strategy": self.name,
                "num_query_variations": len(query_variations),
                "aggregation_method": self.aggregation_method,
                "total_unique_docs": len(doc_scores)
            }
        )
    
    def supports_vector_store(self, vector_store_type: str) -> bool:
        return vector_store_type == "ChromaStore"


class ChromaRerankedStrategy(RetrievalStrategy):
    """ChromaDB strategy with score-based re-ranking."""
    
    def __init__(self, name: str = "ChromaRerankedStrategy", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.initial_k = config.get("initial_k", 20)  # Retrieve more, then re-rank
        self.rerank_factors = config.get("rerank_factors", {
            "recency": 0.1,      # Boost recent documents
            "length": 0.05,      # Slight preference for longer documents
            "metadata_boost": 0.2  # Boost based on metadata
        })
        self.distance_metric = config.get("distance_metric", "cosine")
    
    def retrieve(
        self,
        query_embedding: List[float],
        vector_store,
        top_k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        """Search with re-ranking based on multiple factors."""
        # Get initial results (more than needed)
        initial_k = max(self.initial_k, top_k * 2)
        
        results = vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_k,
            include=["documents", "metadatas", "distances"]
        )
        
        candidates = []
        
        if results["documents"] and results["documents"][0]:
            for i, (content, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0] or [{}] * len(results["documents"][0]),
                results["distances"][0]
            )):
                # Base similarity score
                if self.distance_metric == "cosine":
                    base_score = 1.0 - distance
                else:
                    base_score = -distance
                
                # Apply re-ranking factors
                final_score = self._rerank_score(base_score, content, metadata)
                
                doc = Document(
                    id=metadata.get("id", f"doc_{i}"),
                    content=content,
                    metadata={**metadata, "similarity_score": final_score, "base_score": base_score},
                    source=metadata.get("source")
                )
                
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
                "initial_k": initial_k,
                "rerank_factors": self.rerank_factors,
                "candidates_considered": len(candidates)
            }
        )
    
    def _rerank_score(self, base_score: float, content: str, metadata: Dict[str, Any]) -> float:
        """Apply re-ranking factors to base similarity score."""
        final_score = base_score
        
        # Recency boost (if timestamp available)
        if "timestamp" in metadata and self.rerank_factors.get("recency", 0) > 0:
            # This is a simplified recency boost - in practice you'd calculate actual recency
            recency_boost = self.rerank_factors["recency"] * 0.5  # Placeholder
            final_score += recency_boost
        
        # Length factor (slight preference for longer documents)
        if self.rerank_factors.get("length", 0) > 0:
            length_factor = min(len(content) / 1000, 1.0)  # Normalize by 1000 chars
            length_boost = self.rerank_factors["length"] * length_factor
            final_score += length_boost
        
        # Metadata-based boosts
        if self.rerank_factors.get("metadata_boost", 0) > 0:
            metadata_boost = 0
            
            # Boost high priority items
            if metadata.get("priority") == "high":
                metadata_boost += 0.3
            elif metadata.get("priority") == "medium":
                metadata_boost += 0.1
            
            # Boost certain types
            if metadata.get("type") in ["documentation", "tutorial"]:
                metadata_boost += 0.2
            
            final_score += self.rerank_factors["metadata_boost"] * metadata_boost
        
        return final_score
    
    def supports_vector_store(self, vector_store_type: str) -> bool:
        return vector_store_type == "ChromaStore"


class ChromaHybridStrategy(HybridRetrievalStrategy):
    """Hybrid strategy combining multiple ChromaDB approaches."""
    
    def __init__(self, name: str = "ChromaHybridStrategy", config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Initialize sub-strategies based on config
        strategies_config = config.get("strategies", [
            {"type": "ChromaBasicStrategy", "weight": 0.6},
            {"type": "ChromaMetadataFilterStrategy", "weight": 0.4}
        ])
        
        for strategy_config in strategies_config:
            strategy_type = strategy_config["type"]
            weight = strategy_config.get("weight", 1.0)
            strategy_params = strategy_config.get("config", {})
            
            # Create strategy instance
            if strategy_type == "ChromaBasicStrategy":
                strategy = ChromaBasicStrategy(config=strategy_params)
            elif strategy_type == "ChromaMetadataFilterStrategy":
                strategy = ChromaMetadataFilterStrategy(config=strategy_params)
            elif strategy_type == "ChromaMultiQueryStrategy":
                strategy = ChromaMultiQueryStrategy(config=strategy_params)
            elif strategy_type == "ChromaRerankedStrategy":
                strategy = ChromaRerankedStrategy(config=strategy_params)
            else:
                continue  # Skip unknown strategy types
            
            self.add_strategy(strategy, weight)
    
    def retrieve(
        self,
        query_embedding: List[float],
        vector_store,
        top_k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        """Combine results from multiple strategies."""
        if not self.strategies:
            # Fallback to basic strategy
            basic_strategy = ChromaBasicStrategy()
            return basic_strategy.retrieve(query_embedding, vector_store, top_k, **kwargs)
        
        # Get results from all strategies
        results = []
        for strategy in self.strategies:
            try:
                result = strategy.retrieve(query_embedding, vector_store, top_k * 2, **kwargs)
                results.append(result)
            except Exception as e:
                # Log error and continue with other strategies
                print(f"Strategy {strategy.name} failed: {e}")
                continue
        
        if not results:
            # No strategies succeeded
            return RetrievalResult(
                documents=[],
                scores=[],
                strategy_metadata={"strategy": self.name, "error": "All sub-strategies failed"}
            )
        
        # Combine results
        return self.combine_results(results, top_k)
    
    def supports_vector_store(self, vector_store_type: str) -> bool:
        return vector_store_type == "ChromaStore"