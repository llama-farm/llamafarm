"""Internal API for RAG system search functionality."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict

from core.base import Document
from core.factories import create_embedder_from_config, create_vector_store_from_config, create_retrieval_strategy_from_config
from utils.path_resolver import PathResolver, resolve_paths_in_config


@dataclass
class SearchResult:
    """Search result with document and metadata."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: Optional[str] = None
    
    @classmethod
    def from_document(cls, doc: Document) -> "SearchResult":
        """Create SearchResult from Document."""
        score = doc.metadata.get("similarity_score", 0.0)
        return cls(
            id=doc.id or "unknown",
            content=doc.content,
            score=score,
            metadata={k: v for k, v in doc.metadata.items() if k != "similarity_score"},
            source=doc.source
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class SearchAPI:
    """Internal API for searching the RAG system."""
    
    def __init__(self, config_path: str = "rag_config.json", base_dir: Optional[str] = None):
        """Initialize the search API.
        
        Args:
            config_path: Path to configuration file
            base_dir: Base directory for relative path resolution
        """
        self.config_path = config_path
        self.base_dir = base_dir
        self._load_config()
        self._initialize_components()
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        resolver = PathResolver(self.base_dir)
        
        try:
            resolved_config_path = resolver.resolve_config_path(self.config_path)
            with open(resolved_config_path, "r") as f:
                self.config = json.load(f)
            
            # Resolve any paths within the configuration
            self.config = resolve_paths_in_config(self.config, resolver)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    def _initialize_components(self) -> None:
        """Initialize embedder, vector store, and retrieval strategy from config."""
        try:
            self.embedder = create_embedder_from_config(self.config.get("embedder", {}))
            self.vector_store = create_vector_store_from_config(self.config.get("vector_store", {}))
            
            # Initialize retrieval strategy (with fallback to basic strategy)
            retrieval_config = self.config.get("retrieval_strategy")
            if retrieval_config:
                # Pass database type for optimization
                database_type = type(self.vector_store).__name__
                self.retrieval_strategy = create_retrieval_strategy_from_config(
                    retrieval_config, 
                    database_type=database_type
                )
            else:
                # Fallback to basic universal strategy
                from components.retrievers.strategies.universal import BasicSimilarityStrategy
                self.retrieval_strategy = BasicSimilarityStrategy()
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize components: {e}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        return_raw_documents: bool = False,
        **kwargs
    ) -> Union[List[SearchResult], List[Document]]:
        """Search for documents matching the query using configured retrieval strategy.
        
        Args:
            query: Search query text
            top_k: Number of results to return (default: 5)
            min_score: Minimum similarity score filter (optional)
            metadata_filter: Filter results by metadata fields (optional)
            return_raw_documents: Return Document objects instead of SearchResult (default: False)
            **kwargs: Additional arguments passed to the retrieval strategy
            
        Returns:
            List of SearchResult objects or Document objects if return_raw_documents=True
            
        Example:
            >>> api = SearchAPI()
            >>> results = api.search("password reset", top_k=3)
            >>> for result in results:
            ...     print(f"Score: {result.score:.3f} - {result.content[:100]}...")
        """
        # Embed the query
        query_embedding = self.embedder.embed([query])[0]
        
        # Use retrieval strategy to get results
        retrieval_result = self.retrieval_strategy.retrieve(
            query_embedding=query_embedding,
            vector_store=self.vector_store,
            top_k=top_k,
            metadata_filter=metadata_filter,
            **kwargs
        )
        
        documents = retrieval_result.documents
        
        # Apply min_score filter if specified
        if min_score is not None:
            filtered_docs = []
            filtered_scores = []
            for doc, score in zip(documents, retrieval_result.scores):
                if score >= min_score:
                    filtered_docs.append(doc)
                    filtered_scores.append(score)
            documents = filtered_docs
        
        # Return raw documents if requested
        if return_raw_documents:
            return documents
        
        # Convert to SearchResult objects
        return [SearchResult.from_document(doc) for doc in documents]
    
    def _filter_by_metadata(
        self,
        documents: List[Document],
        metadata_filter: Dict[str, Any]
    ) -> List[Document]:
        """Filter documents by metadata fields.
        
        Args:
            documents: List of documents to filter
            metadata_filter: Dictionary of metadata field filters
            
        Returns:
            Filtered list of documents
        """
        filtered = []
        for doc in documents:
            match = True
            for key, value in metadata_filter.items():
                if key not in doc.metadata:
                    match = False
                    break
                if isinstance(value, list):
                    # Check if metadata value is in the list
                    if doc.metadata[key] not in value:
                        match = False
                        break
                else:
                    # Direct comparison
                    if doc.metadata[key] != value:
                        match = False
                        break
            if match:
                filtered.append(doc)
        return filtered
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection.
        
        Returns:
            Dictionary with collection information including retrieval strategy info
        """
        info = self.vector_store.get_collection_info()
        info["retrieval_strategy"] = {
            "name": self.retrieval_strategy.name,
            "type": type(self.retrieval_strategy).__name__,
            "config": getattr(self.retrieval_strategy, 'config', {})
        }
        return info
    
    def search_with_context(
        self,
        query: str,
        context_size: int = 2,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search and include surrounding context documents.
        
        Args:
            query: Search query text
            context_size: Number of context documents before/after each result
            **kwargs: Additional arguments passed to search()
            
        Returns:
            List of search results with context
        """
        # Get main search results
        main_results = self.search(query, return_raw_documents=True, **kwargs)
        
        # For each result, try to get context documents
        results_with_context = []
        for doc in main_results:
            result = {
                "main": SearchResult.from_document(doc).to_dict(),
                "context_before": [],
                "context_after": []
            }
            
            # This is a simplified version - in a real implementation,
            # you might want to fetch documents by ID or sequence
            results_with_context.append(result)
        
        return results_with_context


# Convenience function for simple searches
def search(
    query: str,
    config_path: str = "rag_config.json",
    top_k: int = 5,
    **kwargs
) -> List[SearchResult]:
    """Convenience function for simple searches.
    
    Args:
        query: Search query text
        config_path: Path to configuration file
        top_k: Number of results to return
        **kwargs: Additional arguments passed to SearchAPI.search()
        
    Returns:
        List of SearchResult objects
        
    Example:
        >>> from api import search
        >>> results = search("login issues", top_k=3)
        >>> print(results[0].content)
    """
    api = SearchAPI(config_path=config_path)
    return api.search(query, top_k=top_k, **kwargs)