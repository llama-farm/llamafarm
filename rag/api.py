"""Internal API for RAG system search functionality."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from core.base import Document
from core.factories import (
    create_embedder_from_config,
    create_vector_store_from_config,
    create_retrieval_strategy_from_config,
)
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
            source=doc.source,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DatabaseSearchAPI:
    """API for searching directly against a database without dataset requirement."""

    def __init__(
        self,
        config_path: str = "rag_config.json",
        base_dir: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """Initialize the database search API.

        Args:
            config_path: Path to llamafarm.yaml configuration file
            base_dir: Base directory for relative path resolution
            database: Database name to search in
        """
        self.config_path = config_path
        self.base_dir = base_dir
        self.database = database
        self._load_database_config()
        self._initialize_components()

    def _load_database_config(self) -> None:
        """Load configuration for database from llamafarm.yaml."""
        resolver = PathResolver(self.base_dir)

        try:
            resolved_config_path = resolver.resolve_config_path(self.config_path)

            # Load llamafarm.yaml
            if yaml is None:
                raise ImportError("PyYAML is required to parse llamafarm.yaml files")

            with open(resolved_config_path, "r") as f:
                llamafarm_config = yaml.safe_load(f)

            if not llamafarm_config:
                raise ValueError("Empty llamafarm config file")

            # Find the database configuration directly from rag section
            rag_config = llamafarm_config.get("rag", {})
            databases = rag_config.get("databases", [])
            
            # If no database specified, use the first one
            if not self.database and databases:
                self.database = databases[0].get("name")
            
            database_config = None
            for db in databases:
                if db.get("name") == self.database:
                    database_config = db
                    break

            if not database_config:
                raise ValueError(f"Database '{self.database}' not found in rag configuration")

            # Store database config for later use
            self._database_config = database_config

            # Build traditional rag config format
            traditional_config = {}

            # Vector store configuration
            db_type = database_config.get("type")
            db_config = database_config.get("config", {})
            
            # Resolve persist_directory if relative
            if "persist_directory" in db_config and not db_config["persist_directory"].startswith("/"):
                import os
                base_path = os.path.dirname(resolved_config_path)
                db_config["persist_directory"] = os.path.join(base_path, db_config["persist_directory"])
                print(f"[DatabaseSearchAPI] Resolved persist_directory to: {db_config['persist_directory']}")
            
            traditional_config["vector_store"] = {"type": db_type, "config": db_config}

            # Embedding configuration - use default embedding strategy
            embedding_strategies = database_config.get("embedding_strategies", [])
            default_embedding_strategy = database_config.get("default_embedding_strategy")

            embedder_config = None
            if default_embedding_strategy:
                # Find the named strategy
                for strategy in embedding_strategies:
                    if strategy.get("name") == default_embedding_strategy:
                        embedder_config = {
                            "type": strategy.get("type"),
                            "config": strategy.get("config", {}),
                        }
                        break
            elif embedding_strategies:
                # Use first available strategy
                first_strategy = embedding_strategies[0]
                embedder_config = {
                    "type": first_strategy.get("type"),
                    "config": first_strategy.get("config", {}),
                }

            if embedder_config:
                traditional_config["embedder"] = embedder_config

            # Retrieval strategy configuration - use default retrieval strategy
            retrieval_strategies = database_config.get("retrieval_strategies", [])
            default_retrieval_strategy = database_config.get("default_retrieval_strategy")

            retrieval_config = None
            if default_retrieval_strategy:
                # Find the named strategy
                for strategy in retrieval_strategies:
                    if strategy.get("name") == default_retrieval_strategy:
                        retrieval_config = {
                            "type": strategy.get("type"),
                            "config": strategy.get("config", {}),
                        }
                        break
            elif retrieval_strategies:
                # Use first strategy or one marked as default
                for strategy in retrieval_strategies:
                    if strategy.get("default", False):
                        retrieval_config = {
                            "type": strategy.get("type"),
                            "config": strategy.get("config", {}),
                        }
                        break
                
                if not retrieval_config and retrieval_strategies:
                    # Use first available strategy as fallback
                    first_strategy = retrieval_strategies[0]
                    retrieval_config = {
                        "type": first_strategy.get("type"),
                        "config": first_strategy.get("config", {}),
                    }

            if retrieval_config:
                traditional_config["retrieval_strategy"] = retrieval_config

            self.config = traditional_config

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found: {e}")
        except (yaml.YAMLError) as e:
            raise ValueError(f"Invalid config file format: {e}")

    def _get_retrieval_strategy_by_name(self, strategy_name: str):
        """Get a retrieval strategy by name from the database config."""
        retrieval_strategies = self._database_config.get("retrieval_strategies", [])
        
        for strategy in retrieval_strategies:
            if strategy.get("name") == strategy_name:
                # Create strategy from config
                strategy_config = {
                    "type": strategy.get("type"),
                    "config": strategy.get("config", {}),
                }
                return create_retrieval_strategy_from_config(strategy_config)
        
        # If not found, return the default strategy
        return self.retrieval_strategy

    def _initialize_components(self) -> None:
        """Initialize RAG components from configuration."""
        try:
            # Initialize embedder
            if "embedder" in self.config:
                self.embedder = create_embedder_from_config(self.config["embedder"])
            else:
                raise ValueError("No embedder configuration found")

            # Initialize vector store
            if "vector_store" in self.config:
                self.vector_store = create_vector_store_from_config(
                    self.config["vector_store"]
                )
            else:
                raise ValueError("No vector store configuration found")

            # Initialize retrieval strategy
            if "retrieval_strategy" in self.config:
                self.retrieval_strategy = create_retrieval_strategy_from_config(
                    self.config["retrieval_strategy"]
                )
            else:
                # Fallback to basic universal strategy
                try:
                    from components.retrievers.strategies.universal import (
                        BasicSimilarityStrategy,
                    )

                    self.retrieval_strategy = BasicSimilarityStrategy()
                except ImportError:
                    # Use basic similarity from the standard location
                    from components.retrievers.basic_similarity.basic_similarity import (
                        BasicSimilarityStrategy,
                    )

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
        retrieval_strategy: Optional[str] = None,
        **kwargs,
    ) -> Union[List[SearchResult], List[Document]]:
        """Search for documents in the database using configured retrieval strategy."""
        # Embed the query
        query_embedding = self.embedder.embed([query])[0]

        # Determine which retrieval strategy to use
        strategy_to_use = self.retrieval_strategy
        if retrieval_strategy:
            # Look for alternative retrieval strategy by name
            strategy_to_use = self._get_retrieval_strategy_by_name(retrieval_strategy)

        # Use retrieval strategy to get results
        retrieval_result = strategy_to_use.retrieve(
            query_embedding=query_embedding,
            vector_store=self.vector_store,
            top_k=top_k,
            **kwargs,
        )

        # Filter by minimum score if specified
        if min_score is not None:
            filtered_docs = []
            filtered_scores = []
            for doc, score in zip(retrieval_result.documents, retrieval_result.scores):
                if score >= min_score:
                    filtered_docs.append(doc)
                    filtered_scores.append(score)
            retrieval_result.documents = filtered_docs
            retrieval_result.scores = filtered_scores

        # Apply metadata filter if specified
        if metadata_filter:
            filtered_docs = []
            filtered_scores = []
            for doc, score in zip(retrieval_result.documents, retrieval_result.scores):
                if self._matches_metadata_filter(doc, metadata_filter):
                    filtered_docs.append(doc)
                    filtered_scores.append(score)
            retrieval_result.documents = filtered_docs
            retrieval_result.scores = filtered_scores

        # Return raw documents if requested
        if return_raw_documents:
            return retrieval_result.documents

        # Convert to SearchResult objects
        results = []
        for doc, score in zip(retrieval_result.documents, retrieval_result.scores):
            # Update score in metadata for SearchResult creation
            doc.metadata["similarity_score"] = score
            results.append(SearchResult.from_document(doc))

        return results

    def _matches_metadata_filter(
        self, doc: Document, metadata_filter: Dict[str, Any]
    ) -> bool:
        """Check if a document matches metadata filter criteria."""
        if not doc.metadata:
            return False

        for key, value in metadata_filter.items():
            if key not in doc.metadata:
                return False
            if doc.metadata[key] != value:
                return False

        return True


class SearchAPI:
    """Internal API for searching the RAG system."""

    def __init__(
        self,
        config_path: str = "rag_config.json",
        base_dir: Optional[str] = None,
        dataset: Optional[str] = None,
    ):
        """Initialize the search API.

        Args:
            config_path: Path to configuration file
            base_dir: Base directory for relative path resolution
            dataset: Dataset name to use when config_path is a llamafarm.yaml file
        """
        self.config_path = config_path
        self.base_dir = base_dir
        self.dataset = dataset
        self._load_config()
        self._initialize_components()

    def _load_config(self) -> None:
        """Load configuration from file."""
        resolver = PathResolver(self.base_dir)

        try:
            resolved_config_path = resolver.resolve_config_path(self.config_path)

            # Check if this is a llamafarm.yaml file that needs dataset parsing
            if resolved_config_path.name.startswith(
                "llamafarm"
            ) or resolved_config_path.suffix.lower() in [".yaml", ".yml"]:
                if not self.dataset:
                    raise ValueError(
                        "Dataset parameter is required when using llamafarm.yaml config files"
                    )
                self.config = self._parse_llamafarm_config(
                    resolved_config_path, self.dataset
                )
            else:
                # Traditional rag config file (JSON)
                with open(resolved_config_path, "r") as f:
                    self.config = json.load(f)

            # Resolve any paths within the configuration
            self.config = resolve_paths_in_config(self.config, resolver)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found: {e}")
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Invalid config file format: {e}")

    def _parse_llamafarm_config(
        self, config_path: Path, dataset_name: str
    ) -> Dict[str, Any]:
        """Parse llamafarm.yaml config and extract rag configuration for the specified dataset.

        Args:
            config_path: Path to llamafarm.yaml file
            dataset_name: Name of the dataset to extract configuration for

        Returns:
            Dictionary in traditional rag config format
        """
        # Load llamafarm.yaml
        if yaml is None:
            raise ImportError("PyYAML is required to parse llamafarm.yaml files")

        with open(config_path, "r") as f:
            llamafarm_config = yaml.safe_load(f)

        if not llamafarm_config:
            raise ValueError("Empty llamafarm config file")

        # Find the dataset configuration
        datasets = llamafarm_config.get("datasets", [])
        dataset_config = None
        for dataset in datasets:
            if dataset.get("name") == dataset_name:
                dataset_config = dataset
                break

        if not dataset_config:
            raise ValueError(f"Dataset '{dataset_name}' not found in config")

        # Get the database name from dataset config
        database_name = dataset_config.get("database")
        if not database_name:
            raise ValueError(f"No database specified for dataset '{dataset_name}'")

        # Find the database configuration in rag section
        rag_config = llamafarm_config.get("rag", {})
        databases = rag_config.get("databases", [])
        database_config = None
        for db in databases:
            if db.get("name") == database_name:
                database_config = db
                break

        if not database_config:
            raise ValueError(
                f"Database '{database_name}' not found in rag configuration"
            )

        # Build traditional rag config format
        traditional_config = {}

        # Vector store configuration
        db_type = database_config.get("type")
        db_config = database_config.get("config", {})
        
        # Resolve persist_directory if relative (for ChromaDB)
        if "persist_directory" in db_config and not db_config["persist_directory"].startswith("/"):
            import os
            base_path = os.path.dirname(config_path)
            db_config["persist_directory"] = os.path.join(base_path, db_config["persist_directory"])
            print(f"[SearchAPI] Resolved persist_directory to: {db_config['persist_directory']}")
        
        traditional_config["vector_store"] = {"type": db_type, "config": db_config}

        # Embedding configuration - use default embedding strategy
        embedding_strategies = database_config.get("embedding_strategies", [])
        default_embedding_strategy = database_config.get("default_embedding_strategy")

        embedder_config = None
        if default_embedding_strategy:
            # Find the named strategy
            for strategy in embedding_strategies:
                if strategy.get("name") == default_embedding_strategy:
                    embedder_config = {
                        "type": strategy.get("type"),
                        "config": strategy.get("config", {}),
                    }
                    break
        elif embedding_strategies:
            # Use first available strategy
            first_strategy = embedding_strategies[0]
            embedder_config = {
                "type": first_strategy.get("type"),
                "config": first_strategy.get("config", {}),
            }

        if embedder_config:
            traditional_config["embedder"] = embedder_config

        # Retrieval strategy configuration - use default retrieval strategy
        retrieval_strategies = database_config.get("retrieval_strategies", [])
        default_retrieval_strategy = database_config.get("default_retrieval_strategy")

        retrieval_config = None
        if default_retrieval_strategy:
            # Find the named strategy
            for strategy in retrieval_strategies:
                if strategy.get("name") == default_retrieval_strategy:
                    retrieval_config = {
                        "type": strategy.get("type"),
                        "config": strategy.get("config", {}),
                    }
                    break
        else:
            # Find default strategy or use first available
            for strategy in retrieval_strategies:
                if strategy.get("default", False):
                    retrieval_config = {
                        "type": strategy.get("type"),
                        "config": strategy.get("config", {}),
                    }
                    break
            if not retrieval_config and retrieval_strategies:
                # Use first available strategy
                first_strategy = retrieval_strategies[0]
                retrieval_config = {
                    "type": first_strategy.get("type"),
                    "config": first_strategy.get("config", {}),
                }

        if retrieval_config:
            traditional_config["retrieval_strategy"] = retrieval_config

        # Store database config for alternative retrieval strategy lookup
        traditional_config["_database_config"] = database_config

        return traditional_config

    def _initialize_components(self) -> None:
        """Initialize embedder, vector store, and retrieval strategy from config."""
        try:
            self.embedder = create_embedder_from_config(self.config.get("embedder", {}))
            self.vector_store = create_vector_store_from_config(
                self.config.get("vector_store", {})
            )

            # Store database config for alternative retrieval strategy lookup
            self._database_config = self.config.get("_database_config")

            # Initialize retrieval strategy (with fallback to basic strategy)
            retrieval_config = self.config.get("retrieval_strategy")
            if retrieval_config:
                # Pass database type for optimization
                database_type = type(self.vector_store).__name__
                self.retrieval_strategy = create_retrieval_strategy_from_config(
                    retrieval_config,
                    # database_type=database_type # TODO: Bobby commented this out because it was causing an error
                )
            else:
                # Fallback to basic universal strategy
                try:
                    from components.retrievers.strategies.universal import (
                        BasicSimilarityStrategy,
                    )

                    self.retrieval_strategy = BasicSimilarityStrategy()
                except ImportError:
                    # Use basic similarity from the standard location
                    from components.retrievers.basic_similarity.basic_similarity import (
                        BasicSimilarityStrategy,
                    )

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
        retrieval_strategy: Optional[str] = None,
        **kwargs,
    ) -> Union[List[SearchResult], List[Document]]:
        """Search for documents matching the query using configured retrieval strategy.

        Args:
            query: Search query text
            top_k: Number of results to return (default: 5)
            min_score: Minimum similarity score filter (optional)
            metadata_filter: Filter results by metadata fields (optional)
            return_raw_documents: Return Document objects instead of SearchResult (default: False)
            retrieval_strategy: Optional retrieval strategy name to override default
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

        # Determine which retrieval strategy to use
        strategy_to_use = self.retrieval_strategy
        if retrieval_strategy and hasattr(self, "_database_config"):
            # Look for alternative retrieval strategy by name
            strategy_to_use = self._get_retrieval_strategy_by_name(retrieval_strategy)

        # Use retrieval strategy to get results
        retrieval_result = strategy_to_use.retrieve(
            query_embedding=query_embedding,
            vector_store=self.vector_store,
            top_k=top_k,
            metadata_filter=metadata_filter,
            **kwargs,
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

    def _get_retrieval_strategy_by_name(self, strategy_name: str):
        """Get a retrieval strategy by name from the database config.

        Args:
            strategy_name: Name of the retrieval strategy to find

        Returns:
            Retrieval strategy instance
        """
        if not self._database_config:
            return self.retrieval_strategy

        retrieval_strategies = self._database_config.get("retrieval_strategies", [])

        # Find strategy by name
        for strategy in retrieval_strategies:
            if strategy.get("name") == strategy_name:
                strategy_config = {
                    "type": strategy.get("type"),
                    "config": strategy.get("config", {}),
                }
                return create_retrieval_strategy_from_config(strategy_config)

        # If not found, return default strategy
        return self.retrieval_strategy

    def _filter_by_metadata(
        self, documents: List[Document], metadata_filter: Dict[str, Any]
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
        try:
            # Try to get collection info if the method exists
            if hasattr(self.vector_store, "get_collection_info"):
                info = self.vector_store.get_collection_info()
            else:
                info = {"error": "get_collection_info not implemented"}
        except Exception as e:
            info = {"error": str(e)}

        info["retrieval_strategy"] = {
            "name": getattr(self.retrieval_strategy, "name", "unknown"),
            "type": type(self.retrieval_strategy).__name__,
            "config": getattr(self.retrieval_strategy, "config", {}),
        }
        return info

    def search_with_context(
        self, query: str, context_size: int = 2, **kwargs
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
        for item in main_results:
            # Ensure item is a Document object for SearchResult conversion
            if isinstance(item, Document):
                # It's a Document, convert to SearchResult
                main_data = SearchResult.from_document(item).to_dict()
            else:
                # It's already a SearchResult, get its dict representation
                main_data = item.to_dict() if hasattr(item, "to_dict") else {}

            result = {
                "main": main_data,
                "context_before": [],
                "context_after": [],
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
    dataset: Optional[str] = None,
    **kwargs,
) -> List[SearchResult]:
    """Convenience function for simple searches.

    Args:
        query: Search query text
        config_path: Path to configuration file
        top_k: Number of results to return
        dataset: Dataset name (required if config_path is llamafarm.yaml)
        **kwargs: Additional arguments passed to SearchAPI.search()

    Returns:
        List of SearchResult objects

    Example:
        >>> from api import search
        >>> results = search("login issues", top_k=3)
        >>> print(results[0].content)

        # With llamafarm.yaml and dataset
        >>> results = search("login issues", config_path="llamafarm.yaml", dataset="my_dataset", top_k=3)
        >>> print(results[0].content)
    """
    api = SearchAPI(config_path=config_path, dataset=dataset)
    results = api.search(query, top_k=top_k, **kwargs)
    # Ensure we always return SearchResult objects
    if not results:
        return []

    # Check the type of the first result to determine conversion needed
    first_result = results[0]
    if isinstance(first_result, SearchResult):
        return results  # type: ignore # Already SearchResult objects
    else:
        # Convert Documents to SearchResult objects
        return [
            SearchResult.from_document(doc)
            for doc in results
            if isinstance(doc, Document)
        ]
