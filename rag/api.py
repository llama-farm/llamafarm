"""Internal API for RAG system search functionality."""

# Use the common config module instead of direct YAML loading
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

from core.base import Document
from core.factories import (
    create_embedder_from_config,
    create_retrieval_strategy_from_config,
    create_vector_store_from_config,
)

# Add the repo root to the path to find the config module
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from config import load_config
    from config.datamodel import Database, LlamaFarmConfig, RetrievalStrategy
except ImportError as e:
    raise ImportError(
        f"Could not import config module. Make sure you're running from the repo root. Error: {e}"
    ) from e


@dataclass
class SearchResult:
    """Search result with document and metadata."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any]
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BaseAPI:
    """Base API for all RAG APIs."""

    config: LlamaFarmConfig
    rag_config: dict[str, Any]
    database: str | None = None
    dataset: str | None = None
    _database_config: Database | None = None

    def __init__(
        self,
        project_dir: str,
        database: str | None = None,
        dataset: str | None = None,
    ):
        self.project_dir = project_dir
        self.database = database
        self.dataset = dataset
        self._load_config()
        self._load_database_config()
        self._initialize_components()

    def _load_config(self) -> None:
        """Load configuration from file."""
        self.config = load_config(config_path=self.project_dir, validate=True)

    def _load_database_config(self) -> None:
        """Load configuration for database from project config."""
        try:
            rag_config = self._validate_and_get_rag_config()
            databases = rag_config.databases

            self._resolve_database_name(databases)
            database_config = self._find_database_config(databases)

            # Store database config for later use
            self._database_config = database_config

            # Build traditional rag config format
            traditional_config = self._build_traditional_config(database_config)
            self.rag_config = traditional_config

        except Exception as e:
            raise ValueError(f"Invalid config file format: {e}") from e

    def _validate_and_get_rag_config(self):
        """Validate and return RAG configuration."""
        rag_config = self.config.rag
        if not rag_config:
            raise ValueError("RAG configuration not found in project config")
        if not rag_config.databases:
            raise ValueError("No databases found in RAG configuration")
        return rag_config

    def _resolve_database_name(self, databases) -> None:
        """Resolve database name from dataset or use first available."""
        if self.dataset:
            dataset_config = next(
                (
                    dataset
                    for dataset in (self.config.datasets or [])
                    if dataset.name == self.dataset
                ),
                None,
            )
            if not dataset_config:
                raise ValueError(f"Dataset '{self.dataset}' not found in config")
            self.database = dataset_config.database

        # If no database specified, use the first one
        if not self.database and databases:
            self.database = databases[0].name

    def _find_database_config(self, databases):
        """Find database configuration by name."""
        database_config = next(
            (db for db in databases if db.name == self.database), None
        )
        if not database_config:
            raise ValueError(
                f"Database '{self.database}' not found in rag configuration"
            )
        return database_config

    def _resolve_model_references(self, strategy_config: dict[str, Any]) -> dict[str, Any]:
        """Resolve model references in strategy config.

        For strategies that reference models by name (e.g., CrossEncoderRerankedStrategy, MultiTurnRAGStrategy),
        this looks up the model in runtime.models and adds the resolved base_url and model ID.
        """
        strategy_type = strategy_config.get("type")

        # Strategies that need model resolution
        if strategy_type in ["CrossEncoderRerankedStrategy", "MultiTurnRAGStrategy"]:
            config = strategy_config.get("config", {})
            model_name = config.get("model_name")

            if model_name and hasattr(self.config, "runtime") and hasattr(self.config.runtime, "models"):
                # Find the model in runtime.models
                model_config = None
                for model in self.config.runtime.models:
                    if model.name == model_name:
                        model_config = model
                        break

                if model_config:
                    # Add resolved model details to the config
                    config["model_base_url"] = model_config.base_url
                    config["model_id"] = model_config.model

            # For MultiTurnRAGStrategy, also resolve the reranker model if present
            if strategy_type == "MultiTurnRAGStrategy" and config.get("enable_reranking"):
                reranker_config = config.get("reranker_config", {})
                reranker_model_name = reranker_config.get("model_name")

                if reranker_model_name and hasattr(self.config, "runtime") and hasattr(self.config.runtime, "models"):
                    # Find the reranker model in runtime.models
                    reranker_model_config = None
                    for model in self.config.runtime.models:
                        if model.name == reranker_model_name:
                            reranker_model_config = model
                            break

                    if reranker_model_config:
                        # Add resolved model details to the reranker config
                        reranker_config["model_base_url"] = reranker_model_config.base_url
                        reranker_config["model_id"] = reranker_model_config.model

        return strategy_config

    def _build_traditional_config(self, database_config: Database) -> dict[str, Any]:
        """Build traditional RAG config format from database config."""
        traditional_config: dict[str, Any] = {}

        # Vector store configuration
        traditional_config["vector_store"] = self._build_vector_store_config(
            database_config
        )

        # Embedding configuration
        embedder_config = self._build_embedder_config(database_config)
        if embedder_config:
            traditional_config["embedder"] = embedder_config

        # Retrieval strategy configuration
        retrieval_config = self._build_retrieval_config(database_config)
        if retrieval_config:
            traditional_config["retrieval_strategy"] = retrieval_config

        return traditional_config

    def _build_vector_store_config(self, database_config: Database) -> dict[str, Any]:
        """Build vector store configuration."""
        db_type = database_config.type.value
        return {
            "type": db_type,
            "config": database_config.config,
        }

    def _build_embedder_config(
        self, database_config: Database
    ) -> Optional[dict[str, Any]]:
        """Build embedder configuration from embedding strategies."""
        embedding_strategies = database_config.embedding_strategies or []
        default_embedding_strategy = database_config.default_embedding_strategy

        if default_embedding_strategy:
            # Find the named strategy
            strategy = next(
                (
                    s
                    for s in embedding_strategies
                    if s.name == default_embedding_strategy
                ),
                None,
            )
            if strategy:
                return self._strategy_to_config(strategy)
        elif embedding_strategies:
            # Use first available strategy
            return self._strategy_to_config(embedding_strategies[0])

        return None

    def _build_retrieval_config(
        self, database_config: Database
    ) -> Optional[dict[str, Any]]:
        """Build retrieval configuration from retrieval strategies."""
        retrieval_strategies = database_config.retrieval_strategies or []
        default_retrieval_strategy = database_config.default_retrieval_strategy

        if default_retrieval_strategy:
            # Find the named strategy
            strategy = next(
                (
                    s
                    for s in retrieval_strategies
                    if s.name == default_retrieval_strategy
                ),
                None,
            )
            if strategy:
                return self._strategy_to_config(strategy)
        elif retrieval_strategies:
            # Use first strategy marked as default, or first available
            strategy = next(
                (s for s in retrieval_strategies if getattr(s, "default", False)),
                retrieval_strategies[0] if retrieval_strategies else None,
            )
            if strategy:
                return self._strategy_to_config(strategy)

        return None

    def _strategy_to_config(self, strategy: RetrievalStrategy) -> dict[str, Any]:
        """Convert strategy object to config dictionary."""
        return {
            "type": strategy.type.value,
            "config": strategy.config,
        }

    def _get_retrieval_strategy_by_name(self, strategy_name: str):
        """Get a retrieval strategy by name from the database config."""
        if not self._database_config:
            return None

        retrieval_strategies = self._database_config.retrieval_strategies or []

        for strategy in retrieval_strategies:
            if strategy.name == strategy_name:
                # Create strategy from config
                strategy_config = {
                    "type": strategy.type.value,
                    "config": strategy.config,
                }
                # Resolve model references
                strategy_config = self._resolve_model_references(strategy_config)
                return create_retrieval_strategy_from_config(strategy_config)

        # If not found, return the default strategy
        return self.retrieval_strategy

    def _initialize_components(self) -> None:
        """Initialize RAG components from configuration."""
        try:
            # Initialize embedder
            if "embedder" in self.rag_config:
                self.embedder = create_embedder_from_config(self.rag_config["embedder"])
            else:
                raise ValueError("No embedder configuration found")

            # Initialize vector store
            if "vector_store" in self.rag_config:
                self.vector_store = create_vector_store_from_config(
                    self.rag_config["vector_store"], project_dir=Path(self.project_dir)
                )
            else:
                raise ValueError("No vector store configuration found")

            # Initialize retrieval strategy
            if "retrieval_strategy" in self.rag_config:
                strategy_config = self.rag_config["retrieval_strategy"]

                # Resolve model references for strategies that need them
                strategy_config = self._resolve_model_references(strategy_config)

                self.retrieval_strategy = create_retrieval_strategy_from_config(
                    strategy_config,
                    project_dir=self.project_dir
                )
            else:
                # Fallback to basic universal strategy
                try:
                    from components.retrievers.basic_similarity import (
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
            raise RuntimeError(
                f"Failed to initialize components: {e}: rag_config: {self.rag_config}"
            ) from e


class DatabaseSearchAPI(BaseAPI):
    """API for searching directly against a database without dataset requirement."""

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        return_raw_documents: bool = False,
        retrieval_strategy: Optional[str] = None,
        **kwargs,
    ) -> Union[list[SearchResult], list[Document]]:
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
            query_text=query,  # Pass original query text for strategies that need it
            embedder=self.embedder,  # Pass embedder for strategies that need to embed sub-queries
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
        self, doc: Document, metadata_filter: dict[str, Any]
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


class SearchAPI(BaseAPI):
    """Internal API for searching the RAG system."""

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        return_raw_documents: bool = False,
        retrieval_strategy: Optional[str] = None,
        **kwargs,
    ) -> Union[list[SearchResult], list[Document]]:
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
            >>> api = SearchAPI(project_dir=".")
            >>> results = api.search("password reset", top_k=3)
            >>> for result in results:
            ...     print(f"Score: {result.score:.3f} - {result.content[:100]}...")
        """
        # Embed the query
        query_embedding = self.embedder.embed([query])[0]

        # Determine which retrieval strategy to use
        strategy_to_use = self.retrieval_strategy
        if retrieval_strategy and self._database_config:
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

        retrieval_strategies = self.rag_config.get("retrieval_strategies", [])

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
        self, documents: list[Document], metadata_filter: dict[str, Any]
    ) -> list[Document]:
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

    def get_collection_info(self) -> dict[str, Any]:
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
    ) -> list[dict[str, Any]]:
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
    project_dir: str,
    top_k: int = 5,
    dataset: Optional[str] = None,
    **kwargs,
) -> list[SearchResult]:
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
        >>> results = search("login issues", project_dir="l.", dataset="my_dataset", top_k=3)
        >>> print(results[0].content)
    """
    api = SearchAPI(project_dir=project_dir, dataset=dataset)
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
