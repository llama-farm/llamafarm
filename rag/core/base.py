"""Base classes for the extensible RAG system."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.core.base")


@dataclass
class Document:
    """Universal document representation."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str | None = None
    embeddings: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "id": self.id,
            "source": self.source,
            "embeddings": self.embeddings,
        }


@dataclass
class ProcessingResult:
    """Result of processing documents through a component."""

    documents: list[Document]
    errors: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class Component(ABC):
    """Base class for all pipeline components."""

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.logger = logger.bind(name=self.name)
        self.project_dir = project_dir

    @abstractmethod
    def process(self, documents: list[Document]) -> ProcessingResult:
        """Process documents and return results."""
        pass

    def validate_config(self) -> bool:
        """Validate component configuration."""
        return True


class Parser(Component):
    """Base class for document parsers."""

    @abstractmethod
    def parse(self, source: str) -> ProcessingResult:
        """Parse documents from source."""
        pass

    def process(self, documents: list[Document]) -> ProcessingResult:
        """Process already parsed documents (pass-through for parsers)."""
        return ProcessingResult(documents)


class Embedder(Component):
    """Base class for embedding generators with circuit breaker protection."""

    # Default circuit breaker settings (can be overridden in subclasses)
    DEFAULT_FAILURE_THRESHOLD = 5
    DEFAULT_RESET_TIMEOUT = 60.0

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ):
        super().__init__(name, config, project_dir)

        # Initialize circuit breaker for this embedder
        from utils.embedding_safety import CircuitBreaker

        circuit_config = (config or {}).get("circuit_breaker", {})
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_config.get(
                "failure_threshold", self.DEFAULT_FAILURE_THRESHOLD
            ),
            reset_timeout=circuit_config.get(
                "reset_timeout", self.DEFAULT_RESET_TIMEOUT
            ),
        )

        # Track whether to fail fast on errors (default: True for safety)
        self._fail_fast = (config or {}).get("fail_fast", True)

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass

    def check_circuit_breaker(self) -> None:
        """
        Check if the circuit breaker allows requests.

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        from utils.embedding_safety import CircuitBreakerOpenError

        if not self._circuit_breaker.can_execute():
            state_info = self._circuit_breaker.get_state_info()
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open for {self.name}. "
                f"Too many consecutive failures ({state_info['failure_count']}/{state_info['failure_threshold']}). "
                f"Will retry in {state_info.get('time_until_reset', 'N/A')} seconds.",
                failures=state_info["failure_count"],
                reset_time=state_info.get("time_until_reset", 0),
            )

    def record_success(self) -> None:
        """Record a successful embedding operation."""
        self._circuit_breaker.record_success()

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed embedding operation."""
        self._circuit_breaker.record_failure(error)

    def get_circuit_state(self) -> dict[str, Any]:
        """Get current circuit breaker state."""
        return self._circuit_breaker.get_state_info()

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        self._circuit_breaker.force_reset()

    def process(self, documents: list[Document]) -> ProcessingResult:
        """Add embeddings to documents."""
        texts = [doc.content for doc in documents]
        embeddings = self.embed(texts)

        for doc, embedding in zip(documents, embeddings, strict=False):
            doc.embeddings = embedding

        # Calculate chunk metrics
        chunked_docs = [
            doc
            for doc in documents
            if "chunk_num" in doc.metadata or "chunk_index" in doc.metadata
        ]

        return ProcessingResult(
            documents=documents,
            metrics={
                "embedded_count": len(documents),
                "chunk_count": len(chunked_docs),
                "non_chunked_count": len(documents) - len(chunked_docs),
            },
        )


class VectorStore(Component):
    """Base class for vector databases."""

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        project_dir: Path | None = None,
    ):
        super().__init__(name, config, project_dir)
        if project_dir is None:
            raise ValueError("project_dir is required")
        self.persist_directory = str(Path(project_dir) / "lf_data" / "stores" / name)

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> bool:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    def get_documents_by_metadata(
        self, metadata_filter: dict[str, Any]
    ) -> list[Document]:
        """Get documents matching a metadata filter.

        Args:
            metadata_filter: Key-value pairs to match against document metadata.

        Returns:
            List of matching documents.
        """
        pass

    @abstractmethod
    def delete_documents(self, doc_ids: list[str]) -> int:
        """Delete documents by their IDs.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            Number of documents deleted.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[Document]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete_collection(self) -> bool:
        """Delete the collection."""
        pass

    def process(self, documents: list[Document]) -> ProcessingResult:
        """Add documents to vector store."""
        success = self.add_documents(documents)

        # Calculate chunk metrics
        chunked_docs = [
            doc
            for doc in documents
            if "chunk_num" in doc.metadata or "chunk_index" in doc.metadata
        ]

        return ProcessingResult(
            documents=documents,
            metrics={
                "stored_count": len(documents) if success else 0,
                "stored_chunks": len(chunked_docs) if success else 0,
                "stored_docs": (len(documents) - len(chunked_docs)) if success else 0,
            },
        )


class Pipeline:
    """Simple pipeline for chaining components."""

    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.components: list[Component] = []
        self.logger = logger.bind(name=self.name)

    def add_component(self, component: Component) -> "Pipeline":
        """Add a component to the pipeline."""
        component.validate_config()
        self.components.append(component)
        return self

    def run(
        self, source: str | None = None, documents: list[Document] | None = None
    ) -> ProcessingResult:
        """Run the pipeline."""
        if source and not documents:
            # Start with parser
            if not self.components or not isinstance(self.components[0], Parser):
                raise ValueError(
                    "Pipeline must start with a Parser when source is provided"
                )
            result = self.components[0].parse(source)
            current_docs = result.documents
            all_errors = result.errors
            start_idx = 1
        elif documents:
            current_docs = documents
            all_errors = []
            start_idx = 0
        else:
            raise ValueError("Either source or documents must be provided")

        # Process through remaining components
        aggregated_metrics = {}
        for component in self.components[start_idx:]:
            try:
                result = component.process(current_docs)
                current_docs = result.documents
                all_errors.extend(result.errors)
                # Aggregate metrics from each component
                if result.metrics:
                    for key, value in result.metrics.items():
                        aggregated_metrics[f"{component.name}_{key}"] = value
            except Exception as e:
                self.logger.error(f"Component {component.name} failed: {e}")
                all_errors.append({"component": component.name, "error": str(e)})

        # Calculate final chunk metrics
        chunked_docs = [
            doc
            for doc in current_docs
            if "chunk_num" in doc.metadata or "chunk_index" in doc.metadata
        ]
        aggregated_metrics["total_chunks"] = len(chunked_docs)
        aggregated_metrics["total_documents"] = len(current_docs)

        return ProcessingResult(
            documents=current_docs, errors=all_errors, metrics=aggregated_metrics
        )
