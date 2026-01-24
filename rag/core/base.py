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


@dataclass
class PreviewChunk:
    """A single chunk with position information for preview."""

    chunk_index: int
    content: str
    start_position: int  # Character position in original text (-1 if not found)
    end_position: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Get character count of the chunk."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Get word count of the chunk."""
        return len(self.content.split())


@dataclass
class PreviewResult:
    """Result of generating a document preview."""

    original_text: str
    chunks: list[PreviewChunk]
    file_info: dict[str, Any]
    parser_used: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    total_chunks: int
    warnings: list[str] = field(default_factory=list)

    @property
    def avg_chunk_size(self) -> float:
        """Calculate average chunk size."""
        if not self.chunks:
            return 0.0
        return sum(c.char_count for c in self.chunks) / len(self.chunks)

    @property
    def total_size_with_overlaps(self) -> int:
        """Calculate total size including overlaps."""
        return sum(c.char_count for c in self.chunks)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_text": self.original_text,
            "chunks": [
                {
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "start_position": c.start_position,
                    "end_position": c.end_position,
                    "char_count": c.char_count,
                    "word_count": c.word_count,
                    "metadata": c.metadata,
                }
                for c in self.chunks
            ],
            "file_info": self.file_info,
            "parser_used": self.parser_used,
            "chunk_strategy": self.chunk_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "total_chunks": self.total_chunks,
            "avg_chunk_size": self.avg_chunk_size,
            "total_size_with_overlaps": self.total_size_with_overlaps,
            "warnings": self.warnings,
        }


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
    """Base class for embedding generators with circuit breaker protection.

    Subclasses should implement:
    - `_call_embedding_api(texts)`: Make the actual API call to generate embeddings
    - `get_embedding_dimension()`: Return the embedding dimension for this model

    The base class handles:
    - Circuit breaker pattern (stops after consecutive failures)
    - Fail-fast behavior (raises exceptions vs returning zero vectors)
    - Embedding validation (rejects zero/invalid embeddings)
    - Success/failure tracking
    """

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

        # Track consecutive failures for logging
        self._consecutive_failures = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass

    @abstractmethod
    def _call_embedding_api(self, text: str) -> list[float]:
        """Make the actual API call to generate an embedding for a single text.

        Subclasses must implement this method to perform the API call.
        This method should NOT handle errors - let them propagate up.

        Args:
            text: The text to embed

        Returns:
            The embedding vector

        Raises:
            Any exception from the underlying API
        """
        pass

    def _get_connection_exceptions(self) -> tuple:
        """Return tuple of exception types that indicate connection failures.

        Subclasses can override to add provider-specific connection exceptions.
        """
        return (ConnectionError, TimeoutError)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string with error handling.

        This method handles circuit breaker, fail-fast, and error recovery.
        Subclasses should implement `_call_embedding_api` instead.

        Args:
            text: The text to embed

        Returns:
            The embedding vector, or zero vector if fail_fast=False and error occurs

        Raises:
            EmbedderUnavailableError: If embedding fails and fail_fast=True
            CircuitBreakerOpenError: If circuit breaker is open
        """
        from utils.embedding_safety import EmbedderUnavailableError, is_zero_vector

        # Handle empty text
        if not text or not text.strip():
            if self._fail_fast:
                raise EmbedderUnavailableError("Cannot embed empty text")
            return [0.0] * self.get_embedding_dimension()

        # Check circuit breaker
        self.check_circuit_breaker()

        try:
            embedding = self._call_embedding_api(text)

            # Validate embedding
            if embedding and not is_zero_vector(embedding):
                self.record_success()
                self._consecutive_failures = 0
                return embedding
            else:
                # Invalid/empty embedding returned
                self._consecutive_failures += 1
                self.record_failure(Exception("Empty or invalid embedding returned"))

                if self._fail_fast:
                    raise EmbedderUnavailableError(
                        f"{self.name} returned empty/invalid embedding. "
                        f"Consecutive failures: {self._consecutive_failures}"
                    )
                return [0.0] * self.get_embedding_dimension()

        except EmbedderUnavailableError:
            # Re-raise our own exceptions
            raise

        except self._get_connection_exceptions() as e:
            self._consecutive_failures += 1
            self.logger.error(f"Connection error embedding text: {e}")
            self.record_failure(e)

            if self._fail_fast:
                raise EmbedderUnavailableError(
                    f"{self.name} is unavailable: {e}. "
                    f"Consecutive failures: {self._consecutive_failures}"
                ) from e
            return [0.0] * self.get_embedding_dimension()

        except Exception as e:
            self._consecutive_failures += 1
            self.logger.error(f"Error embedding text: {e}")
            self.record_failure(e)

            if self._fail_fast:
                raise EmbedderUnavailableError(
                    f"Failed to embed text: {e}. "
                    f"Consecutive failures: {self._consecutive_failures}"
                ) from e
            return [0.0] * self.get_embedding_dimension()

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
