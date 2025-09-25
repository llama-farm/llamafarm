"""Base classes for the extensible RAG system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from core.settings import settings


@dataclass
class Document:
    """Universal document representation."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    source: Optional[str] = None
    embeddings: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
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

    documents: List[Document]
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class Component(ABC):
    """Base class for all pipeline components."""

    def __init__(
        self,
        name: str = None,
        config: Dict[str, Any] = None,
        project_dir: Path | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.logger = logging.getLogger(self.name)
        self.project_dir = project_dir

    @abstractmethod
    def process(self, documents: List[Document]) -> ProcessingResult:
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

    def process(self, documents: List[Document]) -> ProcessingResult:
        """Process already parsed documents (pass-through for parsers)."""
        return ProcessingResult(documents)


class Embedder(Component):
    """Base class for embedding generators."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass

    def process(self, documents: List[Document]) -> ProcessingResult:
        """Add embeddings to documents."""
        texts = [doc.content for doc in documents]
        embeddings = self.embed(texts)

        for doc, embedding in zip(documents, embeddings):
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
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Document]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete_collection(self) -> bool:
        """Delete the collection."""
        pass

    def process(self, documents: List[Document]) -> ProcessingResult:
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
        self.components: List[Component] = []
        self.logger = logging.getLogger(self.name)

    def add_component(self, component: Component) -> "Pipeline":
        """Add a component to the pipeline."""
        component.validate_config()
        self.components.append(component)
        return self

    def run(
        self, source: str = None, documents: List[Document] = None
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
