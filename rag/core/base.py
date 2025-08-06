"""Base classes for the extensible RAG system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging


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

    def __init__(self, name_or_config: Any = None, config: Dict[str, Any] = None):
        # Handle backwards compatibility
        # If first arg is a dict, treat it as config (old style)
        # If first arg is a string, treat it as name (new style)
        if isinstance(name_or_config, dict):
            # Old style: Component(config_dict)
            self.name = self.__class__.__name__
            self.config = name_or_config
        elif isinstance(name_or_config, str):
            # New style: Component(name, config)
            self.name = name_or_config
            self.config = config or {}
        elif name_or_config is None:
            # Default: Component()
            self.name = self.__class__.__name__
            self.config = config or {}
        else:
            # Fallback for any other type
            self.name = self.__class__.__name__
            self.config = {}
            
        self.logger = logging.getLogger(self.name)

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

        return ProcessingResult(
            documents=documents, metrics={"embedded_count": len(documents)}
        )


class VectorStore(Component):
    """Base class for vector databases."""

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
        return ProcessingResult(
            documents=documents,
            metrics={"stored_count": len(documents) if success else 0},
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
        for component in self.components[start_idx:]:
            try:
                result = component.process(current_docs)
                current_docs = result.documents
                all_errors.extend(result.errors)
            except Exception as e:
                self.logger.error(f"Component {component.name} failed: {e}")
                all_errors.append({"component": component.name, "error": str(e)})

        return ProcessingResult(documents=current_docs, errors=all_errors)
