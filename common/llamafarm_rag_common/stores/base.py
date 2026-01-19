"""Base vector store for shared RAG library."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import Document


class VectorStore(ABC):
    """Base class for vector stores."""

    def __init__(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ):
        self.name = name
        self.config = config or {}
        self.project_dir = project_dir

        # Set persist directory
        if project_dir:
            self.persist_directory = str(project_dir / "chroma_db")
        else:
            import os
            data_dir = os.path.expanduser(os.getenv("LF_DATA_DIR", "~/.llamafarm"))
            self.persist_directory = os.path.join(data_dir, "chroma_db")

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    def search(
        self,
        query: str = "",
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        **kwargs,
    ) -> list[Document]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete(self, document_ids: list[str]) -> None:
        """Delete documents by ID."""
        pass

    @abstractmethod
    def get_count(self) -> int:
        """Get total document count."""
        pass
