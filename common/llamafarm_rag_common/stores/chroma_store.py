"""ChromaDB vector store for shared RAG library (embedded mode only)."""

import json
import logging
import threading
from typing import Any

import chromadb

from ..models import Document
from .base import VectorStore

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """ChromaDB vector store implementation (embedded/persistent client only)."""

    # Class-level client cache
    _client_cache: dict[str, chromadb.ClientAPI] = {}
    _client_cache_lock = threading.Lock()

    def __init__(
        self,
        name: str = "ChromaStore",
        config: dict[str, Any] | None = None,
        project_dir: Any = None,
    ):
        super().__init__(name, config, project_dir)
        config = config or {}

        self.collection_name = config.get("collection_name", "documents")
        self.embedding_dimension = max(config.get("embedding_dimension", 768), 1)
        self.distance_metric = config.get("distance_metric", "cosine")

        # Validate distance metric
        valid_metrics = ["cosine", "l2", "ip"]
        if self.distance_metric not in valid_metrics:
            logger.warning(
                f"Invalid distance metric '{self.distance_metric}', using 'cosine'"
            )
            self.distance_metric = "cosine"

        # Initialize ChromaDB client (persistent/embedded only for server)
        client_key = f"persistent://{self.persist_directory}"
        logger.info(
            f"Using ChromaDB persistent client: {self.persist_directory}"
        )
        self.client = self._get_or_create_client(
            client_key,
            lambda: chromadb.PersistentClient(path=self.persist_directory),
        )

        self._setup_collection()

    @classmethod
    def _get_or_create_client(
        cls, client_key: str, client_factory
    ) -> chromadb.ClientAPI:
        """Get or create a ChromaDB client from cache."""
        with cls._client_cache_lock:
            if client_key not in cls._client_cache:
                logger.info(f"Creating new ChromaDB client: {client_key}")
                cls._client_cache[client_key] = client_factory()
            return cls._client_cache[client_key]

    def _setup_collection(self):
        """Setup or get collection."""
        metric_map = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip",
        }

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": metric_map.get(self.distance_metric, "cosine")},
        )
        logger.info(
            f"Collection ready: {self.collection_name} with {self.distance_metric} distance"
        )

    def _parse_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Parse JSON strings in metadata back to Python objects."""
        parsed = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                if value.startswith("{") or value.startswith("["):
                    try:
                        parsed[key] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        parsed[key] = value
                else:
                    parsed[key] = value
            else:
                parsed[key] = value
        return parsed

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return

        # Prepare data for ChromaDB
        ids: list[str] = []
        embeddings = []
        metadatas = []
        contents = []

        for doc in documents:
            if not doc.embeddings:
                logger.warning(f"Document {doc.id} has no embeddings, skipping")
                continue

            ids.append(doc.id)
            embeddings.append(doc.embeddings)
            contents.append(doc.content)

            # Serialize metadata (ChromaDB doesn't accept None values)
            metadata = {}
            for key, value in doc.metadata.items():
                if value is None:
                    continue  # Skip None values - ChromaDB doesn't support them
                if isinstance(value, (dict, list)):
                    metadata[key] = json.dumps(value)
                elif isinstance(value, bool):
                    metadata[key] = value  # ChromaDB supports bools
                elif isinstance(value, (int, float, str)):
                    metadata[key] = value
                else:
                    # Convert other types to string
                    metadata[key] = str(value)
            metadatas.append(metadata)

        if not ids:
            logger.warning("No documents with embeddings to add")
            return

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=contents,
        )

        logger.info(f"Added {len(ids)} documents to {self.collection_name}")

    def search(
        self,
        query: str = "",
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        **kwargs,
    ) -> list[Document]:
        """Search for similar documents."""
        if not query_embedding:
            raise ValueError("query_embedding is required")

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # Convert results to Documents
        documents = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                # Get metadata and parse JSON strings
                metadata = (
                    self._parse_metadata(results["metadatas"][0][i])
                    if results["metadatas"]
                    else {}
                )

                # Add distance/score to metadata
                if results["distances"]:
                    metadata["_score"] = results["distances"][0][i]

                # Create Document
                doc = Document(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=metadata,
                    embeddings=None,  # Don't return embeddings to save memory
                )
                documents.append(doc)

        return documents

    def delete(self, document_ids: list[str]) -> None:
        """Delete documents by ID."""
        if not document_ids:
            return

        self.collection.delete(ids=document_ids)
        logger.info(f"Deleted {len(document_ids)} documents from {self.collection_name}")

    def get_count(self) -> int:
        """Get total document count."""
        return self.collection.count()
