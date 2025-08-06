"""ChromaDB vector store implementation."""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from core.base import VectorStore, Document

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """ChromaDB vector store implementation."""

    def __init__(self, name: str = "ChromaStore", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        config = config or {}
        self.collection_name = config.get("collection_name", "documents")
        self.persist_directory = config.get("persist_directory", "./chroma_db")
        self.host = config.get("host")
        self.port = config.get("port")
        self.embedding_dimension = max(config.get("embedding_dimension", 768), 1)  # Ensure positive

        # Initialize ChromaDB client
        if self.host and self.port:
            # HTTP client
            self.client = chromadb.HttpClient(host=self.host, port=self.port)
        else:
            # Persistent client
            self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.collection = None
        self._setup_collection()

    def validate_config(self) -> bool:
        """Validate configuration."""
        try:
            # Test connection
            self.client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            return False

    def _setup_collection(self):
        """Setup or get collection."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"Created new collection: {self.collection_name}")

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store."""
        try:
            if not documents:
                return True

            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents_content = []

            for doc in documents:
                if not doc.embeddings:
                    logger.warning(f"Document {doc.id} has no embeddings, skipping")
                    continue

                ids.append(doc.id or f"doc_{len(ids)}")
                embeddings.append(doc.embeddings)
                
                # Clean metadata - ChromaDB only accepts str, int, float, bool, None
                cleaned_metadata = {}
                
                # Always include the source if available
                if doc.source:
                    cleaned_metadata['source'] = doc.source
                
                if doc.metadata:
                    for key, value in doc.metadata.items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            cleaned_metadata[key] = value
                        elif isinstance(value, list):
                            # Convert lists to comma-separated strings (no spaces after commas for test compatibility)
                            cleaned_metadata[key] = ",".join(str(v) for v in value)
                        elif isinstance(value, dict):
                            # Convert dicts to JSON string
                            import json
                            cleaned_metadata[key] = json.dumps(value)
                        else:
                            # Convert other types to string
                            cleaned_metadata[key] = str(value)
                
                metadatas.append(cleaned_metadata)
                documents_content.append(doc.content)

            if not ids:
                logger.warning("No valid documents with embeddings to add")
                return True

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_content
            )

            logger.info(f"Added {len(ids)} documents to ChromaDB collection")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            return False

    def search(self, query: str = None, top_k: int = 10, query_embedding: Optional[List[float]] = None, where: Optional[Dict[str, Any]] = None, **kwargs) -> List[Document]:
        """Search for similar documents."""
        try:
            if query_embedding is None:
                # If no embedding provided, we can't search
                # In a real implementation, you'd embed the query here
                logger.warning("No query embedding provided for search")
                return []

            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k
            }
            
            # Add metadata filtering if provided
            if where:
                query_params["where"] = where

            results = self.collection.query(**query_params)

            documents = []
            if results and results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    content = results['documents'][0][i] if results['documents'] and results['documents'][0] else ""
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    
                    # Add distance/score to metadata
                    if results['distances'] and results['distances'][0]:
                        metadata['_score'] = results['distances'][0][i]

                    # Preserve source from metadata if available
                    source = metadata.get('file_path') or metadata.get('source')
                    
                    doc = Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        source=source
                    )
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            return []

    def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            # Recreate collection for continued use
            self._setup_collection()
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a specific document by ID."""
        try:
            results = self.collection.get(ids=[doc_id])
            if results and results['ids'] and results['ids'][0] == doc_id:
                content = results['documents'][0] if results['documents'] else ""
                metadata = results['metadatas'][0] if results['metadatas'] else {}
                
                return Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,  # Keep as "count" for test compatibility
                "document_count": count,  # Also provide "document_count" for other uses
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    @classmethod
    def get_description(cls) -> str:
        """Get store description."""
        return "ChromaDB vector store for persistent document storage and similarity search."