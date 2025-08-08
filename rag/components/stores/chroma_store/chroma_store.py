"""ChromaDB vector store implementation."""

import json
import logging
import math
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
        
        # Get distance metric from config, default to cosine for best compatibility
        self.distance_metric = config.get("distance_metric", "cosine")
        
        # Validate distance metric
        valid_metrics = ["cosine", "l2", "ip"]
        if self.distance_metric not in valid_metrics:
            logger.warning(f"Invalid distance metric '{self.distance_metric}', using 'cosine'")
            self.distance_metric = "cosine"

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
            # Collection doesn't exist, create it with configured distance metric
            # Map our metric names to ChromaDB's HNSW space names
            metric_map = {
                "cosine": "cosine",
                "l2": "l2",
                "ip": "ip"  # inner product
            }
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": metric_map.get(self.distance_metric, "cosine")}
            )
            logger.info(f"Created new collection: {self.collection_name} with {self.distance_metric} distance")
    
    def _parse_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON strings in metadata back to Python objects.
        
        ChromaDB stores nested objects as JSON strings, so we need to parse them
        back when retrieving documents.
        
        Args:
            metadata: Raw metadata from ChromaDB
            
        Returns:
            Parsed metadata with nested objects restored
        """
        parsed = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                # Try to parse as JSON if it looks like JSON
                if value.startswith('{') or value.startswith('['):
                    try:
                        parsed[key] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        # Not valid JSON, keep as string
                        parsed[key] = value
                else:
                    parsed[key] = value
            else:
                parsed[key] = value
        return parsed

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
                    
                    # Parse JSON strings in metadata (ChromaDB stores nested objects as JSON)
                    metadata = self._parse_metadata(metadata)
                    
                    # Add distance/score to metadata
                    # Convert distance to similarity score (ChromaDB returns distances, lower is better)
                    if results['distances'] and results['distances'][0]:
                        distance = results['distances'][0][i]
                        metadata['_score'] = distance  # Keep original distance for reference
                        
                        # Convert distance to similarity score based on metric type
                        if self.distance_metric == "cosine":
                            # Cosine distance: 0 = identical, 2 = opposite
                            # Convert to similarity: 1 - (distance / 2)
                            if distance == 0:
                                similarity = 1.0
                            elif distance >= 2:
                                similarity = 0.0
                            else:
                                similarity = 1.0 - (distance / 2.0)
                        
                        elif self.distance_metric == "l2":
                            # L2 (Euclidean) distance: 0 = identical, larger = more different
                            # For unnormalized embeddings, distances can be very large
                            # Use inverse transformation with scaling
                            if distance == 0:
                                similarity = 1.0
                            else:
                                # Adjust scale based on typical embedding norms
                                # This maps distances to similarities in range (0, 1]
                                similarity = 1.0 / (1.0 + distance / 100.0)
                        
                        elif self.distance_metric == "ip":
                            # Inner product: higher = more similar (opposite of distance)
                            # For normalized embeddings, ranges from -1 to 1
                            # ChromaDB may return negative of IP as distance
                            similarity = max(0.0, min(1.0, (1.0 + distance) / 2.0))
                        
                        else:
                            # Fallback for unknown metrics
                            similarity = 1.0 / (1.0 + distance)
                        
                        metadata['similarity_score'] = similarity

                    # Preserve source from metadata if available
                    # Try multiple fields that might contain the source
                    source = (metadata.get('file_path') or 
                             metadata.get('source') or 
                             metadata.get('file_name') or
                             'unknown')
                    
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
                
                # Parse JSON strings in metadata
                metadata = self._parse_metadata(metadata)
                
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