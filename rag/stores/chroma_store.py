"""ChromaDB vector store implementation."""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from core.base import VectorStore, Document


class ChromaStore(VectorStore):
    """ChromaDB vector store implementation."""

    def __init__(self, name: str = "ChromaStore", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.collection_name = config.get("collection_name", "documents")
        self.persist_directory = config.get("persist_directory", "./chroma_db")
        self.host = config.get("host")
        self.port = config.get("port")

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
            raise ValueError(f"Failed to connect to ChromaDB: {e}")

    def _setup_collection(self):
        """Setup or get collection."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            self.logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(name=self.collection_name)
            self.logger.info(f"Created new collection: {self.collection_name}")

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to ChromaDB."""
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents_content = []

            for doc in documents:
                if not doc.embeddings:
                    self.logger.warning(
                        f"Document {doc.id} has no embeddings, skipping"
                    )
                    continue

                ids.append(doc.id)
                embeddings.append(doc.embeddings)
                documents_content.append(doc.content)

                # ChromaDB metadata must be JSON serializable
                metadata = self._serialize_metadata(doc.metadata)
                metadatas.append(metadata)

            if not ids:
                self.logger.warning("No documents with embeddings to add")
                return False

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents_content,
                metadatas=metadatas,
            )

            self.logger.info(f"Added {len(ids)} documents to collection")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False

    def search(
        self,
        query: str = None,
        query_embedding: List[float] = None,
        top_k: int = 10,
        where: Dict[str, Any] = None,
    ) -> List[Document]:
        """Search for similar documents."""
        try:
            if query_embedding is None and query is None:
                raise ValueError("Either query or query_embedding must be provided")

            # Prepare search parameters
            search_params = {"n_results": top_k}

            if query_embedding:
                search_params["query_embeddings"] = [query_embedding]
            else:
                search_params["query_texts"] = [query]

            if where:
                search_params["where"] = where

            # Search
            results = self.collection.query(**search_params)

            # Convert results to Document objects
            documents = []
            if results["ids"] and results["ids"][0]:  # Check if we got any results
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = Document(
                        id=doc_id,
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] or {},
                        embeddings=(
                            results["embeddings"][0][i]
                            if results.get("embeddings")
                            else None
                        ),
                    )

                    # Add distance/similarity score if available
                    if results.get("distances") and len(results["distances"][0]) > i:
                        doc.metadata["similarity_score"] = (
                            1 - results["distances"][0][i]
                        )  # Convert distance to similarity

                    documents.append(doc)

            return documents

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "client_type": type(self.client).__name__,
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {}

    def delete_documents(self, filter_criteria: Dict[str, Any]) -> int:
        """Delete documents based on filter criteria."""
        try:
            # ChromaDB delete with where clause
            where_clause = self._build_where_clause(filter_criteria)
            if where_clause:
                self.collection.delete(where=where_clause)
                # ChromaDB doesn't return count, so we estimate
                self.logger.info(f"Deleted documents matching filter: {filter_criteria}")
                return 1  # Placeholder count
            else:
                self.logger.warning("No valid filter criteria provided")
                return 0
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            return 0
    
    def update_metadata(self, filter_criteria: Dict[str, Any], update_metadata: Dict[str, Any]) -> int:
        """Update metadata for documents matching filter criteria."""
        try:
            # ChromaDB doesn't support direct metadata updates
            # Would need to query, modify, and upsert
            self.logger.warning("ChromaDB metadata updates require query + upsert pattern")
            return 0
        except Exception as e:
            self.logger.error(f"Failed to update metadata: {e}")
            return 0
    
    def query_documents(
        self, 
        filter_criteria: Dict[str, Any], 
        include_embeddings: bool = False
    ) -> List[Document]:
        """Query documents based on filter criteria without embeddings."""
        try:
            where_clause = self._build_where_clause(filter_criteria)
            
            # Query with metadata filter only (no similarity search)
            results = self.collection.get(
                where=where_clause,
                include=["documents", "metadatas", "embeddings" if include_embeddings else None]
            )
            
            documents = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    doc = Document(
                        id=doc_id,
                        content=results["documents"][i] if results.get("documents") else "",
                        metadata=results["metadatas"][i] or {},
                        embeddings=results["embeddings"][i] if include_embeddings and results.get("embeddings") else None
                    )
                    documents.append(doc)
            
            return documents
        except Exception as e:
            self.logger.error(f"Failed to query documents: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Get a specific document by ID."""
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if results["ids"] and len(results["ids"]) > 0:
                return Document(
                    id=results["ids"][0],
                    content=results["documents"][0] if results.get("documents") else "",
                    metadata=results["metadatas"][0] or {},
                    embeddings=results["embeddings"][0] if results.get("embeddings") else None
                )
            return None
        except Exception as e:
            self.logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def list_documents(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None
    ) -> List[Document]:
        """List documents with optional pagination."""
        try:
            get_params = {
                "include": ["documents", "metadatas"]
            }
            
            if limit:
                get_params["limit"] = limit
            if offset:
                get_params["offset"] = offset
            
            results = self.collection.get(**get_params)
            
            documents = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    doc = Document(
                        id=doc_id,
                        content=results["documents"][i] if results.get("documents") else "",
                        metadata=results["metadatas"][i] or {}
                    )
                    documents.append(doc)
            
            return documents
        except Exception as e:
            self.logger.error(f"Failed to list documents: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed collection statistics."""
        try:
            count = self.collection.count()
            
            # Get sample to analyze metadata patterns
            sample = self.collection.get(limit=100, include=["metadatas"])
            
            # Analyze metadata
            active_count = 0
            deleted_count = 0
            versions = set()
            
            for metadata in sample.get("metadatas", []):
                if metadata:
                    if metadata.get("is_active", True):
                        active_count += 1
                    else:
                        deleted_count += 1
                    
                    if "version" in metadata:
                        versions.add(metadata["version"])
            
            return {
                "total_documents": count,
                "active_documents": active_count,
                "deleted_documents": deleted_count,
                "unique_versions": len(versions),
                "storage_size_mb": 0.0,  # ChromaDB doesn't expose this directly
                "collection_name": self.collection_name,
                "client_type": type(self.client).__name__
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {
                "total_documents": 0,
                "active_documents": 0,
                "deleted_documents": 0,
                "unique_versions": 0,
                "storage_size_mb": 0.0
            }
    
    def _build_where_clause(self, filter_criteria: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause from filter criteria."""
        if not filter_criteria:
            return None
        
        where_conditions = {}
        
        for key, value in filter_criteria.items():
            if isinstance(value, dict):
                # Handle operators like $lt, $gt, $in, etc.
                for op, op_value in value.items():
                    if op == "$lt":
                        where_conditions[key] = {"$lt": op_value}
                    elif op == "$gt":
                        where_conditions[key] = {"$gt": op_value}
                    elif op == "$lte":
                        where_conditions[key] = {"$lte": op_value}
                    elif op == "$gte":
                        where_conditions[key] = {"$gte": op_value}
                    elif op == "$ne":
                        where_conditions[key] = {"$ne": op_value}
                    elif op == "$in":
                        where_conditions[key] = {"$in": op_value}
                    elif op == "$nin":
                        where_conditions[key] = {"$nin": op_value}
            else:
                # Direct equality match
                where_conditions[key] = value
        
        return where_conditions if where_conditions else None
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize metadata for ChromaDB storage."""
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                serialized[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                serialized[key] = ",".join(map(str, value))
            else:
                # Convert other types to string
                serialized[key] = str(value)

        return serialized
