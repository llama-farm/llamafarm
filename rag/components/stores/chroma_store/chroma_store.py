"""ChromaDB vector store implementation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from core.base import Document, VectorStore
from utils.hash_utils import DeduplicationTracker

from core.logging import RAGStructLogger
logger = RAGStructLogger("rag.components.stores.chroma_store.chroma_store")


class ChromaStore(VectorStore):
    """ChromaDB vector store implementation."""

    def __init__(
        self,
        name: str = "ChromaStore",
        config: Optional[dict[str, Any]] = None,
        project_dir: Optional[Path] = None,
    ):
        super().__init__(name, config, project_dir)  # type: ignore
        config = config or {}
        self.collection_name = config.get("collection_name", "documents")
        self.host = config.get("host")
        self.port = config.get("port")
        self.embedding_dimension = max(
            config.get("embedding_dimension", 768), 1
        )  # Ensure positive

        # Get distance metric from config, default to cosine for best compatibility
        self.distance_metric = config.get("distance_metric", "cosine")

        # Validate distance metric
        valid_metrics = ["cosine", "l2", "ip"]
        if self.distance_metric not in valid_metrics:
            logger.warning(
                f"Invalid distance metric '{self.distance_metric}', using 'cosine'"
            )
            self.distance_metric = "cosine"

        # Initialize ChromaDB client
        if self.host and self.port:
            # HTTP client
            self.client = chromadb.HttpClient(host=self.host, port=self.port)
        else:
            # Persistent client
            logger.info(
                f"Using ChromaDB persistent client with persist_directory: {self.persist_directory}"
            )
            self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.collection = None
        self._setup_collection()

        # Initialize deduplication tracker
        self.deduplication_enabled = config.get("enable_deduplication", True)
        self.dedup_tracker = (
            DeduplicationTracker() if self.deduplication_enabled else None
        )

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
                "ip": "ip",  # inner product
            }

            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": metric_map.get(self.distance_metric, "cosine")},
            )
            logger.info(
                f"Created new collection: {self.collection_name} with {self.distance_metric} distance"
            )

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
                if value.startswith("{") or value.startswith("["):
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
        """Add documents to the vector store with deduplication support."""
        try:
            if not documents:
                return True

            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents_content = []
            skipped_duplicates = 0

            for doc in documents:
                if not doc.embeddings:
                    logger.warning(f"Document {doc.id} has no embeddings, skipping")
                    continue

                # Check for duplicates if deduplication is enabled
                if self.deduplication_enabled and self.dedup_tracker and doc.metadata:
                    document_hash = doc.metadata.get("document_hash")
                    chunk_hash = doc.metadata.get("chunk_hash")
                    source_hash = doc.metadata.get("source_hash")

                    # Check if this document or chunk already exists
                    is_duplicate_doc = (
                        document_hash
                        and self.dedup_tracker.is_duplicate_document(document_hash)
                    )
                    is_duplicate_chunk = (
                        chunk_hash and self.dedup_tracker.is_duplicate_chunk(chunk_hash)
                    )
                    is_duplicate_source = (
                        source_hash
                        and self.dedup_tracker.is_duplicate_source(source_hash)
                    )

                    if is_duplicate_doc or is_duplicate_chunk or is_duplicate_source:
                        logger.debug(f"Skipping duplicate document {doc.id}")
                        skipped_duplicates += 1
                        continue

                    # Check if document already exists in ChromaDB by ID
                    if self._document_exists(doc.id):
                        logger.debug(
                            f"Document {doc.id} already exists in collection, skipping"
                        )
                        skipped_duplicates += 1
                        continue

                    # Register in dedup tracker
                    if document_hash:
                        self.dedup_tracker.register_document(
                            document_hash, doc.id, source_hash or ""
                        )
                    if chunk_hash:
                        self.dedup_tracker.register_chunk(chunk_hash, doc.id)

                ids.append(doc.id or f"doc_{len(ids)}")
                embeddings.append(doc.embeddings)

                # Clean metadata - ChromaDB only accepts str, int, float, bool, None
                cleaned_metadata = {}

                # Always include the source if available
                if doc.source:
                    cleaned_metadata["source"] = doc.source

                if doc.metadata:
                    for key, value in doc.metadata.items():
                        if value is None:
                            # Skip None values as ChromaDB doesn't accept them
                            continue
                        elif isinstance(value, (str, int, float, bool)):
                            cleaned_metadata[key] = value
                        elif isinstance(value, list):
                            # Convert lists to comma-separated strings (no spaces after commas for test compatibility)
                            # Filter out None values from lists
                            filtered_list = [str(v) for v in value if v is not None]
                            if (
                                filtered_list
                            ):  # Only add if list is not empty after filtering
                                cleaned_metadata[key] = ",".join(filtered_list)
                        elif isinstance(value, dict):
                            # Convert dicts to JSON string, filtering out None values
                            import json

                            filtered_dict = {
                                k: v for k, v in value.items() if v is not None
                            }
                            if (
                                filtered_dict
                            ):  # Only add if dict is not empty after filtering
                                cleaned_metadata[key] = json.dumps(filtered_dict)
                        else:
                            # Convert other types to string
                            str_value = str(value)
                            if (
                                str_value != "None"
                            ):  # Don't add string representations of None
                                cleaned_metadata[key] = str_value

                metadatas.append(cleaned_metadata)
                documents_content.append(doc.content)

            if not ids:
                logger.warning(
                    "No valid documents with embeddings to add (all may be duplicates)"
                )
                return []  # Return empty list when no documents were added

            # DEBUG: Check for duplicate IDs
            id_counts = {}
            for id in ids:
                id_counts[id] = id_counts.get(id, 0) + 1
            duplicates = {id: count for id, count in id_counts.items() if count > 1}
            if duplicates:
                print(f"DUPLICATE IDS FOUND: {duplicates}")
                # Remove duplicates, keeping only first occurrence
                seen_ids = set()
                unique_data = []
                for i, id in enumerate(ids):
                    if id not in seen_ids:
                        seen_ids.add(id)
                        unique_data.append(
                            (ids[i], embeddings[i], metadatas[i], documents_content[i])
                        )

                ids = [d[0] for d in unique_data]
                embeddings = [d[1] for d in unique_data]
                metadatas = [d[2] for d in unique_data]
                documents_content = [d[3] for d in unique_data]
                print(f"Removed duplicates, now have {len(ids)} unique documents")

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_content,
            )

            if skipped_duplicates > 0:
                logger.info(
                    f"Added {len(ids)} documents, skipped {skipped_duplicates} duplicates"
                )
            else:
                logger.info(f"Added {len(ids)} documents to ChromaDB collection")
            return ids  # Return the list of IDs that were actually added

        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            print(f"ERROR STORING DOCUMENTS: {e}")  # DEBUG
            import traceback

            traceback.print_exc()  # DEBUG
            return False

    def search(
        self,
        query: str = None,
        top_k: int = 10,
        query_embedding: Optional[List[float]] = None,
        where: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """Search for similar documents."""
        try:
            if query_embedding is None:
                # If no embedding provided, we can't search
                # In a real implementation, you'd embed the query here
                logger.warning("No query embedding provided for search")
                return []

            query_params = {"query_embeddings": [query_embedding], "n_results": top_k}

            # Add metadata filtering if provided
            if where:
                query_params["where"] = where

            results = self.collection.query(**query_params)

            documents = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    content = (
                        results["documents"][0][i]
                        if results["documents"] and results["documents"][0]
                        else ""
                    )
                    metadata = (
                        results["metadatas"][0][i]
                        if results["metadatas"] and results["metadatas"][0]
                        else {}
                    )

                    # Parse JSON strings in metadata (ChromaDB stores nested objects as JSON)
                    metadata = self._parse_metadata(metadata)

                    # Add distance/score to metadata
                    # Convert distance to similarity score (ChromaDB returns distances, lower is better)
                    if results["distances"] and results["distances"][0]:
                        distance = results["distances"][0][i]
                        metadata["_score"] = (
                            distance  # Keep original distance for reference
                        )

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

                        metadata["similarity_score"] = similarity

                    # Preserve source from metadata if available
                    # Try multiple fields that might contain the source
                    source = (
                        metadata.get("file_path")
                        or metadata.get("source")
                        or metadata.get("file_name")
                        or "unknown"
                    )

                    doc = Document(
                        id=doc_id, content=content, metadata=metadata, source=source
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
            if results and results["ids"] and results["ids"][0] == doc_id:
                content = results["documents"][0] if results["documents"] else ""
                metadata = results["metadatas"][0] if results["metadatas"] else {}

                # Parse JSON strings in metadata
                metadata = self._parse_metadata(metadata)

                return Document(id=doc_id, content=content, metadata=metadata)
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

    def delete_by_document_hash(self, document_hash: str) -> bool:
        """Delete all chunks belonging to a specific document by its hash."""
        try:
            # Find all documents with this document hash
            results = self.collection.get(
                where={"document_hash": document_hash}, include=["metadatas"]
            )

            if results and results["ids"]:
                doc_ids = results["ids"]
                self.collection.delete(ids=doc_ids)
                logger.info(
                    f"Deleted {len(doc_ids)} documents with hash {document_hash[:12]}..."
                )
                return True
            else:
                logger.info(f"No documents found with hash {document_hash[:12]}...")
                return True

        except Exception as e:
            logger.error(f"Failed to delete documents by hash: {e}")
            return False

    def delete_by_source(self, source_path: str) -> bool:
        """Delete all documents from a specific source file."""
        try:
            # Try both exact match and path-based matching
            where_conditions = [{"source": source_path}, {"file_path": source_path}]

            total_deleted = 0
            for where_condition in where_conditions:
                try:
                    results = self.collection.get(
                        where=where_condition, include=["metadatas"]
                    )

                    if results and results["ids"]:
                        doc_ids = results["ids"]
                        self.collection.delete(ids=doc_ids)
                        total_deleted += len(doc_ids)

                except Exception:
                    # Continue to next condition if this one fails
                    continue

            if total_deleted > 0:
                logger.info(
                    f"Deleted {total_deleted} documents from source {source_path}"
                )
                return True
            else:
                logger.info(f"No documents found from source {source_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete documents by source: {e}")
            return False

    def _document_exists(self, doc_id: str) -> bool:
        """Check if a document with the given ID already exists."""
        try:
            results = self.collection.get(ids=[doc_id], include=[])
            return bool(results and results["ids"] and doc_id in results["ids"])
        except Exception:
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,  # Keep as "count" for test compatibility
                "document_count": count,  # Also provide "document_count" for other uses
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    @classmethod
    def get_description(cls) -> str:
        """Get store description."""
        return "ChromaDB vector store for persistent document storage and similarity search."
