"""
Capability Store - Persistent storage for capability embeddings.

Uses ChromaDB (already in LlamaFarm) for efficient similarity search.
Stores capability vectors separately from the main RAG database.

Usage:
    store = CapabilityStore()
    await store.initialize()
    
    # Add a capability
    await store.add_capability("research", "Deep research and analysis", vector)
    
    # Find matching capabilities
    matches = await store.search("I need to research something", query_vec, top_k=5)
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import chromadb, fall back to in-memory if not available
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    logger.warning("ChromaDB not available, using in-memory store (no persistence)")


@dataclass
class StoredCapability:
    """A capability stored in the database."""
    id: str
    label: str
    description: str
    handler: str
    node_id: str
    models: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "handler": self.handler,
            "node_id": self.node_id,
            "models": self.models,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredCapability":
        return cls(
            id=data["id"],
            label=data["label"],
            description=data["description"],
            handler=data.get("handler", ""),
            node_id=data.get("node_id", ""),
            models=data.get("models", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class SearchResult:
    """Result from capability search."""
    capability: StoredCapability
    score: float
    distance: float


class CapabilityStore:
    """
    Persistent store for capability embeddings.
    
    Uses ChromaDB for vector similarity search with these benefits:
    - Efficient cosine similarity search
    - Persistence across restarts
    - Same technology as LlamaFarm RAG
    
    Collection name: "capabilities" (separate from document RAG)
    """
    
    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = "capabilities",
        embedding_dimension: int = 384  # all-MiniLM-L6-v2
    ):
        self.persist_dir = persist_dir or Path("./data/router")
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        self._client = None
        self._collection = None
        self._initialized = False
        
        # Fallback in-memory store if ChromaDB not available
        self._memory_store: Dict[str, Dict] = {}
        self._memory_vectors: Dict[str, np.ndarray] = {}
    
    async def initialize(self) -> None:
        """Initialize the capability store."""
        if self._initialized:
            return
        
        if HAS_CHROMADB:
            await self._init_chromadb()
        else:
            await self._init_memory()
        
        self._initialized = True
    
    async def _init_chromadb(self) -> None:
        """Initialize ChromaDB backend."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Create persistent client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        
        count = self._collection.count()
        logger.info(f"CapabilityStore initialized with ChromaDB ({count} capabilities)")
    
    async def _init_memory(self) -> None:
        """Initialize in-memory fallback."""
        # Try to load from JSON file
        json_path = self.persist_dir / f"{self.collection_name}.json"
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
                for item in data.get("capabilities", []):
                    cap_id = item["id"]
                    self._memory_store[cap_id] = item
                    self._memory_vectors[cap_id] = np.array(item["vector"], dtype=np.float32)
                logger.info(f"CapabilityStore loaded {len(self._memory_store)} capabilities from JSON")
            except Exception as e:
                logger.error(f"Failed to load capabilities from JSON: {e}")
        else:
            logger.info("CapabilityStore initialized with empty in-memory store")
    
    async def add_capability(
        self,
        cap_id: str,
        label: str,
        description: str,
        vector: np.ndarray,
        handler: str = "",
        node_id: str = "",
        models: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StoredCapability:
        """Add a capability to the store."""
        if not self._initialized:
            await self.initialize()
        
        capability = StoredCapability(
            id=cap_id,
            label=label,
            description=description,
            handler=handler,
            node_id=node_id,
            models=models or [],
            metadata=metadata or {}
        )
        
        if HAS_CHROMADB and self._collection:
            # Store in ChromaDB
            self._collection.upsert(
                ids=[cap_id],
                embeddings=[vector.tolist()],
                metadatas=[capability.to_dict()],
                documents=[f"{label}: {description}"]
            )
        else:
            # Store in memory
            self._memory_store[cap_id] = capability.to_dict()
            self._memory_vectors[cap_id] = vector.copy()
            await self._save_memory()
        
        logger.debug(f"Added capability: {cap_id}")
        return capability
    
    async def remove_capability(self, cap_id: str) -> bool:
        """Remove a capability from the store."""
        if not self._initialized:
            await self.initialize()
        
        if HAS_CHROMADB and self._collection:
            try:
                self._collection.delete(ids=[cap_id])
                return True
            except Exception:
                return False
        else:
            if cap_id in self._memory_store:
                del self._memory_store[cap_id]
                del self._memory_vectors[cap_id]
                await self._save_memory()
                return True
            return False
    
    async def get_capability(self, cap_id: str) -> Optional[StoredCapability]:
        """Get a capability by ID."""
        if not self._initialized:
            await self.initialize()
        
        if HAS_CHROMADB and self._collection:
            try:
                result = self._collection.get(ids=[cap_id], include=["metadatas"])
                if result["ids"]:
                    return StoredCapability.from_dict(result["metadatas"][0])
            except Exception:
                pass
            return None
        else:
            if cap_id in self._memory_store:
                return StoredCapability.from_dict(self._memory_store[cap_id])
            return None
    
    async def search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for capabilities matching a query.
        
        Args:
            query_text: Original query text (for logging)
            query_vector: Query embedding vector
            top_k: Maximum results to return
            min_score: Minimum similarity score (0-1)
        
        Returns:
            List of SearchResult sorted by score descending
        """
        if not self._initialized:
            await self.initialize()
        
        results = []
        
        if HAS_CHROMADB and self._collection:
            # ChromaDB search
            try:
                search_results = self._collection.query(
                    query_embeddings=[query_vector.tolist()],
                    n_results=top_k,
                    include=["metadatas", "distances"]
                )
                
                if search_results["ids"] and search_results["ids"][0]:
                    for i, cap_id in enumerate(search_results["ids"][0]):
                        distance = search_results["distances"][0][i]
                        # ChromaDB cosine distance is 1 - similarity
                        score = 1.0 - distance
                        
                        if score >= min_score:
                            metadata = search_results["metadatas"][0][i]
                            capability = StoredCapability.from_dict(metadata)
                            results.append(SearchResult(
                                capability=capability,
                                score=score,
                                distance=distance
                            ))
            except Exception as e:
                logger.error(f"ChromaDB search failed: {e}")
        else:
            # In-memory search
            for cap_id, vector in self._memory_vectors.items():
                # Cosine similarity (assuming normalized vectors)
                score = float(np.dot(query_vector, vector))
                
                if score >= min_score:
                    capability = StoredCapability.from_dict(self._memory_store[cap_id])
                    results.append(SearchResult(
                        capability=capability,
                        score=score,
                        distance=1.0 - score
                    ))
            
            # Sort by score descending
            results.sort(key=lambda r: r.score, reverse=True)
            results = results[:top_k]
        
        return results
    
    async def list_capabilities(
        self,
        node_id: Optional[str] = None,
        limit: int = 100
    ) -> List[StoredCapability]:
        """List all capabilities, optionally filtered by node."""
        if not self._initialized:
            await self.initialize()
        
        capabilities = []
        
        if HAS_CHROMADB and self._collection:
            try:
                if node_id:
                    results = self._collection.get(
                        where={"node_id": node_id},
                        limit=limit,
                        include=["metadatas"]
                    )
                else:
                    results = self._collection.get(
                        limit=limit,
                        include=["metadatas"]
                    )
                
                for metadata in results["metadatas"]:
                    capabilities.append(StoredCapability.from_dict(metadata))
            except Exception as e:
                logger.error(f"Failed to list capabilities: {e}")
        else:
            for cap_data in list(self._memory_store.values())[:limit]:
                cap = StoredCapability.from_dict(cap_data)
                if node_id is None or cap.node_id == node_id:
                    capabilities.append(cap)
        
        return capabilities
    
    async def count(self) -> int:
        """Get total number of stored capabilities."""
        if not self._initialized:
            await self.initialize()
        
        if HAS_CHROMADB and self._collection:
            return self._collection.count()
        else:
            return len(self._memory_store)
    
    async def clear(self) -> None:
        """Clear all capabilities from the store."""
        if not self._initialized:
            await self.initialize()
        
        if HAS_CHROMADB and self._collection:
            # Delete and recreate collection
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self._memory_store.clear()
            self._memory_vectors.clear()
            await self._save_memory()
        
        logger.info("Capability store cleared")
    
    async def _save_memory(self) -> None:
        """Save in-memory store to JSON."""
        if HAS_CHROMADB:
            return  # ChromaDB handles persistence
        
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.persist_dir / f"{self.collection_name}.json"
        
        data = {
            "capabilities": [
                {**cap_data, "vector": self._memory_vectors[cap_id].tolist()}
                for cap_id, cap_data in self._memory_store.items()
            ]
        }
        json_path.write_text(json.dumps(data, indent=2))
    
    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "backend": "chromadb" if HAS_CHROMADB else "memory",
            "collection": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "embedding_dimension": self.embedding_dimension,
            "initialized": self._initialized,
            "count": self._collection.count() if (HAS_CHROMADB and self._collection) else len(self._memory_store)
        }


# Convenience function to get embeddings from LlamaFarm
async def get_embedding(text: str, host: str = "localhost", port: int = 14345, normalize: bool = True) -> np.ndarray:
    """
    Get embedding from LlamaFarm server.
    
    Uses /v1/nlp/embeddings endpoint with default model.
    Returns L2-normalized vector for cosine similarity.
    """
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://{host}:{port}/v1/nlp/embeddings",
            json={
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "input": text
            }
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Embedding failed: {await resp.text()}")
            data = await resp.json()
            vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
            
            # L2 normalize for cosine similarity
            if normalize:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            
            return vec
