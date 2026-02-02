"""
Gradient table management for semantic routing.

Each node maintains a gradient table - a map from capability embeddings
to routing information (next_hop, hops, latency).
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator
import threading
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Configuration constants
GRADIENT_EXPIRE_SEC = 300  # 5 minutes
MAX_GRADIENT_TABLE_SIZE = 1000
LINK_LATENCY_MS = 10


@dataclass
class GradientEntry:
    """
    A single entry in the gradient table.
    
    Represents a known path to a capability, potentially through
    multiple hops.
    """
    capability_id: str
    capability_label: str
    capability_vector: np.ndarray
    hops: int
    next_hop: str
    via_node: str
    estimated_latency_ms: float
    last_updated: float = field(default_factory=time.time)
    confidence: float = 1.0

    def is_expired(self, expire_sec: float = GRADIENT_EXPIRE_SEC) -> bool:
        """Check if this entry has expired."""
        return (time.time() - self.last_updated) > expire_sec

    def to_dict(self) -> dict:
        """Serialize for transmission."""
        return {
            "capability_id": self.capability_id,
            "capability_label": self.capability_label,
            "capability_vector": self.capability_vector.tolist(),
            "hops": self.hops,
            "next_hop": self.next_hop,
            "via_node": self.via_node,
            "estimated_latency_ms": self.estimated_latency_ms,
            "last_updated": self.last_updated,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GradientEntry":
        """Deserialize from transmission."""
        return cls(
            capability_id=data["capability_id"],
            capability_label=data["capability_label"],
            capability_vector=np.array(data["capability_vector"], dtype=np.float32),
            hops=data["hops"],
            next_hop=data["next_hop"],
            via_node=data["via_node"],
            estimated_latency_ms=data["estimated_latency_ms"],
            last_updated=data.get("last_updated", time.time()),
            confidence=data.get("confidence", 1.0)
        )


class GradientTable:
    """
    Gradient table for semantic routing.
    
    Maps capability IDs to routing information. Updated via gossip protocol
    when receiving capability announcements from peers.
    
    Thread-safe via RLock.
    """

    def __init__(
        self,
        node_id: str,
        max_size: int = MAX_GRADIENT_TABLE_SIZE,
        expire_sec: float = GRADIENT_EXPIRE_SEC
    ):
        self.node_id = node_id
        self.max_size = max_size
        self.expire_sec = expire_sec
        self._entries: Dict[str, GradientEntry] = {}
        self._lock = threading.RLock()
        self._vectors: Optional[np.ndarray] = None
        self._vector_ids: List[str] = []
        self._index_dirty = True

    def update(
        self,
        capability_id: str,
        capability_label: str,
        capability_vector: np.ndarray,
        hops: int,
        next_hop: str,
        via_node: str,
        estimated_latency_ms: Optional[float] = None
    ) -> bool:
        """
        Update gradient table with new routing info.
        
        Only updates if entry doesn't exist or new route has fewer hops.
        """
        if estimated_latency_ms is None:
            estimated_latency_ms = hops * LINK_LATENCY_MS

        with self._lock:
            existing = self._entries.get(capability_id)

            if existing is not None:
                if hops >= existing.hops:
                    if hops == existing.hops and next_hop == existing.next_hop:
                        existing.last_updated = time.time()
                    return False

            entry = GradientEntry(
                capability_id=capability_id,
                capability_label=capability_label,
                capability_vector=np.array(capability_vector, dtype=np.float32),
                hops=hops,
                next_hop=next_hop,
                via_node=via_node,
                estimated_latency_ms=estimated_latency_ms,
                last_updated=time.time(),
                confidence=0.95 ** hops
            )

            if len(self._entries) >= self.max_size and capability_id not in self._entries:
                self._evict_one()

            self._entries[capability_id] = entry
            self._index_dirty = True
            return True

    def _evict_one(self) -> None:
        """Evict the oldest/lowest confidence entry."""
        if not self._entries:
            return

        worst_id = None
        worst_score = float('inf')
        now = time.time()

        for cap_id, entry in self._entries.items():
            age = now - entry.last_updated
            score = entry.confidence / (1 + age / 60)
            if score < worst_score:
                worst_score = score
                worst_id = cap_id

        if worst_id:
            del self._entries[worst_id]
            self._index_dirty = True

    def remove(self, capability_id: str) -> bool:
        """Remove a capability from the table."""
        with self._lock:
            if capability_id in self._entries:
                del self._entries[capability_id]
                self._index_dirty = True
                return True
            return False

    def get(self, capability_id: str) -> Optional[GradientEntry]:
        """Get a specific entry by ID."""
        with self._lock:
            return self._entries.get(capability_id)

    def _rebuild_index(self) -> None:
        """Rebuild the vector index for fast similarity search."""
        if not self._index_dirty:
            return

        with self._lock:
            if not self._entries:
                self._vectors = None
                self._vector_ids = []
            else:
                self._vector_ids = list(self._entries.keys())
                self._vectors = np.stack([
                    self._entries[cap_id].capability_vector
                    for cap_id in self._vector_ids
                ])
            self._index_dirty = False

    def find_best_route(
        self,
        intent_vector: np.ndarray,
        min_score: float = 0.5
    ) -> Optional[GradientEntry]:
        """
        Find the best route for an intent.
        
        Uses cosine similarity with hop penalty.
        """
        with self._lock:
            self._rebuild_index()

            if self._vectors is None or len(self._vectors) == 0:
                return None

            similarities = self._vectors @ intent_vector

            best_idx = None
            best_adjusted = 0.0

            for idx, sim in enumerate(similarities):
                entry = self._entries[self._vector_ids[idx]]
                adjusted = sim * entry.confidence
                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_idx = idx

            if best_idx is not None and best_adjusted >= min_score:
                return self._entries[self._vector_ids[best_idx]]

            return None

    def prune_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            expired = [
                cap_id for cap_id, entry in self._entries.items()
                if entry.is_expired(self.expire_sec)
            ]

            for cap_id in expired:
                del self._entries[cap_id]

            if expired:
                self._index_dirty = True

            return len(expired)

    def export_for_gossip(self, max_hops: int = 5) -> List[dict]:
        """Export entries for gossip announcement."""
        with self._lock:
            return [
                entry.to_dict()
                for entry in self._entries.values()
                if entry.hops < max_hops and not entry.is_expired()
            ]

    def invalidate_node(self, node_id: str) -> int:
        """Remove all entries for a disconnected node."""
        with self._lock:
            to_remove = [
                cap_id for cap_id, entry in self._entries.items()
                if entry.next_hop == node_id
            ]

            for cap_id in to_remove:
                del self._entries[cap_id]

            if to_remove:
                self._index_dirty = True

            return len(to_remove)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __iter__(self) -> Iterator[GradientEntry]:
        with self._lock:
            return iter(list(self._entries.values()))

    def stats(self) -> dict:
        """Get table statistics."""
        with self._lock:
            if not self._entries:
                return {
                    "size": 0,
                    "avg_hops": 0,
                    "avg_latency_ms": 0,
                    "unique_next_hops": 0
                }

            entries = list(self._entries.values())
            return {
                "size": len(entries),
                "avg_hops": sum(e.hops for e in entries) / len(entries),
                "avg_latency_ms": sum(e.estimated_latency_ms for e in entries) / len(entries),
                "unique_next_hops": len(set(e.next_hop for e in entries)),
                "avg_confidence": sum(e.confidence for e in entries) / len(entries)
            }

    def to_routing_tuples(self) -> List[Tuple[str, np.ndarray, int, str, str]]:
        """Export as tuples for matcher."""
        with self._lock:
            return [
                (e.capability_label, e.capability_vector, e.hops, e.next_hop, e.via_node)
                for e in self._entries.values()
                if not e.is_expired()
            ]
