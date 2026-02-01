"""
Gradient table management for semantic routing.

Each node maintains a gradient table - a map from capability embeddings
to routing information (next_hop, hops, latency). This enables O(1)
routing decisions after O(n) gossip convergence.
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
LINK_LATENCY_MS = 10  # Assumed per-hop latency


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
    next_hop: str  # Node ID to forward to
    via_node: str  # Node that actually has the capability
    estimated_latency_ms: float
    last_updated: float = field(default_factory=time.time)
    confidence: float = 1.0  # Decreases with hops

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

    Usage:
        table = GradientTable(node_id="my-node")

        # Update from peer announcement
        table.update(capability_id, label, vector, hops=1,
                     next_hop="peer-1", via_node="peer-1")

        # Find route for an intent
        entry = table.find_best_route(intent_vector)
        if entry:
            forward_to(entry.next_hop)

        # Periodic maintenance
        removed = table.prune_expired()
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

        # Index for fast vector lookup
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

        Only updates if:
        - Entry doesn't exist, or
        - New route has fewer hops

        Args:
            capability_id: Unique ID of the capability
            capability_label: Human-readable label
            capability_vector: 768-dim embedding
            hops: Number of hops to capability
            next_hop: Node to forward to
            via_node: Node that has the capability
            estimated_latency_ms: Optional latency estimate

        Returns:
            True if table was updated, False otherwise
        """
        if estimated_latency_ms is None:
            estimated_latency_ms = hops * LINK_LATENCY_MS

        with self._lock:
            existing = self._entries.get(capability_id)

            if existing is not None:
                # Only update if new route is better
                if hops >= existing.hops:
                    # Refresh timestamp if same route
                    if hops == existing.hops and next_hop == existing.next_hop:
                        existing.last_updated = time.time()
                    return False

            # Create new entry
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

            # Check size limit
            if len(self._entries) >= self.max_size and capability_id not in self._entries:
                self._evict_one()

            self._entries[capability_id] = entry
            self._index_dirty = True

            logger.debug(
                f"Gradient update: {capability_label} via {next_hop} "
                f"({hops} hops, {estimated_latency_ms:.0f}ms)"
            )
            return True

    def _evict_one(self) -> None:
        """Evict the oldest/lowest confidence entry."""
        if not self._entries:
            return

        # Find entry with lowest confidence * recency
        worst_id = None
        worst_score = float('inf')

        now = time.time()
        for cap_id, entry in self._entries.items():
            age = now - entry.last_updated
            score = entry.confidence / (1 + age / 60)  # Decay with age
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

        Args:
            intent_vector: Normalized 768-dim intent embedding
            min_score: Minimum adjusted score to consider

        Returns:
            Best matching GradientEntry or None
        """
        with self._lock:
            self._rebuild_index()

            if self._vectors is None or len(self._vectors) == 0:
                return None

            # Compute similarities
            similarities = self._vectors @ intent_vector

            best_idx = None
            best_adjusted = 0.0

            for idx, sim in enumerate(similarities):
                entry = self._entries[self._vector_ids[idx]]
                adjusted = sim * entry.confidence  # confidence includes hop penalty
                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_idx = idx

            if best_idx is not None and best_adjusted >= min_score:
                return self._entries[self._vector_ids[best_idx]]

            return None

    def find_routes_for_capability(
        self,
        capability_vector: np.ndarray,
        top_k: int = 3,
        min_score: float = 0.7
    ) -> List[Tuple[GradientEntry, float]]:
        """
        Find multiple routes for similar capabilities.

        Useful when primary route is overloaded.
        """
        with self._lock:
            self._rebuild_index()

            if self._vectors is None or len(self._vectors) == 0:
                return []

            similarities = self._vectors @ capability_vector
            indices = np.argsort(similarities)[::-1]

            results = []
            for idx in indices[:top_k]:
                score = float(similarities[idx])
                if score >= min_score:
                    entry = self._entries[self._vector_ids[idx]]
                    results.append((entry, score))

            return results

    def prune_expired(self) -> int:
        """
        Remove expired entries.

        Call periodically (e.g., every minute).

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired = [
                cap_id for cap_id, entry in self._entries.items()
                if entry.is_expired(self.expire_sec)
            ]

            for cap_id in expired:
                del self._entries[cap_id]

            if expired:
                self._index_dirty = True
                logger.debug(f"Pruned {len(expired)} expired gradient entries")

            return len(expired)

    def export_for_gossip(self, max_hops: int = 5) -> List[dict]:
        """
        Export entries for gossip announcement.

        Entries beyond max_hops are not exported (prevents infinite propagation).
        """
        with self._lock:
            return [
                entry.to_dict()
                for entry in self._entries.values()
                if entry.hops < max_hops and not entry.is_expired()
            ]

    def get_entries_via(self, node_id: str) -> List[GradientEntry]:
        """Get all entries routed through a specific node."""
        with self._lock:
            return [
                entry for entry in self._entries.values()
                if entry.next_hop == node_id or entry.via_node == node_id
            ]

    def invalidate_node(self, node_id: str) -> int:
        """
        Remove all entries for a disconnected node.

        Call when a peer disconnects.
        """
        with self._lock:
            to_remove = [
                cap_id for cap_id, entry in self._entries.items()
                if entry.next_hop == node_id
            ]

            for cap_id in to_remove:
                del self._entries[cap_id]

            if to_remove:
                self._index_dirty = True
                logger.info(
                    f"Invalidated {len(to_remove)} routes via {node_id}"
                )

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
        """
        Export as tuples for CapabilityMatcher.

        Returns:
            List of (label, vector, hops, next_hop, via_node)
        """
        with self._lock:
            return [
                (e.capability_label, e.capability_vector, e.hops, e.next_hop, e.via_node)
                for e in self._entries.values()
                if not e.is_expired()
            ]
