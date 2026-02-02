"""
Gossip protocol for capability propagation.

Nodes periodically announce their capabilities to neighbors.
Announcements propagate through the mesh with TTL decrement.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Dict, List, Optional, Set
import logging

import numpy as np

from ..router.gradient import GradientTable, GradientEntry

logger = logging.getLogger(__name__)

# Protocol constants
ANNOUNCE_INTERVAL_SEC = 30
MAX_TTL = 10
MAX_CAPABILITIES_PER_ANNOUNCE = 50
NONCE_CACHE_SEC = 300


@dataclass
class CapabilityInfo:
    """Capability information for announcements."""
    id: str
    label: str
    description: str
    vector: List[float]
    local: bool = True
    hops: int = 0
    via: Optional[str] = None
    models: List[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    estimated_latency_ms: float = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "vector": self.vector,
            "local": self.local,
            "hops": self.hops,
            "via": self.via,
            "models": self.models,
            "constraints": self.constraints,
            "estimated_latency_ms": self.estimated_latency_ms
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CapabilityInfo":
        return cls(
            id=data["id"],
            label=data["label"],
            description=data.get("description", ""),
            vector=data["vector"],
            local=data.get("local", False),
            hops=data.get("hops", 0),
            via=data.get("via"),
            models=data.get("models", []),
            constraints=data.get("constraints", {}),
            estimated_latency_ms=data.get("estimated_latency_ms", 0)
        )


@dataclass
class ResourceInfo:
    """Node resource information."""
    cpu_available: float = 1.0
    memory_available_mb: int = 0
    gpu_available: float = 0.0
    battery_percent: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "cpu_available": self.cpu_available,
            "memory_available_mb": self.memory_available_mb,
            "gpu_available": self.gpu_available,
            "battery_percent": self.battery_percent
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResourceInfo":
        return cls(
            cpu_available=data.get("cpu_available", 1.0),
            memory_available_mb=data.get("memory_available_mb", 0),
            gpu_available=data.get("gpu_available", 0.0),
            battery_percent=data.get("battery_percent")
        )


@dataclass
class Announcement:
    """A capability announcement message."""
    type: str = "announce"
    from_node: str = ""
    capabilities: List[CapabilityInfo] = field(default_factory=list)
    resources: Optional[ResourceInfo] = None
    timestamp: float = field(default_factory=time.time)
    ttl: int = MAX_TTL
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "from": self.from_node,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "resources": self.resources.to_dict() if self.resources else None,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "nonce": self.nonce
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Announcement":
        return cls(
            type=data.get("type", "announce"),
            from_node=data.get("from", ""),
            capabilities=[
                CapabilityInfo.from_dict(c) for c in data.get("capabilities", [])
            ],
            resources=ResourceInfo.from_dict(data["resources"]) if data.get("resources") else None,
            timestamp=data.get("timestamp", time.time()),
            ttl=data.get("ttl", MAX_TTL),
            nonce=data.get("nonce", "")
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> "Announcement":
        return cls.from_dict(json.loads(data))


# Type for broadcast callback
BroadcastCallback = Callable[[str, bytes], Awaitable[None]]


class GossipProtocol:
    """
    Gossip protocol for capability propagation.
    
    Periodically announces capabilities to peers. Processes incoming
    announcements and updates gradient table.
    """

    def __init__(
        self,
        node_id: str,
        gradient_table: GradientTable,
        local_capabilities: List[CapabilityInfo],
        announce_interval: float = ANNOUNCE_INTERVAL_SEC
    ):
        self.node_id = node_id
        self.gradient_table = gradient_table
        self.local_capabilities = local_capabilities
        self.announce_interval = announce_interval

        self._broadcast_callback: Optional[BroadcastCallback] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._nonce_cache: Dict[str, float] = {}
        self._nonce_cache_lock = asyncio.Lock()
        self._known_nodes: Dict[str, float] = {}

        # Metrics
        self._announcements_sent = 0
        self._announcements_received = 0
        self._announcements_forwarded = 0

    def set_broadcast_callback(self, callback: BroadcastCallback) -> None:
        """Set the callback for broadcasting messages to peers."""
        self._broadcast_callback = callback

    def update_local_capabilities(self, capabilities: List[CapabilityInfo]) -> None:
        """Update the list of local capabilities."""
        self.local_capabilities = capabilities

    def get_resource_info(self) -> ResourceInfo:
        """Get current resource usage."""
        import psutil
        try:
            cpu = 1.0 - (psutil.cpu_percent() / 100.0)
            memory = psutil.virtual_memory().available // (1024 * 1024)
            return ResourceInfo(
                cpu_available=cpu,
                memory_available_mb=memory,
                gpu_available=0.8
            )
        except Exception:
            return ResourceInfo()

    def build_announcement(self) -> Announcement:
        """Build an announcement message."""
        capabilities = []

        for cap in self.local_capabilities[:MAX_CAPABILITIES_PER_ANNOUNCE]:
            capabilities.append(CapabilityInfo(
                id=cap.id,
                label=cap.label,
                description=cap.description,
                vector=cap.vector if isinstance(cap.vector, list) else cap.vector.tolist(),
                local=True,
                hops=0,
                models=cap.models,
                constraints=cap.constraints
            ))

        remaining_slots = MAX_CAPABILITIES_PER_ANNOUNCE - len(capabilities)
        for entry_dict in self.gradient_table.export_for_gossip(max_hops=5)[:remaining_slots]:
            entry = GradientEntry.from_dict(entry_dict)
            capabilities.append(CapabilityInfo(
                id=entry.capability_id,
                label=entry.capability_label,
                description="",
                vector=entry.capability_vector.tolist(),
                local=False,
                hops=entry.hops,
                via=entry.via_node,
                estimated_latency_ms=entry.estimated_latency_ms
            ))

        return Announcement(
            from_node=self.node_id,
            capabilities=capabilities,
            resources=self.get_resource_info(),
            ttl=MAX_TTL
        )

    async def announce(self) -> None:
        """Broadcast capability announcement to all peers."""
        if not self._broadcast_callback:
            return

        announcement = self.build_announcement()
        data = announcement.to_json().encode()

        try:
            await self._broadcast_callback(self.node_id, data)
            self._announcements_sent += 1
        except Exception as e:
            logger.error(f"Failed to broadcast announcement: {e}")

    async def handle_announcement(
        self,
        data: bytes,
        from_peer: str,
        forward_callback: Optional[BroadcastCallback] = None
    ) -> None:
        """Handle an incoming announcement."""
        try:
            announcement = Announcement.from_json(data.decode())
        except Exception as e:
            logger.warning(f"Invalid announcement from {from_peer}: {e}")
            return

        if not await self._check_nonce(announcement.nonce, announcement.timestamp):
            return

        self._announcements_received += 1
        self._known_nodes[announcement.from_node] = time.time()

        updates = 0
        for cap in announcement.capabilities:
            vector = np.array(cap.vector, dtype=np.float32)
            new_hops = cap.hops + 1 if not cap.local else 1

            updated = self.gradient_table.update(
                capability_id=cap.id,
                capability_label=cap.label,
                capability_vector=vector,
                hops=new_hops,
                next_hop=from_peer,
                via_node=cap.via or announcement.from_node,
                estimated_latency_ms=cap.estimated_latency_ms + 10
            )
            if updated:
                updates += 1

        if announcement.ttl > 1 and forward_callback:
            forwarded = Announcement(
                from_node=announcement.from_node,
                capabilities=announcement.capabilities,
                resources=announcement.resources,
                timestamp=announcement.timestamp,
                ttl=announcement.ttl - 1,
                nonce=announcement.nonce
            )

            for cap in forwarded.capabilities:
                if not cap.local:
                    cap.hops += 1

            try:
                await forward_callback(self.node_id, forwarded.to_json().encode())
                self._announcements_forwarded += 1
            except Exception as e:
                logger.error(f"Failed to forward announcement: {e}")

    async def _check_nonce(self, nonce: str, timestamp: float) -> bool:
        """Check if nonce is new (not a replay)."""
        now = time.time()

        if abs(now - timestamp) > NONCE_CACHE_SEC:
            return False

        async with self._nonce_cache_lock:
            expired = [
                n for n, t in self._nonce_cache.items()
                if now - t > NONCE_CACHE_SEC
            ]
            for n in expired:
                del self._nonce_cache[n]

            if nonce in self._nonce_cache:
                return False

            self._nonce_cache[nonce] = timestamp
            return True

    async def _announce_loop(self) -> None:
        """Periodic announcement loop."""
        while self._running:
            try:
                await self.announce()
            except Exception as e:
                logger.error(f"Announcement failed: {e}")
            await asyncio.sleep(self.announce_interval)

    async def start(self) -> None:
        """Start periodic announcements."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._announce_loop())
        logger.info("Gossip protocol started")

    async def stop(self) -> None:
        """Stop periodic announcements."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Gossip protocol stopped")

    def known_nodes(self) -> Set[str]:
        """Return set of nodes we've heard from recently."""
        now = time.time()
        return {
            node_id for node_id, last_seen in self._known_nodes.items()
            if now - last_seen < NONCE_CACHE_SEC
        }

    def stats(self) -> dict:
        """Get gossip protocol statistics."""
        return {
            "announcements_sent": self._announcements_sent,
            "announcements_received": self._announcements_received,
            "announcements_forwarded": self._announcements_forwarded,
            "known_nodes": len(self.known_nodes()),
            "gradient_table_size": len(self.gradient_table)
        }
