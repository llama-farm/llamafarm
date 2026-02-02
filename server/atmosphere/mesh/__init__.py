"""
Mesh networking for Atmosphere.

Provides:
- Node identity and management
- Gossip protocol for capability propagation
- mDNS/DNS-SD discovery
- Mesh joining logic
"""

from .node import Node, NodeIdentity, MeshIdentity
from .gossip import GossipProtocol, Announcement, CapabilityInfo
from .discovery import MeshDiscovery
from .join import MeshJoin, JoinCode

__all__ = [
    # Node
    "Node",
    "NodeIdentity",
    "MeshIdentity",
    # Gossip
    "GossipProtocol",
    "Announcement",
    "CapabilityInfo",
    # Discovery
    "MeshDiscovery",
    # Join
    "MeshJoin",
    "JoinCode",
]
