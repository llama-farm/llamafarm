"""
Atmosphere - The Internet of Intent

Distributed intelligence mesh that routes work to capability.
Install LlamaFarm, install Atmosphere, you have a mesh.
"""

__version__ = "0.1.0"

from .mesh.node import Node, NodeIdentity, MeshIdentity
from .mesh.gossip import GossipProtocol, Announcement, CapabilityInfo
from .router.gradient import GradientTable, GradientEntry
from .router.semantic import SemanticRouter

__all__ = [
    "Node",
    "NodeIdentity", 
    "MeshIdentity",
    "GossipProtocol",
    "Announcement",
    "CapabilityInfo",
    "GradientTable",
    "GradientEntry",
    "SemanticRouter",
]
