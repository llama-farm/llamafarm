"""
Semantic Router - Needle protocol integration for LlamaFarm.

This package provides capability-based mesh routing for distributed AI workloads.
Nodes advertise what they can do, and intents are routed to the best-matching 
node via semantic gradient descent.
"""

from .embeddings import EmbeddingEngine, EmbeddingConfig
from .matcher import CapabilityMatcher, MatchResult, RouteDecision, Capability
from .gradient import GradientTable, GradientEntry
from .gossip import GossipProtocol, Announcement, CapabilityInfo
from .learning import RouteLearner, RouteQuality, get_route_learner

__all__ = [
    "EmbeddingEngine",
    "EmbeddingConfig",
    "CapabilityMatcher",
    "MatchResult",
    "RouteDecision",
    "Capability",
    "GradientTable",
    "GradientEntry",
    "GossipProtocol",
    "Announcement",
    "CapabilityInfo",
    "RouteLearner",
    "RouteQuality",
    "get_route_learner",
]
