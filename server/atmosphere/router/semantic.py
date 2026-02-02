"""
Semantic router for intent-based capability matching.

Routes intents to the best available capability using
embedding-based similarity matching.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gradient import GradientTable, GradientEntry
from .embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)

# Matching thresholds
MATCH_THRESHOLD = 0.75
MIN_ROUTE_THRESHOLD = 0.50
HOP_PENALTY = 0.95


class RouteAction(Enum):
    """Actions from routing decision."""
    PROCESS_LOCAL = "process_local"
    FORWARD = "forward"
    NO_MATCH = "no_match"


@dataclass
class Capability:
    """A capability that a node can provide."""
    id: str
    label: str
    description: str
    vector: np.ndarray
    handler: str = "default"
    models: List[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.float32)


@dataclass
class RouteResult:
    """Result of routing decision."""
    action: RouteAction
    capability: Optional[Capability] = None
    score: float = 0.0
    adjusted_score: float = 0.0
    hops: int = 0
    next_hop: Optional[str] = None
    via_node: Optional[str] = None
    reason: str = ""

    @property
    def matched(self) -> bool:
        return self.action != RouteAction.NO_MATCH


class SemanticRouter:
    """
    Semantic router for intent-based capability matching.
    
    Usage:
        router = SemanticRouter()
        await router.initialize()
        
        # Register local capabilities
        router.register_capability("llm", "Language model inference", handler="ollama")
        
        # Route an intent
        result = await router.route("summarize this document")
        
        if result.action == RouteAction.PROCESS_LOCAL:
            # Execute locally
            pass
        elif result.action == RouteAction.FORWARD:
            # Forward to result.next_hop
            pass
    """

    def __init__(
        self,
        node_id: str,
        match_threshold: float = MATCH_THRESHOLD,
        min_route_threshold: float = MIN_ROUTE_THRESHOLD,
        hop_penalty: float = HOP_PENALTY
    ):
        self.node_id = node_id
        self.match_threshold = match_threshold
        self.min_route_threshold = min_route_threshold
        self.hop_penalty = hop_penalty
        
        self.embedding_engine = EmbeddingEngine()
        self.gradient_table = GradientTable(node_id)
        self.local_capabilities: Dict[str, Capability] = {}
        
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the router."""
        if self._initialized:
            return
        
        await self.embedding_engine.initialize()
        self._initialized = True
        logger.info("Semantic router initialized")

    async def close(self) -> None:
        """Close the router."""
        await self.embedding_engine.close()
        self._initialized = False

    async def register_capability(
        self,
        label: str,
        description: str,
        handler: str = "default",
        models: Optional[List[str]] = None,
        constraints: Optional[dict] = None
    ) -> Capability:
        """
        Register a local capability.
        
        Args:
            label: Short label (e.g., "llm", "vision")
            description: Full description for embedding
            handler: Handler function/service name
            models: Available models for this capability
            constraints: Resource constraints
            
        Returns:
            The registered Capability
        """
        if not self._initialized:
            await self.initialize()
        
        # Generate embedding from description
        vector = await self.embedding_engine.embed(description)
        
        cap_id = f"{self.node_id}:{label}"
        capability = Capability(
            id=cap_id,
            label=label,
            description=description,
            vector=vector,
            handler=handler,
            models=models or [],
            constraints=constraints or {}
        )
        
        self.local_capabilities[cap_id] = capability
        logger.info(f"Registered capability: {label}")
        
        return capability

    async def route(self, intent: str) -> RouteResult:
        """
        Route an intent to the best capability.
        
        Args:
            intent: Natural language description of what to do
            
        Returns:
            RouteResult with routing decision
        """
        if not self._initialized:
            await self.initialize()
        
        # Generate intent embedding
        intent_vector = await self.embedding_engine.embed(intent)
        
        # Try local match first
        local_result = self._match_local(intent_vector)
        if local_result.action == RouteAction.PROCESS_LOCAL:
            return local_result
        
        # Check gradient table for remote capabilities
        remote_result = self._match_remote(intent_vector)
        
        # Return the best result
        if remote_result.adjusted_score > local_result.score:
            return remote_result
        
        if local_result.score >= self.min_route_threshold:
            # Local is below threshold but still best option
            return RouteResult(
                action=RouteAction.PROCESS_LOCAL,
                capability=local_result.capability,
                score=local_result.score,
                adjusted_score=local_result.score,
                reason="Best available (below threshold)"
            )
        
        return RouteResult(
            action=RouteAction.NO_MATCH,
            score=max(local_result.score, remote_result.score),
            reason="No capability match above threshold"
        )

    def _match_local(self, intent_vector: np.ndarray) -> RouteResult:
        """Match against local capabilities."""
        if not self.local_capabilities:
            return RouteResult(action=RouteAction.NO_MATCH)
        
        best_cap = None
        best_score = 0.0
        
        for cap in self.local_capabilities.values():
            score = self.embedding_engine.cosine_similarity(intent_vector, cap.vector)
            if score > best_score:
                best_score = score
                best_cap = cap
        
        if best_score >= self.match_threshold and best_cap:
            return RouteResult(
                action=RouteAction.PROCESS_LOCAL,
                capability=best_cap,
                score=best_score,
                adjusted_score=best_score,
                hops=0
            )
        
        return RouteResult(
            action=RouteAction.NO_MATCH,
            capability=best_cap,
            score=best_score
        )

    def _match_remote(self, intent_vector: np.ndarray) -> RouteResult:
        """Match against gradient table (remote capabilities)."""
        entry = self.gradient_table.find_best_route(
            intent_vector,
            min_score=self.min_route_threshold
        )
        
        if entry:
            return RouteResult(
                action=RouteAction.FORWARD,
                score=self.embedding_engine.cosine_similarity(
                    intent_vector, entry.capability_vector
                ),
                adjusted_score=entry.confidence,
                hops=entry.hops,
                next_hop=entry.next_hop,
                via_node=entry.via_node
            )
        
        return RouteResult(action=RouteAction.NO_MATCH)

    async def rank_capabilities(
        self,
        intent: str,
        top_k: int = 5
    ) -> List[Tuple[Capability, float]]:
        """
        Rank all capabilities by match score.
        
        Returns list of (capability, score) tuples.
        """
        if not self._initialized:
            await self.initialize()
        
        intent_vector = await self.embedding_engine.embed(intent)
        
        scored = []
        for cap in self.local_capabilities.values():
            score = self.embedding_engine.cosine_similarity(intent_vector, cap.vector)
            scored.append((cap, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_local_capability_vectors(self) -> List[Tuple[str, str, np.ndarray]]:
        """Get local capabilities for gossip announcements."""
        return [
            (cap.id, cap.label, cap.vector)
            for cap in self.local_capabilities.values()
        ]
