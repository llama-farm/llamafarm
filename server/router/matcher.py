"""
Capability matching for semantic routing.

Matches intent embeddings against capability embeddings using cosine similarity.
Implements hop-distance penalty for preferring local capabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


# Matching thresholds
MATCH_THRESHOLD = 0.75  # Process locally if above this
MIN_ROUTE_THRESHOLD = 0.50  # Drop intent if below this
HOP_PENALTY = 0.95  # Per-hop score reduction (0.95^hops)


class MatchAction(Enum):
    """Actions from matching decision."""
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
    handler: str  # Name of handler function
    models: List[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)

    def __post_init__(self):
        # Ensure vector is numpy array
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.float32)


@dataclass
class MatchResult:
    """Result of capability matching."""
    action: MatchAction
    capability: Optional[Capability] = None
    score: float = 0.0
    adjusted_score: float = 0.0
    hops: int = 0
    next_hop: Optional[str] = None
    via_node: Optional[str] = None
    reason: Optional[str] = None

    @property
    def matched(self) -> bool:
        return self.action != MatchAction.NO_MATCH


@dataclass
class RouteDecision:
    """Decision about how to route an intent."""
    action: MatchAction
    next_hop: Optional[str] = None
    capability_label: Optional[str] = None
    score: float = 0.0
    reason: str = ""


class CapabilityMatcher:
    """
    Matches intents to capabilities using semantic similarity.

    Usage:
        matcher = CapabilityMatcher(
            match_threshold=0.75,
            min_route_threshold=0.50
        )

        result = matcher.match(intent_vec, local_capabilities)
        if result.matched:
            # Process with result.capability
    """

    def __init__(
        self,
        match_threshold: float = MATCH_THRESHOLD,
        min_route_threshold: float = MIN_ROUTE_THRESHOLD,
        hop_penalty: float = HOP_PENALTY
    ):
        self.match_threshold = match_threshold
        self.min_route_threshold = min_route_threshold
        self.hop_penalty = hop_penalty

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between normalized vectors."""
        return float(np.dot(vec1, vec2))

    def _compute_adjusted_score(self, raw_score: float, hops: int) -> float:
        """Apply hop penalty to raw score."""
        return raw_score * (self.hop_penalty ** hops)

    def match_local(
        self,
        intent_vec: np.ndarray,
        capabilities: List[Capability]
    ) -> MatchResult:
        """
        Match intent against local capabilities.

        Args:
            intent_vec: Normalized 768-dim intent embedding
            capabilities: List of local capabilities

        Returns:
            MatchResult with best match or NO_MATCH
        """
        if not capabilities:
            return MatchResult(action=MatchAction.NO_MATCH)

        best_cap = None
        best_score = 0.0

        for cap in capabilities:
            score = self._cosine_similarity(intent_vec, cap.vector)
            if score > best_score:
                best_score = score
                best_cap = cap

        if best_score >= self.match_threshold and best_cap:
            logger.debug(
                f"Local match: {best_cap.label} with score {best_score:.3f}"
            )
            return MatchResult(
                action=MatchAction.PROCESS_LOCAL,
                capability=best_cap,
                score=best_score,
                adjusted_score=best_score,
                hops=0
            )

        return MatchResult(
            action=MatchAction.NO_MATCH,
            score=best_score
        )

    def match_with_routing(
        self,
        intent_vec: np.ndarray,
        local_capabilities: List[Capability],
        gradient_entries: List[Tuple[str, np.ndarray, int, str, str]]
    ) -> MatchResult:
        """
        Match intent against local + remote capabilities.

        Args:
            intent_vec: Normalized 768-dim intent embedding
            local_capabilities: List of local capabilities
            gradient_entries: List of (label, vector, hops, next_hop, via_node)

        Returns:
            MatchResult with routing decision
        """
        # First try local match
        local_result = self.match_local(intent_vec, local_capabilities)
        if local_result.action == MatchAction.PROCESS_LOCAL:
            return local_result

        # Search gradient table for remote match
        best_entry = None
        best_score = 0.0
        best_adjusted = 0.0

        for label, vector, hops, next_hop, via_node in gradient_entries:
            score = self._cosine_similarity(intent_vec, vector)
            adjusted = self._compute_adjusted_score(score, hops)

            if adjusted > best_adjusted:
                best_score = score
                best_adjusted = adjusted
                best_entry = (label, hops, next_hop, via_node)

        if best_entry and best_adjusted >= self.min_route_threshold:
            label, hops, next_hop, via_node = best_entry
            logger.debug(
                f"Route match: {label} via {next_hop} "
                f"(score={best_score:.3f}, hops={hops})"
            )
            return MatchResult(
                action=MatchAction.FORWARD,
                score=best_score,
                adjusted_score=best_adjusted,
                hops=hops,
                next_hop=next_hop,
                via_node=via_node
            )

        return MatchResult(
            action=MatchAction.NO_MATCH,
            score=max(local_result.score, best_score),
            reason="No capability match above threshold"
        )

    def rank_capabilities(
        self,
        intent_vec: np.ndarray,
        capabilities: List[Capability],
        top_k: int = 5
    ) -> List[Tuple[Capability, float]]:
        """
        Rank capabilities by match score.

        Returns:
            List of (capability, score) tuples, sorted by score descending
        """
        scored = []
        for cap in capabilities:
            score = self._cosine_similarity(intent_vec, cap.vector)
            scored.append((cap, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def batch_match(
        self,
        intent_vec: np.ndarray,
        capability_vectors: np.ndarray,
        labels: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Efficiently match against many capability vectors.

        Args:
            intent_vec: Normalized 768-dim intent embedding
            capability_vectors: Array of shape (N, 768)
            labels: List of N capability labels

        Returns:
            List of (label, score) tuples, sorted by score descending
        """
        scores = capability_vectors @ intent_vec
        indices = np.argsort(scores)[::-1]

        return [(labels[i], float(scores[i])) for i in indices]

    def explain_match(
        self,
        intent_vec: np.ndarray,
        capability: Capability
    ) -> dict:
        """
        Explain why a capability matches (or doesn't match) an intent.

        Returns dict with match details for debugging/UI.
        """
        score = self._cosine_similarity(intent_vec, capability.vector)

        return {
            "capability": capability.label,
            "raw_score": score,
            "threshold": self.match_threshold,
            "above_threshold": score >= self.match_threshold,
            "match_quality": (
                "excellent" if score >= 0.9 else
                "good" if score >= 0.75 else
                "fair" if score >= 0.6 else
                "poor" if score >= 0.4 else
                "no match"
            ),
            "vector_norm": float(np.linalg.norm(capability.vector))
        }


def create_capability(
    id: str,
    label: str,
    description: str,
    vector: np.ndarray,
    handler: str = "default",
    **kwargs
) -> Capability:
    """Factory function to create a Capability."""
    return Capability(
        id=id,
        label=label,
        description=description,
        vector=vector,
        handler=handler,
        **kwargs
    )


class MultiIntentMatcher:
    """
    Handles intents that might need multiple capabilities.

    For complex queries like "analyze this image and summarize findings",
    this can decompose into vision + llm capabilities.
    """

    def __init__(self, base_matcher: CapabilityMatcher):
        self.matcher = base_matcher

    def decompose_intent(
        self,
        intent_vec: np.ndarray,
        intent_text: str,
        capabilities: List[Capability]
    ) -> List[MatchResult]:
        """
        For complex intents, find all required capabilities.

        Currently a simple implementation that returns top-N matches.
        Future: Use LLM to decompose intent into sub-tasks.
        """
        ranked = self.matcher.rank_capabilities(
            intent_vec, capabilities, top_k=3
        )

        results = []
        for cap, score in ranked:
            if score >= self.matcher.min_route_threshold:
                results.append(MatchResult(
                    action=MatchAction.PROCESS_LOCAL,
                    capability=cap,
                    score=score,
                    adjusted_score=score,
                    hops=0
                ))

        return results
