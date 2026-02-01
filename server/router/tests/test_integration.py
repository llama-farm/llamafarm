"""
Integration tests for the semantic router.

Tests the full routing flow from intent to decision,
including embedding, matching, and learning.
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

# Test fixtures and helpers


@pytest.fixture
def mock_embedding_engine():
    """Mock embedding engine that returns predictable vectors."""
    engine = MagicMock()
    engine._initialized = True
    engine.backend = "mock"
    
    # Map keywords to specific vector directions
    keyword_vectors = {
        "llm": np.array([1.0] + [0.0] * 767, dtype=np.float32),
        "vision": np.array([0.0, 1.0] + [0.0] * 766, dtype=np.float32),
        "rag": np.array([0.0, 0.0, 1.0] + [0.0] * 765, dtype=np.float32),
        "code": np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 764, dtype=np.float32),
    }
    
    async def mock_embed(text: str) -> np.ndarray:
        text_lower = text.lower()
        for keyword, vec in keyword_vectors.items():
            if keyword in text_lower:
                return vec / np.linalg.norm(vec)
        # Default: random normalized vector
        vec = np.random.randn(768).astype(np.float32)
        return vec / np.linalg.norm(vec)
    
    engine.embed = mock_embed
    engine.initialize = AsyncMock()
    engine.close = AsyncMock()
    
    return engine


@pytest.fixture
def sample_capabilities():
    """Sample local capabilities for testing."""
    from router.matcher import Capability
    
    return [
        Capability(
            id="local:llm:001",
            label="llm",
            description="Large language model for text generation",
            vector=np.array([1.0] + [0.0] * 767, dtype=np.float32),
            handler="llm_handler"
        ),
        Capability(
            id="local:vision:001",
            label="vision",
            description="Computer vision for image analysis",
            vector=np.array([0.0, 1.0] + [0.0] * 766, dtype=np.float32),
            handler="vision_handler"
        ),
    ]


class TestRoutingFlow:
    """Test the full routing flow."""
    
    @pytest.mark.asyncio
    async def test_local_capability_match(
        self,
        mock_embedding_engine,
        sample_capabilities
    ):
        """Test routing to a local capability."""
        from router.matcher import CapabilityMatcher, RouteDecision
        
        matcher = CapabilityMatcher(
            match_threshold=0.75,
            min_route_threshold=0.50
        )
        
        # Generate intent embedding
        intent_vec = await mock_embedding_engine.embed(
            "Generate a summary using LLM"
        )
        
        # Match against local capabilities
        result = matcher.match_with_routing(
            intent_vec,
            sample_capabilities,
            []  # No gradient entries
        )
        
        assert result.matched is True
        assert result.action == RouteDecision.PROCESS_LOCAL
        assert result.capability is not None
        assert result.capability.label == "llm"
        assert result.score > 0.75
    
    @pytest.mark.asyncio
    async def test_remote_capability_routing(
        self,
        mock_embedding_engine,
        sample_capabilities
    ):
        """Test routing to a remote capability via gradient table."""
        from router.matcher import CapabilityMatcher, RouteDecision
        from router.gradient import GradientTable
        
        matcher = CapabilityMatcher(
            match_threshold=0.75,
            min_route_threshold=0.50,
            hop_penalty=0.95
        )
        
        # Set up gradient table with remote capability
        gradient_table = GradientTable(node_id="local-node")
        
        # Add remote "code" capability
        code_vector = np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 764, dtype=np.float32)
        gradient_table.update(
            capability_id="remote:code:001",
            capability_label="code",
            capability_vector=code_vector,
            hops=1,
            next_hop="peer-1",
            via_node="peer-1"
        )
        
        # Generate intent for code
        intent_vec = await mock_embedding_engine.embed(
            "Execute this code snippet"
        )
        
        # Convert gradient entries to tuples
        gradient_entries = gradient_table.to_routing_tuples()
        
        # Match - should route to remote
        result = matcher.match_with_routing(
            intent_vec,
            sample_capabilities,  # Local has llm, vision - not code
            gradient_entries
        )
        
        assert result.matched is True
        assert result.action == RouteDecision.ROUTE_FORWARD
        assert result.next_hop == "peer-1"
        assert result.score > 0.5
    
    @pytest.mark.asyncio
    async def test_no_match_broadcasts(
        self,
        mock_embedding_engine,
        sample_capabilities
    ):
        """Test that unmatched intents trigger broadcast."""
        from router.matcher import CapabilityMatcher, RouteDecision
        
        matcher = CapabilityMatcher(
            match_threshold=0.75,
            min_route_threshold=0.50
        )
        
        # Generate intent for something we don't have
        # Use random text that won't match our keywords
        random_vec = np.random.randn(768).astype(np.float32)
        random_vec = random_vec / np.linalg.norm(random_vec)
        
        result = matcher.match_with_routing(
            random_vec,
            sample_capabilities,
            []
        )
        
        # Should either not match or broadcast
        if result.matched:
            assert result.score < 0.75  # Below threshold
        else:
            assert result.action == RouteDecision.ROUTE_BROADCAST


class TestLearningIntegration:
    """Test learning integration with routing."""
    
    @pytest.mark.asyncio
    async def test_feedback_improves_routing(self):
        """Test that feedback affects route quality."""
        from router.learning import RouteLearner
        
        learner = RouteLearner()
        
        # Simulate two routes to similar capability
        route_a = "peer-a:llm:001"
        route_b = "peer-b:llm:002"
        
        # Route A has good performance
        for _ in range(10):
            learner.record_success(route_a, latency_ms=50)
        
        # Route B has failures
        for _ in range(5):
            learner.record_success(route_b, latency_ms=100)
            learner.record_failure(route_b, error_type="timeout")
        
        # Quality scores should reflect this
        quality_a = learner.get_quality(route_a)
        quality_b = learner.get_quality(route_b)
        
        assert quality_a.quality_score > quality_b.quality_score
        assert quality_a.success_rate() == 1.0
        assert quality_b.success_rate() == 0.5
    
    @pytest.mark.asyncio
    async def test_adjusted_routing_prefers_quality(self):
        """Test that adjusted scores prefer higher quality routes."""
        from router.learning import RouteLearner
        
        learner = RouteLearner()
        
        # Route with failures
        route_id = "degraded-route"
        for _ in range(10):
            learner.record_failure(route_id, error_type="error")
        
        # Base similarity is high
        base_score = 0.9
        
        # Adjusted score should be lower
        adjusted = learner.adjust_score(route_id, base_score, weight=0.3)
        
        assert adjusted < base_score
        # Should be noticeably degraded
        assert adjusted < base_score * 0.9


class TestGossipPropagation:
    """Test gossip-based capability propagation."""
    
    @pytest.mark.asyncio
    async def test_capability_reaches_all_nodes(self):
        """Test that capabilities propagate through gossip."""
        from router.gradient import GradientTable
        
        # Create 3-node linear topology
        nodes = [GradientTable(node_id=f"node-{i}") for i in range(3)]
        
        # Node 0 has the capability
        cap_vector = np.random.randn(768).astype(np.float32)
        cap_vector = cap_vector / np.linalg.norm(cap_vector)
        
        # Simulate gossip: 0 -> 1 -> 2
        
        # Node 1 receives from Node 0
        nodes[1].update(
            capability_id="cap-1",
            capability_label="special",
            capability_vector=cap_vector,
            hops=1,
            next_hop="node-0",
            via_node="node-0"
        )
        
        # Node 1 gossips to Node 2
        for entry_dict in nodes[1].export_for_gossip():
            nodes[2].update(
                capability_id=entry_dict["capability_id"],
                capability_label=entry_dict["capability_label"],
                capability_vector=np.array(entry_dict["capability_vector"]),
                hops=entry_dict["hops"] + 1,
                next_hop="node-1",
                via_node=entry_dict["via_node"]
            )
        
        # Verify all nodes know about capability
        entry_1 = nodes[1].get("cap-1")
        entry_2 = nodes[2].get("cap-1")
        
        assert entry_1 is not None
        assert entry_1.hops == 1
        assert entry_1.next_hop == "node-0"
        
        assert entry_2 is not None
        assert entry_2.hops == 2
        assert entry_2.next_hop == "node-1"
        assert entry_2.via_node == "node-0"
    
    @pytest.mark.asyncio
    async def test_better_route_replaces_worse(self):
        """Test that discovering better routes updates the table."""
        from router.gradient import GradientTable
        
        table = GradientTable(node_id="test-node")
        cap_vector = np.random.randn(768).astype(np.float32)
        
        # Initial route: 3 hops
        table.update(
            "cap-1", "llm", cap_vector,
            hops=3, next_hop="peer-a", via_node="peer-c"
        )
        
        entry = table.get("cap-1")
        assert entry.hops == 3
        assert entry.next_hop == "peer-a"
        
        # Discover better route: 1 hop
        table.update(
            "cap-1", "llm", cap_vector,
            hops=1, next_hop="peer-b", via_node="peer-b"
        )
        
        entry = table.get("cap-1")
        assert entry.hops == 1
        assert entry.next_hop == "peer-b"
    
    @pytest.mark.asyncio
    async def test_node_failure_invalidates_routes(self):
        """Test that node failures remove affected routes."""
        from router.gradient import GradientTable
        
        table = GradientTable(node_id="test-node")
        cap_vector = np.random.randn(768).astype(np.float32)
        
        # Add routes through peer-a
        table.update("cap-1", "llm", cap_vector, 1, "peer-a", "peer-a")
        table.update("cap-2", "vision", cap_vector, 2, "peer-a", "peer-b")
        
        # Add route NOT through peer-a
        table.update("cap-3", "rag", cap_vector, 1, "peer-c", "peer-c")
        
        assert len(table) == 3
        
        # peer-a goes down
        removed = table.invalidate_node("peer-a")
        
        assert removed == 2
        assert len(table) == 1
        assert table.get("cap-3") is not None


class TestEndToEndScenario:
    """End-to-end scenario tests."""
    
    @pytest.mark.asyncio
    async def test_distributed_ai_workload(self):
        """
        Simulate a distributed AI workload across multiple nodes.
        
        Scenario:
        - Node A: LLM, embeddings
        - Node B: Vision, RAG
        - Node C: Code execution
        
        Request: "Analyze this image and generate a description"
        Expected: Route to Node B (vision)
        """
        from router.matcher import CapabilityMatcher, Capability, RouteDecision
        from router.gradient import GradientTable
        
        # Node A's view
        local_capabilities = [
            Capability(
                id="node-a:llm:001",
                label="llm",
                description="Text generation",
                vector=np.array([1.0] + [0.0] * 767, dtype=np.float32)
            ),
            Capability(
                id="node-a:embeddings:001",
                label="embeddings",
                description="Text embeddings",
                vector=np.array([0.5, 0.5] + [0.0] * 766, dtype=np.float32)
            ),
        ]
        
        # Gradient table with remote capabilities
        gradient = GradientTable(node_id="node-a")
        
        # Node B's capabilities (1 hop)
        vision_vec = np.array([0.0, 1.0] + [0.0] * 766, dtype=np.float32)
        gradient.update("node-b:vision:001", "vision", vision_vec, 1, "node-b", "node-b")
        
        rag_vec = np.array([0.0, 0.0, 1.0] + [0.0] * 765, dtype=np.float32)
        gradient.update("node-b:rag:001", "rag", rag_vec, 1, "node-b", "node-b")
        
        # Node C's capabilities (2 hops through B)
        code_vec = np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 764, dtype=np.float32)
        gradient.update("node-c:code:001", "code", code_vec, 2, "node-b", "node-c")
        
        # Matcher
        matcher = CapabilityMatcher(
            match_threshold=0.75,
            min_route_threshold=0.50,
            hop_penalty=0.95
        )
        
        # Intent: vision-related request
        # Simulating embedding that matches vision
        intent_vec = np.array([0.1, 0.9] + [0.0] * 766, dtype=np.float32)
        intent_vec = intent_vec / np.linalg.norm(intent_vec)
        
        result = matcher.match_with_routing(
            intent_vec,
            local_capabilities,
            gradient.to_routing_tuples()
        )
        
        assert result.matched is True
        assert result.action == RouteDecision.ROUTE_FORWARD
        assert result.next_hop == "node-b"
        assert result.score > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
