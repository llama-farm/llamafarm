"""
Multi-node mesh discovery and routing tests.

Tests the gossip protocol and gradient table updates
across multiple simulated nodes.
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
import time

from router.gradient import GradientTable, GradientEntry
from router.gossip import GossipProtocol, CapabilityInfo
from router.matcher import CapabilityMatcher, Capability
from router.learning import RouteLearner, RouteQuality


class TestGradientTable:
    """Test gradient table operations."""
    
    def test_update_new_entry(self):
        """Test adding a new entry to gradient table."""
        table = GradientTable(node_id="test-node")
        vector = np.random.randn(768).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        updated = table.update(
            capability_id="cap-1",
            capability_label="llm",
            capability_vector=vector,
            hops=1,
            next_hop="peer-1",
            via_node="peer-1"
        )
        
        assert updated is True
        assert len(table) == 1
        entry = table.get("cap-1")
        assert entry is not None
        assert entry.capability_label == "llm"
        assert entry.hops == 1
    
    def test_update_better_route(self):
        """Test that better routes replace worse ones."""
        table = GradientTable(node_id="test-node")
        vector = np.random.randn(768).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        # Add initial route with 3 hops
        table.update("cap-1", "llm", vector, hops=3, 
                     next_hop="peer-a", via_node="peer-c")
        
        # Add better route with 1 hop
        updated = table.update("cap-1", "llm", vector, hops=1,
                               next_hop="peer-b", via_node="peer-b")
        
        assert updated is True
        entry = table.get("cap-1")
        assert entry.hops == 1
        assert entry.next_hop == "peer-b"
    
    def test_update_worse_route_rejected(self):
        """Test that worse routes are rejected."""
        table = GradientTable(node_id="test-node")
        vector = np.random.randn(768).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        # Add initial route with 1 hop
        table.update("cap-1", "llm", vector, hops=1,
                     next_hop="peer-a", via_node="peer-a")
        
        # Try to add worse route with 3 hops
        updated = table.update("cap-1", "llm", vector, hops=3,
                               next_hop="peer-b", via_node="peer-c")
        
        assert updated is False
        entry = table.get("cap-1")
        assert entry.hops == 1
        assert entry.next_hop == "peer-a"
    
    def test_find_best_route(self):
        """Test finding best route for an intent."""
        table = GradientTable(node_id="test-node")
        
        # Add multiple capabilities
        for i, label in enumerate(["llm", "vision", "rag"]):
            vector = np.zeros(768, dtype=np.float32)
            vector[i] = 1.0  # Different directions
            table.update(f"cap-{i}", label, vector, hops=1,
                        next_hop=f"peer-{i}", via_node=f"peer-{i}")
        
        # Query for "llm" direction
        query = np.zeros(768, dtype=np.float32)
        query[0] = 1.0
        
        entry = table.find_best_route(query, min_score=0.5)
        assert entry is not None
        assert entry.capability_label == "llm"
    
    def test_prune_expired(self):
        """Test pruning expired entries."""
        table = GradientTable(node_id="test-node", expire_sec=0.1)
        vector = np.random.randn(768).astype(np.float32)
        
        table.update("cap-1", "llm", vector, hops=1,
                     next_hop="peer-1", via_node="peer-1")
        
        assert len(table) == 1
        
        # Wait for expiry
        time.sleep(0.2)
        removed = table.prune_expired()
        
        assert removed == 1
        assert len(table) == 0
    
    def test_invalidate_node(self):
        """Test invalidating routes through a disconnected node."""
        table = GradientTable(node_id="test-node")
        vector = np.random.randn(768).astype(np.float32)
        
        # Add routes through different peers
        table.update("cap-1", "llm", vector, hops=1,
                     next_hop="peer-1", via_node="peer-1")
        table.update("cap-2", "vision", vector, hops=2,
                     next_hop="peer-1", via_node="peer-2")
        table.update("cap-3", "rag", vector, hops=1,
                     next_hop="peer-2", via_node="peer-2")
        
        assert len(table) == 3
        
        # Invalidate peer-1
        removed = table.invalidate_node("peer-1")
        
        assert removed == 2
        assert len(table) == 1
        assert table.get("cap-3") is not None


class TestRouteLearner:
    """Test route learning and quality tracking."""
    
    def test_record_success(self):
        """Test recording successful routes."""
        learner = RouteLearner()
        
        learner.record_success("route-1", latency_ms=50)
        learner.record_success("route-1", latency_ms=60)
        learner.record_success("route-1", latency_ms=55)
        
        quality = learner.get_quality("route-1")
        assert quality is not None
        assert quality.total_requests == 3
        assert quality.successful_requests == 3
        assert quality.failed_requests == 0
        assert 50 <= quality.avg_latency_ms <= 60
    
    def test_record_failure(self):
        """Test recording failed routes."""
        learner = RouteLearner()
        
        learner.record_success("route-1", latency_ms=50)
        learner.record_failure("route-1", error_type="timeout")
        
        quality = learner.get_quality("route-1")
        assert quality.total_requests == 2
        assert quality.successful_requests == 1
        assert quality.failed_requests == 1
        assert quality.success_rate() == 0.5
    
    def test_quality_score(self):
        """Test quality score computation."""
        learner = RouteLearner()
        
        # Good route: 100% success, low latency
        for _ in range(10):
            learner.record_success("good-route", latency_ms=50)
        
        # Bad route: 50% success, high latency
        for _ in range(5):
            learner.record_success("bad-route", latency_ms=500)
            learner.record_failure("bad-route", error_type="error")
        
        good_quality = learner.get_quality("good-route")
        bad_quality = learner.get_quality("bad-route")
        
        assert good_quality.quality_score > bad_quality.quality_score
        assert good_quality.quality_score > 0.8
        assert bad_quality.quality_score < 0.6
    
    def test_adjust_score(self):
        """Test score adjustment based on quality."""
        learner = RouteLearner()
        
        # Route with many failures
        for _ in range(10):
            learner.record_failure("bad-route", error_type="error")
        
        base_score = 0.9
        adjusted = learner.adjust_score("bad-route", base_score, weight=0.3)
        
        # Adjusted should be lower than base
        assert adjusted < base_score
    
    def test_get_degraded_routes(self):
        """Test finding degraded routes."""
        learner = RouteLearner()
        
        # Good route
        for _ in range(10):
            learner.record_success("good", latency_ms=50)
        
        # Degraded route
        for _ in range(10):
            learner.record_failure("degraded", error_type="error")
        
        degraded = learner.get_degraded_routes(threshold=0.7)
        assert "degraded" in degraded
        assert "good" not in degraded


class TestMultiNodeMesh:
    """Test multi-node mesh scenarios."""
    
    @pytest.mark.asyncio
    async def test_capability_propagation(self):
        """Test capability propagation through mesh."""
        # Create 3 nodes
        nodes = [
            GradientTable(node_id=f"node-{i}")
            for i in range(3)
        ]
        
        # Node 0 has a capability
        cap_vector = np.random.randn(768).astype(np.float32)
        cap_vector = cap_vector / np.linalg.norm(cap_vector)
        
        # Simulate gossip: node-0 announces to node-1
        nodes[1].update(
            capability_id="cap-llm",
            capability_label="llm",
            capability_vector=cap_vector,
            hops=1,
            next_hop="node-0",
            via_node="node-0"
        )
        
        # node-1 gossips to node-2 (hops + 1)
        entries = nodes[1].export_for_gossip()
        for entry_dict in entries:
            nodes[2].update(
                capability_id=entry_dict["capability_id"],
                capability_label=entry_dict["capability_label"],
                capability_vector=np.array(entry_dict["capability_vector"]),
                hops=entry_dict["hops"] + 1,
                next_hop="node-1",  # node-2 routes through node-1
                via_node=entry_dict["via_node"]
            )
        
        # Verify propagation
        entry_n1 = nodes[1].get("cap-llm")
        entry_n2 = nodes[2].get("cap-llm")
        
        assert entry_n1 is not None
        assert entry_n1.hops == 1
        assert entry_n1.next_hop == "node-0"
        
        assert entry_n2 is not None
        assert entry_n2.hops == 2
        assert entry_n2.next_hop == "node-1"
        assert entry_n2.via_node == "node-0"
    
    @pytest.mark.asyncio
    async def test_route_quality_feedback(self):
        """Test route quality learning in mesh."""
        table = GradientTable(node_id="node-0")
        learner = RouteLearner()
        
        # Add two routes to same capability
        cap_vector = np.random.randn(768).astype(np.float32)
        cap_vector = cap_vector / np.linalg.norm(cap_vector)
        
        table.update("cap-1", "llm", cap_vector, hops=1,
                     next_hop="node-1", via_node="node-1")
        table.update("cap-2", "llm", cap_vector, hops=2,
                     next_hop="node-2", via_node="node-3")
        
        # Simulate usage: node-1 path is reliable
        for _ in range(10):
            learner.record_success("cap-1", latency_ms=50)
        
        # node-2 path has failures
        for _ in range(5):
            learner.record_success("cap-2", latency_ms=100)
            learner.record_failure("cap-2", error_type="timeout")
        
        # Check quality-adjusted routing
        q1 = learner.get_quality("cap-1")
        q2 = learner.get_quality("cap-2")
        
        assert q1.quality_score > q2.quality_score
        
        # Find best routes - cap-1 should be preferred
        best = learner.get_best_routes(top_k=2)
        assert best[0][0] == "cap-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
