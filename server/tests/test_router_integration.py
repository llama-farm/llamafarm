"""
Integration tests for the semantic router.
"""

import pytest
from router.embeddings import EmbeddingEngine, EmbeddingConfig
from router.matcher import CapabilityMatcher, Capability
from router.gradient import GradientTable
from router.service import RouterService
import numpy as np


@pytest.mark.asyncio
async def test_embedding_engine_initialization():
    """Test that embedding engine can initialize."""
    engine = EmbeddingEngine(EmbeddingConfig())
    
    # Should auto-select backend (Ollama or LlamaFarm)
    try:
        await engine.initialize()
        assert engine._initialized
        assert engine.backend in ["ollama", "llamafarm"]
        await engine.close()
    except RuntimeError as e:
        # If no backend available, that's expected in test env
        assert "No embedding backend available" in str(e)


@pytest.mark.asyncio
async def test_capability_matcher():
    """Test basic capability matching."""
    matcher = CapabilityMatcher()
    
    # Create mock capability
    cap = Capability(
        id="test:vision",
        label="vision",
        description="Analyze images",
        vector=np.random.randn(768).astype(np.float32),
        handler="vision_handler"
    )
    cap.vector = cap.vector / np.linalg.norm(cap.vector)  # Normalize
    
    # Create intent vector (very similar to capability - 98% same)
    intent_vec = cap.vector * 0.98 + np.random.randn(768).astype(np.float32) * 0.02
    intent_vec = intent_vec / np.linalg.norm(intent_vec)
    
    # Should match locally with high similarity
    result = matcher.match_local(intent_vec, [cap])
    
    assert result.matched
    assert result.capability.id == "test:vision"
    assert result.score > 0.75  # Above match threshold


def test_gradient_table():
    """Test gradient table operations."""
    table = GradientTable(node_id="test-node")
    
    # Add entry
    cap_vec = np.random.randn(768).astype(np.float32)
    cap_vec = cap_vec / np.linalg.norm(cap_vec)
    
    updated = table.update(
        capability_id="remote:llm",
        capability_label="llm",
        capability_vector=cap_vec,
        hops=1,
        next_hop="peer-1",
        via_node="peer-1"
    )
    
    assert updated
    assert len(table) == 1
    
    # Get entry
    entry = table.get("remote:llm")
    assert entry is not None
    assert entry.hops == 1
    assert entry.next_hop == "peer-1"
    
    # Find best route
    query_vec = cap_vec + np.random.randn(768).astype(np.float32) * 0.05
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    route = table.find_best_route(query_vec, min_score=0.5)
    assert route is not None


@pytest.mark.asyncio
async def test_router_service_lifecycle():
    """Test router service initialization and shutdown."""
    service = RouterService(
        node_id="test-node",
        enable_discovery=False,  # Don't start UDP server in tests
        enable_gossip=False  # Don't start gossip in tests
    )
    
    try:
        await service.initialize()
        
        assert service._initialized
        assert service.embedding_engine is not None
        assert service.gradient_table is not None
        assert service.matcher is not None
        
        # Should be able to get stats
        stats = service.get_stats()
        assert stats["node_id"] == "test-node"
        assert stats["initialized"] == True
        
    except RuntimeError as e:
        # If embedding engine can't initialize (no backend), that's expected
        if "No embedding backend available" not in str(e):
            raise
    finally:
        await service.shutdown()


def test_router_api_imports():
    """Test that router API can be imported."""
    from router.api import router
    assert router is not None
    assert len(router.routes) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
