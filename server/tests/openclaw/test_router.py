"""Tests for OpenClaw Lite semantic router."""
import pytest
import numpy as np
from router import EmbeddingEngine, CapabilityMatcher


class TestEmbeddingEngine:
    """Test EmbeddingEngine for text vectorization."""
    
    def test_engine_initialization(self):
        """Test embedding engine initializes."""
        engine = EmbeddingEngine(model="all-MiniLM-L6-v2")
        assert engine.model_name == "all-MiniLM-L6-v2"
    
    def test_embed_single_text(self):
        """Test embedding a single text."""
        engine = EmbeddingEngine()
        embedding = engine.embed("Hello world")
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # 1D vector
        assert embedding.shape[0] > 0  # Has dimensions
    
    def test_embed_batch(self):
        """Test embedding multiple texts."""
        engine = EmbeddingEngine()
        texts = ["Hello", "World", "Test"]
        embeddings = engine.embed_batch(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3  # 3 texts
        assert embeddings.shape[1] > 0  # Each has dimensions
    
    def test_embedding_consistency(self):
        """Test same text produces same embedding."""
        engine = EmbeddingEngine()
        emb1 = engine.embed("test phrase")
        emb2 = engine.embed("test phrase")
        
        assert np.allclose(emb1, emb2)
    
    def test_similarity_calculation(self):
        """Test cosine similarity between embeddings."""
        engine = EmbeddingEngine()
        emb1 = engine.embed("cat")
        emb2 = engine.embed("kitten")
        emb3 = engine.embed("database")
        
        sim_similar = engine.similarity(emb1, emb2)
        sim_different = engine.similarity(emb1, emb3)
        
        assert sim_similar > sim_different  # cat/kitten more similar than cat/database


class TestCapabilityMatcher:
    """Test CapabilityMatcher for routing intents."""
    
    def test_matcher_initialization(self):
        """Test matcher initializes with capabilities."""
        capabilities = {
            "search": {
                "description": "Search the web",
                "examples": ["google this", "search for", "find information"]
            },
            "weather": {
                "description": "Get weather info",
                "examples": ["what's the weather", "temperature today", "forecast"]
            }
        }
        
        matcher = CapabilityMatcher(capabilities)
        assert len(matcher.capabilities) == 2
        assert "search" in matcher.capabilities
    
    def test_match_query_to_capability(self):
        """Test matching user query to capability."""
        capabilities = {
            "weather": {
                "description": "Weather information",
                "examples": ["weather in NYC", "temperature", "is it raining"]
            }
        }
        
        matcher = CapabilityMatcher(capabilities)
        result = matcher.match("What's the temperature today?")
        
        assert result is not None
        assert result["capability"] == "weather"
        assert result["confidence"] > 0.5
    
    def test_no_match_low_confidence(self):
        """Test query with no good match returns None or low confidence."""
        capabilities = {
            "email": {
                "description": "Send emails",
                "examples": ["send email", "compose message"]
            }
        }
        
        matcher = CapabilityMatcher(capabilities, threshold=0.7)
        result = matcher.match("What's the square root of 144?")
        
        # Should either be None or have low confidence
        if result:
            assert result["confidence"] < 0.7
    
    def test_top_k_matches(self):
        """Test returning top K capability matches."""
        capabilities = {
            "search": {"description": "Web search", "examples": ["search web"]},
            "calculator": {"description": "Math calc", "examples": ["calculate", "math"]},
            "weather": {"description": "Weather", "examples": ["weather", "forecast"]}
        }
        
        matcher = CapabilityMatcher(capabilities)
        results = matcher.match_top_k("search google", k=2)
        
        assert len(results) <= 2
        assert all("capability" in r and "confidence" in r for r in results)


class TestSemanticRouter:
    """Test end-to-end SemanticRouter."""
    
    def test_router_initialization(self):
        """Test router initializes with engine and matcher."""
        capabilities = {
            "test": {"description": "Test capability", "examples": ["test"]}
        }
        
        router = SemanticRouter(capabilities)
        assert router.engine is not None
        assert router.matcher is not None
    
    def test_route_query(self):
        """Test routing a user query."""
        capabilities = {
            "calculate": {
                "description": "Perform calculations",
                "examples": ["what is 5+5", "calculate pi", "math problem"]
            },
            "search": {
                "description": "Search information",
                "examples": ["search for cats", "google this", "find info"]
            }
        }
        
        router = SemanticRouter(capabilities)
        result = router.route("What is 10 times 20?")
        
        assert result is not None
        assert result["capability"] == "calculate"
        assert result["confidence"] > 0.0
    
    def test_route_with_fallback(self):
        """Test router fallback when no match."""
        capabilities = {
            "specific": {
                "description": "Very specific task",
                "examples": ["specific phrase only"]
            }
        }
        
        router = SemanticRouter(capabilities, fallback="default")
        result = router.route("completely random unrelated query")
        
        # Should fall back to default
        assert result["capability"] == "default"
    
    def test_route_confidence_threshold(self):
        """Test router respects confidence threshold."""
        capabilities = {
            "task": {"description": "Some task", "examples": ["do task"]}
        }
        
        router = SemanticRouter(capabilities, threshold=0.9)  # Very high
        result = router.route("vaguely related query")
        
        # Should not match with high threshold
        assert result is None or result["confidence"] < 0.9
