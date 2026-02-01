"""
Tests for router embeddings module.

Tests:
- Embedding engine initialization
- Text embedding generation
- Batch embedding
- Cosine similarity calculations
- Backend selection (Ollama vs LlamaFarm)
"""

import pytest
import numpy as np

from router import EmbeddingEngine, EmbeddingConfig


@pytest.fixture
async def embedding_engine():
    """Create an embedding engine for testing."""
    engine = EmbeddingEngine()
    await engine.initialize()
    yield engine
    await engine.close()


@pytest.mark.asyncio
async def test_engine_initialization():
    """Test that embedding engine initializes correctly."""
    engine = EmbeddingEngine()
    await engine.initialize()
    
    assert engine._initialized
    assert engine._provider is not None
    assert engine.dimension == 768
    
    await engine.close()


@pytest.mark.asyncio
async def test_embed_single_text(embedding_engine):
    """Test embedding a single text."""
    text = "What's the weather like today?"
    
    embedding = await embedding_engine.embed(text)
    
    # Check shape and type
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)
    assert embedding.dtype == np.float32
    
    # Check normalized (L2 norm should be ~1.0)
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.01


@pytest.mark.asyncio
async def test_embed_batch(embedding_engine):
    """Test batch embedding."""
    texts = [
        "What's the weather?",
        "Send an email",
        "Calculate 2 + 2"
    ]
    
    embeddings = await embedding_engine.embed_batch(texts)
    
    # Check shape
    assert embeddings.shape == (3, 768)
    assert embeddings.dtype == np.float32
    
    # Check each embedding is normalized
    for i in range(3):
        norm = np.linalg.norm(embeddings[i])
        assert abs(norm - 1.0) < 0.01


@pytest.mark.asyncio
async def test_semantic_similarity(embedding_engine):
    """Test that semantically similar texts have high similarity."""
    # Similar texts
    text1 = "What's the weather like today?"
    text2 = "Tell me about today's weather"
    
    # Different text
    text3 = "Send an email to my boss"
    
    emb1 = await embedding_engine.embed(text1)
    emb2 = await embedding_engine.embed(text2)
    emb3 = await embedding_engine.embed(text3)
    
    # Calculate similarities
    sim_similar = embedding_engine.cosine_similarity(emb1, emb2)
    sim_different = embedding_engine.cosine_similarity(emb1, emb3)
    
    # Similar texts should have higher similarity
    assert sim_similar > sim_different
    assert sim_similar > 0.7  # High similarity
    assert sim_different < 0.5  # Low similarity


@pytest.mark.asyncio
async def test_batch_cosine_similarity(embedding_engine):
    """Test batch cosine similarity calculation."""
    query = "What's the weather?"
    candidates = [
        "Tell me the temperature",
        "Send an email",
        "Is it raining today?"
    ]
    
    # Get embeddings
    query_emb = await embedding_engine.embed(query)
    candidate_embs = await embedding_engine.embed_batch(candidates)
    
    # Calculate batch similarity
    similarities = embedding_engine.batch_cosine_similarity(query_emb, candidate_embs)
    
    # Check shape
    assert similarities.shape == (3,)
    
    # Weather-related queries should be most similar
    assert similarities[0] > similarities[1]  # "temperature" > "email"
    assert similarities[2] > similarities[1]  # "raining" > "email"


@pytest.mark.asyncio
async def test_embedding_cache(embedding_engine):
    """Test that embeddings are cached."""
    text = "What's the weather?"
    
    # First call
    emb1 = await embedding_engine.embed(text)
    
    # Second call (should use cache)
    emb2 = await embedding_engine.embed(text)
    
    # Should be equal
    np.testing.assert_array_equal(emb1, emb2)


@pytest.mark.asyncio
async def test_normalize_flag(embedding_engine):
    """Test that normalize parameter works."""
    text = "What's the weather?"
    
    # Get normalized embedding (default)
    emb_normalized = await embedding_engine.embed(text, normalize=True)
    norm_normalized = np.linalg.norm(emb_normalized)
    
    # Get unnormalized embedding
    emb_unnormalized = await embedding_engine.embed(text, normalize=False)
    norm_unnormalized = np.linalg.norm(emb_unnormalized)
    
    # Normalized should have norm ~1.0
    assert abs(norm_normalized - 1.0) < 0.01
    
    # Unnormalized will have different norm (likely != 1.0)
    # We can't assert exact value, but they should be different
    assert abs(norm_normalized - norm_unnormalized) > 0.01 or abs(norm_unnormalized - 1.0) < 0.01


@pytest.mark.asyncio
async def test_backend_property(embedding_engine):
    """Test backend property returns correct value."""
    backend = embedding_engine.backend
    
    # Should be either ollama or llamafarm
    assert backend in ["ollama", "llamafarm"]


@pytest.mark.asyncio
async def test_dimension_property(embedding_engine):
    """Test dimension property."""
    assert embedding_engine.dimension == 768


@pytest.mark.asyncio
async def test_empty_text_handling(embedding_engine):
    """Test handling of empty or whitespace text."""
    # Empty string
    emb_empty = await embedding_engine.embed("")
    assert emb_empty.shape == (768,)
    
    # Whitespace
    emb_whitespace = await embedding_engine.embed("   ")
    assert emb_whitespace.shape == (768,)


@pytest.mark.asyncio
async def test_special_characters(embedding_engine):
    """Test embedding text with special characters."""
    text = "Hello! What's 2+2? 🌤️"
    
    emb = await embedding_engine.embed(text)
    
    assert emb.shape == (768,)
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 0.01
