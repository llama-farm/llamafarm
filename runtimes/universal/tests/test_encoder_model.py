"""
Tests for EncoderModel (embeddings & classification).
"""

import pytest
import numpy as np
from models.encoder_model import EncoderModel


@pytest.mark.asyncio
async def test_encoder_load(device, test_model_ids):
    """Test loading an encoder model."""
    model = EncoderModel(test_model_ids["encoder"], device, task="embedding")
    await model.load()

    assert model.model is not None
    assert model.tokenizer is not None
    assert model.model_type == "encoder_embedding"


@pytest.mark.asyncio
async def test_encoder_embed_single(device, test_model_ids, sample_text):
    """Test embedding a single text."""
    model = EncoderModel(test_model_ids["encoder"], device, task="embedding")
    await model.load()

    embeddings = await model.embed([sample_text], normalize=True)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)
    assert len(embeddings[0]) > 0

    # Check if normalized (L2 norm should be ~1)
    embedding_array = np.array(embeddings[0])
    norm = np.linalg.norm(embedding_array)
    assert abs(norm - 1.0) < 0.01


@pytest.mark.asyncio
async def test_encoder_embed_batch(device, test_model_ids, sample_texts):
    """Test embedding multiple texts."""
    model = EncoderModel(test_model_ids["encoder"], device, task="embedding")
    await model.load()

    embeddings = await model.embed(sample_texts, normalize=True)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)

    # All embeddings should have same dimension
    dims = [len(emb) for emb in embeddings]
    assert len(set(dims)) == 1  # All same dimension


@pytest.mark.asyncio
async def test_encoder_embed_similarity(device, test_model_ids):
    """Test that similar texts have similar embeddings."""
    model = EncoderModel(test_model_ids["encoder"], device, task="embedding")
    await model.load()

    texts = [
        "The cat sat on the mat",
        "A cat was sitting on a mat",
        "The weather is nice today",
    ]

    embeddings = await model.embed(texts, normalize=True)

    # Convert to numpy for easier calculation
    emb_array = np.array(embeddings)

    # Calculate cosine similarities (dot product since normalized)
    sim_0_1 = np.dot(emb_array[0], emb_array[1])  # Similar sentences
    sim_0_2 = np.dot(emb_array[0], emb_array[2])  # Different sentences

    # Similar sentences should have higher similarity
    assert sim_0_1 > sim_0_2


@pytest.mark.asyncio
async def test_encoder_normalize_effect(device, test_model_ids, sample_text):
    """Test normalization effect on embeddings."""
    model = EncoderModel(test_model_ids["encoder"], device, task="embedding")
    await model.load()

    # Get normalized embeddings
    emb_normalized = await model.embed([sample_text], normalize=True)
    emb_norm = np.linalg.norm(np.array(emb_normalized[0]))

    # Get unnormalized embeddings
    emb_unnormalized = await model.embed([sample_text], normalize=False)
    emb_unnorm = np.linalg.norm(np.array(emb_unnormalized[0]))

    # Normalized should be ~1, unnormalized likely different
    assert abs(emb_norm - 1.0) < 0.01
    # Unnormalized is typically much larger
    assert emb_unnorm > 1.0


@pytest.mark.asyncio
async def test_encoder_generate_not_supported(device, test_model_ids):
    """Test that generate() raises NotImplementedError."""
    model = EncoderModel(test_model_ids["encoder"], device, task="embedding")
    await model.load()

    with pytest.raises(NotImplementedError):
        await model.generate("test")


@pytest.mark.asyncio
async def test_encoder_model_info(device, test_model_ids):
    """Test getting model info."""
    model = EncoderModel(test_model_ids["encoder"], device, task="embedding")
    await model.load()

    info = model.get_model_info()

    assert info["model_id"] == test_model_ids["encoder"]
    assert info["model_type"] == "encoder_embedding"
    assert info["device"] == device
