"""
Tests for VisionModel (image classification & understanding).
"""

import pytest
from models.vision_model import VisionModel


@pytest.mark.asyncio
async def test_vision_load_classification(device, test_model_ids):
    """Test loading a vision classification model."""
    model = VisionModel(
        test_model_ids["vision_classification"], device, task="classification"
    )
    await model.load()

    assert model.model is not None
    assert model.processor is not None
    assert model.model_type == "vision_classification"


@pytest.mark.asyncio
async def test_vision_classify(device, test_model_ids, sample_image):
    """Test image classification."""
    model = VisionModel(
        test_model_ids["vision_classification"], device, task="classification"
    )
    await model.load()

    results = await model.classify([sample_image], top_k=3)

    assert isinstance(results, list)
    assert len(results) == 1
    assert "predictions" in results[0]
    assert len(results[0]["predictions"]) <= 3

    # Check prediction format
    pred = results[0]["predictions"][0]
    assert "label" in pred
    assert "score" in pred
    assert isinstance(pred["score"], float)
    assert 0.0 <= pred["score"] <= 1.0


@pytest.mark.asyncio
async def test_vision_classify_batch(device, test_model_ids, sample_image):
    """Test classifying multiple images."""
    model = VisionModel(
        test_model_ids["vision_classification"], device, task="classification"
    )
    await model.load()

    # Use same image twice for testing
    results = await model.classify([sample_image, sample_image], top_k=2)

    assert len(results) == 2
    assert all("predictions" in r for r in results)


@pytest.mark.asyncio
async def test_vision_classify_from_base64(device, test_model_ids, sample_image_base64):
    """Test classification with base64 encoded image."""
    model = VisionModel(
        test_model_ids["vision_classification"], device, task="classification"
    )
    await model.load()

    results = await model.classify([sample_image_base64], top_k=5)

    assert len(results) == 1
    assert "predictions" in results[0]


@pytest.mark.asyncio
@pytest.mark.slow  # CLIP model is larger
async def test_vision_clip_load(device, test_model_ids):
    """Test loading a CLIP model."""
    model = VisionModel(test_model_ids["vision_clip"], device, task="clip")
    await model.load()

    assert model.model is not None
    assert model.processor is not None
    assert model.task == "clip"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_vision_clip_classify(device, test_model_ids, sample_image):
    """Test zero-shot classification with CLIP."""
    model = VisionModel(test_model_ids["vision_clip"], device, task="clip")
    await model.load()

    labels = ["a dog", "a cat", "a bird", "a car"]
    results = await model.clip_classify([sample_image], labels)

    assert len(results) == 1
    assert "predictions" in results[0]
    assert len(results[0]["predictions"]) == len(labels)

    # Check predictions are sorted by score
    scores = [p["score"] for p in results[0]["predictions"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_vision_clip_embed_image(device, test_model_ids, sample_image):
    """Test image embedding with CLIP."""
    model = VisionModel(test_model_ids["vision_clip"], device, task="clip")
    await model.load()

    embeddings = await model.embed_image([sample_image], normalize=True)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)
    assert len(embeddings[0]) > 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_vision_clip_embed_text(device, test_model_ids):
    """Test text embedding with CLIP."""
    model = VisionModel(test_model_ids["vision_clip"], device, task="clip")
    await model.load()

    texts = ["a photo of a cat", "a photo of a dog"]
    embeddings = await model.embed_text(texts, normalize=True)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(emb, list) for emb in embeddings)


@pytest.mark.asyncio
async def test_vision_generate_not_supported(device, test_model_ids):
    """Test that generate() raises NotImplementedError."""
    model = VisionModel(
        test_model_ids["vision_classification"], device, task="classification"
    )
    await model.load()

    with pytest.raises(NotImplementedError):
        await model.generate()


@pytest.mark.asyncio
async def test_vision_model_info(device, test_model_ids):
    """Test getting model info."""
    model = VisionModel(
        test_model_ids["vision_classification"], device, task="classification"
    )
    await model.load()

    info = model.get_model_info()

    assert info["model_id"] == test_model_ids["vision_classification"]
    assert info["model_type"] == "vision_classification"
    assert info["device"] == device
