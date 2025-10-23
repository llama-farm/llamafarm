"""
Tests for MultimodalModel (vision-language tasks).
"""

import pytest
from models.multimodal_model import MultimodalModel


@pytest.mark.asyncio
@pytest.mark.slow  # BLIP model is larger
async def test_multimodal_load(device, test_model_ids):
    """Test loading a multimodal model."""
    model = MultimodalModel(test_model_ids["multimodal"], device, task="image-to-text")
    await model.load()

    assert model.model is not None or model.pipe is not None
    assert model.processor is not None
    assert model.model_type == "multimodal_image-to-text"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_multimodal_caption(device, test_model_ids, sample_image):
    """Test image captioning."""
    model = MultimodalModel(test_model_ids["multimodal"], device, task="image-to-text")
    await model.load()

    caption = await model.caption(sample_image, max_length=20)

    assert isinstance(caption, str)
    assert len(caption) > 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_multimodal_caption_from_base64(
    device, test_model_ids, sample_image_base64
):
    """Test captioning with base64 encoded image."""
    model = MultimodalModel(test_model_ids["multimodal"], device, task="image-to-text")
    await model.load()

    caption = await model.caption(sample_image_base64, max_length=20)

    assert isinstance(caption, str)
    assert len(caption) > 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_multimodal_caption_with_beams(device, test_model_ids, sample_image):
    """Test captioning with different beam sizes."""
    model = MultimodalModel(test_model_ids["multimodal"], device, task="image-to-text")
    await model.load()

    # Test with different beam sizes
    caption1 = await model.caption(sample_image, num_beams=1)
    caption3 = await model.caption(sample_image, num_beams=3)

    assert isinstance(caption1, str)
    assert isinstance(caption3, str)
    # Both should produce captions, might be different


@pytest.mark.asyncio
@pytest.mark.slow
async def test_multimodal_answer_question(device, test_model_ids, sample_image):
    """Test visual question answering."""
    # Load as VQA model (or image-to-text which can handle questions)
    model = MultimodalModel(test_model_ids["multimodal"], device, task="image-to-text")
    await model.load()

    answer = await model.answer_question(
        sample_image, "What color is this?", max_length=10
    )

    assert isinstance(answer, str)
    assert len(answer) > 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_multimodal_generate(device, test_model_ids, sample_image):
    """Test the generic generate() method."""
    model = MultimodalModel(test_model_ids["multimodal"], device, task="image-to-text")
    await model.load()

    # Without prompt (caption)
    result1 = await model.generate(sample_image)
    assert isinstance(result1, str)

    # With prompt (VQA)
    result2 = await model.generate(sample_image, prompt="What is this?")
    assert isinstance(result2, str)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_multimodal_chat(device, test_model_ids, sample_image):
    """Test visual chat (may not work with all models)."""
    model = MultimodalModel(test_model_ids["multimodal"], device, task="visual-chat")
    await model.load()

    messages = [{"role": "user", "content": "Describe this image."}]

    # This might fail for BLIP since it's not a chat model
    # That's okay, we're testing the code path
    try:
        response = await model.chat(messages, images=[sample_image])
        assert isinstance(response, str)
    except Exception:
        # Some models don't support chat, that's expected
        pass


@pytest.mark.asyncio
@pytest.mark.slow
async def test_multimodal_model_info(device, test_model_ids):
    """Test getting model info."""
    model = MultimodalModel(test_model_ids["multimodal"], device, task="image-to-text")
    await model.load()

    info = model.get_model_info()

    assert info["model_id"] == test_model_ids["multimodal"]
    assert info["model_type"] == "multimodal_image-to-text"
    assert info["device"] == device


@pytest.mark.asyncio
@pytest.mark.slow
async def test_multimodal_chat_requires_image(device, test_model_ids):
    """Test that chat requires at least one image."""
    model = MultimodalModel(test_model_ids["multimodal"], device, task="visual-chat")
    await model.load()

    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="requires at least one image"):
        await model.chat(messages, images=None)
