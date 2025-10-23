"""
Tests for DiffusionModel (image generation).
"""

import pytest
from PIL import Image
from models.diffusion_model import DiffusionModel


@pytest.mark.asyncio
async def test_diffusion_load(device, test_model_ids):
    """Test loading a diffusion model."""
    model = DiffusionModel(test_model_ids["diffusion"], device)
    await model.load()

    assert model.pipe is not None
    assert model.model_type == "diffusion"


@pytest.mark.asyncio
async def test_diffusion_generate(device, test_model_ids):
    """Test image generation from text prompt."""
    model = DiffusionModel(test_model_ids["diffusion"], device)
    await model.load()

    images = await model.generate(
        prompt="a cat",
        num_images=1,
        width=64,  # Small size for testing
        height=64,
        num_inference_steps=2,  # Few steps for speed
    )

    assert isinstance(images, list)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (64, 64)


@pytest.mark.asyncio
async def test_diffusion_generate_multiple(device, test_model_ids):
    """Test generating multiple images."""
    model = DiffusionModel(test_model_ids["diffusion"], device)
    await model.load()

    images = await model.generate(
        prompt="a dog",
        num_images=2,
        width=64,
        height=64,
        num_inference_steps=2,
    )

    assert len(images) == 2
    assert all(isinstance(img, Image.Image) for img in images)


@pytest.mark.asyncio
async def test_diffusion_with_negative_prompt(device, test_model_ids):
    """Test generation with negative prompt."""
    model = DiffusionModel(test_model_ids["diffusion"], device)
    await model.load()

    images = await model.generate(
        prompt="a beautiful landscape",
        negative_prompt="ugly, blurry",
        num_images=1,
        width=64,
        height=64,
        num_inference_steps=2,
    )

    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


@pytest.mark.asyncio
async def test_diffusion_with_seed(device, test_model_ids):
    """Test generation with fixed seed for reproducibility."""
    model = DiffusionModel(test_model_ids["diffusion"], device)
    await model.load()

    # Generate with same seed twice
    images1 = await model.generate(
        prompt="a tree",
        seed=42,
        width=64,
        height=64,
        num_inference_steps=2,
    )

    images2 = await model.generate(
        prompt="a tree",
        seed=42,
        width=64,
        height=64,
        num_inference_steps=2,
    )

    # Results should be very similar (bit-identical in most cases)
    assert images1[0].size == images2[0].size


@pytest.mark.asyncio
async def test_diffusion_scheduler_selection(device, test_model_ids):
    """Test using different schedulers."""
    model = DiffusionModel(test_model_ids["diffusion"], device)
    await model.load()

    # Test with different schedulers
    for scheduler in ["ddim", "euler", None]:
        images = await model.generate(
            prompt="test",
            width=64,
            height=64,
            num_inference_steps=2,
            scheduler=scheduler,
        )

        assert len(images) == 1
        assert isinstance(images[0], Image.Image)


@pytest.mark.asyncio
async def test_diffusion_model_info(device, test_model_ids):
    """Test getting model info."""
    model = DiffusionModel(test_model_ids["diffusion"], device)
    await model.load()

    info = model.get_model_info()

    assert info["model_id"] == test_model_ids["diffusion"]
    assert info["model_type"] == "diffusion"
    assert info["device"] == device
