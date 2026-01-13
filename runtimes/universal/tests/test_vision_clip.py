"""Tests for zero-shot image classification with CLIP.

Phase 5 of Universal Runtime ML Enhancements.

Tests verify:
- CLIP model loads successfully
- Zero-shot classification returns probabilities for all labels
- Classification works with various image formats (PNG, JPEG, WebP)
- Labels are case-insensitive
- Batch processing multiple images works
"""

import base64
import os
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def create_test_image(color: tuple = (255, 0, 0), size: tuple = (224, 224)) -> bytes:
    """Create a simple test image with given color.

    Args:
        color: RGB color tuple
        size: Image dimensions

    Returns:
        PNG image bytes
    """
    img = Image.new("RGB", size, color)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def create_test_image_base64(color: tuple = (255, 0, 0), size: tuple = (224, 224)) -> str:
    """Create a base64-encoded test image.

    Args:
        color: RGB color tuple
        size: Image dimensions

    Returns:
        Base64-encoded PNG string
    """
    return base64.b64encode(create_test_image(color, size)).decode("utf-8")


@pytest.fixture
def temp_image_png():
    """Create a temporary PNG image file."""
    img_bytes = create_test_image((0, 0, 255))  # Blue image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(img_bytes)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_image_jpg():
    """Create a temporary JPEG image file."""
    img = Image.new("RGB", (224, 224), (0, 255, 0))  # Green image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f, format="JPEG")
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_image_webp():
    """Create a temporary WebP image file."""
    img = Image.new("RGB", (224, 224), (255, 255, 0))  # Yellow image
    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
        img.save(f, format="WEBP")
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestCLIPModelLoading:
    """Tests for CLIP model loading."""

    @pytest.mark.asyncio
    async def test_clip_model_loads_successfully(self):
        """Test that CLIP model loads without errors."""
        from models.vision_model import CLIPVisionModel

        model = CLIPVisionModel(model_id="test_clip_load")
        await model.load()

        assert model.is_loaded
        assert model.processor is not None
        assert model.model is not None

    @pytest.mark.asyncio
    async def test_clip_model_unload(self):
        """Test that CLIP model can be unloaded."""
        from models.vision_model import CLIPVisionModel

        model = CLIPVisionModel(model_id="test_clip_unload")
        await model.load()
        assert model.is_loaded

        await model.unload()
        assert not model.is_loaded


class TestZeroShotClassification:
    """Tests for zero-shot classification."""

    @pytest.mark.asyncio
    async def test_classification_returns_probabilities(self, temp_image_png):
        """Test that classification returns probabilities for all labels."""
        from models.vision_model import CLIPVisionModel

        model = CLIPVisionModel(model_id="test_clip_probs")
        await model.load()

        labels = ["cat", "dog", "bird"]
        result = await model.classify_zero_shot(temp_image_png, labels)

        assert result is not None
        assert "label" in result
        assert "score" in result
        assert "all_scores" in result
        assert len(result["all_scores"]) == len(labels)

        # Check all labels have scores
        for label in labels:
            assert label in result["all_scores"]
            assert 0 <= result["all_scores"][label] <= 1

        # Probabilities should sum to 1
        total_prob = sum(result["all_scores"].values())
        assert abs(total_prob - 1.0) < 0.01  # Allow small floating point error

    @pytest.mark.asyncio
    async def test_classification_with_base64_image(self):
        """Test classification with base64-encoded image."""
        from models.vision_model import CLIPVisionModel

        model = CLIPVisionModel(model_id="test_clip_base64")
        await model.load()

        img_base64 = create_test_image_base64((255, 0, 0))  # Red image
        labels = ["red object", "blue object", "green object"]

        result = await model.classify_zero_shot(img_base64, labels)

        assert result is not None
        assert result["label"] in labels
        assert 0 <= result["score"] <= 1


class TestImageFormats:
    """Tests for various image formats."""

    @pytest.mark.asyncio
    async def test_png_format(self, temp_image_png):
        """Test classification works with PNG images."""
        from models.vision_model import CLIPVisionModel

        model = CLIPVisionModel(model_id="test_clip_png")
        await model.load()

        labels = ["photo", "drawing"]
        result = await model.classify_zero_shot(temp_image_png, labels)

        assert result is not None
        assert result["label"] in labels

    @pytest.mark.asyncio
    async def test_jpeg_format(self, temp_image_jpg):
        """Test classification works with JPEG images."""
        from models.vision_model import CLIPVisionModel

        model = CLIPVisionModel(model_id="test_clip_jpg")
        await model.load()

        labels = ["photo", "drawing"]
        result = await model.classify_zero_shot(temp_image_jpg, labels)

        assert result is not None
        assert result["label"] in labels

    @pytest.mark.asyncio
    async def test_webp_format(self, temp_image_webp):
        """Test classification works with WebP images."""
        from models.vision_model import CLIPVisionModel

        model = CLIPVisionModel(model_id="test_clip_webp")
        await model.load()

        labels = ["photo", "drawing"]
        result = await model.classify_zero_shot(temp_image_webp, labels)

        assert result is not None
        assert result["label"] in labels


class TestLabelHandling:
    """Tests for label handling."""

    @pytest.mark.asyncio
    async def test_labels_case_insensitive(self, temp_image_png):
        """Test that labels are case-insensitive in results."""
        from models.vision_model import CLIPVisionModel

        model = CLIPVisionModel(model_id="test_clip_case")
        await model.load()

        # Test with mixed case labels
        labels = ["CAT", "Dog", "BIRD"]
        result = await model.classify_zero_shot(temp_image_png, labels)

        assert result is not None
        # Result label should be normalized to lowercase
        assert result["label"].lower() in [l.lower() for l in labels]

    @pytest.mark.asyncio
    async def test_many_labels(self, temp_image_png):
        """Test classification with many labels."""
        from models.vision_model import CLIPVisionModel

        model = CLIPVisionModel(model_id="test_clip_many")
        await model.load()

        labels = [
            "cat", "dog", "bird", "fish", "horse",
            "elephant", "lion", "tiger", "bear", "wolf"
        ]
        result = await model.classify_zero_shot(temp_image_png, labels)

        assert result is not None
        assert len(result["all_scores"]) == len(labels)


class TestBatchProcessing:
    """Tests for batch processing multiple images."""

    @pytest.mark.asyncio
    async def test_batch_classification(self, temp_image_png, temp_image_jpg):
        """Test batch processing multiple images."""
        from models.vision_model import CLIPVisionModel

        model = CLIPVisionModel(model_id="test_clip_batch")
        await model.load()

        images = [temp_image_png, temp_image_jpg]
        labels = ["photo", "drawing", "painting"]

        results = await model.classify_zero_shot_batch(images, labels)

        assert results is not None
        assert len(results) == len(images)
        for result in results:
            assert "label" in result
            assert "score" in result
            assert "all_scores" in result

    @pytest.mark.asyncio
    async def test_batch_with_mixed_formats(
        self, temp_image_png, temp_image_jpg, temp_image_webp
    ):
        """Test batch processing with mixed image formats."""
        from models.vision_model import CLIPVisionModel

        model = CLIPVisionModel(model_id="test_clip_mixed")
        await model.load()

        images = [temp_image_png, temp_image_jpg, temp_image_webp]
        labels = ["colorful", "monochrome"]

        results = await model.classify_zero_shot_batch(images, labels)

        assert results is not None
        assert len(results) == 3
