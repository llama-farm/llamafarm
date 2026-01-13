"""Tests for Background Removal Model (RMBG).

Phase 10 of Universal Runtime ML Enhancements.
"""

import base64
from io import BytesIO

import pytest
from PIL import Image

from models.background_removal_model import BackgroundRemovalModel


def create_test_image(width: int = 224, height: int = 224) -> bytes:
    """Create a simple test image with a colored foreground."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))  # White background
    # Add a colored rectangle in the center (simulating foreground)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Draw a red rectangle in the center
    center_x, center_y = width // 2, height // 2
    draw.rectangle(
        [center_x - 30, center_y - 30, center_x + 30, center_y + 30],
        fill=(255, 0, 0)
    )
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def create_test_image_base64(width: int = 224, height: int = 224) -> str:
    """Create a base64-encoded test image."""
    return base64.b64encode(create_test_image(width, height)).decode("utf-8")


class TestBackgroundRemovalModelLoading:
    """Test RMBG model loading."""

    @pytest.mark.asyncio
    async def test_model_loads_successfully(self):
        """Test that RMBG model loads."""
        model = BackgroundRemovalModel(model_id="test-rmbg", device="cpu")
        await model.load()
        assert model.is_loaded
        await model.unload()

    @pytest.mark.asyncio
    async def test_model_unload(self):
        """Test that model can be unloaded."""
        model = BackgroundRemovalModel(model_id="test-rmbg", device="cpu")
        await model.load()
        assert model.is_loaded
        await model.unload()
        assert not model.is_loaded

    @pytest.mark.asyncio
    async def test_model_reloads(self):
        """Test that model can be reloaded after unload."""
        model = BackgroundRemovalModel(model_id="test-rmbg", device="cpu")
        await model.load()
        await model.unload()
        await model.load()
        assert model.is_loaded
        await model.unload()


class TestBackgroundRemoval:
    """Test background removal functionality."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = BackgroundRemovalModel(model_id="test-rmbg", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_remove_background_returns_dict(self, model):
        """Test that remove_background returns a dictionary."""
        image_bytes = create_test_image()
        result = await model.remove_background(image_bytes)
        assert isinstance(result, dict)
        assert "image" in result
        assert "width" in result
        assert "height" in result

    @pytest.mark.asyncio
    async def test_remove_background_with_base64(self, model):
        """Test background removal with base64-encoded image."""
        image_b64 = create_test_image_base64()
        result = await model.remove_background(image_b64)
        assert isinstance(result, dict)
        assert "image" in result

    @pytest.mark.asyncio
    async def test_remove_background_with_pil_image(self, model):
        """Test background removal with PIL Image."""
        pil_image = Image.new("RGB", (224, 224), color=(100, 100, 100))
        result = await model.remove_background(pil_image)
        assert isinstance(result, dict)
        assert "image" in result

    @pytest.mark.asyncio
    async def test_returns_correct_dimensions(self, model):
        """Test that output dimensions match input."""
        image_bytes = create_test_image(width=320, height=240)
        result = await model.remove_background(image_bytes)
        assert result["width"] == 320
        assert result["height"] == 240


class TestOutputFormat:
    """Test output format and alpha channel."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = BackgroundRemovalModel(model_id="test-rmbg", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_output_is_valid_base64(self, model):
        """Test that output image is valid base64."""
        image_bytes = create_test_image()
        result = await model.remove_background(image_bytes)
        # Should be able to decode the base64
        decoded = base64.b64decode(result["image"])
        assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_output_is_valid_png(self, model):
        """Test that output is a valid PNG image."""
        image_bytes = create_test_image()
        result = await model.remove_background(image_bytes)
        decoded = base64.b64decode(result["image"])
        # Should be able to open as image
        img = Image.open(BytesIO(decoded))
        assert img.format == "PNG"

    @pytest.mark.asyncio
    async def test_output_has_alpha_channel(self, model):
        """Test that output image has alpha channel."""
        image_bytes = create_test_image()
        result = await model.remove_background(image_bytes)
        decoded = base64.b64decode(result["image"])
        img = Image.open(BytesIO(decoded))
        assert img.mode == "RGBA"


class TestMaskOutput:
    """Test mask output functionality."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = BackgroundRemovalModel(model_id="test-rmbg", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_return_mask_false_no_mask(self, model):
        """Test that mask is not returned when return_mask=False."""
        image_bytes = create_test_image()
        result = await model.remove_background(image_bytes, return_mask=False)
        assert "mask" not in result

    @pytest.mark.asyncio
    async def test_return_mask_true_includes_mask(self, model):
        """Test that mask is returned when return_mask=True."""
        image_bytes = create_test_image()
        result = await model.remove_background(image_bytes, return_mask=True)
        assert "mask" in result
        assert len(result["mask"]) > 0

    @pytest.mark.asyncio
    async def test_mask_is_valid_image(self, model):
        """Test that mask is a valid grayscale image."""
        image_bytes = create_test_image()
        result = await model.remove_background(image_bytes, return_mask=True)
        decoded = base64.b64decode(result["mask"])
        mask = Image.open(BytesIO(decoded))
        # Mask should be grayscale
        assert mask.mode == "L"


class TestBatchProcessing:
    """Test batch background removal."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = BackgroundRemovalModel(model_id="test-rmbg", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_batch_removal(self, model):
        """Test batch background removal returns list."""
        images = [create_test_image() for _ in range(3)]
        results = await model.remove_background_batch(images)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_batch_removal_each_result(self, model):
        """Test each batch result has correct format."""
        images = [create_test_image() for _ in range(2)]
        results = await model.remove_background_batch(images)

        for result in results:
            assert "image" in result
            assert "width" in result
            assert "height" in result


class TestImageLoading:
    """Test image loading from various formats."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = BackgroundRemovalModel(model_id="test-rmbg", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_load_from_bytes(self, model):
        """Test loading image from bytes."""
        image_bytes = create_test_image()
        result = await model.remove_background(image_bytes)
        assert result is not None

    @pytest.mark.asyncio
    async def test_load_from_base64(self, model):
        """Test loading image from base64 string."""
        image_b64 = create_test_image_base64()
        result = await model.remove_background(image_b64)
        assert result is not None

    @pytest.mark.asyncio
    async def test_load_from_pil(self, model):
        """Test loading image from PIL Image."""
        pil_image = Image.new("RGB", (224, 224), color=(50, 100, 150))
        result = await model.remove_background(pil_image)
        assert result is not None

    @pytest.mark.asyncio
    async def test_invalid_image_raises(self, model):
        """Test that invalid image data raises error."""
        with pytest.raises(ValueError):
            await model.remove_background("not-a-valid-image-or-base64")


class TestModelNotLoaded:
    """Test behavior when model is not loaded."""

    @pytest.mark.asyncio
    async def test_remove_background_without_load_raises(self):
        """Test that remove_background without load raises error."""
        model = BackgroundRemovalModel(model_id="test-rmbg", device="cpu")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await model.remove_background(create_test_image())


class TestModelInfo:
    """Test model information retrieval."""

    @pytest.mark.asyncio
    async def test_model_info(self):
        """Test getting model info."""
        model = BackgroundRemovalModel(model_id="test-rmbg", device="cpu")
        await model.load()

        info = model.get_model_info()
        assert "hf_model_name" in info
        assert "is_loaded" in info
        assert "output_format" in info
        assert info["is_loaded"] is True
        assert info["output_format"] == "PNG with alpha channel"

        await model.unload()
