"""Tests for Object Detection Model (YOLOS).

Phase 9 of Universal Runtime ML Enhancements.
"""

import base64
from io import BytesIO

import pytest
from PIL import Image

from models.object_detection_model import COCO_CLASSES, ObjectDetectionModel


def create_test_image(width: int = 224, height: int = 224) -> bytes:
    """Create a simple test image."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def create_test_image_base64(width: int = 224, height: int = 224) -> str:
    """Create a base64-encoded test image."""
    return base64.b64encode(create_test_image(width, height)).decode("utf-8")


class TestObjectDetectionModelLoading:
    """Test YOLOS model loading."""

    @pytest.mark.asyncio
    async def test_model_loads_successfully(self):
        """Test that YOLOS model loads."""
        model = ObjectDetectionModel(model_id="test-yolos", device="cpu")
        await model.load()
        assert model.is_loaded
        await model.unload()

    @pytest.mark.asyncio
    async def test_model_unload(self):
        """Test that model can be unloaded."""
        model = ObjectDetectionModel(model_id="test-yolos", device="cpu")
        await model.load()
        assert model.is_loaded
        await model.unload()
        assert not model.is_loaded

    @pytest.mark.asyncio
    async def test_model_reloads(self):
        """Test that model can be reloaded after unload."""
        model = ObjectDetectionModel(model_id="test-yolos", device="cpu")
        await model.load()
        await model.unload()
        await model.load()
        assert model.is_loaded
        await model.unload()


class TestObjectDetection:
    """Test object detection functionality."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = ObjectDetectionModel(model_id="test-yolos", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_detect_returns_dict(self, model):
        """Test that detect returns a dictionary."""
        image_bytes = create_test_image()
        result = await model.detect(image_bytes)
        assert isinstance(result, dict)
        assert "objects" in result
        assert "count" in result
        assert "image_size" in result

    @pytest.mark.asyncio
    async def test_detect_with_base64(self, model):
        """Test detection with base64-encoded image."""
        image_b64 = create_test_image_base64()
        result = await model.detect(image_b64)
        assert isinstance(result, dict)
        assert "objects" in result

    @pytest.mark.asyncio
    async def test_detect_with_pil_image(self, model):
        """Test detection with PIL Image."""
        pil_image = Image.new("RGB", (224, 224), color=(100, 100, 100))
        result = await model.detect(pil_image)
        assert isinstance(result, dict)
        assert "objects" in result

    @pytest.mark.asyncio
    async def test_detect_returns_image_size(self, model):
        """Test that image size is returned correctly."""
        image_bytes = create_test_image(width=320, height=240)
        result = await model.detect(image_bytes)
        assert result["image_size"]["width"] == 320
        assert result["image_size"]["height"] == 240

    @pytest.mark.asyncio
    async def test_detect_with_threshold(self, model):
        """Test detection with custom threshold."""
        image_bytes = create_test_image()
        # High threshold should return fewer results
        high_result = await model.detect(image_bytes, threshold=0.9)
        # Low threshold should return more results
        low_result = await model.detect(image_bytes, threshold=0.1)
        # Count from low threshold should be >= high threshold
        assert low_result["count"] >= high_result["count"]

    @pytest.mark.asyncio
    async def test_detect_with_label_filter(self, model):
        """Test detection with label filter."""
        image_bytes = create_test_image()
        result = await model.detect(image_bytes, labels=["cat", "dog"])
        # All detected objects should be in the filter list
        for obj in result["objects"]:
            assert obj["label"] in ["cat", "dog"]


class TestObjectDetectionOutput:
    """Test object detection output format."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = ObjectDetectionModel(model_id="test-yolos", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_object_format(self, model):
        """Test that detected objects have correct format."""
        image_bytes = create_test_image()
        result = await model.detect(image_bytes, threshold=0.01)

        # If any objects detected, check their format
        for obj in result["objects"]:
            assert "label" in obj
            assert "score" in obj
            assert "box" in obj
            assert isinstance(obj["label"], str)
            assert isinstance(obj["score"], float)
            assert 0 <= obj["score"] <= 1
            # Box should have x1, y1, x2, y2
            box = obj["box"]
            assert "x1" in box
            assert "y1" in box
            assert "x2" in box
            assert "y2" in box

    @pytest.mark.asyncio
    async def test_objects_sorted_by_score(self, model):
        """Test that objects are sorted by score descending."""
        image_bytes = create_test_image()
        result = await model.detect(image_bytes, threshold=0.01)

        scores = [obj["score"] for obj in result["objects"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_count_matches_objects_length(self, model):
        """Test that count matches the number of objects."""
        image_bytes = create_test_image()
        result = await model.detect(image_bytes)
        assert result["count"] == len(result["objects"])


class TestBatchDetection:
    """Test batch object detection."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = ObjectDetectionModel(model_id="test-yolos", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_batch_detection(self, model):
        """Test batch detection returns list."""
        images = [create_test_image() for _ in range(3)]
        results = await model.detect_batch(images)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_batch_detection_each_result(self, model):
        """Test each batch result has correct format."""
        images = [create_test_image() for _ in range(2)]
        results = await model.detect_batch(images)

        for result in results:
            assert "objects" in result
            assert "count" in result
            assert "image_size" in result


class TestImageLoading:
    """Test image loading from various formats."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = ObjectDetectionModel(model_id="test-yolos", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_load_from_bytes(self, model):
        """Test loading image from bytes."""
        image_bytes = create_test_image()
        result = await model.detect(image_bytes)
        assert result is not None

    @pytest.mark.asyncio
    async def test_load_from_base64(self, model):
        """Test loading image from base64 string."""
        image_b64 = create_test_image_base64()
        result = await model.detect(image_b64)
        assert result is not None

    @pytest.mark.asyncio
    async def test_load_from_pil(self, model):
        """Test loading image from PIL Image."""
        pil_image = Image.new("RGB", (224, 224), color=(50, 100, 150))
        result = await model.detect(pil_image)
        assert result is not None

    @pytest.mark.asyncio
    async def test_invalid_image_raises(self, model):
        """Test that invalid image data raises error."""
        with pytest.raises(ValueError):
            await model.detect("not-a-valid-image-or-base64")


class TestModelNotLoaded:
    """Test behavior when model is not loaded."""

    @pytest.mark.asyncio
    async def test_detect_without_load_raises(self):
        """Test that detect without load raises error."""
        model = ObjectDetectionModel(model_id="test-yolos", device="cpu")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await model.detect(create_test_image())


class TestModelInfo:
    """Test model information retrieval."""

    @pytest.mark.asyncio
    async def test_model_info(self):
        """Test getting model info."""
        model = ObjectDetectionModel(model_id="test-yolos", device="cpu")
        await model.load()

        info = model.get_model_info()
        assert "hf_model_name" in info
        assert "is_loaded" in info
        assert "supported_classes" in info
        assert info["is_loaded"] is True
        assert len(info["supported_classes"]) == 80  # COCO has 80 classes

        await model.unload()


class TestConstants:
    """Test module constants."""

    def test_coco_classes_count(self):
        """Test that COCO classes list has 80 items."""
        assert len(COCO_CLASSES) == 80

    def test_coco_classes_content(self):
        """Test that common COCO classes are present."""
        assert "person" in COCO_CLASSES
        assert "car" in COCO_CLASSES
        assert "dog" in COCO_CLASSES
        assert "cat" in COCO_CLASSES
        assert "bicycle" in COCO_CLASSES
