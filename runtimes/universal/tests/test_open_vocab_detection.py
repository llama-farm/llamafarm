"""Tests for open-vocabulary object detection with OWL-ViT.

Tests verify:
- OpenVocabDetectionModel loads successfully
- Text-conditioned detection finds objects based on queries
- Image-conditioned detection finds similar objects
- Batch detection works
- Threshold filtering works correctly
- top_k limiting works
"""

import base64
import os
import tempfile
from io import BytesIO

import pytest
from PIL import Image


def create_test_image(color: tuple = (255, 0, 0), size: tuple = (640, 480)) -> bytes:
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


def create_test_image_base64(color: tuple = (255, 0, 0), size: tuple = (640, 480)) -> str:
    """Create a base64-encoded test image.

    Args:
        color: RGB color tuple
        size: Image dimensions

    Returns:
        Base64-encoded PNG string
    """
    return base64.b64encode(create_test_image(color, size)).decode("utf-8")


def create_test_image_with_shapes(size: tuple = (640, 480)) -> bytes:
    """Create a test image with colored shapes for detection testing.

    Args:
        size: Image dimensions

    Returns:
        PNG image bytes
    """
    from PIL import ImageDraw

    img = Image.new("RGB", size, (255, 255, 255))  # White background
    draw = ImageDraw.Draw(img)

    # Draw a red rectangle (simulating an object)
    draw.rectangle([50, 50, 200, 150], fill=(255, 0, 0))

    # Draw a blue circle (simulating another object)
    draw.ellipse([300, 200, 450, 350], fill=(0, 0, 255))

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def temp_image_simple():
    """Create a temporary simple image file."""
    img_bytes = create_test_image((128, 128, 128))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(img_bytes)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_image_with_shapes():
    """Create a temporary image with shapes."""
    img_bytes = create_test_image_with_shapes()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(img_bytes)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_image_red():
    """Create a temporary red image for query."""
    img_bytes = create_test_image((255, 0, 0), (100, 100))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(img_bytes)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestOpenVocabModelLoading:
    """Tests for OWL-ViT model loading."""

    @pytest.mark.asyncio
    async def test_model_loads_successfully(self):
        """Test that OWL-ViT model loads without errors."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_load")
        await model.load()

        assert model.is_loaded
        assert model.processor is not None
        assert model.model is not None

    @pytest.mark.asyncio
    async def test_model_unload(self):
        """Test that OWL-ViT model can be unloaded."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_unload")
        await model.load()
        assert model.is_loaded

        await model.unload()
        assert not model.is_loaded


class TestTextConditionedDetection:
    """Tests for text-conditioned detection."""

    @pytest.mark.asyncio
    async def test_detection_returns_correct_structure(self, temp_image_simple):
        """Test that detection returns correct response structure."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_struct")
        await model.load()

        queries = ["a red object", "a blue object"]
        result = await model.detect_by_text(
            temp_image_simple, queries, threshold=0.01
        )

        assert "objects" in result
        assert "count" in result
        assert "queries" in result
        assert "image_size" in result
        assert result["queries"] == queries

    @pytest.mark.asyncio
    async def test_detection_objects_have_required_fields(self, temp_image_with_shapes):
        """Test that detected objects have required fields."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_fields")
        await model.load()

        queries = ["a red rectangle", "a blue circle"]
        result = await model.detect_by_text(
            temp_image_with_shapes, queries, threshold=0.01
        )

        # Even if no detections, structure should be correct
        for obj in result["objects"]:
            assert "query" in obj
            assert "label" in obj
            assert "score" in obj
            assert "box" in obj
            assert "x1" in obj["box"]
            assert "y1" in obj["box"]
            assert "x2" in obj["box"]
            assert "y2" in obj["box"]
            assert obj["query"] in queries
            assert 0 <= obj["score"] <= 1

    @pytest.mark.asyncio
    async def test_detection_with_base64_image(self):
        """Test detection with base64-encoded image."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_base64")
        await model.load()

        img_base64 = create_test_image_base64((200, 100, 50))
        queries = ["an object"]

        result = await model.detect_by_text(img_base64, queries, threshold=0.01)

        assert result is not None
        assert "objects" in result

    @pytest.mark.asyncio
    async def test_threshold_filtering(self, temp_image_simple):
        """Test that threshold filters out low-confidence detections."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_thresh")
        await model.load()

        queries = ["anything"]

        # Very low threshold - should get more detections
        low_result = await model.detect_by_text(
            temp_image_simple, queries, threshold=0.001
        )

        # High threshold - should get fewer detections
        high_result = await model.detect_by_text(
            temp_image_simple, queries, threshold=0.9
        )

        # All detections should be above their respective thresholds
        for obj in low_result["objects"]:
            assert obj["score"] >= 0.001

        for obj in high_result["objects"]:
            assert obj["score"] >= 0.9

    @pytest.mark.asyncio
    async def test_top_k_limiting(self, temp_image_simple):
        """Test that top_k limits the number of results."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_topk")
        await model.load()

        queries = ["a shape", "an object", "a thing"]

        # Very low threshold to get many detections
        result = await model.detect_by_text(
            temp_image_simple, queries, threshold=0.001, top_k=3
        )

        assert len(result["objects"]) <= 3

    @pytest.mark.asyncio
    async def test_image_size_in_response(self, temp_image_simple):
        """Test that image size is correctly reported."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_size")
        await model.load()

        result = await model.detect_by_text(
            temp_image_simple, ["anything"], threshold=0.01
        )

        # Our test image is 640x480
        assert result["image_size"]["width"] == 640
        assert result["image_size"]["height"] == 480


class TestImageConditionedDetection:
    """Tests for image-conditioned detection."""

    @pytest.mark.asyncio
    async def test_image_guided_detection_structure(
        self, temp_image_with_shapes, temp_image_red
    ):
        """Test image-guided detection returns correct structure."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_img_struct")
        await model.load()

        result = await model.detect_by_image(
            temp_image_with_shapes,
            query_images=[temp_image_red],
            threshold=0.1,
        )

        assert "objects" in result
        assert "count" in result
        assert "num_queries" in result
        assert "image_size" in result
        assert result["num_queries"] == 1

    @pytest.mark.asyncio
    async def test_image_guided_detection_objects(
        self, temp_image_with_shapes, temp_image_red
    ):
        """Test image-guided detection object fields."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_img_fields")
        await model.load()

        result = await model.detect_by_image(
            temp_image_with_shapes,
            query_images=[temp_image_red],
            threshold=0.1,
        )

        for obj in result["objects"]:
            assert "query_index" in obj
            assert "score" in obj
            assert "box" in obj
            assert isinstance(obj["query_index"], int)
            assert 0 <= obj["score"] <= 1


class TestBatchDetection:
    """Tests for batch detection."""

    @pytest.mark.asyncio
    async def test_batch_detection(self, temp_image_simple, temp_image_with_shapes):
        """Test batch detection on multiple images."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_batch")
        await model.load()

        images = [temp_image_simple, temp_image_with_shapes]
        queries = ["a shape", "an object"]

        results = await model.detect_batch_by_text(
            images, queries, threshold=0.01
        )

        assert len(results) == 2
        for result in results:
            assert "objects" in result
            assert "count" in result
            assert "queries" in result


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_detect_before_load_raises_error(self, temp_image_simple):
        """Test that detecting before loading raises an error."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_no_load")

        with pytest.raises(RuntimeError, match="not loaded"):
            await model.detect_by_text(temp_image_simple, ["test"])

    @pytest.mark.asyncio
    async def test_empty_queries_raises_error(self, temp_image_simple):
        """Test that empty queries raises an error."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_no_queries")
        await model.load()

        with pytest.raises(ValueError, match="At least one"):
            await model.detect_by_text(temp_image_simple, [])

    @pytest.mark.asyncio
    async def test_empty_query_images_raises_error(self, temp_image_simple):
        """Test that empty query images raises an error."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_no_query_img")
        await model.load()

        with pytest.raises(ValueError, match="At least one"):
            await model.detect_by_image(temp_image_simple, [])


class TestModelInfo:
    """Tests for model info."""

    @pytest.mark.asyncio
    async def test_model_info(self):
        """Test model info returns expected fields."""
        from models.open_vocab_detection_model import OpenVocabDetectionModel

        model = OpenVocabDetectionModel(model_id="test_owl_info")
        await model.load()

        info = model.get_model_info()

        assert info["is_loaded"]
        assert "hf_model_name" in info
        assert "capabilities" in info
        assert "text_conditioned_detection" in info["capabilities"]
        assert "image_conditioned_detection" in info["capabilities"]
