"""Tests for few-shot image classification with CLIP linear probe.

Tests verify:
- FewShotImageClassifier loads successfully
- Training works with minimal images (2+ images)
- Classification returns accurate predictions
- Refinement can add new classes
- State can be saved and loaded
- Batch prediction works
"""

import base64
import os
import tempfile
from io import BytesIO

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
def temp_image_red():
    """Create a temporary red image file."""
    img_bytes = create_test_image((255, 0, 0))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(img_bytes)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_image_blue():
    """Create a temporary blue image file."""
    img_bytes = create_test_image((0, 0, 255))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(img_bytes)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_image_green():
    """Create a temporary green image file."""
    img_bytes = create_test_image((0, 255, 0))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(img_bytes)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestFewShotModelLoading:
    """Tests for few-shot classifier loading."""

    @pytest.mark.asyncio
    async def test_model_loads_successfully(self):
        """Test that few-shot classifier loads without errors."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_load")
        await model.load()

        assert model.is_loaded
        assert model.processor is not None
        assert model.model is not None
        assert not model.is_trained  # Should not be trained yet

    @pytest.mark.asyncio
    async def test_model_unload(self):
        """Test that few-shot classifier can be unloaded."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_unload")
        await model.load()
        assert model.is_loaded

        await model.unload()
        assert not model.is_loaded
        assert not model.is_trained


class TestFewShotTraining:
    """Tests for few-shot training."""

    @pytest.mark.asyncio
    async def test_training_with_minimal_images(self, temp_image_red, temp_image_blue):
        """Test training with just 2 images."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_minimal")
        await model.load()

        images = [temp_image_red, temp_image_blue]
        labels = ["red", "blue"]

        result = await model.fit(images, labels)

        assert result["success"]
        assert result["num_samples"] == 2
        assert result["num_classes"] == 2
        assert set(result["classes"]) == {"red", "blue"}
        assert model.is_trained

    @pytest.mark.asyncio
    async def test_training_returns_accuracy(self, temp_image_red, temp_image_blue):
        """Test that training returns accuracy metric."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_accuracy")
        await model.load()

        images = [temp_image_red, temp_image_blue]
        labels = ["red", "blue"]

        result = await model.fit(images, labels, epochs=100)

        assert "final_accuracy" in result
        assert 0 <= result["final_accuracy"] <= 1
        assert "training_time_ms" in result

    @pytest.mark.asyncio
    async def test_training_with_base64_images(self):
        """Test training with base64-encoded images."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_base64")
        await model.load()

        images = [
            create_test_image_base64((255, 0, 0)),  # Red
            create_test_image_base64((0, 0, 255)),  # Blue
        ]
        labels = ["red", "blue"]

        result = await model.fit(images, labels)

        assert result["success"]
        assert model.is_trained


class TestFewShotPrediction:
    """Tests for few-shot prediction."""

    @pytest.mark.asyncio
    async def test_prediction_returns_label_and_score(
        self, temp_image_red, temp_image_blue
    ):
        """Test that prediction returns correct structure."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_predict")
        await model.load()

        await model.fit([temp_image_red, temp_image_blue], ["red", "blue"])

        result = await model.predict(temp_image_red)

        assert "label" in result
        assert "score" in result
        assert "all_scores" in result
        assert result["label"] in ["red", "blue"]
        assert 0 <= result["score"] <= 1

    @pytest.mark.asyncio
    async def test_prediction_accuracy_on_training_data(
        self, temp_image_red, temp_image_blue
    ):
        """Test that model correctly classifies training images."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_train_acc")
        await model.load()

        await model.fit(
            [temp_image_red, temp_image_blue],
            ["red", "blue"],
            epochs=200,  # More epochs for better fit
        )

        # Should correctly classify training images
        red_result = await model.predict(temp_image_red)
        blue_result = await model.predict(temp_image_blue)

        # With distinct colors and enough training, should get these right
        assert red_result["label"] == "red"
        assert blue_result["label"] == "blue"

    @pytest.mark.asyncio
    async def test_batch_prediction(self, temp_image_red, temp_image_blue):
        """Test batch prediction."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_batch")
        await model.load()

        await model.fit([temp_image_red, temp_image_blue], ["red", "blue"], epochs=100)

        results = await model.predict_batch([temp_image_red, temp_image_blue])

        assert len(results) == 2
        for result in results:
            assert "label" in result
            assert "score" in result
            assert "all_scores" in result


class TestFewShotRefinement:
    """Tests for refining a trained classifier."""

    @pytest.mark.asyncio
    async def test_refinement_adds_new_class(
        self, temp_image_red, temp_image_blue, temp_image_green
    ):
        """Test that refinement can add new classes."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_refine")
        await model.load()

        # Initial training with 2 classes
        await model.fit([temp_image_red, temp_image_blue], ["red", "blue"])
        assert len(model.classes) == 2

        # Refine with new class
        result = await model.refine([temp_image_green], ["green"])

        assert result["success"]
        assert len(model.classes) == 3
        assert "green" in model.classes
        assert "green" in result["new_classes_added"]

    @pytest.mark.asyncio
    async def test_refinement_with_more_examples(
        self, temp_image_red, temp_image_blue
    ):
        """Test refinement with additional examples of existing classes."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_refine_more")
        await model.load()

        # Initial training
        await model.fit([temp_image_red, temp_image_blue], ["red", "blue"])

        # Add more red examples
        more_red = create_test_image_base64((250, 10, 10))
        result = await model.refine([more_red], ["red"])

        assert result["success"]
        assert result["new_classes_added"] == []  # No new classes
        assert len(model.classes) == 2


class TestStateSaveLoad:
    """Tests for saving and loading classifier state."""

    @pytest.mark.asyncio
    async def test_save_state(self, temp_image_red, temp_image_blue):
        """Test that trained state can be saved."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_save")
        await model.load()
        await model.fit([temp_image_red, temp_image_blue], ["red", "blue"])

        state = model.save_state()

        assert state is not None
        assert isinstance(state, bytes)
        assert len(state) > 0

    @pytest.mark.asyncio
    async def test_load_state_restores_classifier(
        self, temp_image_red, temp_image_blue
    ):
        """Test that loaded state restores a working classifier."""
        from models.few_shot_classifier import FewShotImageClassifier

        # Train and save
        model1 = FewShotImageClassifier(model_id="test_fs_load_state1")
        await model1.load()
        await model1.fit([temp_image_red, temp_image_blue], ["red", "blue"])
        state = model1.save_state()

        # Create new model and load state
        model2 = FewShotImageClassifier(model_id="test_fs_load_state2")
        await model2.load()
        model2.load_state(state)

        assert model2.is_trained
        assert model2.classes == model1.classes

        # Should produce same predictions
        result = await model2.predict(temp_image_red)
        assert result["label"] in ["red", "blue"]


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_predict_before_training_raises_error(self, temp_image_red):
        """Test that predicting before training raises an error."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_no_train")
        await model.load()

        with pytest.raises(RuntimeError, match="not trained"):
            await model.predict(temp_image_red)

    @pytest.mark.asyncio
    async def test_train_without_load_raises_error(self, temp_image_red, temp_image_blue):
        """Test that training before loading raises an error."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_no_load")

        with pytest.raises(RuntimeError, match="not loaded"):
            await model.fit([temp_image_red, temp_image_blue], ["red", "blue"])

    @pytest.mark.asyncio
    async def test_mismatched_images_labels_raises_error(
        self, temp_image_red, temp_image_blue
    ):
        """Test that mismatched images/labels raises an error."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_mismatch")
        await model.load()

        with pytest.raises(ValueError, match="must match"):
            await model.fit(
                [temp_image_red, temp_image_blue],
                ["red"],  # Only one label for two images
            )

    @pytest.mark.asyncio
    async def test_too_few_images_raises_error(self, temp_image_red):
        """Test that training with fewer than 2 images raises an error."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_too_few")
        await model.load()

        with pytest.raises(ValueError, match="at least 2"):
            await model.fit([temp_image_red], ["red"])


class TestModelInfo:
    """Tests for model info."""

    @pytest.mark.asyncio
    async def test_model_info_before_training(self):
        """Test model info before training."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_info_before")
        await model.load()

        info = model.get_model_info()

        assert info["is_loaded"]
        assert not info["is_trained"]
        assert info["classes"] == []
        assert info["num_classes"] == 0

    @pytest.mark.asyncio
    async def test_model_info_after_training(
        self, temp_image_red, temp_image_blue
    ):
        """Test model info after training."""
        from models.few_shot_classifier import FewShotImageClassifier

        model = FewShotImageClassifier(model_id="test_fs_info_after")
        await model.load()
        await model.fit([temp_image_red, temp_image_blue], ["red", "blue"])

        info = model.get_model_info()

        assert info["is_loaded"]
        assert info["is_trained"]
        assert set(info["classes"]) == {"red", "blue"}
        assert info["num_classes"] == 2
