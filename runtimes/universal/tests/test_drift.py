"""Tests for Alibi Detect drift detection functionality.

These tests verify:
- Drift detector types (KS, MMD, Chi-squared)
- Model initialization and configuration
- Drift detection on new data
- Model save/load operations
- Helper functions
"""
# ruff: noqa: E402  # imports must come after pytest.importorskip

import pytest

alibi_detect = pytest.importorskip("alibi_detect", reason="drift addon not installed")

import numpy as np

from api_types.drift import (
    DriftDetectRequest,
    DriftFitRequest,
    DriftResult,
)
from models.drift_model import (
    DETECTOR_TYPES,
    DriftModel,
    delete_model,
    get_all_detectors,
    get_detectors_info,
    is_valid_detector,
    list_saved_models,
)


class TestDetectorInfo:
    """Test detector information functions."""

    def test_get_all_detectors(self):
        """Test that all detectors are returned."""
        detectors = get_all_detectors()
        assert len(detectors) == 3
        assert "ks" in detectors
        assert "mmd" in detectors
        assert "chi2" in detectors

    def test_is_valid_detector(self):
        """Test detector validation."""
        assert is_valid_detector("ks")
        assert is_valid_detector("mmd")
        assert is_valid_detector("chi2")
        assert not is_valid_detector("invalid")
        assert not is_valid_detector("")

    def test_get_detectors_info(self):
        """Test detector info structure."""
        info = get_detectors_info()
        assert len(info) == 3

        # Check KS info
        ks = next(d for d in info if d["name"] == "ks")
        assert ks["multivariate"] is False
        assert "description" in ks
        assert "default_params" in ks

        # Check MMD info (multivariate)
        mmd = next(d for d in info if d["name"] == "mmd")
        assert mmd["multivariate"] is True

    def test_detector_types_have_required_fields(self):
        """Test that all detector types have required configuration fields."""
        for name, config in DETECTOR_TYPES.items():
            assert "description" in config, f"{name} missing description"
            assert "class" in config, f"{name} missing class"
            assert "multivariate" in config, f"{name} missing multivariate"
            assert "default_params" in config, f"{name} missing default_params"


class TestDriftModel:
    """Test DriftModel class."""

    def test_init_with_ks_detector(self):
        """Test model initialization with KS detector."""
        model = DriftModel(
            model_id="test-model",
            device="cpu",
            detector="ks",
        )
        assert model.detector_type == "ks"
        assert model.model_type == "drift_ks"
        assert model.is_fitted is False

    def test_init_with_mmd_detector(self):
        """Test model initialization with MMD detector."""
        model = DriftModel(
            model_id="test-model",
            device="cpu",
            detector="mmd",
        )
        assert model.detector_type == "mmd"
        assert model.model_type == "drift_mmd"

    def test_init_with_chi2_detector(self):
        """Test model initialization with Chi-squared detector."""
        model = DriftModel(
            model_id="test-model",
            device="cpu",
            detector="chi2",
        )
        assert model.detector_type == "chi2"
        assert model.model_type == "drift_chi2"

    def test_init_with_invalid_detector(self):
        """Test model initialization with invalid detector raises error."""
        with pytest.raises(ValueError, match="Unknown detector"):
            DriftModel(
                model_id="test-model",
                device="cpu",
                detector="invalid",
            )

    def test_reference_size_before_fit(self):
        """Test reference size is 0 before fitting."""
        model = DriftModel(model_id="test", detector="ks")
        assert model.reference_size == 0


class TestDriftModelLoad:
    """Test DriftModel load method."""

    @pytest.mark.asyncio
    async def test_load_initializes_model(self):
        """Test load initializes the model."""
        model = DriftModel(model_id="test-model", detector="ks")
        await model.load()
        # Load is a no-op for drift models, just logs readiness
        assert model.is_fitted is False


class TestDriftModelFit:
    """Test DriftModel fit method."""

    @pytest.mark.asyncio
    async def test_fit_with_numpy_array(self):
        """Test fitting with numpy array."""
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (100, 3))

        model = DriftModel(model_id="test-fit", detector="ks")
        await model.load()
        result = await model.fit(reference_data, autosave=False)

        assert model.is_fitted is True
        assert model.reference_size == 100
        assert result["reference_size"] == 100
        assert result["n_features"] == 3
        assert "training_time_ms" in result

    @pytest.mark.asyncio
    async def test_fit_with_list(self):
        """Test fitting with list of lists."""
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (50, 2)).tolist()

        model = DriftModel(model_id="test-fit-list", detector="ks")
        await model.load()
        await model.fit(reference_data, autosave=False)

        assert model.is_fitted is True
        assert model.reference_size == 50

    @pytest.mark.asyncio
    async def test_fit_with_feature_names(self):
        """Test fitting with feature names."""
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (30, 2))

        model = DriftModel(model_id="test-fit-names", detector="ks")
        await model.load()
        result = await model.fit(
            reference_data,
            feature_names=["feature_a", "feature_b"],
            autosave=False,
        )

        assert model.is_fitted is True
        assert result["n_features"] == 2

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="MMD detector requires TensorFlow which is not installed")
    async def test_fit_mmd_detector(self):
        """Test fitting MMD detector."""
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (100, 4))

        model = DriftModel(model_id="test-fit-mmd", detector="mmd")
        await model.load()
        result = await model.fit(reference_data, autosave=False)

        assert model.is_fitted is True
        assert result["detector"] == "mmd"

    @pytest.mark.asyncio
    async def test_fit_chi2_detector(self):
        """Test fitting Chi-squared detector with categorical data."""
        np.random.seed(42)
        # Chi-squared works on discrete/categorical data
        reference_data = np.random.randint(0, 5, (100, 3)).astype(float)

        model = DriftModel(model_id="test-fit-chi2", detector="chi2")
        await model.load()
        result = await model.fit(reference_data, autosave=False)

        assert model.is_fitted is True
        assert result["detector"] == "chi2"


class TestDriftModelDetect:
    """Test DriftModel detect method."""

    @pytest.mark.asyncio
    async def test_detect_requires_fit(self):
        """Test that detection requires fitting first."""
        model = DriftModel(model_id="test-detect", detector="ks")
        await model.load()

        np.random.seed(42)
        test_data = np.random.normal(0, 1, (10, 3))

        with pytest.raises(RuntimeError, match="must be fitted"):
            await model.detect(test_data)

    @pytest.mark.asyncio
    async def test_detect_no_drift(self):
        """Test detection with same distribution (no drift)."""
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (100, 3))

        model = DriftModel(model_id="test-detect-no-drift", detector="ks")
        await model.load()
        await model.fit(reference_data, autosave=False)

        # Test data from same distribution
        np.random.seed(123)
        test_data = np.random.normal(0, 1, (50, 3))
        result = await model.detect(test_data)

        assert isinstance(result.is_drift, bool)
        assert isinstance(result.p_value, float)
        assert isinstance(result.threshold, float)
        # Should likely not detect drift with same distribution
        # (though not guaranteed due to random sampling)

    @pytest.mark.asyncio
    async def test_detect_with_drift(self):
        """Test detection with shifted distribution (drift)."""
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (100, 3))

        model = DriftModel(model_id="test-detect-drift", detector="ks")
        await model.load()
        await model.fit(reference_data, autosave=False)

        # Test data from shifted distribution
        np.random.seed(456)
        test_data = np.random.normal(5, 1, (50, 3))  # Shifted mean
        result = await model.detect(test_data)

        assert result.is_drift is True
        assert result.p_value < 0.05  # Low p-value indicates drift

    @pytest.mark.asyncio
    async def test_detect_updates_statistics(self):
        """Test that detection updates internal statistics."""
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (50, 2))
        test_data = np.random.normal(5, 1, (20, 2))

        model = DriftModel(model_id="test-stats", detector="ks")
        await model.load()
        await model.fit(reference_data, autosave=False)
        await model.detect(test_data)

        status = model.get_status()
        assert status.detection_count == 1
        assert status.last_detection is not None


class TestDriftModelStatus:
    """Test DriftModel get_status method."""

    @pytest.mark.asyncio
    async def test_status_before_fit(self):
        """Test status before fitting."""
        model = DriftModel(model_id="test-status", detector="ks")
        status = model.get_status()

        assert status.detector_id == "test-status"
        assert status.detector_type == "ks"
        assert status.is_fitted is False
        assert status.reference_size == 0
        assert status.detection_count == 0
        assert status.drift_count == 0
        assert status.last_detection is None

    @pytest.mark.asyncio
    async def test_status_after_fit(self):
        """Test status after fitting."""
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (50, 2))

        model = DriftModel(model_id="test-status-fit", detector="ks")
        await model.load()
        await model.fit(reference_data, autosave=False)

        status = model.get_status()
        assert status.is_fitted is True
        assert status.reference_size == 50

    @pytest.mark.asyncio
    async def test_status_after_multiple_detections(self):
        """Test status after multiple detections."""
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (100, 3))

        model = DriftModel(model_id="test-multi", detector="ks")
        await model.load()
        await model.fit(reference_data, autosave=False)

        # Run multiple detections
        for i in range(3):
            np.random.seed(i * 100)
            test_data = np.random.normal(i, 1, (20, 3))
            await model.detect(test_data)

        status = model.get_status()
        assert status.detection_count == 3


class TestDriftModelReset:
    """Test DriftModel reset method."""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """Test that reset clears all state."""
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (50, 2))

        model = DriftModel(model_id="test-reset", detector="ks")
        await model.load()
        await model.fit(reference_data, autosave=False)

        # Verify fitted state
        assert model.is_fitted is True
        assert model.reference_size == 50

        # Reset
        await model.reset()

        # Verify cleared state
        assert model.is_fitted is False
        assert model.reference_size == 0


class TestDriftModelSaveLoad:
    """Test DriftModel save/load methods."""

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """Test saving and loading a model."""
        import uuid
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (50, 2))

        # Generate unique model id for this test
        unique_id = f"test-save-{uuid.uuid4().hex[:8]}"

        # Create and fit model
        model = DriftModel(model_id=unique_id, detector="ks")
        await model.load()
        await model.fit(reference_data, autosave=False)

        # Save model (uses default path in DRIFT_MODELS_DIR)
        saved_path = await model.save()
        assert saved_path.exists()

        try:
            # Create new model and load
            new_model = DriftModel(model_id=unique_id, detector="ks")
            await new_model.load_from_path(saved_path)

            assert new_model.is_fitted is True
            assert new_model.reference_size == 50
        finally:
            # Cleanup: delete the saved model
            if saved_path.exists():
                saved_path.unlink()
            metadata_path = saved_path.with_suffix(".metadata.json")
            if metadata_path.exists():
                metadata_path.unlink()


class TestDriftModelUnload:
    """Test DriftModel unload method."""

    @pytest.mark.asyncio
    async def test_unload_clears_resources(self):
        """Test that unload clears resources."""
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (30, 2))

        model = DriftModel(model_id="test-unload", detector="ks")
        await model.load()
        await model.fit(reference_data, autosave=False)

        assert model.is_fitted is True

        await model.unload()

        assert model.is_fitted is False
        assert model.reference_size == 0


class TestDriftAPITypes:
    """Test drift API type models."""

    def test_fit_request_validation(self):
        """Test DriftFitRequest validation."""
        request = DriftFitRequest(
            model="test-model",
            detector="ks",
            reference_data=[[1.0, 2.0], [3.0, 4.0]],
        )
        assert request.model == "test-model"
        assert request.detector == "ks"
        assert len(request.reference_data) == 2

    def test_fit_request_default_detector(self):
        """Test DriftFitRequest default detector."""
        request = DriftFitRequest(
            reference_data=[[1.0, 2.0]],
        )
        assert request.detector == "ks"

    def test_detect_request_validation(self):
        """Test DriftDetectRequest validation."""
        request = DriftDetectRequest(
            model="test-model",
            data=[[1.0, 2.0, 3.0]],
        )
        assert request.model == "test-model"
        assert len(request.data) == 1

    def test_drift_result_structure(self):
        """Test DriftResult structure."""
        result = DriftResult(
            is_drift=True,
            p_value=0.01,
            threshold=0.05,
            distance=1.5,
        )
        assert result.is_drift is True
        assert result.p_value == 0.01
        assert result.threshold == 0.05
        assert result.distance == 1.5

    def test_drift_result_optional_distance(self):
        """Test DriftResult with optional distance."""
        result = DriftResult(
            is_drift=False,
            p_value=0.3,
            threshold=0.05,
        )
        assert result.distance is None


class TestListAndDeleteModels:
    """Test list and delete model functions."""

    def test_list_saved_models(self):
        """Test listing saved models returns a list."""
        models = list_saved_models()
        assert isinstance(models, list)

    def test_delete_nonexistent_model(self):
        """Test deleting nonexistent model returns False."""
        result = delete_model("nonexistent-model-xyz-123")
        assert result is False
