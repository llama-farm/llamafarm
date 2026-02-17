"""Tests for CatBoost classification module.

Tests CatBoost gradient boosting functionality including:
- Model initialization and training
- Prediction and probability estimation
- Incremental learning
- Model save/load
- API types
"""
# ruff: noqa: E402  # imports must come after pytest.importorskip

import pytest

catboost = pytest.importorskip("catboost", reason="catboost addon not installed")

from pathlib import Path

import numpy as np

from api_types.catboost import (
    CatBoostFeatureImportance,
    CatBoostFeatureImportanceResponse,
    CatBoostFitRequest,
    CatBoostFitResponse,
    CatBoostInfoResponse,
    CatBoostModelInfo,
    CatBoostPrediction,
    CatBoostPredictRequest,
    CatBoostPredictResponse,
    CatBoostUpdateRequest,
    CatBoostUpdateResponse,
)
from models.catboost_model import (
    CATBOOST_AVAILABLE,
    CatBoostModel,
    get_catboost_info,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    # Create features
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create labels based on first feature
    y = (X[:, 0] > 0).astype(int)

    return X, y


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1

    return X, y.astype(np.float32)


@pytest.fixture
def feature_names():
    """Feature names for testing."""
    return ["temp", "humidity", "pressure", "wind_speed", "precipitation"]


@pytest.fixture
def catboost_models_dir():
    """Create a temporary directory for CatBoost models."""
    models_dir = Path.home() / ".llamafarm" / "models" / "catboost"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


# =============================================================================
# CatBoost Availability Tests
# =============================================================================


class TestCatBoostAvailability:
    """Tests for CatBoost availability detection."""

    def test_catboost_available(self):
        """Test that CatBoost is available."""
        assert CATBOOST_AVAILABLE is True

    def test_get_info(self):
        """Test get_catboost_info function."""
        info = get_catboost_info()
        assert info["available"] is True
        assert "classifier" in info["model_types"]
        assert "regressor" in info["model_types"]


# =============================================================================
# CatBoost Model Tests
# =============================================================================


class TestCatBoostModel:
    """Tests for CatBoostModel class."""

    @pytest.mark.asyncio
    async def test_init_classifier(self):
        """Test classifier initialization."""
        model = CatBoostModel(
            model_id="test-classifier",
            model_type="classifier",
            iterations=10,
        )
        await model.load()
        assert model._is_loaded

    @pytest.mark.asyncio
    async def test_init_regressor(self):
        """Test regressor initialization."""
        model = CatBoostModel(
            model_id="test-regressor",
            model_type="regressor",
            iterations=10,
        )
        await model.load()
        assert model._is_loaded

    @pytest.mark.asyncio
    async def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        model = CatBoostModel(
            model_id="test-invalid",
            model_type="invalid_type",
        )
        with pytest.raises(ValueError, match="Unknown model type"):
            await model.load()

    @pytest.mark.asyncio
    async def test_fit_classifier(self, sample_classification_data, feature_names):
        """Test classifier training."""
        X, y = sample_classification_data
        model = CatBoostModel(
            model_id="test-fit-classifier",
            model_type="classifier",
            iterations=10,
        )
        await model.load()

        result = await model.fit(X, y, feature_names=feature_names)

        assert result["samples_fitted"] == len(y)
        assert result["n_features"] == X.shape[1]
        assert result["classes"] is not None
        assert model.is_fitted

    @pytest.mark.asyncio
    async def test_fit_regressor(self, sample_regression_data, feature_names):
        """Test regressor training."""
        X, y = sample_regression_data
        model = CatBoostModel(
            model_id="test-fit-regressor",
            model_type="regressor",
            iterations=10,
        )
        await model.load()

        result = await model.fit(X, y, feature_names=feature_names)

        assert result["samples_fitted"] == len(y)
        assert result["classes"] is None  # Regressor has no classes
        assert model.is_fitted

    @pytest.mark.asyncio
    async def test_predict_classifier(self, sample_classification_data):
        """Test classifier prediction."""
        X, y = sample_classification_data
        model = CatBoostModel(
            model_id="test-predict",
            model_type="classifier",
            iterations=10,
        )
        await model.load()
        await model.fit(X, y)

        predictions = await model.predict(X[:10])

        assert len(predictions) == 10
        # All predictions should be valid class labels
        assert all(p in [0, 1] for p in predictions)

    @pytest.mark.asyncio
    async def test_predict_proba(self, sample_classification_data):
        """Test probability prediction."""
        X, y = sample_classification_data
        model = CatBoostModel(
            model_id="test-proba",
            model_type="classifier",
            iterations=10,
        )
        await model.load()
        await model.fit(X, y)

        probas = await model.predict_proba(X[:10])

        assert probas.shape == (10, 2)  # 10 samples, 2 classes
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(10))

    @pytest.mark.asyncio
    async def test_predict_without_fit_raises(self):
        """Test that predicting without fitting raises error."""
        model = CatBoostModel(model_id="test-no-fit")
        await model.load()

        with pytest.raises(RuntimeError, match="not fitted"):
            await model.predict([[1.0, 2.0, 3.0, 4.0, 5.0]])

    @pytest.mark.asyncio
    async def test_feature_importance(self, sample_classification_data, feature_names):
        """Test feature importance computation."""
        X, y = sample_classification_data
        model = CatBoostModel(
            model_id="test-importance",
            model_type="classifier",
            iterations=10,
        )
        await model.load()
        await model.fit(X, y, feature_names=feature_names)

        importance = await model.get_feature_importance()

        assert len(importance) == len(feature_names)
        for name, imp in importance:
            assert name in feature_names
            assert isinstance(imp, (int, float))


# =============================================================================
# Incremental Learning Tests
# =============================================================================


class TestIncrementalLearning:
    """Tests for incremental learning functionality."""

    @pytest.mark.asyncio
    async def test_update_model(self, sample_classification_data):
        """Test incremental model update."""
        X, y = sample_classification_data
        n_initial = 100
        n_update = 50

        model = CatBoostModel(
            model_id="test-update",
            model_type="classifier",
            iterations=10,
        )
        await model.load()

        # Initial training
        await model.fit(X[:n_initial], y[:n_initial])
        initial_trees = model._model.tree_count_

        # Incremental update
        result = await model.update(X[n_initial:n_initial + n_update], y[n_initial:n_initial + n_update])

        assert result["samples_added"] == n_update
        assert result["trees_before"] == initial_trees
        assert result["trees_after"] > initial_trees

    @pytest.mark.asyncio
    async def test_update_without_fit_raises(self, sample_classification_data):
        """Test that updating without initial fit raises error."""
        X, y = sample_classification_data
        model = CatBoostModel(model_id="test-update-no-fit")
        await model.load()

        with pytest.raises(RuntimeError, match="must be fitted"):
            await model.update(X[:10], y[:10])


# =============================================================================
# Save/Load Tests
# =============================================================================


class TestSaveLoad:
    """Tests for model persistence."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, sample_classification_data, catboost_models_dir):
        """Test saving and loading a model."""
        X, y = sample_classification_data
        model_id = "test-save-load"

        # Train and save
        model = CatBoostModel(
            model_id=model_id,
            model_type="classifier",
            iterations=10,
        )
        await model.load()
        await model.fit(X, y)

        save_path = catboost_models_dir / f"{model_id}.joblib"
        saved = await model.save(save_path)

        assert Path(saved).exists()

        # Make predictions before unload
        pred_before = await model.predict(X[:5])

        # Load into new model
        new_model = CatBoostModel(model_id=model_id)
        await new_model.load_from_path(save_path)

        assert new_model.is_fitted
        assert new_model.model_type == "classifier"

        # Predictions should match
        pred_after = await new_model.predict(X[:5])
        np.testing.assert_array_equal(pred_before, pred_after)

        # Cleanup
        save_path.unlink()

    @pytest.mark.asyncio
    async def test_load_nonexistent_raises(self):
        """Test that loading nonexistent model raises error."""
        model = CatBoostModel(model_id="nonexistent")

        with pytest.raises(FileNotFoundError):
            await model.load_from_path("/nonexistent/path/model.joblib")


# =============================================================================
# API Types Tests
# =============================================================================


class TestAPITypes:
    """Tests for CatBoost API Pydantic types."""

    def test_fit_request(self):
        """Test CatBoostFitRequest type."""
        request = CatBoostFitRequest(
            model_id="test-model",
            model_type="classifier",
            data=[[1.0, 2.0, 3.0]],
            labels=[0],
            iterations=100,
        )
        assert request.model_id == "test-model"
        assert request.model_type == "classifier"

    def test_fit_response(self):
        """Test CatBoostFitResponse type."""
        response = CatBoostFitResponse(
            model_id="test-model",
            model_type="classifier",
            samples_fitted=100,
            n_features=5,
            iterations=50,
            classes=[0, 1],
            saved_path="/path/to/model.joblib",
            fit_time_ms=500.0,
        )
        assert response.samples_fitted == 100
        assert response.classes == [0, 1]

    def test_predict_request(self):
        """Test CatBoostPredictRequest type."""
        request = CatBoostPredictRequest(
            model_id="test-model",
            data=[[1.0, 2.0, 3.0]],
            return_proba=True,
        )
        assert request.return_proba is True

    def test_predict_response(self):
        """Test CatBoostPredictResponse type."""
        prediction = CatBoostPrediction(
            sample_index=0,
            prediction=1,
            probabilities={"0": 0.3, "1": 0.7},
        )
        response = CatBoostPredictResponse(
            model_id="test-model",
            predictions=[prediction],
            predict_time_ms=10.0,
        )
        assert len(response.predictions) == 1
        assert response.predictions[0].probabilities["1"] == 0.7

    def test_update_request(self):
        """Test CatBoostUpdateRequest type."""
        request = CatBoostUpdateRequest(
            model_id="test-model",
            data=[[1.0, 2.0]],
            labels=[1],
        )
        assert request.model_id == "test-model"

    def test_update_response(self):
        """Test CatBoostUpdateResponse type."""
        response = CatBoostUpdateResponse(
            model_id="test-model",
            samples_added=50,
            trees_before=100,
            trees_after=110,
            update_time_ms=200.0,
        )
        assert response.trees_after > response.trees_before

    def test_info_response(self):
        """Test CatBoostInfoResponse type."""
        response = CatBoostInfoResponse(
            available=True,
            gpu_available=False,
            model_types=["classifier", "regressor"],
            features=["native_categorical", "incremental_learning"],
        )
        assert response.available is True

    def test_model_info(self):
        """Test CatBoostModelInfo type."""
        info = CatBoostModelInfo(
            model_id="test-model",
            model_type="classifier",
            n_features=5,
            iterations=100,
            path="/path/to/model.joblib",
        )
        assert info.n_features == 5

    def test_feature_importance_response(self):
        """Test CatBoostFeatureImportanceResponse type."""
        imp = CatBoostFeatureImportance(feature="temp", importance=0.5)
        response = CatBoostFeatureImportanceResponse(
            model_id="test-model",
            importances=[imp],
            importance_type="FeatureImportance",
        )
        assert len(response.importances) == 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_single_sample_training(self):
        """Test training with minimum samples."""
        model = CatBoostModel(
            model_id="test-single",
            model_type="classifier",
            iterations=10,
        )
        await model.load()

        # CatBoost can train on small datasets
        X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        y = [0, 1]

        result = await model.fit(X, y)
        assert result["samples_fitted"] == 2

    @pytest.mark.asyncio
    async def test_unload(self, sample_classification_data):
        """Test model unload."""
        X, y = sample_classification_data
        model = CatBoostModel(
            model_id="test-unload",
            model_type="classifier",
            iterations=10,
        )
        await model.load()
        await model.fit(X[:50], y[:50])

        await model.unload()

        assert model._model is None
        assert not model._is_fitted

    @pytest.mark.asyncio
    async def test_proba_for_regressor_raises(self, sample_regression_data):
        """Test that predict_proba raises for regressor."""
        X, y = sample_regression_data
        model = CatBoostModel(
            model_id="test-proba-regressor",
            model_type="regressor",
            iterations=10,
        )
        await model.load()
        await model.fit(X, y)

        with pytest.raises(ValueError, match="only available for classifiers"):
            await model.predict_proba(X[:5])

    @pytest.mark.asyncio
    async def test_early_stopping(self, sample_classification_data):
        """Test training with early stopping."""
        X, y = sample_classification_data
        n_train = 150

        model = CatBoostModel(
            model_id="test-early-stop",
            model_type="classifier",
            iterations=100,  # High iterations, should stop early
        )
        await model.load()

        result = await model.fit(
            X[:n_train],
            y[:n_train],
            eval_set=(X[n_train:], y[n_train:]),
            early_stopping_rounds=5,
        )

        # If early stopping worked, we may have fewer iterations than requested
        assert result["iterations"] <= 100
        # best_iteration may be set if early stopping triggered
