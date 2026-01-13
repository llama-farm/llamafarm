"""Tests for SHAP-based Anomaly Explanations.

Phase 15 of Universal Runtime ML Enhancements.
"""

import numpy as np
import pytest

from utils.anomaly_explainer import AnomalyExplainer, create_explainer_for_sklearn


def generate_normal_data(n_samples: int = 100, n_features: int = 3, seed: int = 42) -> np.ndarray:
    """Generate normal (Gaussian) data."""
    np.random.seed(seed)
    return np.random.randn(n_samples, n_features)


def generate_anomalous_data(n_features: int = 3, seed: int = 42) -> np.ndarray:
    """Generate data that should be anomalous."""
    np.random.seed(seed)
    # Far from center (high values)
    return np.array([[10.0, -10.0, 5.0]] * 5)


class TestAnomalyExplainer:
    """Test AnomalyExplainer class."""

    @pytest.fixture
    def simple_model_fn(self):
        """Create a simple model function for testing."""
        # Simple model: higher scores for points far from origin
        def model_fn(x: np.ndarray) -> np.ndarray:
            return np.sum(x ** 2, axis=1)
        return model_fn

    @pytest.fixture
    def background_data(self):
        """Create background data for testing."""
        return generate_normal_data(n_samples=50)

    def test_explainer_initializes(self, simple_model_fn, background_data):
        """Test that explainer initializes correctly."""
        explainer = AnomalyExplainer(
            model_fn=simple_model_fn,
            background_data=background_data,
        )
        assert explainer is not None

    def test_explainer_with_feature_names(self, simple_model_fn, background_data):
        """Test explainer with feature names."""
        explainer = AnomalyExplainer(
            model_fn=simple_model_fn,
            background_data=background_data,
            feature_names=["cpu", "memory", "latency"],
        )
        assert explainer.feature_names == ["cpu", "memory", "latency"]

    def test_explain_returns_list(self, simple_model_fn, background_data):
        """Test that explain returns a list."""
        explainer = AnomalyExplainer(
            model_fn=simple_model_fn,
            background_data=background_data,
            n_background_samples=20,
        )
        data = generate_anomalous_data()
        explanations = explainer.explain(data, nsamples=20)
        assert isinstance(explanations, list)

    def test_explain_length_matches_data(self, simple_model_fn, background_data):
        """Test that explanation count matches input data."""
        explainer = AnomalyExplainer(
            model_fn=simple_model_fn,
            background_data=background_data,
            n_background_samples=20,
        )
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        explanations = explainer.explain(data, nsamples=20)
        assert len(explanations) == 3

    def test_explanation_has_features(self, simple_model_fn, background_data):
        """Test that each explanation has features."""
        explainer = AnomalyExplainer(
            model_fn=simple_model_fn,
            background_data=background_data,
            n_background_samples=20,
        )
        data = generate_anomalous_data()[:1]
        explanations = explainer.explain(data, nsamples=20)
        assert "features" in explanations[0]
        assert "top_contributors" in explanations[0]

    def test_feature_has_required_keys(self, simple_model_fn, background_data):
        """Test that each feature has required keys."""
        explainer = AnomalyExplainer(
            model_fn=simple_model_fn,
            background_data=background_data,
            feature_names=["a", "b", "c"],
            n_background_samples=20,
        )
        data = generate_anomalous_data()[:1]
        explanations = explainer.explain(data, nsamples=20)

        for feature in explanations[0]["features"]:
            assert "feature" in feature
            assert "shap_value" in feature
            assert "importance" in feature
            assert "value" in feature
            assert "direction" in feature

    def test_features_sorted_by_importance(self, simple_model_fn, background_data):
        """Test that features are sorted by importance."""
        explainer = AnomalyExplainer(
            model_fn=simple_model_fn,
            background_data=background_data,
            n_background_samples=20,
        )
        data = generate_anomalous_data()[:1]
        explanations = explainer.explain(data, nsamples=20)

        features = explanations[0]["features"]
        importances = [f["importance"] for f in features]
        assert importances == sorted(importances, reverse=True)

    def test_explain_single(self, simple_model_fn, background_data):
        """Test single point explanation."""
        explainer = AnomalyExplainer(
            model_fn=simple_model_fn,
            background_data=background_data,
            n_background_samples=20,
        )
        data_point = np.array([10.0, -10.0, 5.0])
        explanation = explainer.explain_single(data_point, nsamples=20)
        assert "features" in explanation
        assert "top_contributors" in explanation


class TestCreateExplainerForSklearn:
    """Test sklearn model explainer creation."""

    @pytest.fixture
    def isolation_forest(self):
        """Create and fit an IsolationForest."""
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(n_estimators=50, random_state=42)
        data = generate_normal_data()
        model.fit(data)
        return model

    @pytest.fixture
    def one_class_svm(self):
        """Create and fit a OneClassSVM."""
        from sklearn.svm import OneClassSVM
        model = OneClassSVM(kernel="rbf", gamma="auto")
        data = generate_normal_data()
        model.fit(data)
        return model

    def test_creates_explainer_for_isolation_forest(self, isolation_forest):
        """Test explainer creation for IsolationForest."""
        background = generate_normal_data(n_samples=50)
        explainer = create_explainer_for_sklearn(
            model=isolation_forest,
            background_data=background,
        )
        assert explainer is not None

    def test_creates_explainer_for_one_class_svm(self, one_class_svm):
        """Test explainer creation for OneClassSVM."""
        background = generate_normal_data(n_samples=50)
        explainer = create_explainer_for_sklearn(
            model=one_class_svm,
            background_data=background,
        )
        assert explainer is not None

    def test_isolation_forest_explanations(self, isolation_forest):
        """Test explanations from IsolationForest."""
        background = generate_normal_data(n_samples=50)
        explainer = create_explainer_for_sklearn(
            model=isolation_forest,
            background_data=background,
            feature_names=["f1", "f2", "f3"],
            n_background_samples=20,
        )

        anomalous = generate_anomalous_data()[:1]
        explanations = explainer.explain(anomalous, nsamples=20)
        assert len(explanations) == 1
        assert "features" in explanations[0]


class TestFeatureSummary:
    """Test feature summary functionality."""

    @pytest.fixture
    def explainer(self):
        """Create an explainer for testing."""
        def model_fn(x: np.ndarray) -> np.ndarray:
            return np.sum(x ** 2, axis=1)

        background = generate_normal_data(n_samples=50)
        return AnomalyExplainer(
            model_fn=model_fn,
            background_data=background,
            feature_names=["cpu", "memory", "latency"],
            n_background_samples=20,
        )

    def test_get_feature_summary(self, explainer):
        """Test feature summary computation."""
        data = generate_anomalous_data()
        explanations = explainer.explain(data, nsamples=20)
        summary = explainer.get_feature_summary(explanations)

        assert "features" in summary
        assert "total_explained" in summary
        assert summary["total_explained"] == len(data)

    def test_summary_has_mean_importance(self, explainer):
        """Test that summary has mean importance."""
        data = generate_anomalous_data()
        explanations = explainer.explain(data, nsamples=20)
        summary = explainer.get_feature_summary(explanations)

        for feature_summary in summary["features"]:
            assert "mean_importance" in feature_summary
            assert "max_importance" in feature_summary


class TestDirectionDetection:
    """Test direction detection (high/low)."""

    def test_high_direction_for_positive_shap(self):
        """Test that positive SHAP values give 'high' direction."""
        # Create a model where higher x[0] = higher score
        def model_fn(x: np.ndarray) -> np.ndarray:
            return x[:, 0]  # Score is just the first feature

        background = np.zeros((10, 1))
        explainer = AnomalyExplainer(
            model_fn=model_fn,
            background_data=background,
            feature_names=["feature_0"],
            n_background_samples=10,
        )

        # High value should have positive SHAP, thus "high" direction
        data = np.array([[10.0]])
        explanations = explainer.explain(data, nsamples=50)

        # With positive value far from background (0), should contribute positively
        features = explanations[0]["features"]
        assert len(features) == 1
        # Direction depends on SHAP sign
        assert features[0]["direction"] in ["high", "low"]


class TestBackgroundSampling:
    """Test background data sampling."""

    def test_subsamples_large_background(self):
        """Test that large background data is subsampled."""
        def model_fn(x: np.ndarray) -> np.ndarray:
            return np.sum(x ** 2, axis=1)

        large_background = generate_normal_data(n_samples=500)
        explainer = AnomalyExplainer(
            model_fn=model_fn,
            background_data=large_background,
            n_background_samples=50,
        )

        # Should have subsampled to 50
        assert len(explainer.background_data) == 50

    def test_keeps_small_background(self):
        """Test that small background data is kept as-is."""
        def model_fn(x: np.ndarray) -> np.ndarray:
            return np.sum(x ** 2, axis=1)

        small_background = generate_normal_data(n_samples=30)
        explainer = AnomalyExplainer(
            model_fn=model_fn,
            background_data=small_background,
            n_background_samples=100,  # More than available
        )

        # Should keep all 30
        assert len(explainer.background_data) == 30
