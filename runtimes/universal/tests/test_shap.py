"""Tests for SHAP explainability module.

Tests SHAP explainer functionality including:
- Explainer type detection
- SHAP value computation
- Natural language narrative generation
- Feature importance
- API types
"""

import numpy as np
import pytest

pytest.importorskip("shap", reason="shap addon not installed")

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from api_types.explain import (
    ExplainerInfo,
    ExplainersResponse,
    FeatureContribution,
    FeatureImportance,
    FeatureImportanceRequest,
    FeatureImportanceResponse,
    NarrativeExplanation,
    SHAPExplainRequest,
    SHAPExplainResponse,
    SHAPExplanation,
)
from models.shap_explainer import (
    FeatureContribution as ExplainerFeatureContribution,
)
from models.shap_explainer import (
    NarrativeExplanation as ExplainerNarrativeExplanation,
)
from models.shap_explainer import (
    SHAPExplainer,
    auto_detect_explainer_type,
    get_explainer_types,
)
from models.shap_explainer import (
    SHAPExplanation as ExplainerSHAPExplanation,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    return X


@pytest.fixture
def trained_iforest(sample_data):
    """Create and fit an Isolation Forest model."""
    model = IsolationForest(n_estimators=10, random_state=42)
    model.fit(sample_data)
    return model


@pytest.fixture
def trained_logistic(sample_data):
    """Create and fit a Logistic Regression model."""
    # Create binary labels
    y = (sample_data[:, 0] > 0).astype(int)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(sample_data, y)
    return model


@pytest.fixture
def feature_names():
    """Feature names for testing."""
    return ["temp", "humidity", "pressure", "wind_speed", "precipitation"]


# =============================================================================
# Explainer Type Detection Tests
# =============================================================================


class TestExplainerTypeDetection:
    """Tests for auto_detect_explainer_type function."""

    def test_tree_detection_isolation_forest(self, trained_iforest):
        """Test tree type detection for Isolation Forest."""
        detected = auto_detect_explainer_type(trained_iforest)
        assert detected == "tree"

    def test_linear_detection_logistic(self, trained_logistic):
        """Test linear type detection for Logistic Regression."""
        detected = auto_detect_explainer_type(trained_logistic)
        assert detected == "linear"

    def test_kernel_fallback(self):
        """Test kernel fallback for unknown model types."""

        class UnknownModel:
            pass

        model = UnknownModel()
        detected = auto_detect_explainer_type(model)
        assert detected == "kernel"


class TestExplainerTypes:
    """Tests for get_explainer_types function."""

    def test_returns_list(self):
        """Test that get_explainer_types returns a list."""
        types = get_explainer_types()
        assert isinstance(types, list)
        assert len(types) > 0

    def test_contains_tree_linear_kernel(self):
        """Test that all three explainer types are present."""
        types = get_explainer_types()
        names = {t["name"] for t in types}
        assert "tree" in names
        assert "linear" in names
        assert "kernel" in names

    def test_type_structure(self):
        """Test that each type has required fields."""
        types = get_explainer_types()
        for t in types:
            assert "name" in t
            assert "description" in t
            assert "supported_models" in t
            assert isinstance(t["supported_models"], list)


# =============================================================================
# SHAP Explainer Tests
# =============================================================================


class TestSHAPExplainer:
    """Tests for SHAPExplainer class."""

    @pytest.mark.asyncio
    async def test_init_with_auto_detect(self, trained_iforest, feature_names):
        """Test explainer initialization with auto-detection."""
        explainer = SHAPExplainer(
            model=trained_iforest,
            feature_names=feature_names,
        )
        assert explainer.explainer_type == "tree"

    @pytest.mark.asyncio
    async def test_init_with_explicit_type(self, trained_iforest, feature_names, sample_data):
        """Test explainer initialization with explicit type."""
        explainer = SHAPExplainer(
            model=trained_iforest,
            explainer_type="kernel",
            feature_names=feature_names,
            background_data=sample_data[:10],
        )
        assert explainer.explainer_type == "kernel"

    @pytest.mark.asyncio
    async def test_init_invalid_type(self, trained_iforest):
        """Test that invalid explainer type raises error."""
        with pytest.raises(ValueError, match="Unknown explainer type"):
            SHAPExplainer(
                model=trained_iforest,
                explainer_type="invalid_type",
            )

    @pytest.mark.asyncio
    async def test_tree_explainer_load(self, trained_iforest, feature_names):
        """Test loading tree explainer."""
        explainer = SHAPExplainer(
            model=trained_iforest,
            feature_names=feature_names,
        )
        await explainer.load()
        assert explainer._explainer is not None

    @pytest.mark.asyncio
    async def test_tree_explainer_explain(self, trained_iforest, sample_data, feature_names):
        """Test SHAP explanation with tree explainer."""
        explainer = SHAPExplainer(
            model=trained_iforest,
            feature_names=feature_names,
        )
        await explainer.load()

        # Explain a few samples
        test_data = sample_data[:5]
        explanations = await explainer.explain(test_data, top_k=3)

        assert len(explanations) == 5
        for exp in explanations:
            assert isinstance(exp, ExplainerSHAPExplanation)
            assert exp.sample_index >= 0
            assert isinstance(exp.base_value, float)
            assert isinstance(exp.prediction, float)
            assert len(exp.contributions) == 3
            for contrib in exp.contributions:
                assert isinstance(contrib, ExplainerFeatureContribution)
                assert contrib.feature in feature_names
                assert contrib.direction in ["increases", "decreases"]

        await explainer.unload()

    @pytest.mark.asyncio
    async def test_feature_importance(self, trained_iforest, sample_data, feature_names):
        """Test feature importance computation."""
        explainer = SHAPExplainer(
            model=trained_iforest,
            feature_names=feature_names,
        )
        await explainer.load()

        importance = await explainer.get_feature_importance(sample_data)

        assert len(importance) == len(feature_names)
        for name, imp in importance:
            assert name in feature_names
            assert isinstance(imp, float)
            assert imp >= 0  # Importance is always non-negative

        await explainer.unload()

    @pytest.mark.asyncio
    async def test_generate_narrative(self, trained_iforest, sample_data, feature_names):
        """Test narrative generation."""
        explainer = SHAPExplainer(
            model=trained_iforest,
            feature_names=feature_names,
        )
        await explainer.load()

        explanations = await explainer.explain(sample_data[:1], top_k=3)
        narrative = await explainer.generate_narrative(explanations[0])

        assert isinstance(narrative, ExplainerNarrativeExplanation)
        assert isinstance(narrative.summary, str)
        assert len(narrative.summary) > 0
        assert isinstance(narrative.details, list)

        await explainer.unload()

    @pytest.mark.asyncio
    async def test_explain_without_load_raises(self, trained_iforest, sample_data):
        """Test that explaining without loading raises error."""
        explainer = SHAPExplainer(model=trained_iforest)

        with pytest.raises(RuntimeError, match="not loaded"):
            await explainer.explain(sample_data[:5])

    @pytest.mark.asyncio
    async def test_default_feature_names(self, trained_iforest, sample_data):
        """Test that default feature names are generated."""
        explainer = SHAPExplainer(model=trained_iforest)
        await explainer.load()

        explanations = await explainer.explain(sample_data[:1], top_k=3)

        for contrib in explanations[0].contributions:
            assert contrib.feature.startswith("feature_")

        await explainer.unload()


# =============================================================================
# Kernel Explainer Tests
# =============================================================================


class TestKernelExplainer:
    """Tests for Kernel SHAP explainer."""

    @pytest.mark.asyncio
    async def test_kernel_requires_background_data(self, trained_iforest):
        """Test that kernel explainer requires background data."""
        explainer = SHAPExplainer(
            model=trained_iforest,
            explainer_type="kernel",
        )

        with pytest.raises(ValueError, match="Background data required"):
            await explainer.load()

    @pytest.mark.asyncio
    async def test_kernel_explainer_works(self, trained_logistic, sample_data, feature_names):
        """Test kernel explainer with logistic regression."""
        explainer = SHAPExplainer(
            model=trained_logistic,
            explainer_type="kernel",
            feature_names=feature_names,
            background_data=sample_data[:20],
        )
        await explainer.load()

        explanations = await explainer.explain(sample_data[:2], top_k=3)

        assert len(explanations) == 2
        for exp in explanations:
            assert len(exp.contributions) == 3

        await explainer.unload()


# =============================================================================
# API Types Tests
# =============================================================================


class TestAPITypes:
    """Tests for SHAP API Pydantic types."""

    def test_feature_contribution(self):
        """Test FeatureContribution type."""
        contrib = FeatureContribution(
            feature="temp",
            value=25.5,
            shap_value=0.123,
            direction="increases",
        )
        assert contrib.feature == "temp"
        assert contrib.value == 25.5
        assert contrib.shap_value == 0.123
        assert contrib.direction == "increases"

    def test_shap_explanation(self):
        """Test SHAPExplanation type."""
        contrib = FeatureContribution(
            feature="temp",
            value=25.5,
            shap_value=0.123,
            direction="increases",
        )
        exp = SHAPExplanation(
            sample_index=0,
            base_value=-0.5,
            prediction=-0.1,
            contributions=[contrib],
        )
        assert exp.sample_index == 0
        assert exp.base_value == -0.5
        assert exp.prediction == -0.1
        assert len(exp.contributions) == 1

    def test_narrative_explanation(self):
        """Test NarrativeExplanation type."""
        narrative = NarrativeExplanation(
            summary="This is a summary.",
            details=["Detail 1", "Detail 2"],
        )
        assert narrative.summary == "This is a summary."
        assert len(narrative.details) == 2

    def test_shap_explain_request(self):
        """Test SHAPExplainRequest type."""
        request = SHAPExplainRequest(
            model_type="anomaly",
            model_id="test-model",
            data=[[1.0, 2.0, 3.0]],
            feature_names=["a", "b", "c"],
            top_k=5,
            generate_narrative=True,
        )
        assert request.model_type == "anomaly"
        assert request.model_id == "test-model"
        assert len(request.data) == 1
        assert request.top_k == 5

    def test_shap_explain_response(self):
        """Test SHAPExplainResponse type."""
        contrib = FeatureContribution(
            feature="temp",
            value=25.5,
            shap_value=0.123,
            direction="increases",
        )
        exp = SHAPExplanation(
            sample_index=0,
            base_value=-0.5,
            prediction=-0.1,
            contributions=[contrib],
        )
        response = SHAPExplainResponse(
            model_type="anomaly",
            model_id="test-model",
            explainer_type="tree",
            explanations=[exp],
            explain_time_ms=50.0,
        )
        assert response.model_type == "anomaly"
        assert response.explainer_type == "tree"
        assert len(response.explanations) == 1

    def test_feature_importance_request(self):
        """Test FeatureImportanceRequest type."""
        request = FeatureImportanceRequest(
            model_type="anomaly",
            model_id="test-model",
            data=[[1.0, 2.0], [3.0, 4.0]],
            feature_names=["a", "b"],
        )
        assert request.model_type == "anomaly"
        assert len(request.data) == 2

    def test_feature_importance_response(self):
        """Test FeatureImportanceResponse type."""
        imp = FeatureImportance(feature="temp", importance=0.5)
        response = FeatureImportanceResponse(
            model_type="anomaly",
            model_id="test-model",
            importances=[imp],
            compute_time_ms=25.0,
        )
        assert response.model_type == "anomaly"
        assert len(response.importances) == 1

    def test_explainer_info(self):
        """Test ExplainerInfo type."""
        info = ExplainerInfo(
            name="tree",
            description="Fast explainer for tree-based models",
            supported_models=["isolation_forest", "catboost"],
        )
        assert info.name == "tree"
        assert len(info.supported_models) == 2

    def test_explainers_response(self):
        """Test ExplainersResponse type."""
        info = ExplainerInfo(
            name="tree",
            description="Fast explainer for tree-based models",
            supported_models=["isolation_forest"],
        )
        response = ExplainersResponse(explainers=[info])
        assert len(response.explainers) == 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_single_sample_explain(self, trained_iforest, sample_data):
        """Test explaining a single sample."""
        explainer = SHAPExplainer(model=trained_iforest)
        await explainer.load()

        explanations = await explainer.explain(sample_data[:1], top_k=3)

        assert len(explanations) == 1
        assert explanations[0].sample_index == 0

        await explainer.unload()

    @pytest.mark.asyncio
    async def test_1d_array_input(self, trained_iforest, sample_data):
        """Test explaining with 1D array (single sample)."""
        explainer = SHAPExplainer(model=trained_iforest)
        await explainer.load()

        # Pass single sample as 1D array
        single_sample = sample_data[0]
        explanations = await explainer.explain(single_sample, top_k=3)

        assert len(explanations) == 1

        await explainer.unload()

    @pytest.mark.asyncio
    async def test_top_k_larger_than_features(self, trained_iforest, sample_data, feature_names):
        """Test top_k larger than number of features."""
        explainer = SHAPExplainer(
            model=trained_iforest,
            feature_names=feature_names,
        )
        await explainer.load()

        # Request more top features than available
        explanations = await explainer.explain(sample_data[:1], top_k=100)

        # Should return at most n_features contributions
        assert len(explanations[0].contributions) <= len(feature_names)

        await explainer.unload()

    @pytest.mark.asyncio
    async def test_unload_multiple_times(self, trained_iforest):
        """Test that unload can be called multiple times safely."""
        explainer = SHAPExplainer(model=trained_iforest)
        await explainer.load()
        await explainer.unload()
        await explainer.unload()  # Should not raise

    @pytest.mark.asyncio
    async def test_narrative_with_context(self, trained_iforest, sample_data, feature_names):
        """Test narrative generation with context."""
        explainer = SHAPExplainer(
            model=trained_iforest,
            feature_names=feature_names,
        )
        await explainer.load()

        explanations = await explainer.explain(sample_data[:1], top_k=3)

        # Provide context with feature statistics
        context = {
            "temp": {"mean": 0.0, "std": 1.0},
            "humidity": {"mean": 0.0, "std": 1.0},
        }

        narrative = await explainer.generate_narrative(explanations[0], context=context)

        assert isinstance(narrative.summary, str)
        assert len(narrative.details) > 0

        await explainer.unload()
