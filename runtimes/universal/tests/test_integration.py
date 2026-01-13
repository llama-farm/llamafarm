"""Integration Tests for all ML Features.

Phase 19 of Universal Runtime ML Enhancements.

Tests end-to-end workflows across all new ML capabilities.
"""

import numpy as np
import pytest


class TestAnomalyPipeline:
    """Test complete anomaly detection pipeline."""

    @pytest.mark.asyncio
    async def test_train_score_explain(self):
        """Test: train → score → explain workflow."""
        from models.anomaly_model import AnomalyModel
        from utils.anomaly_explainer import create_explainer_for_sklearn

        # 1. Create and train model
        model = AnomalyModel(
            model_id="integration-test",
            device="cpu",
            backend="isolation_forest",
            contamination=0.1,
        )
        await model.load()

        # Generate training data
        np.random.seed(42)
        normal_data = np.random.randn(100, 3) * 2 + [30, 50, 25]
        await model.fit(normal_data.tolist())

        # 2. Score some test data
        test_data = np.array([
            [30, 50, 25],  # Normal
            [95, 95, 200],  # Anomaly
        ])
        scores = await model.score(test_data.tolist())

        assert len(scores) == 2
        # Anomaly should have higher score
        assert scores[1].score > scores[0].score

        # 3. Get SHAP explanations
        sklearn_model = model.get_sklearn_model()
        training_data = model.get_training_data()

        assert sklearn_model is not None
        assert training_data is not None

        explainer = create_explainer_for_sklearn(
            model=sklearn_model,
            background_data=training_data,
            feature_names=["cpu", "memory", "latency"],
            n_background_samples=50,
        )

        explanations = explainer.explain(test_data, nsamples=20)
        assert len(explanations) == 2
        assert "features" in explanations[0]
        assert "top_contributors" in explanations[0]


class TestDataQualityPipeline:
    """Test dataset quality audit pipeline."""

    def test_audit_and_quality_scores(self):
        """Test: generate data → audit → get quality scores."""
        from utils.dataset_auditor import DatasetAuditor

        # Generate dataset with some label noise
        np.random.seed(42)
        n_samples = 50
        n_classes = 3

        true_labels = np.random.randint(0, n_classes, n_samples)

        # Create prediction probabilities (high confidence for true labels)
        pred_probs = np.zeros((n_samples, n_classes))
        for i, label in enumerate(true_labels):
            pred_probs[i, label] = 0.85 + np.random.uniform(0, 0.1)
            remaining = 1 - pred_probs[i, label]
            for j in range(n_classes):
                if j != label:
                    pred_probs[i, j] = remaining / (n_classes - 1)

        # Add label noise (flip 20% of labels)
        labels = true_labels.tolist()
        for i in range(0, n_samples, 5):  # Every 5th sample
            wrong = [l for l in range(n_classes) if l != true_labels[i]]
            labels[i] = np.random.choice(wrong)

        # 1. Run audit
        auditor = DatasetAuditor(label_names=["cat", "dog", "bird"])
        result = auditor.audit(labels, pred_probs)

        assert "label_issues" in result
        assert "summary" in result
        assert len(result["label_issues"]) > 0

        # 2. Get quality scores
        scores = auditor.get_label_quality_scores(labels, pred_probs)
        assert len(scores) == n_samples
        assert np.all((scores >= 0) & (scores <= 1))


class TestDriftDetection:
    """Test streaming drift detection."""

    def test_detect_drift_in_stream(self):
        """Test: stream data → detect drift."""
        from utils.drift_detector import DriftDetector

        # Create detector
        detector = DriftDetector(algorithm="adwin")

        # Stream normal data
        np.random.seed(42)
        normal_values = np.random.randn(50) * 0.5 + 5

        drifts_normal = []
        for val in normal_values:
            result = detector.update(float(val))
            drifts_normal.append(result["drift_detected"])

        # Stream shifted data (drift)
        shifted_values = np.random.randn(50) * 0.5 + 10  # Mean shifted by 5

        drifts_shifted = []
        for val in shifted_values:
            result = detector.update(float(val))
            drifts_shifted.append(result["drift_detected"])

        # Should detect drift in shifted data
        assert any(drifts_shifted), "Failed to detect drift after mean shift"


class TestChangePointDetection:
    """Test change point detection."""

    def test_detect_changepoints(self):
        """Test: generate data with breaks → detect changepoints."""
        from utils.changepoint_detector import ChangePointDetector

        # Generate data with clear change point at position 50
        np.random.seed(42)
        segment1 = np.random.randn(50) * 0.5 + 5
        segment2 = np.random.randn(50) * 0.5 + 15  # Different mean

        data = np.concatenate([segment1, segment2])

        # Detect change points
        detector = ChangePointDetector(algorithm="binseg")
        result = detector.detect(data.tolist(), n_changepoints=1)

        assert "change_points" in result
        assert len(result["change_points"]) == 1

        # Change point should be near position 50
        detected = result["change_points"][0]
        assert 45 <= detected <= 55, f"Expected ~50, got {detected}"


class TestPyODBackends:
    """Test all PyOD backends work correctly."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ["copod", "hbos", "ecod"])
    async def test_pyod_end_to_end(self, backend):
        """Test each PyOD backend can train and score."""
        from models.anomaly_model import AnomalyModel

        model = AnomalyModel(
            model_id=f"test-{backend}",
            device="cpu",
            backend=backend,
            contamination=0.1,
        )
        await model.load()

        # Train
        np.random.seed(42)
        data = np.random.randn(100, 5)
        result = await model.fit(data.tolist())
        assert result.samples_fitted == 100

        # Score
        test = np.array([[0, 0, 0, 0, 0], [10, 10, 10, 10, 10]])
        scores = await model.score(test.tolist())
        assert len(scores) == 2
        # Extreme point should have higher score
        assert scores[1].score > scores[0].score


class TestEndpointResponses:
    """Test that endpoints respond with correct structure."""

    @pytest.mark.asyncio
    async def test_anomaly_model_fit_response(self):
        """Test anomaly fit response structure."""
        from models.anomaly_model import AnomalyModel

        model = AnomalyModel(
            model_id="test-response",
            device="cpu",
            backend="isolation_forest",
        )
        await model.load()

        np.random.seed(42)
        data = np.random.randn(50, 3).tolist()
        result = await model.fit(data)

        # Check response structure
        assert hasattr(result, "samples_fitted")
        assert hasattr(result, "training_time_ms")
        assert hasattr(result, "model_params")

        assert result.samples_fitted == 50
        assert result.training_time_ms > 0
        assert "backend" in result.model_params

    @pytest.mark.asyncio
    async def test_anomaly_model_score_response(self):
        """Test anomaly score response structure."""
        from models.anomaly_model import AnomalyModel

        model = AnomalyModel(
            model_id="test-score-response",
            device="cpu",
            backend="isolation_forest",
        )
        await model.load()

        np.random.seed(42)
        data = np.random.randn(50, 3).tolist()
        await model.fit(data)

        scores = await model.score([[0, 0, 0], [10, 10, 10]])

        # Check score structure
        assert len(scores) == 2
        for score in scores:
            assert hasattr(score, "index")
            assert hasattr(score, "score")
            assert hasattr(score, "is_anomaly")
            assert hasattr(score, "raw_score")


class TestMemoryBounds:
    """Test memory usage stays within bounds."""

    @pytest.mark.asyncio
    async def test_training_data_limit(self):
        """Test that training data is limited for SHAP."""
        from models.anomaly_model import AnomalyModel

        model = AnomalyModel(
            model_id="test-memory",
            device="cpu",
            backend="isolation_forest",
        )
        await model.load()

        # Train with large dataset
        np.random.seed(42)
        large_data = np.random.randn(5000, 10).tolist()
        await model.fit(large_data)

        # Training data should be limited
        training_data = model.get_training_data()
        assert training_data is not None
        assert len(training_data) <= 1000, "Training data should be capped at 1000"


class TestConcurrentOperations:
    """Test that multiple operations can run."""

    @pytest.mark.asyncio
    async def test_multiple_models(self):
        """Test creating and using multiple models."""
        from models.anomaly_model import AnomalyModel

        models = []
        for backend in ["isolation_forest", "copod", "hbos"]:
            model = AnomalyModel(
                model_id=f"concurrent-{backend}",
                device="cpu",
                backend=backend,
            )
            await model.load()
            models.append(model)

        # Train all models
        np.random.seed(42)
        data = np.random.randn(100, 5).tolist()

        for model in models:
            await model.fit(data)

        # Score with all models
        test_data = [[0, 0, 0, 0, 0]]
        scores = []
        for model in models:
            s = await model.score(test_data)
            scores.append(s[0].score)

        # All should produce valid scores
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)
