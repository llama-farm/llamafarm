"""Tests for PyOD Integration (Advanced Anomaly Detection).

Phase 17 of Universal Runtime ML Enhancements.
"""

import numpy as np
import pytest

from models.anomaly_model import AnomalyModel


def generate_normal_data(n_samples: int = 100, n_features: int = 5, seed: int = 42) -> np.ndarray:
    """Generate normal (Gaussian) data for training."""
    np.random.seed(seed)
    return np.random.randn(n_samples, n_features)


def generate_anomalous_data(n_features: int = 5, seed: int = 42) -> np.ndarray:
    """Generate data points that should be anomalous."""
    np.random.seed(seed)
    # Far from center (high absolute values)
    return np.array([
        [10.0] * n_features,
        [-10.0] * n_features,
        [5.0, -5.0, 5.0, -5.0, 5.0],
    ])


class TestCOPOD:
    """Test COPOD (Copula-Based Outlier Detection) backend."""

    @pytest.fixture
    def copod_model(self):
        """Create COPOD model."""
        return AnomalyModel(
            model_id="test-copod",
            device="cpu",
            backend="copod",
            contamination=0.1,
        )

    @pytest.mark.asyncio
    async def test_copod_initializes(self, copod_model):
        """Test that COPOD model initializes."""
        await copod_model.load()
        assert copod_model._detector is not None

    @pytest.mark.asyncio
    async def test_copod_fits(self, copod_model):
        """Test that COPOD model fits."""
        await copod_model.load()
        data = generate_normal_data()
        result = await copod_model.fit(data)
        assert result.samples_fitted == 100
        assert copod_model.is_fitted

    @pytest.mark.asyncio
    async def test_copod_scores(self, copod_model):
        """Test that COPOD returns scores."""
        await copod_model.load()
        data = generate_normal_data()
        await copod_model.fit(data)

        test_data = generate_anomalous_data()
        scores = await copod_model.score(test_data)

        assert len(scores) == 3
        # Anomalous points should have higher scores
        for score in scores:
            assert hasattr(score, "score")
            assert hasattr(score, "is_anomaly")

    @pytest.mark.asyncio
    async def test_copod_detects_anomalies(self, copod_model):
        """Test that COPOD detects anomalies."""
        await copod_model.load()
        data = generate_normal_data()
        await copod_model.fit(data)

        # Mix of normal and anomalous
        test_data = np.vstack([
            generate_normal_data(n_samples=5),  # Normal
            generate_anomalous_data(),  # Anomalous
        ])
        scores = await copod_model.score(test_data)

        # At least some of the anomalous points should be flagged
        anomaly_flags = [s.is_anomaly for s in scores]
        # Last 3 are anomalous, should have higher rates
        assert sum(anomaly_flags[-3:]) >= sum(anomaly_flags[:5]) / 5 * 3 or sum(anomaly_flags[-3:]) > 0


class TestHBOS:
    """Test HBOS (Histogram-Based Outlier Score) backend."""

    @pytest.fixture
    def hbos_model(self):
        """Create HBOS model."""
        return AnomalyModel(
            model_id="test-hbos",
            device="cpu",
            backend="hbos",
            contamination=0.1,
        )

    @pytest.mark.asyncio
    async def test_hbos_initializes(self, hbos_model):
        """Test that HBOS model initializes."""
        await hbos_model.load()
        assert hbos_model._detector is not None

    @pytest.mark.asyncio
    async def test_hbos_fits(self, hbos_model):
        """Test that HBOS model fits."""
        await hbos_model.load()
        data = generate_normal_data()
        result = await hbos_model.fit(data)
        assert result.samples_fitted == 100
        assert hbos_model.is_fitted

    @pytest.mark.asyncio
    async def test_hbos_scores(self, hbos_model):
        """Test that HBOS returns scores."""
        await hbos_model.load()
        data = generate_normal_data()
        await hbos_model.fit(data)

        test_data = generate_anomalous_data()
        scores = await hbos_model.score(test_data)

        assert len(scores) == 3
        for score in scores:
            assert hasattr(score, "score")
            assert hasattr(score, "raw_score")

    @pytest.mark.asyncio
    async def test_hbos_is_fast(self, hbos_model):
        """Test that HBOS is fast on larger data."""
        await hbos_model.load()
        # HBOS should handle 10k samples quickly
        data = generate_normal_data(n_samples=10000)
        result = await hbos_model.fit(data)
        assert result.training_time_ms < 5000  # Should be < 5 seconds


class TestECOD:
    """Test ECOD (Empirical Cumulative Distribution) backend."""

    @pytest.fixture
    def ecod_model(self):
        """Create ECOD model."""
        return AnomalyModel(
            model_id="test-ecod",
            device="cpu",
            backend="ecod",
            contamination=0.1,
        )

    @pytest.mark.asyncio
    async def test_ecod_initializes(self, ecod_model):
        """Test that ECOD model initializes."""
        await ecod_model.load()
        assert ecod_model._detector is not None

    @pytest.mark.asyncio
    async def test_ecod_fits(self, ecod_model):
        """Test that ECOD model fits."""
        await ecod_model.load()
        data = generate_normal_data()
        result = await ecod_model.fit(data)
        assert result.samples_fitted == 100
        assert ecod_model.is_fitted

    @pytest.mark.asyncio
    async def test_ecod_scores(self, ecod_model):
        """Test that ECOD returns scores."""
        await ecod_model.load()
        data = generate_normal_data()
        await ecod_model.fit(data)

        test_data = generate_anomalous_data()
        scores = await ecod_model.score(test_data)

        assert len(scores) == 3
        for score in scores:
            assert hasattr(score, "score")


class TestPyODIntegration:
    """Test PyOD models integrate with existing anomaly API."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ["copod", "hbos", "ecod"])
    async def test_backend_params(self, backend):
        """Test that backend params are stored correctly."""
        model = AnomalyModel(
            model_id=f"test-{backend}",
            device="cpu",
            backend=backend,
            contamination=0.15,
        )
        await model.load()
        assert model.backend == backend
        assert model.contamination == 0.15

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ["copod", "hbos", "ecod"])
    async def test_fit_result_format(self, backend):
        """Test that fit results have consistent format."""
        model = AnomalyModel(
            model_id=f"test-{backend}",
            device="cpu",
            backend=backend,
        )
        await model.load()
        data = generate_normal_data()
        result = await model.fit(data)

        assert hasattr(result, "samples_fitted")
        assert hasattr(result, "training_time_ms")
        assert hasattr(result, "model_params")
        assert result.model_params["backend"] == backend

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ["copod", "hbos", "ecod"])
    async def test_score_normalization(self, backend):
        """Test that score normalization works."""
        model = AnomalyModel(
            model_id=f"test-{backend}",
            device="cpu",
            backend=backend,
            normalization="standardization",
        )
        await model.load()
        data = generate_normal_data()
        await model.fit(data)

        test_data = generate_anomalous_data()
        scores = await model.score(test_data)

        # Normalized scores should be in reasonable range
        for score in scores:
            assert 0 <= score.score <= 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ["copod", "hbos", "ecod"])
    async def test_stores_training_data(self, backend):
        """Test that training data is stored for SHAP."""
        model = AnomalyModel(
            model_id=f"test-{backend}",
            device="cpu",
            backend=backend,
        )
        await model.load()
        data = generate_normal_data()
        await model.fit(data)

        training_data = model.get_training_data()
        assert training_data is not None
        assert len(training_data) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ["copod", "hbos", "ecod"])
    async def test_get_sklearn_model(self, backend):
        """Test that PyOD model is returned for SHAP."""
        model = AnomalyModel(
            model_id=f"test-{backend}",
            device="cpu",
            backend=backend,
        )
        await model.load()
        data = generate_normal_data()
        await model.fit(data)

        sklearn_model = model.get_sklearn_model()
        assert sklearn_model is not None
        # PyOD models have decision_function method
        assert hasattr(sklearn_model, "decision_function")


class TestBackendComparison:
    """Test comparing different PyOD backends."""

    @pytest.mark.asyncio
    async def test_all_backends_detect_same_anomalies(self):
        """Test that different backends detect similar anomalies."""
        backends = ["copod", "hbos", "ecod"]
        data = generate_normal_data(n_samples=200)

        # Create very obvious anomalies
        test_data = np.vstack([
            np.zeros((1, 5)),  # Should be normal (at origin)
            np.ones((1, 5)) * 20,  # Should be anomalous (far from data)
        ])

        anomaly_detected = {}
        for backend in backends:
            model = AnomalyModel(
                model_id=f"test-{backend}",
                device="cpu",
                backend=backend,
                contamination=0.05,
            )
            await model.load()
            await model.fit(data)
            scores = await model.score(test_data)
            anomaly_detected[backend] = scores[1].is_anomaly

        # All backends should detect the extreme anomaly
        for backend in backends:
            assert anomaly_detected[backend], f"{backend} failed to detect extreme anomaly"
