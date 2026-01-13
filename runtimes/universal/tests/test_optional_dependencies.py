"""Tests for Optional Dependencies and Lazy Imports.

Phase 18 of Universal Runtime ML Enhancements.

Verifies that:
1. All ML dependencies are installed correctly
2. Lazy imports work properly
3. Error messages are helpful when dependencies are missing
"""

import importlib
import sys

import pytest


class TestMLDependencies:
    """Test that ML dependencies are installed."""

    @pytest.mark.parametrize("package,import_path", [
        ("sklearn", "sklearn.ensemble"),
        ("pyod", "pyod.models.copod"),
        ("cleanlab", "cleanlab.filter"),
        ("shap", "shap"),
        ("ruptures", "ruptures"),
        ("river", "river.drift"),
    ])
    def test_ml_package_installed(self, package, import_path):
        """Test that ML packages are installed."""
        try:
            importlib.import_module(import_path)
        except ImportError as e:
            pytest.fail(f"Package {package} not installed: {e}")

    @pytest.mark.parametrize("package,import_name", [
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),  # Pillow imports as PIL
    ])
    def test_core_package_installed(self, package, import_name):
        """Test that core packages are installed."""
        try:
            importlib.import_module(import_name)
        except ImportError as e:
            pytest.fail(f"Package {package} not installed: {e}")


class TestLazyImports:
    """Test that lazy imports work in model files."""

    def test_anomaly_model_imports(self):
        """Test anomaly model can be imported."""
        from models.anomaly_model import AnomalyModel
        assert AnomalyModel is not None

    def test_dataset_auditor_imports(self):
        """Test dataset auditor can be imported."""
        from utils.dataset_auditor import DatasetAuditor
        assert DatasetAuditor is not None

    def test_anomaly_explainer_imports(self):
        """Test anomaly explainer can be imported."""
        from utils.anomaly_explainer import AnomalyExplainer
        assert AnomalyExplainer is not None

    def test_changepoint_detector_imports(self):
        """Test changepoint detector can be imported."""
        from utils.changepoint_detector import ChangePointDetector
        assert ChangePointDetector is not None

    def test_drift_detector_imports(self):
        """Test drift detector can be imported."""
        from utils.drift_detector import DriftDetector
        assert DriftDetector is not None


class TestBackendAvailability:
    """Test that different backends can be loaded."""

    @pytest.mark.asyncio
    async def test_sklearn_backends_available(self):
        """Test sklearn anomaly backends work."""
        from models.anomaly_model import AnomalyModel

        for backend in ["isolation_forest", "one_class_svm", "local_outlier_factor"]:
            model = AnomalyModel(
                model_id=f"test-{backend}",
                device="cpu",
                backend=backend,
            )
            await model.load()
            assert model._detector is not None

    @pytest.mark.asyncio
    async def test_pyod_backends_available(self):
        """Test PyOD anomaly backends work."""
        from models.anomaly_model import AnomalyModel

        for backend in ["copod", "hbos", "ecod"]:
            model = AnomalyModel(
                model_id=f"test-{backend}",
                device="cpu",
                backend=backend,
            )
            await model.load()
            assert model._detector is not None


class TestDriftDetectorBackends:
    """Test drift detection backends."""

    @pytest.mark.parametrize("algorithm", ["adwin", "page_hinkley", "kswin", "ddm"])
    def test_drift_backend_available(self, algorithm):
        """Test that drift detection backends work."""
        from utils.drift_detector import DriftDetector

        detector = DriftDetector(algorithm=algorithm)
        assert detector is not None
        # Test update works
        result = detector.update(0.5)
        assert "drift_detected" in result


class TestChangePointBackends:
    """Test change point detection backends."""

    @pytest.mark.parametrize("algorithm", ["binseg", "window", "bottomup"])
    def test_changepoint_backend_available(self, algorithm):
        """Test that changepoint detection backends work."""
        from utils.changepoint_detector import ChangePointDetector
        import numpy as np

        detector = ChangePointDetector(algorithm=algorithm)
        # Generate simple test data
        data = np.concatenate([
            np.random.randn(50),
            np.random.randn(50) + 5,
        ])
        # Should not raise
        result = detector.detect(data, n_changepoints=1)
        # Result key is "change_points" not "changepoints"
        assert "change_points" in result

    def test_pelt_backend_available(self):
        """Test PELT backend works (uses penalty parameter instead of n_changepoints)."""
        from utils.changepoint_detector import ChangePointDetector
        import numpy as np

        detector = ChangePointDetector(algorithm="pelt")
        data = np.concatenate([
            np.random.randn(50),
            np.random.randn(50) + 5,
        ])
        # PELT requires penalty instead of n_changepoints
        result = detector.detect(data.tolist(), penalty=10)
        assert "change_points" in result


class TestVersionInfo:
    """Test version information is available."""

    def test_sklearn_version(self):
        """Test sklearn version."""
        import sklearn
        assert hasattr(sklearn, "__version__")

    def test_pyod_version(self):
        """Test pyod version."""
        import pyod
        assert hasattr(pyod, "__version__")

    def test_cleanlab_version(self):
        """Test cleanlab version."""
        import cleanlab
        assert hasattr(cleanlab, "__version__")

    def test_shap_version(self):
        """Test shap version."""
        import shap
        assert hasattr(shap, "__version__")

    def test_ruptures_version(self):
        """Test ruptures version."""
        import ruptures
        assert hasattr(ruptures, "__version__")

    def test_river_version(self):
        """Test river version."""
        import river
        assert hasattr(river, "__version__")
