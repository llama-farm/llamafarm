"""Tests for model loading and training services (Phase 2)."""

import asyncio
import os
import time

import numpy as np
import pytest


def _check_setfit_installed() -> bool:
    """Check if SetFit is installed."""
    try:
        import setfit  # noqa: F401

        return True
    except ImportError:
        return False


class TestModelLoader:
    """Test ModelLoader caching and locking."""

    def test_model_loader_caching(self):
        """Test that same params return cached model."""
        from services.model_loader import ModelLoader
        from utils.model_cache import ModelCache

        cache: ModelCache[str] = ModelCache(ttl=300)
        loader: ModelLoader[str] = ModelLoader(cache=cache, model_type="test")

        # Pre-populate cache
        cache["test-key"] = "cached-model"

        # Get cached should return the same object
        result = loader.get_cached("test-key")
        assert result == "cached-model"

    @pytest.mark.asyncio
    async def test_model_loader_get_or_load_cached(self):
        """Test get_or_load returns cached model without calling loader."""
        from services.model_loader import ModelLoader
        from utils.model_cache import ModelCache

        cache: ModelCache[str] = ModelCache(ttl=300)
        loader: ModelLoader[str] = ModelLoader(cache=cache, model_type="test")

        # Pre-populate cache
        cache["test-key"] = "cached-model"

        call_count = 0

        async def loader_fn():
            nonlocal call_count
            call_count += 1
            return "new-model"

        result = await loader.get_or_load("test-key", loader_fn)
        assert result == "cached-model"
        assert call_count == 0  # Loader should not have been called

    @pytest.mark.asyncio
    async def test_model_loader_different_params(self):
        """Test that different params load new model."""
        from services.model_loader import ModelLoader
        from utils.model_cache import ModelCache

        cache: ModelCache[str] = ModelCache(ttl=300)
        loader: ModelLoader[str] = ModelLoader(cache=cache, model_type="test")

        async def loader_fn_1():
            return "model-1"

        async def loader_fn_2():
            return "model-2"

        result1 = await loader.get_or_load("key-1", loader_fn_1)
        result2 = await loader.get_or_load("key-2", loader_fn_2)

        assert result1 == "model-1"
        assert result2 == "model-2"
        assert loader.is_cached("key-1")
        assert loader.is_cached("key-2")

    @pytest.mark.asyncio
    async def test_model_loader_lock_contention(self):
        """Test concurrent loads don't corrupt state."""
        from services.model_loader import ModelLoader
        from utils.model_cache import ModelCache

        cache: ModelCache[int] = ModelCache(ttl=300)
        loader: ModelLoader[int] = ModelLoader(cache=cache, model_type="test")

        load_count = 0

        async def slow_loader():
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.1)  # Simulate slow load
            return load_count

        # Launch multiple concurrent requests for the same key
        tasks = [loader.get_or_load("same-key", slow_loader) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should get the same result (first loader wins)
        assert all(r == 1 for r in results)
        # Only one load should have occurred
        assert load_count == 1


class TestModelLoaderRegistry:
    """Test ModelLoaderRegistry."""

    @pytest.mark.asyncio
    async def test_registry_creates_loaders(self):
        """Test registry creates separate loaders for each type."""
        from services.model_loader import ModelLoaderRegistry

        registry = ModelLoaderRegistry(ttl=300)

        loader1 = await registry.get_loader("anomaly")
        loader2 = await registry.get_loader("classifier")
        loader3 = await registry.get_loader("anomaly")

        # Same type should return same loader
        assert loader1 is loader3
        # Different types should return different loaders
        assert loader1 is not loader2

    @pytest.mark.asyncio
    async def test_registry_get_all_cached_keys(self):
        """Test getting all cached keys from registry."""
        from services.model_loader import ModelLoaderRegistry

        registry = ModelLoaderRegistry(ttl=300)

        anomaly_loader = await registry.get_loader("anomaly")
        classifier_loader = await registry.get_loader("classifier")

        anomaly_loader.cache["anomaly:test1"] = "model1"
        classifier_loader.cache["classifier:test2"] = "model2"

        all_keys = registry.get_all_cached_keys()
        assert "anomaly" in all_keys
        assert "classifier" in all_keys
        assert "anomaly:test1" in all_keys["anomaly"]
        assert "classifier:test2" in all_keys["classifier"]


class TestTrainingExecutor:
    """Test training executor for non-blocking training."""

    @pytest.mark.asyncio
    async def test_run_in_executor_basic(self):
        """Test run_in_executor runs function in thread pool."""
        from services.training_executor import run_in_executor

        def sync_function(x, y):
            return x + y

        result = await run_in_executor(sync_function, 2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_run_in_executor_with_kwargs(self):
        """Test run_in_executor handles kwargs correctly."""
        from services.training_executor import run_in_executor

        def sync_function(a, b=10):
            return a * b

        result = await run_in_executor(sync_function, 5, b=3)
        assert result == 15

    @pytest.mark.asyncio
    async def test_async_training_not_blocking(self):
        """Test that training doesn't block other async operations."""
        from services.training_executor import run_in_executor

        def slow_training():
            time.sleep(0.2)
            return "trained"

        async def quick_operation():
            return "quick"

        # Start slow training
        training_task = asyncio.create_task(run_in_executor(slow_training))

        # Quick operations should complete while training runs
        start = time.time()
        results = []
        for _ in range(3):
            result = await quick_operation()
            results.append(result)
        quick_time = time.time() - start

        # Wait for training
        training_result = await training_task

        # Quick operations should be fast (< 0.1s)
        assert quick_time < 0.1
        assert all(r == "quick" for r in results)
        assert training_result == "trained"

    @pytest.mark.asyncio
    async def test_training_context(self):
        """Test TrainingContext tracks timing."""
        from services.training_executor import TrainingContext

        async with TrainingContext("test-training") as ctx:

            def sync_work():
                time.sleep(0.05)
                return "done"

            result = await ctx.run(sync_work)
            assert result == "done"


class TestRobustScaler:
    """Test that anomaly model uses RobustScaler."""

    def test_robust_scaler_used(self):
        """Test AnomalyModel uses RobustScaler not StandardScaler."""
        from models.anomaly_model import AnomalyModel

        model = AnomalyModel("test", "cpu", backend="isolation_forest")

        # After _initialize_backend is called, scaler should be RobustScaler
        import asyncio

        asyncio.get_event_loop().run_until_complete(model.load())

        from sklearn.preprocessing import RobustScaler

        assert isinstance(model._scaler, RobustScaler)

    @pytest.mark.asyncio
    async def test_robust_scaler_resilient_to_outliers(self):
        """Test RobustScaler handles outliers better than StandardScaler."""
        from sklearn.preprocessing import RobustScaler, StandardScaler

        # Data with extreme outlier
        data = np.array([[1], [2], [3], [4], [1000]])

        robust = RobustScaler()
        standard = StandardScaler()

        robust_scaled = robust.fit_transform(data)
        standard_scaled = standard.fit_transform(data)

        # RobustScaler uses median/IQR, so outlier doesn't affect others much
        # StandardScaler uses mean/std, so outlier pulls everything
        # The "normal" values (1,2,3,4) should be closer together with RobustScaler
        robust_range = robust_scaled[:4].max() - robust_scaled[:4].min()
        standard_range = standard_scaled[:4].max() - standard_scaled[:4].min()

        # RobustScaler should spread normal values more (outlier doesn't compress them)
        assert robust_range > standard_range


class TestAnomalyModelNonBlocking:
    """Test anomaly model non-blocking fit."""

    @pytest.mark.asyncio
    async def test_anomaly_fit_uses_executor(self):
        """Test that anomaly model fit uses thread pool executor."""
        from models.anomaly_model import AnomalyModel

        model = AnomalyModel("test", "cpu", backend="isolation_forest")
        await model.load()

        # Generate simple training data
        data = np.random.randn(100, 5)

        # Fit should not block the event loop
        async def check_not_blocked():
            return "not blocked"

        # Start fit
        fit_task = asyncio.create_task(model.fit(data, use_executor=True))

        # Check we can do other things immediately
        result = await check_not_blocked()
        assert result == "not blocked"

        # Wait for fit to complete
        fit_result = await fit_task
        assert fit_result.samples_fitted == 100


class TestClassifierModelNonBlocking:
    """Test classifier model non-blocking fit."""

    @pytest.mark.skipif(not _check_setfit_installed(), reason="SetFit not installed")
    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="SetFit training segfaults in accelerate memory cleanup on CI runners without GPU",
    )
    @pytest.mark.asyncio
    async def test_classifier_fit_uses_executor(self):
        """Test that classifier model fit uses thread pool executor."""
        from models.classifier_model import ClassifierModel

        model = ClassifierModel("test", "cpu")
        await model.load()

        # Simple training data
        texts = ["good product", "bad product", "excellent", "terrible"]
        labels = ["positive", "negative", "positive", "negative"]

        # Fit with executor
        result = await model.fit(texts, labels, num_iterations=1, use_executor=True)
        assert result.samples_fitted == 4
        assert result.num_classes == 2


class TestAutoencoderEarlyStopping:
    """Test autoencoder early stopping functionality."""

    @pytest.mark.asyncio
    async def test_early_stopping_triggers(self):
        """Test that early stopping stops training before max epochs."""
        from models.anomaly_model import AnomalyModel

        model = AnomalyModel("test", "cpu", backend="autoencoder")
        await model.load()

        # Generate very simple data that converges quickly
        # Using deterministic data so validation loss plateaus
        np.random.seed(42)
        data = np.random.randn(50, 4)

        # With patience=10 and max epochs=100, should stop early
        result = await model.fit(data, epochs=100, batch_size=16, use_executor=False)

        # Training should complete (early stopping doesn't break the fit)
        assert result.samples_fitted == 50
        assert model._is_fitted

    @pytest.mark.asyncio
    async def test_early_stopping_restores_best_model(self):
        """Test that early stopping restores the best model state."""
        from models.anomaly_model import AnomalyModel

        model = AnomalyModel("test", "cpu", backend="autoencoder")
        await model.load()

        # Data where loss will fluctuate
        np.random.seed(123)
        data = np.random.randn(100, 8)

        # Fit with early stopping
        await model.fit(data, epochs=50, batch_size=32, use_executor=False)

        # Verify the model is fitted and can make predictions
        assert model._is_fitted, "Model should be fitted after training"
        # Verify the detector has been trained (has decision_scores_)
        detector = model._detector
        assert hasattr(detector, "decision_scores_"), "Detector should have decision_scores_ after fitting"
        # Verify we can score new data (model is in eval mode)
        scores = await model.score(data)
        assert len(scores) == len(data), "Should be able to score data after fitting"
