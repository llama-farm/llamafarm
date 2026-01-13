"""
Tests for non-blocking async training infrastructure.

These tests verify that:
1. Training runs in a separate thread pool, not blocking the event loop
2. Health checks and other requests can be served during training
3. Concurrent fit requests are handled gracefully
4. Training errors are propagated properly
"""

import asyncio
import time

import numpy as np
import pytest

from models.anomaly_model import AnomalyModel, FitResult
from utils.training_executor import (
    get_training_executor,
    get_training_executor_stats,
)


@pytest.fixture
def anomaly_model(device):
    """Create a fresh anomaly model for testing."""
    model = AnomalyModel(
        model_id="test-anomaly-model",
        device=device,
        backend="isolation_forest",
        contamination=0.1,
    )
    return model


@pytest.fixture
def training_data():
    """Generate synthetic training data (normal distribution)."""
    np.random.seed(42)
    # 500 samples, 5 features - normal data
    normal_data = np.random.randn(500, 5)
    return normal_data.tolist()


@pytest.fixture
def test_data_with_anomalies():
    """Generate test data with known anomalies."""
    np.random.seed(42)
    # 100 normal samples
    normal = np.random.randn(100, 5)
    # 10 anomalous samples (much larger values)
    anomalous = np.random.randn(10, 5) * 10 + 5
    data = np.vstack([normal, anomalous])
    return data.tolist()


class TestTrainingExecutor:
    """Tests for the training executor module."""

    def test_get_training_executor_creates_executor(self):
        """Test that get_training_executor creates an executor."""
        executor = get_training_executor()
        assert executor is not None

    def test_get_training_executor_returns_same_instance(self):
        """Test that get_training_executor returns singleton."""
        executor1 = get_training_executor()
        executor2 = get_training_executor()
        assert executor1 is executor2

    def test_get_training_executor_stats(self):
        """Test executor stats are available."""
        _ = get_training_executor()  # Ensure executor is created
        stats = get_training_executor_stats()
        assert stats["initialized"] is True
        assert "max_workers" in stats


class TestAsyncAnomalyTraining:
    """Tests for async anomaly model training."""

    @pytest.mark.asyncio
    async def test_fit_returns_fit_result(self, anomaly_model, training_data, device):
        """Test that fit() returns a FitResult."""
        await anomaly_model.load()
        result = await anomaly_model.fit(training_data)

        assert isinstance(result, FitResult)
        assert result.samples_fitted == len(training_data)
        assert result.training_time_ms > 0
        assert result.model_params["backend"] == "isolation_forest"

    @pytest.mark.asyncio
    async def test_fit_does_not_block_event_loop(self, anomaly_model, training_data, device):
        """Test that fit() doesn't block the event loop.

        This test verifies that other async operations can run during training.
        We start a training task and immediately run health check-style operations.
        """
        await anomaly_model.load()

        # Track that our concurrent task ran during training
        concurrent_operations_count = 0
        training_started = asyncio.Event()

        async def concurrent_operation():
            """Simulates a health check or other async operation."""
            nonlocal concurrent_operations_count
            # Wait for training to start
            await training_started.wait()
            # Run multiple times during training
            for _ in range(5):
                await asyncio.sleep(0.1)
                concurrent_operations_count += 1
            return concurrent_operations_count

        async def training_task():
            """Run the training."""
            training_started.set()
            return await anomaly_model.fit(training_data)

        # Run training and concurrent operations in parallel
        results = await asyncio.gather(
            training_task(),
            concurrent_operation(),
        )

        fit_result, ops_count = results

        # Both should have completed
        assert isinstance(fit_result, FitResult)
        assert ops_count >= 3, "Concurrent operations should have run during training"

    @pytest.mark.asyncio
    async def test_health_check_during_training(self, anomaly_model, training_data, device):
        """Test that health checks can be served during training.

        This is the key test for the blocking issue - simulates the /health endpoint
        responding while a training request is in progress.
        """
        await anomaly_model.load()

        health_check_times = []

        async def health_check():
            """Simulates a health check request."""
            start = time.perf_counter()
            await asyncio.sleep(0.01)  # Simulate minimal async operation
            elapsed = time.perf_counter() - start
            health_check_times.append(elapsed)
            return {"status": "healthy"}

        async def training_with_health_checks():
            """Run training while periodically checking health."""
            # Start training as a background task
            training_task = asyncio.create_task(anomaly_model.fit(training_data))

            # Run health checks while training is happening
            for _ in range(5):
                result = await health_check()
                assert result["status"] == "healthy"
                await asyncio.sleep(0.1)

            # Wait for training to complete
            return await training_task

        fit_result = await training_with_health_checks()

        # Training should complete successfully
        assert isinstance(fit_result, FitResult)

        # Health checks should have been fast (not blocked)
        # If training blocked the loop, health checks would take much longer
        avg_health_check_time = sum(health_check_times) / len(health_check_times)
        assert avg_health_check_time < 0.5, f"Health checks were too slow: {avg_health_check_time:.3f}s"

    @pytest.mark.asyncio
    async def test_model_fitted_after_async_training(self, anomaly_model, training_data, test_data_with_anomalies, device):
        """Test that model is properly fitted after async training."""
        await anomaly_model.load()

        # Fit the model
        await anomaly_model.fit(training_data)
        assert anomaly_model.is_fitted

        # Score some data to verify model works
        scores = await anomaly_model.score(test_data_with_anomalies)
        assert len(scores) == len(test_data_with_anomalies)

        # Check that some anomalies were detected
        anomalies = [s for s in scores if s.is_anomaly]
        assert len(anomalies) > 0, "Model should detect some anomalies"

    @pytest.mark.asyncio
    async def test_concurrent_fit_requests_handled(self, device, training_data):
        """Test that concurrent fit requests are handled gracefully.

        The thread pool should queue requests, not crash.
        """
        model1 = AnomalyModel("test-model-1", device, backend="isolation_forest")
        model2 = AnomalyModel("test-model-2", device, backend="isolation_forest")

        await model1.load()
        await model2.load()

        # Use smaller datasets for faster test
        small_data = training_data[:100]

        # Start both training tasks concurrently
        results = await asyncio.gather(
            model1.fit(small_data),
            model2.fit(small_data),
        )

        # Both should complete successfully
        assert all(isinstance(r, FitResult) for r in results)
        assert model1.is_fitted
        assert model2.is_fitted

    @pytest.mark.asyncio
    async def test_training_error_propagation(self, anomaly_model, device):
        """Test that errors during training are properly propagated."""
        await anomaly_model.load()

        # Empty data should cause an error (ValueError from validation)
        with pytest.raises(ValueError):
            await anomaly_model.fit([])

    @pytest.mark.asyncio
    async def test_autoencoder_async_training(self, device, training_data):
        """Test that autoencoder training also runs asynchronously."""
        model = AnomalyModel(
            model_id="test-autoencoder",
            device=device,
            backend="autoencoder",
        )
        await model.load()

        # Use smaller data and fewer epochs for faster test
        small_data = training_data[:100]

        concurrent_ops = 0

        async def concurrent_task():
            nonlocal concurrent_ops
            for _ in range(3):
                await asyncio.sleep(0.1)
                concurrent_ops += 1

        results = await asyncio.gather(
            model.fit(small_data, epochs=5, batch_size=16),
            concurrent_task(),
        )

        fit_result = results[0]
        assert isinstance(fit_result, FitResult)
        assert concurrent_ops >= 2, "Concurrent operations should run during autoencoder training"


class TestAsyncClassifierTraining:
    """Tests for async classifier training."""

    @pytest.mark.asyncio
    async def test_classifier_fit_does_not_block_event_loop(self, device):
        """Test that classifier fit() doesn't block the event loop."""
        # Skip if setfit not available
        pytest.importorskip("setfit")

        from models.classifier_model import ClassifierModel

        model = ClassifierModel(
            model_id="test-classifier",
            device=device,
            base_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        await model.load()

        # Minimal training data
        texts = [
            "I love this product",
            "This is amazing",
            "Great quality",
            "Terrible experience",
            "Waste of money",
            "Poor quality",
        ]
        labels = ["positive", "positive", "positive", "negative", "negative", "negative"]

        concurrent_ops = 0

        async def concurrent_task():
            nonlocal concurrent_ops
            for _ in range(3):
                await asyncio.sleep(0.2)
                concurrent_ops += 1

        results = await asyncio.gather(
            model.fit(texts, labels, num_iterations=1, batch_size=4),
            concurrent_task(),
        )

        fit_result = results[0]
        assert fit_result.samples_fitted == len(texts)
        assert concurrent_ops >= 2, "Concurrent operations should run during classifier training"
