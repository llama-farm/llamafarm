"""Tests for streaming anomaly detection with Polars integration.

Tests the auto-rolling detector that uses:
- Polars as the internal data substrate (automatic)
- PyOD for anomaly detection
- Tick-Tock pattern for background retraining
"""

import asyncio

import pytest


class TestStreamingAnomalyDetectorInit:
    """Test StreamingAnomalyDetector initialization."""

    def test_initializes_with_cold_start_status(self):
        """Test detector starts in collecting (cold start) status."""
        from models.streaming_anomaly import DetectorStatus, StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="test-detector",
            backend="ecod",
            min_samples=50,
        )

        assert detector.status == DetectorStatus.COLLECTING
        assert detector.model_version == 0
        assert detector.samples_collected == 0
        assert not detector.is_ready

    def test_initializes_with_default_values(self):
        """Test detector has sensible defaults."""
        from models.streaming_anomaly import StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(model_id="test")

        assert detector.min_samples == 50
        assert detector.retrain_interval == 100
        assert detector.window_size == 1000
        assert detector.threshold == 0.5
        assert detector.contamination == 0.1
        assert detector.backend == "ecod"

    def test_initializes_with_custom_values(self):
        """Test detector accepts custom configuration."""
        from models.streaming_anomaly import StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="custom",
            backend="hbos",
            min_samples=100,
            retrain_interval=50,
            window_size=500,
            contamination=0.05,
            threshold=0.7,
        )

        assert detector.backend == "hbos"
        assert detector.min_samples == 100
        assert detector.retrain_interval == 50
        assert detector.window_size == 500
        assert detector.contamination == 0.05
        assert detector.threshold == 0.7


class TestStreamingAnomalyColdStart:
    """Test cold start behavior."""

    @pytest.mark.asyncio
    async def test_cold_start_collects_minimum_samples(self):
        """Test cold start phase collects min_samples before training."""
        from models.streaming_anomaly import DetectorStatus, StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="cold-start-test",
            min_samples=10,
        )

        # Process 9 samples (less than min_samples)
        for i in range(9):
            result = await detector.process({"value": float(i)})
            assert result.status == DetectorStatus.COLLECTING
            assert result.score is None
            assert result.is_anomaly is None
            assert result.samples_until_ready == 10 - (i + 1)

        assert detector.samples_collected == 9
        assert not detector.is_ready

    @pytest.mark.asyncio
    async def test_transitions_to_ready_after_min_samples(self):
        """Test detector transitions to ready after collecting min_samples."""
        from models.streaming_anomaly import DetectorStatus, StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="transition-test",
            min_samples=10,
        )

        # Process exactly min_samples
        for i in range(10):
            await detector.process({"value": float(i)})

        assert detector.status == DetectorStatus.READY
        assert detector.is_ready
        assert detector.model_version == 1

    @pytest.mark.asyncio
    async def test_returns_scores_after_warm_up(self):
        """Test scores are returned after warm-up phase."""
        from models.streaming_anomaly import DetectorStatus, StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="score-test",
            min_samples=10,
        )

        # Warm up
        for i in range(10):
            await detector.process({"value": float(i)})

        # Process one more - should get scores
        result = await detector.process({"value": 5.0})

        assert result.status == DetectorStatus.READY
        assert result.score is not None
        assert result.is_anomaly is not None
        assert result.raw_score is not None
        assert result.samples_until_ready == 0


class TestStreamingAnomalyScoring:
    """Test anomaly scoring after warm-up."""

    @pytest.mark.asyncio
    async def test_detects_obvious_anomaly(self):
        """Test that obvious anomalies are detected."""
        from models.streaming_anomaly import StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="anomaly-detection-test",
            min_samples=50,
            threshold=0.5,
        )

        # Train on normal data (values around 10)
        for i in range(50):
            await detector.process({"value": 10.0 + (i % 5)})

        # Test with normal data
        normal_result = await detector.process({"value": 12.0})
        assert normal_result.score is not None

        # Test with obvious anomaly (way outside normal range)
        anomaly_result = await detector.process({"value": 1000.0})
        assert anomaly_result.score is not None

        # Anomaly should have higher score
        assert anomaly_result.score > normal_result.score

    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of multiple data points."""
        from models.streaming_anomaly import StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="batch-test",
            min_samples=10,
        )

        # Warm up
        batch_data = [{"value": float(i)} for i in range(10)]
        batch_result = await detector.process_batch(batch_data)

        assert len(batch_result.results) == 10
        assert batch_result.model_version == 1
        assert batch_result.processing_time_ms > 0

        # Process more batch - should get scores
        more_data = [{"value": float(i)} for i in range(5)]
        more_result = await detector.process_batch(more_data)

        assert len(more_result.results) == 5
        for r in more_result.results:
            assert r.score is not None


class TestStreamingAnomalyRetraining:
    """Test auto-rolling retraining."""

    @pytest.mark.asyncio
    async def test_retrain_triggers_after_interval(self):
        """Test that retraining triggers after retrain_interval samples."""
        from models.streaming_anomaly import StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="retrain-test",
            min_samples=10,
            retrain_interval=20,
        )

        # Warm up (10 samples)
        for i in range(10):
            await detector.process({"value": float(i)})

        assert detector.model_version == 1
        initial_version = detector.model_version

        # Process retrain_interval more samples
        for i in range(20):
            await detector.process({"value": float(i + 10)})

        # Poll for background retrain to complete (max 2 seconds)
        for _ in range(20):
            if detector.model_version > initial_version:
                break
            await asyncio.sleep(0.1)

        # Should have triggered retrain
        assert detector.model_version > initial_version

    @pytest.mark.asyncio
    async def test_continues_scoring_during_retrain(self):
        """Test that scoring continues during background retraining."""
        from models.streaming_anomaly import StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="score-during-retrain-test",
            min_samples=10,
            retrain_interval=15,
        )

        # Warm up
        for i in range(10):
            await detector.process({"value": float(i)})

        # Process until retrain triggers
        for i in range(15):
            result = await detector.process({"value": float(i)})
            # Should always get scores after warm-up
            assert result.score is not None


class TestStreamingAnomalySlidingWindow:
    """Test sliding window behavior."""

    @pytest.mark.asyncio
    async def test_window_maintains_correct_size(self):
        """Test sliding window maintains window_size."""
        from models.streaming_anomaly import StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="window-test",
            min_samples=10,
            window_size=50,
        )

        # Process more than window_size samples
        for i in range(100):
            await detector.process({"value": float(i)})

        # Window should be at most window_size
        assert detector.samples_collected <= 50


class TestStreamingAnomalyRollingFeatures:
    """Test optional rolling feature computation."""

    @pytest.mark.asyncio
    async def test_rolling_features_computed_when_configured(self):
        """Test rolling features are computed when rolling_windows is set."""
        from models.streaming_anomaly import StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="rolling-features-test",
            min_samples=20,
            rolling_windows=[5, 10],
            include_lags=True,
            lag_periods=[1, 2],
        )

        # The detector should use rolling features internally
        # Verify it initializes with the config
        assert detector.rolling_windows == [5, 10]
        assert detector.include_lags is True
        assert detector.lag_periods == [1, 2]

        # Process enough data
        for i in range(25):
            result = await detector.process({"value": float(i * 10)})

        # Should still work and produce scores
        assert result.score is not None


class TestStreamingDetectorManager:
    """Test the detector manager for session management."""

    @pytest.mark.asyncio
    async def test_get_or_create_returns_same_detector(self):
        """Test that get_or_create returns the same detector instance."""
        from models.streaming_anomaly import StreamingDetectorManager

        manager = StreamingDetectorManager()

        detector1 = await manager.get_or_create("test-model", backend="ecod")
        detector2 = await manager.get_or_create("test-model", backend="ecod")

        assert detector1 is detector2
        await manager.clear_all()

    @pytest.mark.asyncio
    async def test_different_model_ids_create_different_detectors(self):
        """Test that different model_ids create separate detectors."""
        from models.streaming_anomaly import StreamingDetectorManager

        manager = StreamingDetectorManager()

        detector1 = await manager.get_or_create("model-1")
        detector2 = await manager.get_or_create("model-2")

        assert detector1 is not detector2
        assert detector1.model_id == "model-1"
        assert detector2.model_id == "model-2"
        await manager.clear_all()

    @pytest.mark.asyncio
    async def test_list_detectors_returns_all_active(self):
        """Test list_detectors returns all active detector stats."""
        from models.streaming_anomaly import StreamingDetectorManager

        manager = StreamingDetectorManager()

        await manager.get_or_create("detector-a")
        await manager.get_or_create("detector-b")
        await manager.get_or_create("detector-c")

        detectors = await manager.list_detectors()

        assert len(detectors) == 3
        model_ids = [d["model_id"] for d in detectors]
        assert "detector-a" in model_ids
        assert "detector-b" in model_ids
        assert "detector-c" in model_ids
        await manager.clear_all()

    @pytest.mark.asyncio
    async def test_delete_removes_detector(self):
        """Test delete removes a detector."""
        from models.streaming_anomaly import StreamingDetectorManager

        manager = StreamingDetectorManager()

        await manager.get_or_create("to-delete")
        assert await manager.get("to-delete") is not None

        deleted = await manager.delete("to-delete")
        assert deleted is True
        assert await manager.get("to-delete") is None
        await manager.clear_all()

    @pytest.mark.asyncio
    async def test_clear_all_removes_all_detectors(self):
        """Test clear_all removes all detectors."""
        from models.streaming_anomaly import StreamingDetectorManager

        manager = StreamingDetectorManager()

        await manager.get_or_create("d1")
        await manager.get_or_create("d2")
        await manager.get_or_create("d3")

        count = await manager.clear_all()
        assert count == 3

        detectors = await manager.list_detectors()
        assert len(detectors) == 0


class TestStreamingAnomalyReset:
    """Test detector reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_returns_to_cold_start(self):
        """Test reset returns detector to cold start state."""
        from models.streaming_anomaly import DetectorStatus, StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="reset-test",
            min_samples=10,
        )

        # Warm up and process data
        for i in range(20):
            await detector.process({"value": float(i)})

        assert detector.is_ready
        assert detector.model_version >= 1
        assert detector.samples_collected > 0

        # Reset
        detector.reset()

        assert detector.status == DetectorStatus.COLLECTING
        assert detector.model_version == 0
        assert detector.samples_collected == 0
        assert not detector.is_ready


class TestStreamingAnomalyStats:
    """Test detector statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_returns_all_info(self):
        """Test get_stats returns comprehensive detector info."""
        from models.streaming_anomaly import StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="stats-test",
            backend="ecod",
            min_samples=10,
            retrain_interval=20,
            window_size=100,
            threshold=0.6,
        )

        # Process some data
        for i in range(15):
            await detector.process({"value": float(i)})

        stats = detector.get_stats()

        assert stats["model_id"] == "stats-test"
        assert stats["backend"] == "ecod"
        assert stats["status"] == "ready"
        assert stats["model_version"] == 1
        assert stats["samples_collected"] == 15
        assert stats["total_processed"] == 15
        assert stats["min_samples"] == 10
        assert stats["retrain_interval"] == 20
        assert stats["window_size"] == 100
        assert stats["threshold"] == 0.6
        assert stats["is_ready"] is True


class TestStreamingAnomalyDefaults:
    """Test that default configuration works without any optional params."""

    @pytest.mark.asyncio
    async def test_works_with_all_defaults(self):
        """Test streaming detection works with only required params."""
        from models.streaming_anomaly import StreamingAnomalyDetector

        # Create with only model_id
        detector = StreamingAnomalyDetector(model_id="defaults-test")

        # Should work with default values
        # Process min_samples (default 50)
        for i in range(50):
            await detector.process({"value": float(i)})

        # Should be ready
        assert detector.is_ready

        # Should return scores
        result = await detector.process({"value": 25.0})
        assert result.score is not None


class TestStreamingAnomalyMultipleFeatures:
    """Test with multiple feature columns."""

    @pytest.mark.asyncio
    async def test_multiple_features(self):
        """Test detector works with multiple feature columns."""
        from models.streaming_anomaly import StreamingAnomalyDetector

        detector = StreamingAnomalyDetector(
            model_id="multi-feature-test",
            min_samples=20,
        )

        # Process data with multiple features
        for i in range(25):
            await detector.process({
                "amount": float(i * 10),
                "count": float(i % 5),
                "duration_ms": float(i * 100),
            })

        assert detector.is_ready

        # Normal transaction
        normal_result = await detector.process({
            "amount": 100.0,
            "count": 2.0,
            "duration_ms": 500.0,
        })
        assert normal_result.score is not None

        # Anomalous transaction (all values way off)
        anomaly_result = await detector.process({
            "amount": 10000.0,
            "count": 100.0,
            "duration_ms": 50000.0,
        })
        assert anomaly_result.score is not None
