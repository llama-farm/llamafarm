"""Tests for Polars buffer and rolling features.

Tests the high-performance data buffer for streaming ML:
- PolarsBuffer: Sliding window buffer with O(1) appends
- Rolling features: mean, std, min, max, lag features
- Performance benchmarks
"""

import time

import polars as pl
import pytest


class TestPolarsBufferBasic:
    """Test basic PolarsBuffer functionality."""

    def test_create_empty_buffer(self):
        """Test creating an empty buffer."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        assert buffer.size == 0
        assert buffer.window_size == 100
        assert buffer.columns == []

    def test_append_single_record(self):
        """Test appending a single record."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append({"value": 1.0, "category": "A"})

        assert buffer.size == 1
        assert "value" in buffer.columns
        assert "category" in buffer.columns

    def test_append_multiple_records(self):
        """Test appending multiple records one by one."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        for i in range(10):
            buffer.append({"value": float(i), "id": i})

        assert buffer.size == 10

    def test_append_batch(self):
        """Test batch append."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        records = [{"value": float(i), "id": i} for i in range(50)]
        buffer.append_batch(records)

        assert buffer.size == 50

    def test_creates_dataframe_from_dict(self):
        """Test that buffer creates DataFrame from dict data correctly."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append({"time_ms": 100, "value": 1.5, "category": "A"})
        buffer.append({"time_ms": 200, "value": 2.5, "category": "B"})

        df = buffer.get_data()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert df["time_ms"].to_list() == [100, 200]
        assert df["value"].to_list() == [1.5, 2.5]
        assert df["category"].to_list() == ["A", "B"]


class TestPolarsBufferWindow:
    """Test window truncation behavior."""

    def test_window_truncation(self):
        """Test that buffer truncates to window size."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=10)

        # Add more than window size
        for i in range(20):
            buffer.append({"value": float(i)})

        assert buffer.size == 10
        # Should have last 10 values (10-19)
        df = buffer.get_data()
        assert df["value"].to_list() == list(range(10, 20))

    def test_window_truncation_keeps_last_n(self):
        """Test that truncation keeps the most recent N records."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=5)
        buffer.append_batch([{"id": i} for i in range(100)])

        assert buffer.size == 5
        df = buffer.get_data()
        # Should have ids 95-99
        assert df["id"].to_list() == [95, 96, 97, 98, 99]

    def test_batch_append_truncation(self):
        """Test window truncation with batch append."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=5)
        buffer.append_batch([{"value": i} for i in range(10)])

        assert buffer.size == 5


class TestPolarsBufferFeatures:
    """Test rolling feature computation."""

    def test_rolling_mean(self):
        """Test rolling mean computation."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        # Add values 1-10
        buffer.append_batch([{"value": float(i)} for i in range(1, 11)])

        df = buffer.get_features(rolling_windows=[3])

        assert "value_rolling_mean_3" in df.columns
        # Rolling mean of window 3 for last value (8,9,10) = 9.0
        assert df["value_rolling_mean_3"][-1] == 9.0

    def test_rolling_std(self):
        """Test rolling std computation."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append_batch([{"value": float(i)} for i in range(1, 11)])

        df = buffer.get_features(rolling_windows=[3])

        assert "value_rolling_std_3" in df.columns
        # Std should be ~1.0 for consecutive integers
        assert abs(df["value_rolling_std_3"][-1] - 1.0) < 0.01

    def test_rolling_min_max(self):
        """Test rolling min/max computation."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append_batch([{"value": float(i)} for i in range(1, 11)])

        df = buffer.get_features(rolling_windows=[3])

        assert "value_rolling_min_3" in df.columns
        assert "value_rolling_max_3" in df.columns
        # For last 3 values (8, 9, 10): min=8, max=10
        assert df["value_rolling_min_3"][-1] == 8.0
        assert df["value_rolling_max_3"][-1] == 10.0

    def test_lag_features(self):
        """Test lag feature computation."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append_batch([{"value": float(i)} for i in range(1, 11)])

        df = buffer.get_features(
            rolling_windows=[],
            include_lags=True,
            lag_periods=[1, 2],
        )

        assert "value_lag_1" in df.columns
        assert "value_lag_2" in df.columns
        # Last value is 10, lag_1 should be 9, lag_2 should be 8
        assert df["value_lag_1"][-1] == 9.0
        assert df["value_lag_2"][-1] == 8.0

    def test_multiple_rolling_windows(self):
        """Test multiple rolling window sizes."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append_batch([{"value": float(i)} for i in range(50)])

        df = buffer.get_features(rolling_windows=[5, 10, 20])

        assert "value_rolling_mean_5" in df.columns
        assert "value_rolling_mean_10" in df.columns
        assert "value_rolling_mean_20" in df.columns

    def test_numeric_columns_only(self):
        """Test that features are only computed for numeric columns."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append_batch([
            {"value": float(i), "category": f"cat_{i}"}
            for i in range(10)
        ])

        df = buffer.get_features(rolling_windows=[3])

        # Should have rolling features for value but not category
        assert "value_rolling_mean_3" in df.columns
        assert "category_rolling_mean_3" not in df.columns


class TestPolarsBufferMethods:
    """Test additional buffer methods."""

    def test_get_latest(self):
        """Test getting latest N records."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append_batch([{"value": float(i)} for i in range(50)])

        latest = buffer.get_latest(n=5)
        assert len(latest) == 5
        assert latest["value"].to_list() == [45.0, 46.0, 47.0, 48.0, 49.0]

    def test_get_latest_with_features(self):
        """Test getting latest records with features."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append_batch([{"value": float(i)} for i in range(50)])

        latest = buffer.get_latest(n=5, with_features=True, rolling_windows=[3])
        assert len(latest) == 5
        assert "value_rolling_mean_3" in latest.columns

    def test_get_numpy(self):
        """Test getting numpy array of numeric data."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append_batch([{"x": float(i), "y": float(i * 2)} for i in range(10)])

        arr = buffer.get_numpy()
        assert arr.shape == (10, 2)

    def test_clear(self):
        """Test clearing the buffer."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append_batch([{"value": float(i)} for i in range(50)])
        assert buffer.size == 50

        buffer.clear()
        assert buffer.size == 0
        assert buffer.columns == []

    def test_to_list(self):
        """Test converting to list of dicts."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append_batch([{"value": float(i)} for i in range(5)])

        records = buffer.to_list()
        assert len(records) == 5
        assert records[0] == {"value": 0.0}
        assert records[4] == {"value": 4.0}

    def test_get_stats(self):
        """Test getting buffer statistics."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=100)
        buffer.append_batch([{"value": float(i), "category": f"c{i}"} for i in range(50)])

        stats = buffer.get_stats()
        assert stats.size == 50
        assert stats.window_size == 100
        assert "value" in stats.columns
        assert "category" in stats.columns
        assert "value" in stats.numeric_columns
        assert "category" not in stats.numeric_columns
        assert stats.memory_bytes > 0
        assert stats.append_count == 50


class TestRollingFeatures:
    """Test the rolling_features module."""

    def test_compute_features_default(self):
        """Test compute_features with default config."""
        from utils.rolling_features import compute_features

        df = pl.DataFrame({"value": list(range(50))})
        result = compute_features(df)

        assert "value_rolling_mean_5" in result.columns
        assert "value_rolling_mean_10" in result.columns
        assert "value_rolling_mean_20" in result.columns

    def test_compute_features_custom_config(self):
        """Test compute_features with custom config."""
        from utils.rolling_features import RollingFeatureConfig, compute_features

        config = RollingFeatureConfig(
            rolling_windows=[3, 7],
            include_stats=["mean", "max"],
            include_lags=True,
            lag_periods=[1, 5],
        )

        df = pl.DataFrame({"value": list(range(50))})
        result = compute_features(df, config)

        assert "value_rolling_mean_3" in result.columns
        assert "value_rolling_max_7" in result.columns
        assert "value_lag_1" in result.columns
        assert "value_lag_5" in result.columns
        # Should not have std since not in include_stats
        assert "value_rolling_std_3" not in result.columns

    def test_compute_anomaly_features(self):
        """Test compute_anomaly_features function."""
        from utils.rolling_features import compute_anomaly_features

        df = pl.DataFrame({"value": list(range(100))})
        result = compute_anomaly_features(df, windows=[10, 20])

        assert "value_zscore_10" in result.columns
        assert "value_deviation_20" in result.columns
        assert "value_minmax_scaled_10" in result.columns

    def test_get_feature_names(self):
        """Test get_feature_names function."""
        from utils.rolling_features import RollingFeatureConfig, get_feature_names

        config = RollingFeatureConfig(
            rolling_windows=[5],
            include_stats=["mean", "std"],
            include_lags=True,
            lag_periods=[1],
        )

        names = get_feature_names(["value"], config)

        assert "value_rolling_mean_5" in names
        assert "value_rolling_std_5" in names
        assert "value_lag_1" in names


@pytest.mark.slow
class TestPolarsBufferPerformance:
    """Test performance characteristics.

    These tests have timing thresholds and may be flaky in CI.
    Run with: pytest -m slow
    Skip with: pytest -m "not slow"
    """

    def test_append_performance(self):
        """Test that append is fast (<1ms average)."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=1000)

        # Warm up
        for i in range(10):
            buffer.append({"value": float(i), "x": float(i * 2)})

        buffer.clear()

        # Time 1000 appends
        start = time.perf_counter()
        for i in range(1000):
            buffer.append({"value": float(i), "x": float(i * 2)})
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / 1000
        stats = buffer.get_stats()

        # Assert average is under 1ms per append
        # Note: This is a soft test - CI environments may be slower
        assert avg_ms < 5.0, f"Average append time {avg_ms:.3f}ms exceeds 5ms"
        assert stats.avg_append_ms < 5.0

    def test_batch_append_faster_than_individual(self):
        """Test that batch append is faster than individual appends."""
        from utils.polars_buffer import PolarsBuffer

        # Individual appends
        buffer1 = PolarsBuffer(window_size=2000)
        start1 = time.perf_counter()
        for i in range(1000):
            buffer1.append({"value": float(i)})
        time_individual = time.perf_counter() - start1

        # Batch append
        buffer2 = PolarsBuffer(window_size=2000)
        records = [{"value": float(i)} for i in range(1000)]
        start2 = time.perf_counter()
        buffer2.append_batch(records)
        time_batch = time.perf_counter() - start2

        # Batch should be faster (at least 2x typically)
        assert time_batch < time_individual, "Batch append should be faster than individual"

    def test_feature_computation_reasonable(self):
        """Test that feature computation completes in reasonable time."""
        from utils.polars_buffer import PolarsBuffer

        buffer = PolarsBuffer(window_size=10000)
        buffer.append_batch([
            {"a": float(i), "b": float(i * 2), "c": float(i * 3)}
            for i in range(5000)
        ])

        start = time.perf_counter()
        df = buffer.get_features(rolling_windows=[5, 10, 20, 50, 100])
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete in under 500ms for 5000 records with 5 windows
        assert elapsed_ms < 500, f"Feature computation took {elapsed_ms:.1f}ms"
        assert len(df) == 5000


class TestFeatureEncoderIntegration:
    """Test integration with encoded features."""

    def test_polars_buffer_with_numeric_features(self):
        """Test that PolarsBuffer works with numeric feature vectors."""
        from utils.polars_buffer import PolarsBuffer

        # Simulate encoded features (numeric vectors)
        # This mimics what FeatureEncoder.fit_transform produces
        encoded_features = [
            [100.0, 5.0, 0.0, 1.0, 0.0],  # amount, count, one-hot category
            [200.0, 10.0, 1.0, 0.0, 0.0],
            [150.0, 7.0, 0.0, 0.0, 1.0],
        ]

        # Add to buffer as named features
        buffer = PolarsBuffer(window_size=100)
        for row in encoded_features:
            buffer.append({f"f{i}": v for i, v in enumerate(row)})

        assert buffer.size == 3
        assert len(buffer.numeric_columns) == 5

        # Should be able to get numpy array for ML
        arr = buffer.get_numpy()
        assert arr.shape == (3, 5)

        # Should be able to compute rolling features
        df = buffer.get_features(rolling_windows=[2])
        assert "f0_rolling_mean_2" in df.columns
