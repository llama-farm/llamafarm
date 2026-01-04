"""Tests for DuckDB Store - Time-series and spatial data storage.

These tests are written FIRST following TDD methodology.
The DuckDBStore implementation should make these tests pass.
"""

import tempfile
import time
from datetime import datetime, timedelta

import pytest


class TestDuckDBStoreInitialization:
    """Test DuckDBStore initialization and extension loading."""

    def test_duckdb_store_initializes_with_default_config(self):
        """Test DuckDBStore initializes with default configuration."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            assert store is not None
            assert store.is_connected()
            store.close()

    def test_duckdb_store_creates_database_file(self):
        """Test DuckDBStore creates the database file on disk."""
        import os

        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = f"{temp_dir}/test.duckdb"
            config = {"path": db_path}
            store = DuckDBStore(config=config)

            assert os.path.exists(db_path)
            store.close()

    def test_duckdb_store_loads_spatial_extension(self):
        """Test DuckDBStore loads the spatial extension."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb", "extensions": ["spatial"]}
            store = DuckDBStore(config=config)

            # Verify spatial extension is loaded by checking for ST_Point function
            result = store.execute("SELECT ST_Point(0, 0) IS NOT NULL AS has_spatial")
            assert result[0][0] is True
            store.close()

    def test_duckdb_store_creates_timeseries_table(self):
        """Test DuckDBStore creates the time-series table with proper schema."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb", "table_name": "telemetry"}
            store = DuckDBStore(config=config)

            # Check table exists
            result = store.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'telemetry'"
            )
            assert result[0][0] == 1

            # Check schema has required columns
            result = store.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'telemetry'
                ORDER BY ordinal_position
                """
            )
            columns = [row[0] for row in result]
            assert "id" in columns
            assert "ts" in columns
            assert "source" in columns
            assert "data" in columns
            assert "metadata" in columns
            store.close()


class TestDuckDBStoreRecordOperations:
    """Test adding and querying records."""

    def test_add_records_inserts_data_correctly(self):
        """Test add_records inserts time-series data correctly."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            records = [
                {
                    "source": "soldier_001",
                    "data": {"heart_rate": 72, "temperature": 98.6},
                    "metadata": {"unit": "alpha"},
                },
                {
                    "source": "soldier_002",
                    "data": {"heart_rate": 85, "temperature": 99.1},
                    "metadata": {"unit": "bravo"},
                },
            ]

            result = store.add_records(records)
            assert result == 2  # Number of records inserted

            # Verify records are in database
            count = store.execute("SELECT COUNT(*) FROM telemetry")
            assert count[0][0] == 2
            store.close()

    def test_add_records_with_custom_timestamp(self):
        """Test add_records accepts custom timestamps."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            custom_ts = datetime(2024, 1, 15, 10, 30, 0)
            records = [
                {
                    "source": "soldier_001",
                    "ts": custom_ts,
                    "data": {"heart_rate": 72},
                },
            ]

            store.add_records(records)

            result = store.execute("SELECT ts FROM telemetry")
            assert result[0][0].year == 2024
            assert result[0][0].month == 1
            assert result[0][0].day == 15
            store.close()

    def test_add_records_batch_insert_performance(self):
        """Test batch insert handles 1000+ records efficiently."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            # Generate 1000 records
            records = [
                {
                    "source": f"soldier_{i:03d}",
                    "data": {
                        "heart_rate": 70 + (i % 30),
                        "temperature": 98.0 + (i % 5) * 0.2,
                    },
                }
                for i in range(1000)
            ]

            start = time.time()
            result = store.add_records(records)
            elapsed = time.time() - start

            assert result == 1000
            assert elapsed < 5.0  # Should complete in under 5 seconds
            store.close()


class TestDuckDBStoreTimeRangeQueries:
    """Test time-range query operations."""

    def test_query_time_range_retrieves_correct_records(self):
        """Test query_time_range retrieves data within time window."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            # Insert records with different timestamps
            base_time = datetime.now()
            records = [
                {
                    "source": "s1",
                    "ts": base_time - timedelta(minutes=10),
                    "data": {"value": 1},
                },
                {
                    "source": "s2",
                    "ts": base_time - timedelta(minutes=5),
                    "data": {"value": 2},
                },
                {
                    "source": "s3",
                    "ts": base_time - timedelta(minutes=2),
                    "data": {"value": 3},
                },
                {"source": "s4", "ts": base_time, "data": {"value": 4}},
            ]
            store.add_records(records)

            # Query last 3 minutes
            results = store.query_time_range(
                start_time=base_time - timedelta(minutes=3),
                end_time=base_time + timedelta(minutes=1),
            )

            assert len(results) == 2  # s3 and s4
            sources = [r["source"] for r in results]
            assert "s3" in sources
            assert "s4" in sources
            store.close()

    def test_query_time_range_with_source_filter(self):
        """Test query_time_range can filter by source."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            base_time = datetime.now()
            records = [
                {"source": "soldier_001", "ts": base_time, "data": {"value": 1}},
                {"source": "soldier_002", "ts": base_time, "data": {"value": 2}},
                {"source": "soldier_001", "ts": base_time, "data": {"value": 3}},
            ]
            store.add_records(records)

            results = store.query_time_range(
                start_time=base_time - timedelta(minutes=1),
                end_time=base_time + timedelta(minutes=1),
                source="soldier_001",
            )

            assert len(results) == 2
            assert all(r["source"] == "soldier_001" for r in results)
            store.close()

    def test_query_time_range_with_rolling_aggregation(self):
        """Test query with rolling average calculation."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            base_time = datetime.now()
            records = [
                {
                    "source": "s1",
                    "ts": base_time - timedelta(seconds=i),
                    "data": {"value": 10 * i},
                }
                for i in range(10)
            ]
            store.add_records(records)

            results = store.query_time_range(
                start_time=base_time - timedelta(minutes=1),
                end_time=base_time + timedelta(minutes=1),
                aggregation="rolling_avg",
                window_size=3,
            )

            # Should have rolling averages calculated
            assert len(results) == 10
            assert "rolling_avg" in results[0] or "value" in results[0]
            store.close()


class TestDuckDBStoreSpatialQueries:
    """Test spatial/geo query operations."""

    @pytest.mark.integration
    def test_query_spatial_finds_records_within_radius(self):
        """Test query_spatial finds records within distance radius."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "path": f"{temp_dir}/test.duckdb",
                "extensions": ["spatial"],
                "enable_spatial": True,
            }
            store = DuckDBStore(config=config)

            # Insert records with locations (lat, lon)
            records = [
                {
                    "source": "near",
                    "data": {"value": 1},
                    "location": {"lat": 37.7749, "lon": -122.4194},  # SF
                },
                {
                    "source": "far",
                    "data": {"value": 2},
                    "location": {"lat": 40.7128, "lon": -74.0060},  # NYC
                },
                {
                    "source": "close",
                    "data": {"value": 3},
                    "location": {"lat": 37.8044, "lon": -122.2712},  # Oakland
                },
            ]
            store.add_records(records)

            # Query within 50km of SF
            results = store.query_spatial(
                center_lat=37.7749,
                center_lon=-122.4194,
                radius_meters=50000,  # 50km
            )

            assert len(results) == 2  # near and close
            sources = [r["source"] for r in results]
            assert "near" in sources
            assert "close" in sources
            assert "far" not in sources
            store.close()

    @pytest.mark.integration
    def test_query_spatial_returns_distance(self):
        """Test query_spatial returns distance in results."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "path": f"{temp_dir}/test.duckdb",
                "extensions": ["spatial"],
                "enable_spatial": True,
            }
            store = DuckDBStore(config=config)

            records = [
                {
                    "source": "point1",
                    "data": {"value": 1},
                    "location": {"lat": 37.7749, "lon": -122.4194},
                },
            ]
            store.add_records(records)

            results = store.query_spatial(
                center_lat=37.7849,  # Slightly north
                center_lon=-122.4194,
                radius_meters=5000,
            )

            assert len(results) == 1
            assert "distance_meters" in results[0]
            # Distance should be roughly 1.1km (0.01 degrees lat ~ 1.1km)
            assert results[0]["distance_meters"] < 2000
            store.close()


class TestDuckDBStoreErrorHandling:
    """Test error handling and edge cases."""

    def test_handles_connection_error_gracefully(self):
        """Test store handles invalid path gracefully."""
        from components.stores.duckdb_store import DuckDBStore

        # Try to create store in non-existent nested directory
        config = {"path": "/nonexistent/deeply/nested/path/test.duckdb"}

        # Should either create the path or raise a clear error
        try:
            store = DuckDBStore(config=config)
            # If it succeeds, it should have created the directory
            store.close()
        except Exception as e:
            # Should be a clear, descriptive error
            assert "path" in str(e).lower() or "directory" in str(e).lower()

    def test_handles_empty_records_list(self):
        """Test add_records handles empty list gracefully."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            result = store.add_records([])
            assert result == 0
            store.close()

    def test_handles_malformed_record_data(self):
        """Test add_records handles malformed data gracefully."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            # Record missing required fields
            records = [{"invalid": "data"}]

            # Should handle gracefully - either skip or use defaults
            result = store.add_records(records)
            assert result >= 0  # Should not crash
            store.close()

    def test_query_empty_time_range_returns_empty_list(self):
        """Test querying empty time range returns empty list."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            # Query with no data
            results = store.query_time_range(
                start_time=datetime.now() - timedelta(hours=1),
                end_time=datetime.now(),
            )

            assert results == []
            store.close()

    def test_close_is_idempotent(self):
        """Test close() can be called multiple times safely."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            store.close()
            store.close()  # Should not raise
            store.close()  # Should not raise


class TestDuckDBStoreDeleteOperations:
    """Test delete operations for memory management."""

    def test_delete_by_id(self):
        """Test deleting records by ID."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            records = [
                {"source": "s1", "data": {"value": 1}},
                {"source": "s2", "data": {"value": 2}},
            ]
            store.add_records(records)

            # Get the IDs
            result = store.execute("SELECT id FROM telemetry")
            ids = [row[0] for row in result]

            # Delete first record
            deleted = store.delete_records([ids[0]])
            assert deleted == 1

            # Verify only one record remains
            count = store.execute("SELECT COUNT(*) FROM telemetry")
            assert count[0][0] == 1
            store.close()

    def test_delete_by_time_range(self):
        """Test deleting records older than a timestamp."""
        from components.stores.duckdb_store import DuckDBStore

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"path": f"{temp_dir}/test.duckdb"}
            store = DuckDBStore(config=config)

            base_time = datetime.now()
            records = [
                {
                    "source": "old",
                    "ts": base_time - timedelta(days=2),
                    "data": {"value": 1},
                },
                {"source": "recent", "ts": base_time, "data": {"value": 2}},
            ]
            store.add_records(records)

            # Delete records older than 1 day
            deleted = store.delete_older_than(base_time - timedelta(days=1))
            assert deleted == 1

            # Verify only recent record remains
            result = store.execute("SELECT source FROM telemetry")
            assert result[0][0] == "recent"
            store.close()
