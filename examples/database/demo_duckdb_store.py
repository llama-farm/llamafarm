#!/usr/bin/env python3
"""DuckDB Store Demo - Time-Series and Spatial Queries.

Demonstrates:
1. Time-series data storage with timestamps
2. Batch ingestion performance
3. Time-range queries with window functions
4. Spatial queries (find records within radius)
5. Data retention policies

This is part of the Embedded Trinity Memory System.
"""

import sys
import tempfile
from datetime import datetime, timedelta

# Add rag to path
sys.path.insert(0, ".")


def print_section(title: str) -> None:
    print(f"\n\033[0;34m{'─' * 50}\033[0m")
    print(f"\033[0;34m{title}\033[0m")
    print(f"\033[0;34m{'─' * 50}\033[0m")


def print_success(msg: str) -> None:
    print(f"\033[0;32m✓\033[0m {msg}")


def print_info(msg: str) -> None:
    print(f"\033[0;36m→\033[0m {msg}")


def main() -> int:
    from components.stores.duckdb_store import DuckDBStore

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = f"{temp_dir}/demo_timeseries.duckdb"

        # ─────────────────────────────────────────────────
        # Step 1: Initialize DuckDB Store
        # ─────────────────────────────────────────────────
        print_section("Step 1: Initialize DuckDB Store")

        config = {"path": db_path}
        store = DuckDBStore(config=config)
        print_success(f"DuckDB Store initialized at: {db_path}")

        # ─────────────────────────────────────────────────
        # Step 2: Add Time-Series Records
        # ─────────────────────────────────────────────────
        print_section("Step 2: Add Time-Series Records")

        now = datetime.now()

        # Simulated biometric telemetry
        records = []
        for i in range(10):
            ts = now - timedelta(minutes=10 - i)
            records.append({
                "source": "biometric",
                "data": {"heart_rate": 70 + i, "blood_oxygen": 98 - i * 0.1},
                "timestamp": ts,
                "latitude": 35.7796 + (i * 0.001),
                "longitude": -78.6382 + (i * 0.001),
                "metadata": {"device": "watch", "user_id": "soldier_001"},
            })

        count = store.add_records(records)
        print_success(f"Added {count} biometric records")

        # Add some location updates
        location_records = []
        for i in range(5):
            ts = now - timedelta(minutes=5 - i)
            location_records.append({
                "source": "location",
                "data": {"checkpoint": f"delta_{i}", "status": "passed"},
                "timestamp": ts,
                "latitude": 35.7800 + (i * 0.002),
                "longitude": -78.6400 + (i * 0.002),
                "metadata": {"vehicle": "humvee_07"},
            })

        count = store.add_records(location_records)
        print_success(f"Added {count} location records")

        # ─────────────────────────────────────────────────
        # Step 3: Time-Range Queries
        # ─────────────────────────────────────────────────
        print_section("Step 3: Time-Range Queries")

        # Query last 5 minutes
        start_time = now - timedelta(minutes=5)
        results = store.query_time_range(
            start_time=start_time,
            end_time=now,
            source="biometric",
        )
        print_success(f"Found {len(results)} biometric records in last 5 minutes")

        if results:
            print_info("Sample record:")
            r = results[0]
            print(f"    Time: {r['timestamp']}")
            print(f"    Data: {r['data']}")
            print(f"    Location: ({r.get('latitude')}, {r.get('longitude')})")

        # ─────────────────────────────────────────────────
        # Step 4: Spatial Queries
        # ─────────────────────────────────────────────────
        print_section("Step 4: Spatial Queries")

        # Find records within 1km of a point
        center_lat, center_lon = 35.7800, -78.6390
        results = store.query_spatial(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_meters=1000,
        )
        print_success(f"Found {len(results)} records within 1km of ({center_lat}, {center_lon})")

        for r in results[:3]:
            print_info(f"  {r['source']}: record at {r.get('ts', 'N/A')}")

        # ─────────────────────────────────────────────────
        # Step 5: Statistics
        # ─────────────────────────────────────────────────
        print_section("Step 5: Statistics")

        stats = store.get_stats()
        print_success(f"Total records: {stats['record_count']}")
        print_success(f"Unique sources: {stats.get('unique_sources', 0)}")
        print_success(f"Oldest record: {stats.get('oldest_record', 'N/A')}")
        print_success(f"Newest record: {stats.get('newest_record', 'N/A')}")

        # ─────────────────────────────────────────────────
        # Step 6: Data Retention
        # ─────────────────────────────────────────────────
        print_section("Step 6: Data Retention")

        # Delete records older than 8 minutes
        cutoff = now - timedelta(minutes=8)
        deleted = store.delete_older_than(cutoff)
        print_success(f"Deleted {deleted} records older than 8 minutes")

        # Verify
        stats = store.get_stats()
        print_success(f"Remaining records: {stats['record_count']}")

        # ─────────────────────────────────────────────────
        # Cleanup
        # ─────────────────────────────────────────────────
        store.close()
        print_success("\nDuckDB Store closed")

    print("")
    print(f"\033[0;32m{'═' * 50}\033[0m")
    print("\033[0;32m  Demo completed successfully!\033[0m")
    print(f"\033[0;32m{'═' * 50}\033[0m")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(f"\033[0;31m✗ Import error: {e}\033[0m")
        print(
            "\033[0;33m  Run from the 'rag' directory: cd rag && uv run python ../examples/database/demo_duckdb_store.py\033[0m"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\033[0;31m✗ Error: {e}\033[0m")
        import traceback

        traceback.print_exc()
        sys.exit(1)
