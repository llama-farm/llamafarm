#!/usr/bin/env python3
"""
Step 4: Polars Buffer API Demo

Demonstrates direct access to Polars data buffers for:
- Creating named buffers
- Appending streaming data
- Computing rolling features
- Getting buffer statistics

This is useful for:
- Custom feature engineering pipelines
- Integration with external ML systems
- Manual control over data lifecycle

Prerequisites:
- LlamaFarm servers running (nx start universal-runtime && nx start server)
"""

import asyncio
import os
import random
import httpx
from datetime import datetime
from pathlib import Path

# Configuration - uses environment variable or .env file, falls back to default
def get_llamafarm_url():
    """Get LlamaFarm server URL from environment or .env file."""
    if url := os.environ.get("LLAMAFARM_URL"):
        return url.rstrip("/") + "/v1/ml"
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("LLAMAFARM_URL="):
                    url = line.split("=", 1)[1].strip().strip('"\'')
                    return url.rstrip("/") + "/v1/ml"
    return "http://localhost:8000/v1/ml"

LLAMAFARM_URL = get_llamafarm_url()
BUFFER_ID = "sensor-analysis"
WINDOW_SIZE = 100
NUM_SAMPLES = 50


def generate_sensor_reading() -> dict:
    """Generate a sensor reading."""
    return {
        "timestamp": datetime.now().isoformat(),
        "temperature": round(72 + random.gauss(0, 2), 2),
        "humidity": round(45 + random.gauss(0, 3), 2),
        "pressure": round(1013 + random.gauss(0, 5), 2),
    }


async def polars_demo():
    """Demonstrate Polars buffer API."""
    print("=" * 60)
    print("Step 4: Polars Buffer API Demo")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Create a buffer
        print(f"Creating buffer '{BUFFER_ID}' with window_size={WINDOW_SIZE}...")
        create_response = await client.post(
            f"{LLAMAFARM_URL}/polars/buffers",
            json={
                "buffer_id": BUFFER_ID,
                "window_size": WINDOW_SIZE
            }
        )

        if create_response.status_code == 409:
            print("Buffer already exists, clearing it...")
            clear_response = await client.post(f"{LLAMAFARM_URL}/polars/buffers/{BUFFER_ID}/clear")
            if clear_response.status_code != 200:
                print(f"❌ Error clearing buffer: {clear_response.text}")
                return
        elif create_response.status_code != 200:
            print(f"❌ Error creating buffer: {create_response.text}")
            return

        print("✅ Buffer created")
        print()

        # Step 2: Append data in batches
        print(f"Appending {NUM_SAMPLES} samples...")
        batch_size = 10
        for batch_start in range(0, NUM_SAMPLES, batch_size):
            batch = [generate_sensor_reading() for _ in range(batch_size)]

            append_response = await client.post(
                f"{LLAMAFARM_URL}/polars/append",
                json={
                    "buffer_id": BUFFER_ID,
                    "data": batch
                }
            )

            if append_response.status_code == 200:
                result = append_response.json()
                print(f"   Batch {batch_start//batch_size + 1}: "
                      f"appended {result['appended']}, "
                      f"total size: {result['buffer_size']}, "
                      f"avg append: {result['avg_append_ms']:.3f}ms")
            else:
                print(f"   ❌ Error: {append_response.text}")

        print()

        # Step 3: Get buffer statistics
        print("Buffer Statistics:")
        stats_response = await client.get(f"{LLAMAFARM_URL}/polars/buffers/{BUFFER_ID}")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"  Buffer ID: {stats['buffer_id']}")
            print(f"  Size: {stats['size']} records")
            print(f"  Window size: {stats['window_size']}")
            print(f"  Columns: {', '.join(stats['columns'])}")
            print(f"  Numeric columns: {', '.join(stats['numeric_columns'])}")
            print(f"  Memory: {stats['memory_bytes']} bytes")
            print(f"  Total appends: {stats['append_count']}")
            print(f"  Avg append time: {stats['avg_append_ms']:.3f}ms")
        print()

        # Step 4: Compute rolling features
        print("Computing rolling features...")
        features_response = await client.post(
            f"{LLAMAFARM_URL}/polars/features",
            json={
                "buffer_id": BUFFER_ID,
                "rolling_windows": [5, 10, 20],
                "include_rolling_stats": ["mean", "std", "min", "max"],
                "include_lags": True,
                "lag_periods": [1, 2, 5],
                "tail": 5  # Just last 5 rows
            }
        )

        if features_response.status_code == 200:
            features = features_response.json()
            print(f"✅ Computed features for {features['rows']} rows")
            print(f"   Total columns: {len(features['columns'])}")
            print()

            # Show column types
            original_cols = ["timestamp", "temperature", "humidity", "pressure"]
            rolling_cols = [c for c in features['columns'] if "rolling" in c]
            lag_cols = [c for c in features['columns'] if "lag" in c]

            print("   Column breakdown:")
            print(f"     Original: {len(original_cols)}")
            print(f"     Rolling features: {len(rolling_cols)}")
            print(f"     Lag features: {len(lag_cols)}")
            print()

            # Show sample of computed features
            print("   Sample rolling features (last row):")
            if features['data']:
                last_row = features['data'][-1]
                for col in sorted(rolling_cols)[:6]:  # Show first 6
                    value = last_row.get(col)
                    if value is not None:
                        print(f"     {col}: {value:.3f}")
        else:
            print(f"❌ Error computing features: {features_response.text}")
        print()

        # Step 5: Get raw data
        print("Getting raw data (last 5 rows)...")
        data_response = await client.get(
            f"{LLAMAFARM_URL}/polars/buffers/{BUFFER_ID}/data",
            params={"tail": 5}
        )

        if data_response.status_code == 200:
            data = data_response.json()
            print(f"✅ Retrieved {data['rows']} rows")
            print()
            print("   Last 5 readings:")
            for i, row in enumerate(data['data']):
                print(f"     {i+1}. temp={row['temperature']:.1f}, "
                      f"humid={row['humidity']:.1f}, "
                      f"pressure={row['pressure']:.1f}")
        print()

        # Step 6: List all buffers
        print("Listing all active buffers...")
        list_response = await client.get(f"{LLAMAFARM_URL}/polars/buffers")
        if list_response.status_code == 200:
            buffers = list_response.json()
            print(f"Found {buffers['total']} buffer(s):")
            for buf in buffers['data']:
                print(f"   - {buf['buffer_id']}: {buf['size']} records, "
                      f"{buf['memory_bytes']} bytes")
        print()

        # Step 7: Cleanup
        print("Cleaning up...")
        await client.delete(f"{LLAMAFARM_URL}/polars/buffers/{BUFFER_ID}")
        print(f"✅ Buffer '{BUFFER_ID}' deleted")
        print()
        print("Polars demo complete!")


def main():
    asyncio.run(polars_demo())


if __name__ == "__main__":
    main()
