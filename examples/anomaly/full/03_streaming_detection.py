#!/usr/bin/env python3
"""
Step 3: Streaming Anomaly Detection

Demonstrates real-time anomaly detection with:
- Cold start handling
- Automatic retraining
- Anomaly injection (every 50 samples)
- Rolling features via Polars
- Detector statistics

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
MODEL_ID = "factory-stream"

DETECTOR_CONFIG = {
    "model": MODEL_ID,
    "backend": "ecod",           # Fast, parameter-free
    "min_samples": 50,           # Samples before first training
    "retrain_interval": 100,     # Retrain every 100 samples
    "window_size": 500,          # Keep last 500 samples
    "threshold": 0.5,            # Anomaly threshold
    "contamination": 0.05,       # Expected 5% anomalies
    # Schema for features
    "schema": {
        "temperature": "numeric",
        "humidity": "numeric",
        "pressure": "numeric",
        "motor_rpm": "numeric"
    },
    # Rolling features for trend detection
    "rolling_windows": [5, 10, 20],
    "include_lags": True,
    "lag_periods": [1, 2, 5],
}

# Demo parameters
NUM_SAMPLES = 200
ANOMALY_INTERVAL = 50  # Inject anomaly every N samples
SAMPLES_PER_SECOND = 10


def generate_sensor_reading(inject_anomaly: bool = False) -> dict:
    """Generate sensor reading, optionally injecting an anomaly."""
    if inject_anomaly:
        # Anomalous readings
        return {
            "temperature": round(random.choice([50, 95, 120, 150]), 2),
            "humidity": round(random.choice([10, 15, 85, 95]), 2),
            "pressure": round(random.choice([900, 950, 1080, 1100]), 2),
            "motor_rpm": round(random.choice([1500, 2000, 4500, 5000]), 1),
        }
    else:
        # Normal readings
        return {
            "temperature": round(72 + random.gauss(0, 2), 2),
            "humidity": round(45 + random.gauss(0, 3), 2),
            "pressure": round(1013 + random.gauss(0, 5), 2),
            "motor_rpm": round(3000 + random.gauss(0, 50), 1),
        }


async def stream_detection():
    """Run streaming anomaly detection demo."""
    print("=" * 60)
    print("Step 3: Streaming Anomaly Detection")
    print("=" * 60)
    print()
    print(f"Model: {MODEL_ID}")
    print(f"Backend: {DETECTOR_CONFIG['backend']}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Anomaly injection: every {ANOMALY_INTERVAL} samples")
    print()

    # Explain Polars feature engineering
    num_cols = 4  # temperature, humidity, pressure, motor_rpm
    num_windows = len(DETECTOR_CONFIG.get('rolling_windows', []))
    num_lags = len(DETECTOR_CONFIG.get('lag_periods', []))
    rolling_features = num_cols * num_windows * 4  # 4 stats: mean, std, min, max
    lag_features = num_cols * num_lags if DETECTOR_CONFIG.get('include_lags') else 0
    total_features = num_cols + rolling_features + lag_features

    print("Polars Feature Engineering:")
    print(f"  Original features: {num_cols}")
    print(f"  Rolling windows: {DETECTOR_CONFIG.get('rolling_windows', [])} → {rolling_features} features")
    print(f"  Lag periods: {DETECTOR_CONFIG.get('lag_periods', [])} → {lag_features} features")
    print(f"  Total features per sample: {total_features}")
    print()
    print("  Benefits: SIMD vectorization + parallel execution + cold start handling")
    print()
    print("Starting stream...")
    print("-" * 60)

    async with httpx.AsyncClient(timeout=30.0) as client:
        sample_count = 0
        anomaly_count = 0
        detected_count = 0
        missed_count = 0
        false_positives = 0
        last_model_version = 0

        while sample_count < NUM_SAMPLES:
            sample_count += 1

            # Decide if we should inject an anomaly
            inject_anomaly = (
                sample_count > DETECTOR_CONFIG["min_samples"] and
                sample_count % ANOMALY_INTERVAL == 0
            )

            sensor_data = generate_sensor_reading(inject_anomaly)

            if inject_anomaly:
                anomaly_count += 1
                print(f"\n💉 INJECTING ANOMALY #{anomaly_count}: "
                      f"temp={sensor_data['temperature']}, rpm={sensor_data['motor_rpm']}")

            # Send to streaming detector
            response = await client.post(
                f"{LLAMAFARM_URL}/anomaly/stream",
                json={**DETECTOR_CONFIG, "data": sensor_data}
            )

            if response.status_code != 200:
                print(f"❌ Error: {response.text}")
                continue

            result = response.json()

            # Check for model version changes
            if result["model_version"] > last_model_version:
                if last_model_version > 0:
                    print(f"\n🔄 Model retrained: version {result['model_version']} "
                          f"(samples: {result['samples_collected']})")
                last_model_version = result["model_version"]

            # Handle results based on status
            if result["status"] == "collecting":
                if sample_count % 10 == 0:
                    print(f"⏳ Collecting... {result['samples_until_ready']} samples until ready")
            else:
                for r in result["results"]:
                    # Check if we have detection results (ready state)
                    if "is_anomaly" not in r:
                        continue
                    if r["is_anomaly"]:
                        if inject_anomaly:
                            detected_count += 1
                            print(f"🚨 DETECTED! Sample {sample_count}: "
                                  f"score={r['score']:.3f} ✅ TRUE POSITIVE")
                        else:
                            false_positives += 1
                            if sample_count % 20 == 0:  # Don't spam false positives
                                print(f"⚠️  False positive at sample {sample_count}: "
                                      f"score={r['score']:.3f}")
                    elif inject_anomaly:
                        missed_count += 1
                        print(f"❌ MISSED! Sample {sample_count}: "
                              f"score={r['score']:.3f}")

                # Periodic status update
                if sample_count % 50 == 0:
                    print(f"\n📊 Progress: {sample_count}/{NUM_SAMPLES} samples, "
                          f"version={result['model_version']}")

            await asyncio.sleep(1.0 / SAMPLES_PER_SECOND)

        # Final statistics
        print()
        print("-" * 60)
        print("RESULTS")
        print("-" * 60)
        print(f"Total samples processed: {sample_count}")
        print(f"Anomalies injected: {anomaly_count}")
        print(f"Anomalies detected: {detected_count}")
        print(f"Anomalies missed: {missed_count}")
        print(f"False positives: {false_positives}")
        if anomaly_count > 0:
            print(f"Detection rate: {detected_count/anomaly_count*100:.1f}%")
        print()

        # Get detector statistics
        print("Detector Statistics:")
        stats_response = await client.get(f"{LLAMAFARM_URL}/anomaly/stream/{MODEL_ID}")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"  Status: {stats['status']}")
            print(f"  Model version: {stats['model_version']}")
            print(f"  Samples collected: {stats['samples_collected']}")
            print(f"  Total processed: {stats['total_processed']}")
            print(f"  Samples since retrain: {stats['samples_since_retrain']}")
        print()

        # Cleanup option
        print("Cleaning up detector...")
        await client.delete(f"{LLAMAFARM_URL}/anomaly/stream/{MODEL_ID}")
        print("✅ Detector deleted")
        print()
        print("Streaming demo complete! Try 04_polars_features.py for direct buffer access.")


def main():
    asyncio.run(stream_detection())


if __name__ == "__main__":
    main()
