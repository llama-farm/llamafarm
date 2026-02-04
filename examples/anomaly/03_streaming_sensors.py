#!/usr/bin/env python3
"""Streaming Sensor Monitoring: Real-Time IoT Anomaly Detection

This example demonstrates streaming anomaly detection for IoT sensor data.
Uses the StreamingAnomalyDetector for continuous monitoring with:

1. Cold start phase (collects initial samples)
2. Automatic warm-up (trains first model)
3. Real-time scoring with auto-retraining
4. Rolling features for temporal patterns

This runs locally without requiring the server.

Requirements:
    - Install dependencies: uv add polars pyod

Run:
    cd /path/to/llamafarm/runtimes/universal
    uv run python ../../examples/anomaly/03_streaming_sensors.py
"""

import asyncio
import random
import sys
from pathlib import Path

# Add the runtime to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "runtimes" / "universal"))

from models.streaming_anomaly import StreamingAnomalyDetector, DetectorStatus


async def simulate_sensor(
    detector: StreamingAnomalyDetector,
    sensor_name: str,
    duration_samples: int = 100,
):
    """Simulate sensor readings with occasional anomalies."""
    print(f"\n{'='*60}")
    print(f"SENSOR: {sensor_name}")
    print(f"{'='*60}")

    anomaly_indices = {20, 45, 75, 90}  # Inject anomalies at these positions

    for i in range(duration_samples):
        # Generate sensor reading
        if i in anomaly_indices:
            # Anomaly: spike or drop
            temp = random.choice([random.uniform(80, 100), random.uniform(-20, 0)])
            vibration = random.uniform(5, 10)  # High vibration
        else:
            # Normal reading
            temp = random.gauss(45, 5)  # Normal temp around 45C
            vibration = random.gauss(0.5, 0.2)  # Low vibration

        # Process through detector
        result = await detector.process({
            "temperature": temp,
            "vibration": vibration,
        })

        # Report status during cold start
        if result.status == DetectorStatus.COLLECTING:
            if i % 10 == 0:
                print(f"[Cold Start] Sample {i+1}: "
                      f"Collecting data... {result.samples_until_ready} more needed")
            continue

        # Report anomalies during detection
        if result.is_anomaly:
            expected = "EXPECTED" if i in anomaly_indices else "UNEXPECTED"
            print(f"[ANOMALY] Sample {i+1}: "
                  f"temp={temp:.1f}C, vib={vibration:.2f} | "
                  f"score={result.score:.3f} ({expected})")

        # Report retraining
        if result.status == DetectorStatus.RETRAINING:
            print(f"[Retrain] Model updating in background (version {result.model_version})...")

    # Final stats
    stats = detector.get_stats()
    print(f"\n[Stats] Processed: {stats['total_processed']}, "
          f"Model Version: {stats['model_version']}, "
          f"Status: {stats['status']}")


async def main():
    print("=" * 60)
    print("STREAMING SENSOR MONITORING")
    print("Real-Time IoT Anomaly Detection")
    print("=" * 60)

    # Example 1: Basic streaming detector
    print("\n" + "=" * 60)
    print("Example 1: Basic Temperature/Vibration Monitoring")
    print("=" * 60)

    detector = StreamingAnomalyDetector(
        model_id="sensor-monitor",
        backend="ecod",  # Fast, parameter-free
        min_samples=30,  # Cold start threshold
        retrain_interval=50,  # Retrain every 50 new samples
        window_size=200,  # Keep last 200 samples
        threshold=0.7,  # Higher threshold = fewer false positives
    )

    print(f"Detector Config:")
    print(f"  Backend: {detector.backend}")
    print(f"  Min samples for warm-up: {detector.min_samples}")
    print(f"  Retrain interval: {detector.retrain_interval}")
    print(f"  Window size: {detector.window_size}")
    print(f"  Anomaly threshold: {detector.threshold}")

    random.seed(42)
    await simulate_sensor(detector, "Factory-Pump-001", duration_samples=100)

    # Example 2: Streaming with rolling features
    print("\n" + "=" * 60)
    print("Example 2: Rolling Features for Temporal Patterns")
    print("=" * 60)

    detector_with_features = StreamingAnomalyDetector(
        model_id="sensor-temporal",
        backend="hbos",  # Histogram-based, very fast
        min_samples=40,
        retrain_interval=60,
        window_size=300,
        threshold=0.65,
        # Rolling feature configuration
        rolling_windows=[5, 10],  # Compute mean/std over 5 and 10 samples
        include_lags=True,  # Include lagged values
        lag_periods=[1, 2],  # t-1 and t-2 values
    )

    print(f"Detector Config (with rolling features):")
    print(f"  Backend: {detector_with_features.backend}")
    print(f"  Rolling windows: {detector_with_features.rolling_windows}")
    print(f"  Lag periods: {detector_with_features.lag_periods}")
    print(f"  This captures temporal patterns like sudden changes!")

    random.seed(123)
    await simulate_sensor(detector_with_features, "HVAC-Unit-042", duration_samples=100)

    # Example 3: Multiple sensors
    print("\n" + "=" * 60)
    print("Example 3: Multi-Sensor Dashboard")
    print("=" * 60)

    sensors = {
        "Compressor-A": StreamingAnomalyDetector(
            model_id="compressor-a",
            backend="ecod",
            min_samples=20,
            retrain_interval=40,
        ),
        "Compressor-B": StreamingAnomalyDetector(
            model_id="compressor-b",
            backend="ecod",
            min_samples=20,
            retrain_interval=40,
        ),
    }

    print("Simulating 50 readings from 2 compressors...")
    random.seed(456)

    for i in range(50):
        for name, det in sensors.items():
            # Normal readings with occasional spikes for compressor B
            pressure = random.gauss(100, 5)
            if name == "Compressor-B" and i in {25, 35}:
                pressure = random.uniform(150, 180)  # Pressure spike

            result = await det.process({"pressure": pressure})

            if result.is_anomaly:
                print(f"[ALERT] {name} @ sample {i}: "
                      f"pressure={pressure:.1f} PSI | score={result.score:.3f}")

    print("\nDashboard Summary:")
    for name, det in sensors.items():
        stats = det.get_stats()
        print(f"  {name}: {stats['total_processed']} samples, "
              f"version {stats['model_version']}, status={stats['status']}")

    print("\n" + "=" * 60)
    print("STREAMING ANOMALY DETECTION COMPLETE!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Cold start phase collects initial data automatically")
    print("- Models retrain in background without blocking detection")
    print("- Rolling features capture temporal patterns")
    print("- Multiple detectors can monitor different sensors")


if __name__ == "__main__":
    asyncio.run(main())
