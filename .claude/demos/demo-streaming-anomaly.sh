#!/bin/bash
# Demo: Streaming Anomaly Detection with Polars + PyOD
#
# This demo shows:
# 1. Cold start phase (collecting initial samples)
# 2. Warm-up transition (model trained)
# 3. Real-time anomaly detection
# 4. Auto-rolling retraining
# 5. Rolling features computed automatically
#
# Architecture: Polars (data substrate) + PyOD (ML utensils)
#
# This is a local-only demo (no server required).

set -e

echo "========================================"
echo "Streaming Anomaly Detection Demo"
echo "Polars + PyOD Integration"
echo "========================================"
echo ""

# Run the demo using Python directly
cd /Users/robthelen/llamafarm-head/llamafarm/runtimes/universal

uv run python3 << 'PYTHON_DEMO'
import asyncio
import random
import time

from models.streaming_anomaly import StreamingAnomalyDetector, DetectorStatus


async def main():
    print("=" * 50)
    print("Step 1: Create Streaming Detector")
    print("=" * 50)
    print()

    # Create detector with configuration
    detector = StreamingAnomalyDetector(
        model_id="fraud-detector-demo",
        backend="ecod",  # Fast, parameter-free
        min_samples=50,  # Cold start threshold
        retrain_interval=30,  # Retrain every 30 samples
        window_size=200,  # Keep last 200 samples
        threshold=0.6,  # Anomaly threshold
    )

    print(f"Model ID: {detector.model_id}")
    print(f"Backend: {detector.backend}")
    print(f"Min samples for warm-up: {detector.min_samples}")
    print(f"Retrain interval: {detector.retrain_interval}")
    print(f"Window size: {detector.window_size}")
    print(f"Initial status: {detector.status.value}")
    print()

    print("=" * 50)
    print("Step 2: Cold Start Phase")
    print("=" * 50)
    print()

    random.seed(42)

    # Stream normal transactions during cold start
    print("Streaming transactions during cold start...")
    for i in range(50):
        # Normal transaction pattern
        amount = random.gauss(100, 15)
        count = random.randint(1, 5)

        result = await detector.process({
            "amount": amount,
            "count": float(count),
        })

        if i % 10 == 0:
            print(f"  Transaction {i+1}: status={result.status.value}, samples_until_ready={result.samples_until_ready}")

    print()
    print(f"Cold start complete! Status: {detector.status.value}")
    print(f"Model version: {detector.model_version}")
    print()

    print("=" * 50)
    print("Step 3: Real-Time Detection")
    print("=" * 50)
    print()

    # Process normal transactions
    print("Processing 10 normal transactions...")
    for i in range(10):
        amount = random.gauss(100, 15)
        result = await detector.process({"amount": amount, "count": float(random.randint(1, 5))})
        if result.is_anomaly:
            print(f"  Transaction {i+1}: ANOMALY detected! score={result.score:.4f}")
        else:
            print(f"  Transaction {i+1}: Normal (score={result.score:.4f})")

    print()

    print("=" * 50)
    print("Step 4: Injecting Anomalies")
    print("=" * 50)
    print()

    # Inject obvious anomalies
    anomalies = [
        {"amount": 5000.0, "count": 50.0},  # Very large amount
        {"amount": 10000.0, "count": 100.0},  # Extremely large
        {"amount": -500.0, "count": 1.0},  # Negative amount
    ]

    print("Injecting 3 obvious anomalies...")
    for i, anomaly_data in enumerate(anomalies):
        result = await detector.process(anomaly_data)
        status = "ANOMALY!" if result.is_anomaly else "Normal"
        print(f"  Anomaly {i+1} ({anomaly_data}): {status} (score={result.score:.4f})")

    print()

    print("=" * 50)
    print("Step 5: Auto-Rolling Retraining")
    print("=" * 50)
    print()

    initial_version = detector.model_version
    print(f"Current model version: {initial_version}")
    print(f"Streaming more transactions to trigger retrain...")
    print()

    # Process enough to trigger retrain
    for i in range(40):
        amount = random.gauss(100, 15)
        result = await detector.process({"amount": amount, "count": float(random.randint(1, 5))})

        # Check if retrain happened
        if detector.model_version > initial_version:
            print(f"  Transaction {i+1}: Retrain triggered!")
            print(f"  Model version: {initial_version} -> {detector.model_version}")
            break

    # Wait for background retrain to complete
    await asyncio.sleep(0.5)
    print()
    print(f"Final model version: {detector.model_version}")
    print()

    print("=" * 50)
    print("Step 6: Detector Statistics")
    print("=" * 50)
    print()

    stats = detector.get_stats()
    print(f"Model ID: {stats['model_id']}")
    print(f"Backend: {stats['backend']}")
    print(f"Status: {stats['status']}")
    print(f"Model version: {stats['model_version']}")
    print(f"Samples collected: {stats['samples_collected']}")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Samples since retrain: {stats['samples_since_retrain']}")
    print(f"Is ready: {stats['is_ready']}")
    print()

    print("=" * 50)
    print("Step 7: Rolling Features Demo")
    print("=" * 50)
    print()

    # Create detector with rolling features
    detector_with_features = StreamingAnomalyDetector(
        model_id="rolling-features-demo",
        backend="ecod",
        min_samples=30,
        rolling_windows=[5, 10],  # Compute rolling mean/std for windows 5 and 10
        include_lags=True,
        lag_periods=[1, 2],
    )

    print(f"Created detector with rolling features:")
    print(f"  Rolling windows: {detector_with_features.rolling_windows}")
    print(f"  Include lags: {detector_with_features.include_lags}")
    print(f"  Lag periods: {detector_with_features.lag_periods}")
    print()

    # Warm up
    print("Warming up detector...")
    for i in range(35):
        await detector_with_features.process({"value": float(i * 10 + random.gauss(0, 5))})

    print(f"Status: {detector_with_features.status.value}")
    print(f"Ready for scoring with rolling features!")
    print()

    # Score a few more
    for i in range(5):
        result = await detector_with_features.process({"value": float(i * 10 + random.gauss(0, 5))})
        print(f"  Sample {i+1}: score={result.score:.4f}")

    print()
    print("=" * 50)
    print("Demo Complete!")
    print("=" * 50)
    print()
    print("Summary:")
    print("- Polars is the data substrate (internal, automatic)")
    print("- PyOD provides anomaly detection algorithms")
    print("- Cold start collects min_samples before first model")
    print("- Auto-rolling retraining keeps model fresh")
    print("- Rolling features computed automatically when configured")
    print()
    print("Use /v1/anomaly/stream endpoint for API access!")

asyncio.run(main())
PYTHON_DEMO

echo ""
echo "Demo finished successfully!"
