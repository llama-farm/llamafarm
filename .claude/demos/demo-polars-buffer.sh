#!/bin/bash
# Demo: Polars Buffer for Streaming ML
#
# This demo shows:
# 1. Creating a PolarsBuffer with sliding window
# 2. Streaming 1000 transactions through buffer
# 3. Computing rolling features
# 4. Performance metrics
#
# This is a local-only demo (no server required) since PolarsBuffer is a utility class.

set -e

echo "========================================"
echo "Polars Buffer for Streaming ML Demo"
echo "========================================"
echo ""

# Run the demo using Python directly
cd /Users/robthelen/llamafarm-head/llamafarm/runtimes/universal

uv run python3 << 'PYTHON_DEMO'
import time
import random
from utils.polars_buffer import PolarsBuffer
from utils.rolling_features import RollingFeatureConfig, compute_features, compute_anomaly_features

print("=" * 50)
print("Step 1: Create PolarsBuffer")
print("=" * 50)

buffer = PolarsBuffer(window_size=1000)
print(f"Created buffer with window_size={buffer.window_size}")
print(f"Initial size: {buffer.size}")
print()

print("=" * 50)
print("Step 2: Stream 1000 Transactions")
print("=" * 50)

# Generate synthetic transaction data
random.seed(42)
start_time = time.perf_counter()

for i in range(1000):
    # Normal transaction
    amount = random.gauss(100, 20)
    count = random.randint(1, 10)
    time_ms = i * 100

    # Inject some anomalies at specific points
    if i in [250, 500, 750]:
        amount = random.gauss(500, 50)  # Large anomaly

    buffer.append({
        "time_ms": time_ms,
        "amount": amount,
        "count": count,
        "category": random.choice(["A", "B", "C"]),
    })

elapsed_ms = (time.perf_counter() - start_time) * 1000
print(f"Streamed 1000 transactions in {elapsed_ms:.2f}ms")
print(f"Average per transaction: {elapsed_ms/1000:.4f}ms")
print(f"Buffer size: {buffer.size}")
print()

print("=" * 50)
print("Step 3: Compute Rolling Features")
print("=" * 50)

start_time = time.perf_counter()
df = buffer.get_features(
    rolling_windows=[5, 10, 20],
    include_lags=True,
    lag_periods=[1, 2, 3],
)
elapsed_ms = (time.perf_counter() - start_time) * 1000

print(f"Computed features in {elapsed_ms:.2f}ms")
print(f"Original columns: {['time_ms', 'amount', 'count', 'category']}")
print(f"Total columns after features: {len(df.columns)}")
print()

# Show sample of feature columns
feature_cols = [c for c in df.columns if "rolling" in c or "lag" in c]
print(f"Rolling feature columns (sample):")
for col in feature_cols[:8]:
    print(f"  - {col}")
print(f"  ... and {len(feature_cols) - 8} more")
print()

print("=" * 50)
print("Step 4: Compute Anomaly Features")
print("=" * 50)

start_time = time.perf_counter()
df_anomaly = compute_anomaly_features(
    buffer.get_data(),
    value_columns=["amount"],
    windows=[10, 20, 50],
)
elapsed_ms = (time.perf_counter() - start_time) * 1000

print(f"Computed anomaly features in {elapsed_ms:.2f}ms")
anomaly_cols = [c for c in df_anomaly.columns if "zscore" in c or "deviation" in c]
print(f"Anomaly detection columns:")
for col in anomaly_cols:
    print(f"  - {col}")
print()

print("=" * 50)
print("Step 5: Get Latest Records with Features")
print("=" * 50)

latest = buffer.get_latest(n=5, with_features=True, rolling_windows=[5])
print("Latest 5 records with rolling features:")
print(latest.select(["time_ms", "amount", "amount_rolling_mean_5", "amount_rolling_std_5"]))
print()

print("=" * 50)
print("Step 6: Buffer Statistics")
print("=" * 50)

stats = buffer.get_stats()
print(f"Buffer size: {stats.size}")
print(f"Window size: {stats.window_size}")
print(f"Memory usage: {stats.memory_bytes / 1024:.2f} KB")
print(f"Total appends: {stats.append_count}")
print(f"Average append time: {stats.avg_append_ms:.4f}ms")
print(f"Numeric columns: {stats.numeric_columns}")
print()

print("=" * 50)
print("Step 7: Window Truncation Test")
print("=" * 50)

# Add more data to test truncation
buffer.append_batch([{"time_ms": i, "amount": float(i)} for i in range(500)])
print(f"After adding 500 more records: buffer.size = {buffer.size}")
print("Window truncation keeps only the most recent records!")
print()

print("=" * 50)
print("Demo Complete!")
print("=" * 50)
print()
print("Summary:")
print("- PolarsBuffer provides O(1) append with automatic truncation")
print("- Rolling features computed lazily with Polars expressions")
print("- Sub-millisecond append times for streaming ML workloads")
print("- Anomaly detection features (z-scores, deviations) built-in")
print()
print("Use PolarsBuffer with PyOD for real-time anomaly detection!")
PYTHON_DEMO

echo ""
echo "Demo finished successfully!"
