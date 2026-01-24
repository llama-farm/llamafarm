#!/usr/bin/env python3
"""Quick Start: Anomaly Detection in 5 Lines

This is the simplest possible anomaly detection example.
No configuration, no tuning - just detect anomalies.

Requirements:
    - Universal Runtime running: nx start universal-runtime

Run:
    cd /path/to/llamafarm/runtimes/universal
    uv run python ../../examples/anomaly/01_quick_start.py
"""

import httpx

# Training data - normal examples
train_data = [
    [1.0, 2.0], [1.1, 2.1], [0.9, 1.9], [1.2, 2.2], [0.8, 1.8],
    [1.0, 2.0], [1.1, 2.1], [0.9, 1.9], [1.2, 2.2], [0.8, 1.8],
]

# Test data - includes an anomaly
test_data = [
    [1.0, 2.0],      # Normal
    [1.1, 2.1],      # Normal
    [0.9, 1.9],      # Normal
    [100.0, 200.0],  # Anomaly!
    [1.0, 2.0],      # Normal
]

# Step 1: Fit the model on normal data only
client = httpx.Client(timeout=30)
client.post(
    "http://localhost:11545/v1/anomaly/fit",
    json={"data": train_data, "backend": "ecod", "model": "quickstart"},
)

# Step 2: Score test data (includes anomaly)
response = client.post(
    "http://localhost:11545/v1/anomaly/score",
    json={"data": test_data, "backend": "ecod", "model": "quickstart"},
)
result = response.json()

print("=== Quick Start: Anomaly Detection ===\n")
for item in result["data"]:
    original = test_data[item["index"]]
    status = "ANOMALY!" if item["is_anomaly"] else "Normal"
    print(f"Point {item['index']} {original}: {status} (score: {item['score']:.3f})")

print(f"\nSummary: {result['summary']['anomaly_count']} anomalies found "
      f"out of {result['summary']['total_points']} points")
