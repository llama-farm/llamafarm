---
title: "IoT Sensor Monitoring"
sidebar_position: 2
---

# IoT Sensor Monitoring with Streaming Anomaly Detection

This use case demonstrates how to build a production-ready IoT sensor monitoring system using LlamaFarm's streaming anomaly detection and Polars buffer APIs.

## Scenario

You're monitoring a fleet of industrial sensors (temperature, humidity, pressure) that produce data every second. You need to:

1. Detect equipment malfunctions in real-time
2. Handle concept drift as seasons change
3. Alert operators to anomalies within seconds
4. Run on edge devices with limited resources

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Sensors    │───▶│  LlamaFarm  │───▶│   Alerts    │
│ (1000/sec)  │    │  Streaming  │    │  Dashboard  │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                   ┌──────┴──────┐
                   │  Polars     │
                   │  Buffer     │
                   │  (rolling)  │
                   └─────────────┘
```

## Why These Components?

| Component | Why |
|-----------|-----|
| **Streaming API** | Handles cold start, auto-retraining, sliding window |
| **ECOD Backend** | Fast (5000+ samples/sec), parameter-free, handles drift |
| **Polars Buffers** | Sub-millisecond append, efficient rolling features |
| **Rolling Features** | Detect gradual trends and sudden spikes |

## Backend Selection Guide

For IoT sensor monitoring:

| Backend | When to Use | Speed | Accuracy |
|---------|-------------|-------|----------|
| `ecod` | **Default choice** - works great for most sensors | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ |
| `hbos` | Ultra-high throughput (>10k samples/sec) | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ |
| `isolation_forest` | Complex multi-sensor correlations | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| `loda` | Memory-constrained edge devices | ⚡⚡⚡⚡ | ⭐⭐⭐ |

**For this use case, we'll use `ecod`** because:
- Parameter-free (no tuning needed)
- Fast enough for 1000 samples/second
- Handles seasonal drift naturally

## Step 1: Configure the Detector

```python
# sensor_monitor.py
import httpx
import asyncio
import random
from datetime import datetime

# Configuration
LLAMAFARM_URL = "http://localhost:14345/v1/ml"
MODEL_ID = "factory-sensors"
BACKEND = "ecod"

# Streaming parameters tuned for IoT
DETECTOR_CONFIG = {
    "model": MODEL_ID,
    "backend": BACKEND,
    "min_samples": 100,        # Enough for initial pattern learning
    "retrain_interval": 500,   # Retrain every 500 samples (~8 minutes)
    "window_size": 3600,       # Keep 1 hour of history
    "threshold": 0.5,          # Balanced sensitivity
    "contamination": 0.05,     # Expect ~5% anomalies
    # Enable rolling features for trend detection
    "rolling_windows": [10, 60, 300],  # 10s, 1min, 5min windows
    "include_lags": True,
    "lag_periods": [1, 5, 10]  # Compare to recent values
}
```

## Step 2: Create the Streaming Loop

```python
async def stream_sensor_data():
    """Main streaming loop for sensor monitoring."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        sample_count = 0
        anomaly_count = 0

        while True:
            # Simulate sensor reading
            sensor_data = read_sensors()

            # Send to streaming detector
            response = await client.post(
                f"{LLAMAFARM_URL}/anomaly/stream",
                json={**DETECTOR_CONFIG, "data": sensor_data}
            )
            result = response.json()

            sample_count += 1

            # Check status
            if result["status"] == "collecting":
                print(f"Cold start: {result['samples_until_ready']} samples until ready")
            else:
                # Process results
                for r in result["results"]:
                    if r["is_anomaly"]:
                        anomaly_count += 1
                        await send_alert(sensor_data, r["score"])

            # Log progress every 100 samples
            if sample_count % 100 == 0:
                print(f"Processed: {sample_count}, Anomalies: {anomaly_count}, "
                      f"Model version: {result['model_version']}")

            await asyncio.sleep(1)  # 1 sample per second


def read_sensors() -> dict:
    """Simulate sensor readings (replace with real sensor code)."""
    # Normal operating ranges
    temp = 72 + random.gauss(0, 2)      # 72°F ± 2
    humidity = 45 + random.gauss(0, 3)  # 45% ± 3
    pressure = 1013 + random.gauss(0, 5)  # 1013 hPa ± 5

    # Simulate occasional real anomalies
    if random.random() < 0.02:  # 2% chance
        temp = random.choice([50, 95])  # Equipment malfunction

    return {
        "temperature": round(temp, 2),
        "humidity": round(humidity, 2),
        "pressure": round(pressure, 2),
        "timestamp": datetime.now().isoformat()
    }


async def send_alert(data: dict, score: float):
    """Send alert to monitoring system."""
    print(f"🚨 ANOMALY DETECTED! Score: {score:.3f}")
    print(f"   Data: temp={data['temperature']}, humidity={data['humidity']}")
    # In production: send to PagerDuty, Slack, email, etc.
```

## Step 3: Understanding the Rolling Features

When `rolling_windows` is enabled, the detector automatically computes:

| Feature | Window | What It Detects |
|---------|--------|-----------------|
| `temperature_rolling_mean_10` | 10 samples | Short-term average |
| `temperature_rolling_std_10` | 10 samples | Short-term volatility |
| `temperature_rolling_mean_60` | 60 samples | 1-minute trend |
| `temperature_rolling_mean_300` | 5 minutes | Longer baseline |
| `temperature_lag_1` | Previous | Sudden jumps |
| `temperature_lag_10` | 10 ago | Compare to recent |

**Why this matters:**
- A sensor reading of 80°F might be normal if it's been trending up
- The same 80°F is anomalous if it jumped from 72°F suddenly
- Rolling features capture this context automatically

## Step 4: Handling Concept Drift

Sensors naturally drift with:
- Seasonal temperature changes
- Equipment aging
- Calibration shifts

The streaming API handles this automatically:

```python
# The sliding window naturally adapts
window_size = 3600  # Keep 1 hour

# Periodic retraining incorporates new patterns
retrain_interval = 500  # Every ~8 minutes

# Result: the model continuously adapts to gradual changes
# while still detecting sudden anomalies
```

**For sudden operational changes** (e.g., switching to night shift):

```python
# Reset the detector to restart learning
await client.post(f"{LLAMAFARM_URL}/anomaly/stream/{MODEL_ID}/reset")
print("Detector reset - starting fresh learning phase")
```

## Step 5: Monitoring Detector Health

```python
async def monitor_detector_health():
    """Periodically check detector status."""
    async with httpx.AsyncClient() as client:
        while True:
            response = await client.get(
                f"{LLAMAFARM_URL}/anomaly/stream/{MODEL_ID}"
            )
            stats = response.json()

            print(f"""
            Detector Health:
            - Status: {stats['status']}
            - Model Version: {stats['model_version']}
            - Samples Collected: {stats['samples_collected']}
            - Total Processed: {stats['total_processed']}
            - Samples Since Retrain: {stats['samples_since_retrain']}
            - Is Ready: {stats['is_ready']}
            """)

            await asyncio.sleep(60)  # Check every minute
```

## Step 6: Using Polars Buffers for Custom Analysis

For advanced analysis beyond anomaly detection:

```python
async def analyze_with_polars(sensor_data: dict):
    """Use Polars buffers for custom feature engineering.

    Args:
        sensor_data: Sensor reading dict with values like temperature, humidity, etc.
    """
    async with httpx.AsyncClient() as client:
        # Create dedicated analysis buffer
        await client.post(f"{LLAMAFARM_URL}/polars/buffers", json={
            "buffer_id": "sensor-analysis",
            "window_size": 3600  # 1 hour
        })

        # In your streaming loop, also append to analysis buffer
        await client.post(f"{LLAMAFARM_URL}/polars/append", json={
            "buffer_id": "sensor-analysis",
            "data": sensor_data
        })

        # Get custom features for dashboard/reporting
        features = await client.post(f"{LLAMAFARM_URL}/polars/features", json={
            "buffer_id": "sensor-analysis",
            "rolling_windows": [60, 300, 900],  # 1min, 5min, 15min
            "include_rolling_stats": ["mean", "std", "min", "max"],
            "include_lags": True,
            "tail": 100  # Last 100 readings with features
        })

        return features.json()["data"]
```

## Complete Example

```python
#!/usr/bin/env python3
"""Complete IoT sensor monitoring example."""

import asyncio
import httpx
import random
from datetime import datetime

LLAMAFARM_URL = "http://localhost:14345/v1/ml"
MODEL_ID = "factory-sensors"

async def main():
    """Run the sensor monitoring system."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("🏭 Starting IoT Sensor Monitor")
        print(f"   Model: {MODEL_ID}")
        print(f"   Backend: ecod")
        print("-" * 50)

        sample_count = 0
        anomaly_count = 0

        while True:
            # Generate sensor reading
            temp = 72 + random.gauss(0, 2)
            humidity = 45 + random.gauss(0, 3)
            pressure = 1013 + random.gauss(0, 5)

            # Inject anomaly every ~50 samples
            if sample_count > 100 and random.random() < 0.02:
                temp = random.choice([50, 95])
                print(f"💉 Injecting anomaly: temp={temp}")

            sensor_data = {
                "temperature": round(temp, 2),
                "humidity": round(humidity, 2),
                "pressure": round(pressure, 2)
            }

            # Stream to detector
            response = await client.post(
                f"{LLAMAFARM_URL}/anomaly/stream",
                json={
                    "model": MODEL_ID,
                    "data": sensor_data,
                    "backend": "ecod",
                    "min_samples": 50,
                    "retrain_interval": 100,
                    "window_size": 1000,
                    "threshold": 0.5,
                    "rolling_windows": [5, 10, 20],
                    "include_lags": True
                }
            )
            result = response.json()
            sample_count += 1

            # Handle response
            if result["status"] == "collecting":
                if sample_count % 10 == 0:
                    print(f"⏳ Collecting... {result['samples_until_ready']} until ready")
            else:
                for r in result["results"]:
                    if r["is_anomaly"]:
                        anomaly_count += 1
                        print(f"🚨 ANOMALY #{anomaly_count}: score={r['score']:.3f}, "
                              f"data={sensor_data}")

                if sample_count % 50 == 0:
                    print(f"✅ Sample {sample_count}: "
                          f"anomalies={anomaly_count}, "
                          f"version={result['model_version']}")

            await asyncio.sleep(0.1)  # 10 samples/second for demo

if __name__ == "__main__":
    asyncio.run(main())
```

## Performance Considerations

### Edge Device Optimization

For resource-constrained edge devices:

```python
# Use lighter backend
"backend": "loda",  # Lower memory than ecod

# Smaller windows
"window_size": 500,  # 8 minutes instead of 1 hour
"min_samples": 30,   # Faster cold start

# Disable rolling features to save memory
# (let the backend handle pattern detection)
"rolling_windows": None,
"include_lags": False
```

### High-Throughput Optimization

For >1000 samples/second:

```python
# Batch multiple readings
batch = [read_sensors() for _ in range(100)]
await client.post(url, json={**config, "data": batch})

# Use fastest backend
"backend": "hbos",  # Histogram-based, extremely fast

# Increase retrain interval
"retrain_interval": 5000,  # Retrain less frequently
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Too many false positives | Increase `threshold` (0.6-0.7) or `contamination` (0.1-0.2) |
| Missing real anomalies | Decrease `threshold` (0.3-0.4) |
| Slow cold start | Decrease `min_samples` (30-50) |
| Memory growing | Decrease `window_size` |
| Model not adapting | Decrease `retrain_interval` |

## Next Steps

- [Streaming Anomaly Detection](../models/streaming-anomaly-detection.md) - API reference
- [Polars Buffer API](../models/polars-buffers.md) - Custom feature engineering
- [Financial Fraud Detection](./financial-fraud-detection.md) - Another use case
