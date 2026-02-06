---
title: Streaming Anomaly Detection
sidebar_position: 4
---

# Streaming Anomaly Detection

For real-time data streams, LlamaFarm provides a streaming anomaly detection API that handles the complexities of online learning: cold start, automatic retraining, sliding windows, and seamless model updates.

## Overview

The streaming API is designed for scenarios where:
- Data arrives continuously (sensors, logs, metrics)
- You need immediate anomaly scores
- The underlying data distribution may drift over time
- You can't wait to collect a large batch before training.

**Key Features:**
- **Cold Start Handling**: Collects initial samples before first training
- **Tick-Tock Retraining**: Seamlessly swaps models during retrain
- **Sliding Window**: Maintains recent history, discards old data
- **Automatic Feature Engineering**: Optional Polars-based rolling features
- **12 PyOD Backends**: Choose the right algorithm for your data

## Quick Start

```bash
# Send streaming data point
curl -X POST http://localhost:14345/v1/ml/anomaly/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sensor-stream",
    "data": {"temperature": 72.5, "humidity": 45.2, "pressure": 1013.25},
    "backend": "ecod",
    "min_samples": 50,
    "retrain_interval": 100,
    "window_size": 1000,
    "threshold": 0.5
  }'
```

**During cold start (collecting samples):**
```json
{
  "object": "streaming_result",
  "model": "sensor-stream",
  "status": "collecting",
  "results": [
    {"index": 0, "score": null, "is_anomaly": null, "samples_until_ready": 49}
  ],
  "model_version": 0,
  "samples_collected": 1,
  "samples_until_ready": 49
}
```

**After model is trained:**
```json
{
  "object": "streaming_result",
  "model": "sensor-stream",
  "status": "ready",
  "results": [
    {"index": 0, "score": 0.32, "is_anomaly": false, "raw_score": 0.28, "samples_until_ready": 0}
  ],
  "model_version": 1,
  "samples_collected": 150,
  "samples_until_ready": 0
}
```

---

## How Streaming Works

### The Tick-Tock Pattern

LlamaFarm uses a tick-tock pattern for seamless model updates:

```
┌──────────────────────────────────────────────────────────┐
│  Time ──────────────────────────────────────────────────>│
│                                                          │
│  Model A: [ACTIVE]────────────────[ACTIVE]──────────    │
│                    \                     \               │
│  Model B:          [TRAINING]──[ACTIVE]   [TRAINING]──  │
│                                                          │
│  Data:    ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●    │
│           ↑                    ↑                         │
│           Retrain triggered    Retrain triggered         │
└──────────────────────────────────────────────────────────┘
```

1. **Active Model**: Scores incoming data immediately
2. **Background Training**: New model trains on updated window
3. **Atomic Swap**: Once trained, new model becomes active
4. **Zero Downtime**: Scoring continues throughout

### Lifecycle States

| Status | Description | Scoring |
|--------|-------------|---------|
| `collecting` | Cold start - gathering initial samples | Returns `null` scores |
| `ready` | Model trained and scoring | Returns anomaly scores |
| `retraining` | Background retrain in progress | Returns scores (using current model) |

---

## Parameters

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | "default-stream" | Unique identifier for this detector |
| `data` | dict or list | **required** | Single data point or batch of points |
| `backend` | string | "ecod" | PyOD algorithm (see Backend Selection) |
| `min_samples` | int | 50 | Samples needed before first training |
| `retrain_interval` | int | 100 | Retrain after this many new samples |
| `window_size` | int | 1000 | Sliding window size (keeps most recent N) |
| `threshold` | float | 0.5 | Anomaly score threshold (0-1) |
| `contamination` | float | 0.1 | Expected proportion of anomalies (0-0.5) |

### Rolling Feature Parameters (Polars-Powered)

Enable automatic feature engineering using high-performance Polars:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rolling_windows` | list[int] | null | Window sizes, e.g., [5, 10, 20] |
| `include_lags` | bool | false | Include lag features |
| `lag_periods` | list[int] | null | Lag periods, e.g., [1, 2, 3] |

When `rolling_windows` is provided, the detector computes:
- `{col}_rolling_mean_{window}` - Rolling mean
- `{col}_rolling_std_{window}` - Rolling standard deviation
- `{col}_rolling_min_{window}` - Rolling minimum
- `{col}_rolling_max_{window}` - Rolling maximum
- `{col}_lag_{period}` - Lagged values (if `include_lags=true`)

**Why Rolling Features Matter:**

A raw value like `$500` is meaningless without context. The model needs to know: "Is $500 normal for this user?" Rolling features provide this context automatically.

**Polars Performance Advantage:**

| Feature Engineering | Pandas | Polars |
|--------------------|--------|--------|
| Rolling std (10K rows) | 10-50ms | \<1ms |
| Multi-column parallel | Sequential | Parallel (all CPU cores) |
| Memory on append | Copy each time | Arrow (no-copy) |

**Cold Start Handling:**

During the first few samples, rolling statistics have insufficient history. Polars automatically fills null values with `0.0`, ensuring your model always receives valid numeric vectors—no crashes from NaN values.

**Example with Rolling Features:**

```bash
curl -X POST http://localhost:14345/v1/ml/anomaly/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fraud-detector",
    "data": {"amount": 500, "merchant_category": 5},
    "backend": "ecod",
    "min_samples": 50,
    "window_size": 1000,
    "rolling_windows": [5, 10, 20],
    "include_lags": true,
    "lag_periods": [1, 2, 5],
    "schema": {"amount": "numeric", "merchant_category": "numeric"}
  }'
```

This transforms each transaction from 2 features to **26 features**:
- 2 original: `amount`, `merchant_category`
- 24 rolling: 4 stats × 3 windows × 2 columns
- Plus lag features if enabled

See [Polars Buffer API](./polars-buffers.md) for detailed information about the underlying Polars mechanics.

---

## Backend Selection

Choose the right algorithm for your use case:

### Recommended for Streaming

| Backend | Speed | Why Use It |
|---------|-------|------------|
| `ecod` | ⚡⚡⚡ | **Recommended default** - Fast, parameter-free, handles drift |
| `hbos` | ⚡⚡⚡⚡ | Fastest algorithm, great for high-throughput |
| `loda` | ⚡⚡⚡ | Designed for streaming, lightweight |
| `copod` | ⚡⚡⚡ | Parameter-free, interpretable |

### When to Use Other Backends

| Backend | Use Case |
|---------|----------|
| `isolation_forest` | General purpose, reliable baseline |
| `knn` | When local density matters |
| `cblof` | Data has natural clusters |
| `suod` | Maximum robustness (ensemble of multiple algorithms) |
| `autoencoder` | Complex non-linear patterns (slower) |

### All 12 Backends

```bash
# Get full backend information
curl http://localhost:14345/v1/ml/anomaly/backends
```

| Backend | Category | Speed | Memory | Best For |
|---------|----------|-------|--------|----------|
| `isolation_forest` | Legacy | Fast | Low | General purpose |
| `one_class_svm` | Legacy | Slow | Medium | Tight boundaries |
| `local_outlier_factor` | Legacy | Medium | Medium | Density anomalies |
| `autoencoder` | Deep Learning | Slow | High | Complex patterns |
| `ecod` | Fast | Very Fast | Low | Streaming (recommended) |
| `hbos` | Fast | Very Fast | Low | High throughput |
| `copod` | Fast | Very Fast | Low | Interpretability |
| `knn` | Distance | Medium | Medium | Density-based |
| `mcd` | Distance | Medium | Medium | Gaussian data |
| `cblof` | Clustering | Medium | Medium | Clustered data |
| `suod` | Ensemble | Medium | Medium | Maximum robustness |
| `loda` | Streaming | Fast | Low | Online scenarios |

---

## Managing Streaming Detectors

### List All Active Detectors

```bash
curl http://localhost:14345/v1/ml/anomaly/stream/detectors
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "model_id": "sensor-stream",
      "backend": "ecod",
      "status": "ready",
      "model_version": 3,
      "samples_collected": 450,
      "total_processed": 450,
      "samples_since_retrain": 50,
      "is_ready": true
    }
  ],
  "total": 1
}
```

### Get Specific Detector Stats

```bash
curl http://localhost:14345/v1/ml/anomaly/stream/sensor-stream
```

### Reset Detector

Clear all data and restart cold start:

```bash
curl -X POST http://localhost:14345/v1/ml/anomaly/stream/sensor-stream/reset
```

### Delete Detector

Remove detector and free memory:

```bash
curl -X DELETE http://localhost:14345/v1/ml/anomaly/stream/sensor-stream
```

---

## Batch Processing

Send multiple data points in one request:

```bash
curl -X POST http://localhost:14345/v1/ml/anomaly/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sensor-stream",
    "data": [
      {"temperature": 72.5, "humidity": 45.2},
      {"temperature": 73.1, "humidity": 44.8},
      {"temperature": 71.9, "humidity": 46.0},
      {"temperature": 150.0, "humidity": 10.0}
    ],
    "backend": "ecod"
  }'
```

Response includes results for each point:
```json
{
  "object": "streaming_result",
  "model": "sensor-stream",
  "status": "ready",
  "results": [
    {"index": 0, "score": 0.28, "is_anomaly": false, "raw_score": 0.22},
    {"index": 1, "score": 0.31, "is_anomaly": false, "raw_score": 0.25},
    {"index": 2, "score": 0.26, "is_anomaly": false, "raw_score": 0.21},
    {"index": 3, "score": 0.89, "is_anomaly": true, "raw_score": 0.85}
  ],
  "model_version": 2,
  "threshold": 0.5
}
```

---

## With Rolling Features

Enable automatic feature engineering for time-series patterns:

```bash
curl -X POST http://localhost:14345/v1/ml/anomaly/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "enhanced-sensor",
    "data": {"temperature": 72.5, "humidity": 45.2},
    "backend": "ecod",
    "rolling_windows": [5, 10, 20],
    "include_lags": true,
    "lag_periods": [1, 2, 3],
    "min_samples": 30
  }'
```

This automatically computes and includes:
- `temperature_rolling_mean_5`, `temperature_rolling_mean_10`, `temperature_rolling_mean_20`
- `temperature_rolling_std_5`, `temperature_rolling_std_10`, `temperature_rolling_std_20`
- `temperature_lag_1`, `temperature_lag_2`, `temperature_lag_3`
- Same for humidity

The detector learns patterns like "temperature suddenly diverging from recent mean" or "humidity spiking compared to 5 minutes ago".

---

## Python Client Example

```python
import httpx
import asyncio
import random

async def stream_sensor_data():
    """Simulate streaming sensor data with occasional anomalies."""
    async with httpx.AsyncClient() as client:
        base_url = "http://localhost:14345/v1/ml/anomaly/stream"

        for i in range(200):
            # Normal data
            temp = 72 + random.gauss(0, 2)
            humidity = 45 + random.gauss(0, 3)

            # Inject anomaly every 50 samples
            if i > 0 and i % 50 == 0:
                temp = 150  # Anomalous spike
                print(f"Injecting anomaly at sample {i}")

            response = await client.post(base_url, json={
                "model": "python-sensor",
                "data": {"temperature": temp, "humidity": humidity},
                "backend": "ecod",
                "min_samples": 30,
                "retrain_interval": 50,
                "threshold": 0.5
            })

            result = response.json()

            if result["status"] == "ready":
                for r in result["results"]:
                    if r["is_anomaly"]:
                        print(f"⚠️  ANOMALY at sample {i}: score={r['score']:.3f}")
            else:
                print(f"Collecting... {result['samples_until_ready']} samples until ready")

            await asyncio.sleep(0.1)  # 10 samples/second

if __name__ == "__main__":
    asyncio.run(stream_sensor_data())
```

---

## Integration with Polars Buffers

For advanced use cases, you can use Polars buffers directly alongside streaming detection:

```python
import httpx

async def advanced_streaming():
    async with httpx.AsyncClient() as client:
        base_url = "http://localhost:14345/v1/ml"

        # Create a Polars buffer for custom feature engineering
        await client.post(f"{base_url}/polars/buffers", json={
            "buffer_id": "custom-features",
            "window_size": 500
        })

        for data_point in data_stream:
            # Append to Polars buffer
            await client.post(f"{base_url}/polars/append", json={
                "buffer_id": "custom-features",
                "data": data_point
            })

            # Get custom features
            features_resp = await client.post(f"{base_url}/polars/features", json={
                "buffer_id": "custom-features",
                "rolling_windows": [5, 10],
                "include_lags": True,
                "tail": 1  # Just the latest row
            })

            if features_resp.json()["rows"] > 0:
                enhanced_data = features_resp.json()["data"][0]

                # Stream to anomaly detector with enhanced features
                await client.post(f"{base_url}/anomaly/stream", json={
                    "model": "custom-enhanced",
                    "data": enhanced_data,
                    "backend": "ecod"
                })
```

See [Polars Buffer API](./polars-buffers.md) for more details on direct buffer access.

---

## Best Practices

### Choosing Parameters

1. **min_samples**: Set high enough for meaningful training
   - Simple univariate: 30-50
   - Multivariate (5-10 features): 100-200
   - Complex patterns: 500+

2. **retrain_interval**: Balance freshness vs. stability
   - Fast-changing data: 50-100
   - Stable distributions: 500-1000
   - Very stable: 5000+

3. **window_size**: Balance memory vs. history
   - Typical: 1000-10000
   - Memory constrained: 500
   - Long patterns: 50000+

4. **threshold**: Start with 0.5 and adjust
   - More sensitive: 0.3-0.4
   - Fewer false positives: 0.6-0.7

### Performance Tips

1. **Use batch requests** when possible (fewer HTTP calls)
2. **Choose fast backends** (ecod, hbos) for high-throughput
3. **Limit rolling windows** to what you need
4. **Set appropriate window_size** to bound memory

### Handling Concept Drift

The sliding window and periodic retraining naturally handle drift:
- Old data falls out of the window
- New data influences the model
- Retraining incorporates new patterns

For sudden distribution shifts:
```bash
# Reset and restart cold start
curl -X POST http://localhost:14345/v1/ml/anomaly/stream/my-detector/reset
```

---

## Next Steps

- [Polars Buffer API](./polars-buffers.md) - Direct access to data buffers
- [Anomaly Detection Guide](./anomaly-detection.md) - Batch anomaly detection
- [Use Cases: IoT Sensor Monitoring](../use-cases/iot-sensor-monitoring.md) - Complete example
