---
title: Polars Buffer API
sidebar_position: 5
---

# Polars Buffer API

LlamaFarm provides direct access to high-performance Polars-based data buffers for advanced streaming and feature engineering use cases.

## Why Polars Matters for Anomaly Detection

In streaming anomaly detection, **Feature Engineering Latency** is the biggest bottleneck—not model inference:

| Operation | Typical Latency |
|-----------|----------------|
| Isolation Forest inference | ~1ms |
| Rolling std of 10,000 rows (Pandas) | 10-50ms |
| Rolling std of 10,000 rows (Polars) | \<1ms |

**The Problem:** To detect if "$500" is anomalous, the model needs context—"Is $500 normal for this user?" This requires calculating rolling statistics over a history window. In Pandas, this creates a major latency bottleneck.

**The Solution:** Polars is a columnar data store written in Rust with:

1. **Apache Arrow Memory Format** - No copy-on-append, efficient memory allocation
2. **SIMD Vectorization** - Calculate mean of 2,000 numbers in the same CPU cycles as 2 numbers
3. **Parallel Execution** - Compute rolling stats for ALL columns simultaneously across CPU cores
4. **Cold Start Handling** - `fill_null(0.0)` ensures valid feature vectors from the very first transaction

## Overview

Polars buffers are the "data substrate" underlying LlamaFarm's streaming anomaly detection. While most users should use the [Streaming Anomaly Detection API](./streaming-anomaly-detection.md) directly, the Polars Buffer API is available for:

- **Custom feature engineering** pipelines
- **Integration with external ML systems**
- **Manual control** over data lifecycle
- **Building custom streaming analytics**

**Key Features:**
- **High-performance columnar storage** using Polars DataFrames
- **Automatic sliding window** truncation (memory never grows indefinitely)
- **Rolling feature computation** (mean, std, min, max) with SIMD + parallel execution
- **Lag features** for temporal patterns
- **Sub-millisecond append performance**
- **Cold start handling** with null filling

---

## How the Data Flows

Understanding the internal data flow helps you optimize your pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW VISUALIZATION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1: INIT          →  []                                                │
│                                                                              │
│  Step 2: INGEST        →  {"amount": 100}                                   │
│          (Tick 1)          ↓                                                │
│                         [100]    ← Polars converts dict to 1-row DataFrame  │
│                                                                              │
│  Step 3: STACK         →  {"amount": 150}                                   │
│          (Tick 2)          ↓                                                │
│                         [100, 150]  ← pl.concat (Arrow memory - no copy)    │
│                                                                              │
│  ...                                                                         │
│                                                                              │
│  Step 101: TRUNCATE    →  {"amount": 200}                                   │
│            (Tick 101)      ↓                                                │
│                         [150, ..., 200]  ← tail(window_size) drops $100     │
│                                                                              │
│  Step 102: COMPUTE     →  rolling_mean = 175.0                              │
│            (SIMD)          (Calculated instantly with vectorization)        │
│                                                                              │
│  Step 103: EXPORT      →  [200.0, 175.0, ...]                               │
│            (to NumPy)      (Raw value + Context features for model)         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cold Start Behavior

During the first few ticks, rolling features have insufficient history:

| Tick | Buffer | rolling_mean_5 | rolling_std_5 | What Happens |
|------|--------|----------------|---------------|--------------|
| 1 | [100] | 100.0 | **0.0** | std is null → filled with 0.0 |
| 2 | [100, 150] | 125.0 | 35.36 | 2 values, real std |
| 5 | [100, 150, 120, 80, 200] | 130.0 | 45.83 | Full 5-value window |

The `fill_null(0.0)` ensures your model always receives valid numeric vectors, even from the very first transaction. No crashes, no NaN errors.

---

## When to Use Polars Buffers

| Use Case | Recommended API |
|----------|-----------------|
| Real-time anomaly detection | [Streaming Anomaly Detection](./streaming-anomaly-detection.md) |
| Custom feature engineering | **Polars Buffers** |
| Feeding features to external ML | **Polars Buffers** |
| Simple anomaly detection | [Batch Anomaly Detection](./anomaly-detection.md) |
| Rolling statistics only | **Polars Buffers** |

---

## Quick Start

### Create a Buffer

```bash
curl -X POST http://localhost:14345/v1/ml/polars/buffers \
  -H "Content-Type: application/json" \
  -d '{
    "buffer_id": "sensor-data",
    "window_size": 1000
  }'
```

Response:
```json
{
  "object": "buffer_created",
  "buffer_id": "sensor-data",
  "window_size": 1000,
  "status": "created"
}
```

### Append Data

```bash
# Single record
curl -X POST http://localhost:14345/v1/ml/polars/append \
  -H "Content-Type: application/json" \
  -d '{
    "buffer_id": "sensor-data",
    "data": {"temperature": 72.5, "humidity": 45.2, "pressure": 1013.25}
  }'

# Batch append (more efficient)
curl -X POST http://localhost:14345/v1/ml/polars/append \
  -H "Content-Type: application/json" \
  -d '{
    "buffer_id": "sensor-data",
    "data": [
      {"temperature": 72.5, "humidity": 45.2, "pressure": 1013.25},
      {"temperature": 73.1, "humidity": 44.8, "pressure": 1013.10},
      {"temperature": 71.9, "humidity": 46.0, "pressure": 1013.50}
    ]
  }'
```

Response:
```json
{
  "object": "append_result",
  "buffer_id": "sensor-data",
  "appended": 3,
  "buffer_size": 3,
  "avg_append_ms": 0.15
}
```

### Compute Rolling Features

```bash
curl -X POST http://localhost:14345/v1/ml/polars/features \
  -H "Content-Type: application/json" \
  -d '{
    "buffer_id": "sensor-data",
    "rolling_windows": [5, 10, 20],
    "include_rolling_stats": ["mean", "std", "min", "max"],
    "include_lags": true,
    "lag_periods": [1, 2, 3],
    "tail": 5
  }'
```

Response:
```json
{
  "object": "polars_data",
  "buffer_id": "sensor-data",
  "rows": 5,
  "columns": [
    "temperature", "humidity", "pressure",
    "temperature_rolling_mean_5", "temperature_rolling_std_5",
    "temperature_rolling_min_5", "temperature_rolling_max_5",
    "temperature_rolling_mean_10", "temperature_rolling_std_10",
    "humidity_rolling_mean_5", "humidity_rolling_std_5",
    "temperature_lag_1", "temperature_lag_2", "temperature_lag_3",
    "humidity_lag_1", "humidity_lag_2", "humidity_lag_3"
  ],
  "data": [
    {
      "temperature": 72.5,
      "humidity": 45.2,
      "temperature_rolling_mean_5": 72.8,
      "temperature_rolling_std_5": 0.45,
      "temperature_lag_1": 73.1,
      "temperature_lag_2": 71.9
    }
  ]
}
```

---

## API Reference

### Create Buffer

`POST /v1/ml/polars/buffers`

Creates a new named buffer with automatic sliding window.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `buffer_id` | string | Yes | Unique identifier |
| `window_size` | int | No | Max records to keep (default: 1000) |

### List Buffers

`GET /v1/ml/polars/buffers`

Returns all active buffers with statistics.

Response:
```json
{
  "object": "list",
  "data": [
    {
      "buffer_id": "sensor-data",
      "size": 500,
      "window_size": 1000,
      "columns": ["temperature", "humidity", "pressure"],
      "numeric_columns": ["temperature", "humidity", "pressure"],
      "memory_bytes": 12000,
      "append_count": 500,
      "avg_append_ms": 0.12
    }
  ],
  "total": 1
}
```

### Get Buffer Stats

`GET /v1/ml/polars/buffers/{buffer_id}`

Returns detailed statistics for a specific buffer.

### Delete Buffer

`DELETE /v1/ml/polars/buffers/{buffer_id}`

Removes buffer and frees memory.

### Clear Buffer

`POST /v1/ml/polars/buffers/{buffer_id}/clear`

Clears data but keeps buffer structure.

### Append Data

`POST /v1/ml/polars/append`

Appends single record or batch.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `buffer_id` | string | Yes | Target buffer |
| `data` | dict or list | Yes | Single record or batch |

### Compute Features

`POST /v1/ml/polars/features`

Computes rolling statistics and lag features.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer_id` | string | Required | Source buffer |
| `rolling_windows` | list[int] | [5, 10, 20] | Window sizes |
| `include_rolling_stats` | list[str] | ["mean", "std", "min", "max"] | Stats to compute |
| `include_lags` | bool | true | Include lag features |
| `lag_periods` | list[int] | [1, 2, 3] | Lag periods |
| `tail` | int | null | Return only last N rows |

### Get Buffer Data

`GET /v1/ml/polars/buffers/{buffer_id}/data`

Returns raw buffer contents.

Query parameters:
- `tail`: Return only last N rows
- `with_features`: Compute and include rolling features (bool)

---

## Rolling Features Explained

### Rolling Statistics

For each numeric column and window size:

| Feature | Formula | Use Case |
|---------|---------|----------|
| `{col}_rolling_mean_{w}` | Mean of last w values | Smoothing, trend |
| `{col}_rolling_std_{w}` | Std dev of last w values | Volatility, noise |
| `{col}_rolling_min_{w}` | Min of last w values | Lower bound |
| `{col}_rolling_max_{w}` | Max of last w values | Upper bound |

**Example:** With `rolling_windows: [5, 10]` and column `temperature`:
- `temperature_rolling_mean_5` - 5-point moving average
- `temperature_rolling_mean_10` - 10-point moving average
- `temperature_rolling_std_5` - 5-point volatility
- `temperature_rolling_std_10` - 10-point volatility

### Lag Features

| Feature | Description |
|---------|-------------|
| `{col}_lag_1` | Previous value |
| `{col}_lag_2` | Value 2 steps ago |
| `{col}_lag_n` | Value n steps ago |

**Use cases:**
- Detecting sudden changes (current vs lag_1)
- Identifying trends (compare lag_1, lag_2, lag_3)
- Cyclic patterns (compare to periodic lags)

---

## Performance Characteristics

### Append Performance

| Operation | Typical Latency |
|-----------|-----------------|
| Single append | 0.1-0.3 ms |
| Batch append (100 records) | 0.5-1.0 ms |
| Batch append (1000 records) | 2-5 ms |

### Memory Usage

Approximate memory per 1000 records:
- 5 columns: ~40 KB
- 10 columns: ~80 KB
- 20 columns: ~160 KB

Plus overhead for rolling features during computation.

### Window Truncation

The buffer automatically truncates to `window_size`:
- Oldest records are dropped first
- No manual cleanup needed
- Bounded memory usage

---

## Python Example: Custom Feature Pipeline

```python
import httpx
import asyncio

class CustomFeaturePipeline:
    """Build custom features from streaming data."""

    def __init__(self, buffer_id: str = "custom-features"):
        self.buffer_id = buffer_id
        self.base_url = "http://localhost:14345/v1/ml/polars"
        self.client = httpx.AsyncClient()

    async def initialize(self, window_size: int = 500):
        """Create the buffer."""
        await self.client.post(f"{self.base_url}/buffers", json={
            "buffer_id": self.buffer_id,
            "window_size": window_size
        })

    async def add_data(self, data: dict | list):
        """Append data to buffer."""
        return await self.client.post(f"{self.base_url}/append", json={
            "buffer_id": self.buffer_id,
            "data": data
        })

    async def get_features(self, windows: list[int] = None) -> dict:
        """Get latest features."""
        resp = await self.client.post(f"{self.base_url}/features", json={
            "buffer_id": self.buffer_id,
            "rolling_windows": windows or [5, 10, 20],
            "include_lags": True,
            "lag_periods": [1, 2, 5],
            "tail": 1  # Just the latest row
        })
        result = resp.json()
        return result["data"][0] if result["rows"] > 0 else None

    async def get_raw_data(self, tail: int = 10) -> list:
        """Get recent raw data."""
        resp = await self.client.get(
            f"{self.base_url}/buffers/{self.buffer_id}/data",
            params={"tail": tail}
        )
        return resp.json()["data"]

    async def cleanup(self):
        """Delete buffer."""
        await self.client.delete(f"{self.base_url}/buffers/{self.buffer_id}")
        await self.client.aclose()


async def main():
    pipeline = CustomFeaturePipeline()
    await pipeline.initialize(window_size=1000)

    try:
        # Simulate streaming data
        import random
        for i in range(100):
            data = {
                "sensor_id": "temp-001",
                "value": 72 + random.gauss(0, 2),
                "timestamp": i
            }
            await pipeline.add_data(data)

            if i >= 20:  # Need enough data for features
                features = await pipeline.get_features()
                if features:
                    # Use features for your ML model
                    print(f"Sample {i}: rolling_mean_5={features.get('value_rolling_mean_5', 'N/A'):.2f}")

            await asyncio.sleep(0.1)

    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Integration Patterns

### Pattern 1: Polars + External ML

Use Polars for feature engineering, feed to your own model:

```python
# 1. Collect data in Polars buffer
await append_to_buffer(raw_data)

# 2. Get engineered features
features = await get_features(rolling_windows=[5, 10, 30])

# 3. Feed to your scikit-learn/XGBoost/etc model
prediction = your_model.predict([list(features.values())])
```

### Pattern 2: Polars + Streaming Anomaly

Combine custom features with LlamaFarm anomaly detection:

```python
# 1. Build custom features
features = await polars_get_features(buffer_id="custom")

# 2. Stream to anomaly detector
result = await anomaly_stream(
    model="my-detector",
    data=features,
    backend="ecod"
)
```

### Pattern 3: Multiple Feature Sets

Use multiple buffers for different purposes:

```python
# Fast-moving features (short windows)
await append(buffer_id="fast-features", data=raw)
fast = await get_features("fast-features", rolling_windows=[3, 5, 10])

# Slow-moving features (long windows)
await append(buffer_id="slow-features", data=raw)
slow = await get_features("slow-features", rolling_windows=[50, 100, 200])

# Combine for multi-scale analysis
combined = {**fast, **slow}
```

---

## Best Practices

### Buffer Sizing

| Data Rate | Recommended window_size |
|-----------|------------------------|
| 1 sample/sec | 3600 (1 hour) |
| 10 samples/sec | 6000 (10 min) |
| 100 samples/sec | 10000 (100 sec) |
| 1000 samples/sec | 10000-50000 |

### Rolling Window Selection

- **Short windows (3-10)**: Detect sudden changes, noise
- **Medium windows (20-50)**: Trend detection
- **Long windows (100+)**: Baseline comparison

### Memory Management

1. Set appropriate `window_size`
2. Delete unused buffers
3. Use `clear` instead of delete/recreate
4. Monitor with `GET /v1/ml/polars/buffers`

---

## Next Steps

- [Streaming Anomaly Detection](./streaming-anomaly-detection.md) - High-level streaming API
- [Anomaly Detection Guide](./anomaly-detection.md) - Batch detection
- [Use Cases: Financial Fraud Detection](../use-cases/financial-fraud-detection.md) - Complete example
