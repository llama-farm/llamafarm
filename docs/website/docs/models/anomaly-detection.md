---
title: Anomaly Detection Guide
sidebar_position: 3
---

# Anomaly Detection Guide

The Universal Runtime provides production-ready anomaly detection for monitoring APIs, sensors, financial transactions, and any time-series or tabular data.

## Overview

Anomaly detection learns what "normal" looks like from training data and then identifies deviations in new data. LlamaFarm supports:

- **Multiple algorithms**: Isolation Forest, One-Class SVM, Local Outlier Factor, Autoencoder
- **Three normalization methods**: Standardization (0-1), Z-Score (std devs), Raw scores
- **Mixed data types**: Numeric values, categorical features, and combinations
- **Production workflow**: Train, save, load, and score with persistence
- **Feature encoding**: Automatic encoding of categorical data (hash, label, one-hot, etc.)
- **Model versioning**: Automatic timestamped versions with `-latest` resolution

:::tip Using the LlamaFarm API (Recommended)
The LlamaFarm API (`/v1/ml/anomaly/*`) provides the same functionality as the Universal Runtime with added features:
- **Model Versioning**: When `overwrite: false` (default), models are saved with timestamps like `my-model_20251215_160000`
- **Latest Resolution**: Use `model-name-latest` to auto-resolve to the newest version

```bash
# Via LlamaFarm API (port 14345) - with versioning
curl -X POST http://localhost:14345/v1/ml/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{"model": "sensor-monitor", "backend": "isolation_forest", "data": [...], "overwrite": false}'

# Use -latest suffix to auto-resolve
curl -X POST http://localhost:14345/v1/ml/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{"model": "sensor-monitor-latest", "backend": "isolation_forest", "data": [...]}'
```
:::

## Quick Start

### 1. Train on Normal Data

```bash
curl -X POST http://localhost:14345/v1/ml/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sensor-monitor",
    "backend": "isolation_forest",
    "normalization": "zscore",
    "data": [
      [22.1], [23.5], [21.8], [24.2], [22.7],
      [23.1], [21.5], [24.8], [22.3], [23.9]
    ],
    "contamination": 0.1
  }'
```

### 2. Detect Anomalies

```bash
curl -X POST http://localhost:14345/v1/ml/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sensor-monitor",
    "normalization": "zscore",
    "data": [[22.0], [5.0], [50.0], [-10.0]],
    "threshold": 2.0
  }'
```

Response:
```json
{
  "object": "list",
  "data": [
    {"index": 1, "score": 3.25, "raw_score": 0.65},
    {"index": 2, "score": 2.91, "raw_score": 0.64},
    {"index": 3, "score": 3.25, "raw_score": 0.65}
  ],
  "summary": {
    "anomalies_detected": 3,
    "threshold": 2.0
  }
}
```

With z-score normalization, scores represent standard deviations from normal. The readings at 5°C, 50°C, and -10°C are all flagged as anomalies (>2 std devs from the training mean of ~22°C).

---

## Score Normalization Methods

The `normalization` parameter controls how raw anomaly scores are transformed. This is crucial for interpreting results and setting thresholds.

### Comparison Table

| Method | Score Range | Default Threshold | Best For |
|--------|-------------|-------------------|----------|
| `standardization` | 0-1 | 0.5 | General use, bounded scores |
| `zscore` | Unbounded (std devs) | 2.0 | Statistical interpretation |
| `raw` | Backend-specific | 0.0 (set your own) | Debugging, advanced users |

### 1. Standardization (Default)

Sigmoid transformation to a 0-1 range using median and IQR from training data.

```json
{
  "normalization": "standardization",
  "threshold": 0.5
}
```

**How it works:**
- Scores near 0.5 are "normal"
- Scores approaching 1.0 are increasingly anomalous
- Uses median and interquartile range (IQR) for robustness to outliers

**When to use:**
- General-purpose anomaly detection
- When you want bounded, easy-to-interpret scores
- When comparing across different datasets

**Threshold guidance:**
- 0.5: Default, catches moderate anomalies
- 0.6-0.7: More conservative, fewer false positives
- 0.8-0.9: Very conservative, only extreme anomalies

**Example output:**
```
Normal reading (22°C):  score = 0.51
Cold anomaly (5°C):     score = 0.96
Hot anomaly (50°C):     score = 0.95
```

### 2. Z-Score (Standard Deviations)

Scores represent how many standard deviations a point is from the training mean.

```json
{
  "normalization": "zscore",
  "threshold": 2.0
}
```

**How it works:**
- Score of 0 = exactly at the training mean
- Score of 2.0 = 2 standard deviations above normal
- Score of 3.0 = 3 standard deviations (statistically rare)

**When to use:**
- When you want statistically meaningful scores
- When domain experts understand standard deviations
- For scientific/engineering applications
- When you need to explain "how anomalous" something is

**Threshold guidance:**
- 2.0: Unusual (~5% of normal distribution)
- 3.0: Rare (~0.3% of normal distribution)
- 4.0+: Extreme (very rare)

**Example output:**
```
Normal reading (22°C):    score = -0.22 std devs (normal)
Cold anomaly (5°C):       score = 3.25 std devs (rare)
Hot anomaly (50°C):       score = 2.91 std devs (unusual)
Freezing anomaly (-10°C): score = 3.25 std devs (rare)
```

**Key insight:** With z-score, you can immediately see that -10°C and 5°C are equally anomalous (both 3.25 std devs), while 50°C is slightly less extreme (2.91 std devs).

### 3. Raw (Backend Native)

No normalization - returns the backend's native anomaly scores.

```json
{
  "normalization": "raw",
  "threshold": 0.1
}
```

**How it works:**
- Scores are passed through unchanged from the underlying algorithm
- Range and meaning vary by backend (see table below)

**Raw score ranges by backend:**

| Backend | Typical Range | Normal Value | Anomaly Indicator |
|---------|--------------|--------------|-------------------|
| `isolation_forest` | -0.5 to 0.5 | ~0 or negative | Higher = more anomalous |
| `one_class_svm` | Unbounded | ~0 or negative | Higher = more anomalous |
| `local_outlier_factor` | 1 to 10+ | ~1 | Higher = more anomalous |

**When to use:**
- Debugging anomaly detection behavior
- When you understand the specific backend's scoring
- When you need maximum control over thresholding
- For research or algorithm comparison

**Example output (Isolation Forest):**
```
Normal reading (22°C):  raw_score = 0.49
Cold anomaly (5°C):     raw_score = 0.65
Hot anomaly (50°C):     raw_score = 0.64
```

### Choosing a Normalization Method

| Scenario | Recommended Method | Why |
|----------|-------------------|-----|
| First time using anomaly detection | `standardization` | Easy to understand 0-1 scores |
| Production monitoring dashboards | `standardization` | Consistent scale across models |
| Scientific/engineering analysis | `zscore` | Statistically meaningful |
| Explaining to domain experts | `zscore` | "3 std devs from normal" is clear |
| Comparing different backends | `zscore` | Normalizes different score scales |
| Debugging detection issues | `raw` | See actual algorithm output |
| Replicating research papers | `raw` | Match paper's methodology |

---

## Algorithms (Backends)

### Isolation Forest (Recommended)

Best general-purpose algorithm. Fast, works well out of the box.

```json
{
  "backend": "isolation_forest",
  "contamination": 0.1
}
```

**How it works:**
Isolates anomalies by randomly partitioning data using decision trees. Anomalies require fewer splits to isolate because they are "few and different."

**Strengths:**
- Fast training and inference
- Handles high-dimensional data well
- No assumptions about data distribution
- Works with small to large datasets

**Limitations:**
- May struggle with local anomalies in dense regions
- Less effective when anomalies are clustered together

**Best for:**
- General-purpose anomaly detection
- Large datasets (scales linearly)
- Unknown anomaly patterns
- First choice when unsure which algorithm to use

**Training data requirements:**
- Minimum: 50-100 samples
- Recommended: 500+ samples for robust models
- Should be mostly normal with some contamination

**Raw score range:** ~-0.5 to 0.5 (higher = more anomalous after negation)

### One-Class SVM

Support vector machine for outlier detection.

```json
{
  "backend": "one_class_svm",
  "contamination": 0.1
}
```

**How it works:**
Learns a boundary (hyperplane) around normal data in a high-dimensional feature space. Points outside the boundary are anomalies.

**Strengths:**
- Effective when normal data is well-clustered
- Works well with clear separation between normal and anomalous
- Good for small to medium datasets

**Limitations:**
- Slower training on large datasets (O(n²) to O(n³))
- Sensitive to kernel and hyperparameter choices
- May not handle multiple clusters of normal data well

**Best for:**
- Small to medium datasets (&lt;10,000 samples)
- When normal data forms tight clusters
- High-precision requirements (few false positives)

**Training data requirements:**
- Minimum: 50-100 samples
- Recommended: 200-1000 samples
- Works best with clean, well-defined normal data

**Raw score range:** Unbounded real numbers (higher = more anomalous after negation)

### Local Outlier Factor (LOF)

Density-based anomaly detection.

```json
{
  "backend": "local_outlier_factor",
  "contamination": 0.1
}
```

**How it works:**
Compares the local density of each point to the density of its neighbors. Points with substantially lower density than their neighbors are anomalies.

**Strengths:**
- Detects local anomalies (outliers relative to their neighborhood)
- Handles data with varying densities
- Good for clustered data with different cluster sizes

**Limitations:**
- Requires setting number of neighbors (k)
- Computationally expensive for large datasets
- May produce extreme scores for very isolated points

**Best for:**
- Data with multiple clusters of different densities
- When anomalies are "locally unusual" but not globally extreme
- Spatial data, network data, clustered distributions

**Training data requirements:**
- Minimum: 100+ samples (needs enough neighbors)
- Recommended: 500+ samples
- Benefits from representative sampling of all normal regions

**Raw score range:** ~1 to 10+ (higher = more anomalous; can be very large for extreme outliers)

### Autoencoder (Neural Network)

Deep learning approach for complex patterns.

```json
{
  "backend": "autoencoder",
  "epochs": 100,
  "batch_size": 32
}
```

**How it works:**
Neural network learns to compress (encode) and reconstruct (decode) normal data. Anomalies have high reconstruction error because the network hasn't learned to represent them.

**Strengths:**
- Captures complex, non-linear patterns
- Excellent for high-dimensional data
- Can learn hierarchical features
- Scales well with GPU acceleration

**Limitations:**
- Requires more training data
- Needs hyperparameter tuning (architecture, epochs)
- Training is slower than sklearn methods
- May overfit on small datasets

**Best for:**
- Complex patterns (images, time series, multi-modal data)
- Large datasets (10,000+ samples)
- When simpler methods underperform
- GPU-accelerated environments

**Training data requirements:**
- Minimum: 1000+ samples
- Recommended: 10,000+ samples
- More data generally improves performance

**Raw score range:** 0 to unbounded (reconstruction error; higher = more anomalous)

### Algorithm Comparison

| Feature | Isolation Forest | One-Class SVM | LOF | Autoencoder |
|---------|-----------------|---------------|-----|-------------|
| Speed (training) | Fast | Slow | Medium | Slow |
| Speed (inference) | Fast | Fast | Medium | Fast |
| Min samples | 50 | 50 | 100 | 1000 |
| High dimensions | Good | Good | Poor | Excellent |
| Local anomalies | Poor | Poor | Excellent | Good |
| Complex patterns | Medium | Medium | Medium | Excellent |
| GPU support | No | No | No | Yes |

### All Available Backends (PyOD)

LlamaFarm supports 12 anomaly detection backends powered by PyOD. Use `GET /v1/ml/anomaly/backends` to list all with metadata.

| Backend | Category | Speed | Memory | Best For |
|---------|----------|-------|--------|----------|
| `isolation_forest` | Legacy | Fast | Low | General purpose (recommended default) |
| `one_class_svm` | Legacy | Slow | Medium | Small datasets, tight boundaries |
| `local_outlier_factor` | Legacy | Medium | Medium | Local/density-based anomalies |
| `autoencoder` | Deep Learning | Slow | High | Complex patterns, large datasets |
| `ecod` | Fast | Very Fast | Low | **Recommended for new projects** - parameter-free |
| `hbos` | Fast | Very Fast | Low | Fastest algorithm, high dimensions |
| `copod` | Fast | Very Fast | Low | Parameter-free, interpretable |
| `knn` | Distance | Medium | Medium | When density matters |
| `mcd` | Distance | Medium | Medium | Gaussian-distributed data |
| `cblof` | Clustering | Medium | Medium | Clustered data |
| `suod` | Ensemble | Medium | Medium | Most robust, combines multiple algorithms |
| `loda` | Streaming | Fast | Low | Online/streaming scenarios |

**Backend Categories:**
- **Legacy**: Original 4 backends with backward compatibility
- **Fast**: Parameter-free or minimal tuning, good for quick experiments
- **Distance**: Use distance metrics for anomaly scoring
- **Clustering**: Leverage cluster structure in data
- **Ensemble**: Combine multiple algorithms
- **Deep Learning**: Neural network approaches

---

## Understanding Contamination

The `contamination` parameter is one of the most important settings for anomaly detection. It tells the algorithm what percentage of your **training data** might already contain anomalies.

### What Contamination Does

- **Sets the decision boundary**: The algorithm uses contamination to determine where to draw the line between normal and anomalous
- **Affects threshold calculation**: During training, the model computes an anomaly threshold such that approximately `contamination × 100%` of training samples would be flagged
- **Impacts sensitivity**: Lower contamination = stricter definition of "normal" = more sensitive to deviations

### How to Choose Contamination

| Scenario | Contamination | When to Use |
|----------|---------------|-------------|
| **Very clean data** | 0.01 - 0.05 | Curated datasets, lab conditions, known-good samples |
| **Typical production** | 0.05 - 0.15 | API logs, sensor readings, user activity |
| **Noisy data** | 0.15 - 0.30 | Raw logs with errors, unfiltered data streams |
| **Unknown** | 0.10 (default) | Start here and tune based on results |

### Impact on Detection

```
Training data: [normal, normal, normal, anomaly, normal, ...]
                                         ↑
                           If contamination=0.1, model expects
                           ~10% of training data to be anomalies
```

**Contamination too low** (e.g., 0.01 when true rate is 0.10):
- Model assumes almost all training data is normal
- Decision boundary is too tight around training distribution
- Result: **High false negatives** (misses real anomalies that look like the "contaminated" training samples)

**Contamination too high** (e.g., 0.30 when true rate is 0.05):
- Model assumes many normal samples are actually anomalies
- Decision boundary is too loose
- Result: **High false positives** (flags normal variations as anomalies)

### Per-Algorithm Behavior

| Algorithm | How Contamination is Used |
|-----------|-----------------------------|
| **Isolation Forest** | Sets the `contamination` parameter directly, which determines the threshold on the anomaly score distribution |
| **One-Class SVM** | Maps to the `nu` parameter (upper bound on training error fraction) |
| **Local Outlier Factor** | Sets the contamination parameter for decision threshold |
| **Autoencoder** | Sets the reconstruction error threshold at the contamination percentile |

### Best Practices

1. **Start with 0.1** (10%) if you don't know the true anomaly rate
2. **Use domain knowledge**: If you know ~5% of API requests are errors, set `contamination: 0.05`
3. **Prefer clean training data**: If possible, curate a dataset of known-normal samples and use `contamination: 0.01-0.05`
4. **Tune empirically**: Run detection on labeled test data and adjust based on precision/recall
5. **Consider the cost of errors**: High-stakes (security) → lower contamination; low-stakes (monitoring) → higher contamination

### Example: Tuning Contamination

```bash
# Start conservative (assume clean training data)
curl -X POST http://localhost:14345/v1/ml/anomaly/fit \
  -d '{"model": "test", "data": [...], "contamination": 0.05}'

# Test on data with known anomalies
curl -X POST http://localhost:14345/v1/ml/anomaly/score \
  -d '{"model": "test", "data": [known_normal, known_anomaly, ...]}'

# If too many false positives → increase contamination
# If missing anomalies → decrease contamination (or clean training data)
```

---

## Mixed Data Types

Real-world data often includes both numeric and categorical features. Use the `schema` parameter to automatically encode mixed data.

### Schema Encoding Types

| Type | Description | Example |
|------|-------------|---------|
| `numeric` | Pass through as-is | Response time, bytes |
| `hash` | MD5 hash to integer | User agents, IPs (high cardinality) |
| `label` | Category to integer | HTTP methods, status codes |
| `onehot` | One-hot encoding | Low cardinality categoricals |
| `binary` | Boolean to 0/1 | yes/no, true/false |
| `frequency` | Encode as occurrence frequency | Rare vs common values |

### Example: API Log Monitoring

```bash
# Train with mixed data
curl -X POST http://localhost:14345/v1/ml/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-log-detector",
    "backend": "isolation_forest",
    "normalization": "zscore",
    "data": [
      {"response_time_ms": 100, "bytes": 1024, "method": "GET", "user_agent": "Mozilla/5.0"},
      {"response_time_ms": 105, "bytes": 1100, "method": "POST", "user_agent": "Chrome/90.0"},
      {"response_time_ms": 98, "bytes": 980, "method": "GET", "user_agent": "Safari/14.0"},
      {"response_time_ms": 102, "bytes": 1050, "method": "GET", "user_agent": "Mozilla/5.0"}
    ],
    "schema": {
      "response_time_ms": "numeric",
      "bytes": "numeric",
      "method": "label",
      "user_agent": "hash"
    },
    "contamination": 0.1
  }'
```

```bash
# Detect anomalies (schema already learned)
curl -X POST http://localhost:14345/v1/ml/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-log-detector",
    "normalization": "zscore",
    "threshold": 2.0,
    "data": [
      {"response_time_ms": 100, "bytes": 1024, "method": "GET", "user_agent": "Mozilla/5.0"},
      {"response_time_ms": 9000, "bytes": 500000, "method": "DELETE", "user_agent": "sqlmap/1.0"}
    ]
  }'
```

The encoder is automatically cached with the model - no need to pass the schema again for detection.

---

## Production Workflow

### Save Trained Model

After training, save the model for production use:

```bash
curl -X POST http://localhost:14345/v1/ml/anomaly/save \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-log-detector",
    "backend": "isolation_forest"
  }'
```

Response:
```json
{
  "object": "save_result",
  "model": "api-log-detector",
  "backend": "isolation_forest",
  "filename": "api-log-detector_isolation_forest.joblib",
  "path": "~/.llamafarm/models/anomaly/api-log-detector_isolation_forest.joblib",
  "encoder_path": "~/.llamafarm/models/anomaly/api-log-detector_isolation_forest_encoder.json",
  "status": "saved"
}
```

Models are saved to `~/.llamafarm/models/anomaly/` with auto-generated filenames based on the model name and backend. The normalization method and statistics are persisted with the model.

### Load Saved Model

Load a pre-trained model (e.g., after server restart):

```bash
curl -X POST http://localhost:14345/v1/ml/anomaly/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-log-detector",
    "backend": "isolation_forest"
  }'
```

The model is loaded from the standard location based on its name. The encoder and normalization settings are automatically restored.

### List Saved Models

```bash
curl http://localhost:14345/v1/ml/anomaly/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {"filename": "api-log-detector_isolation_forest.joblib", "size_bytes": 45678, "modified": 1705312345.0, "backend": "sklearn"},
    {"filename": "sensor-model_autoencoder.pt", "size_bytes": 123456, "modified": 1705312000.0, "backend": "autoencoder"}
  ],
  "models_dir": "~/.llamafarm/models/anomaly",
  "total": 2
}
```

### Delete Model

```bash
curl -X DELETE http://localhost:14345/v1/ml/anomaly/models/api_detector_v1.joblib
```

---

## Streaming Anomaly Detection

For real-time data streams, LlamaFarm provides a streaming API that handles:
- **Cold start**: Collects samples before first training
- **Auto-retraining**: Periodically retrains on new data
- **Sliding window**: Maintains recent history for context
- **Tick-Tock pattern**: Seamless model updates without downtime

### Quick Start

```bash
# Send streaming data points
curl -X POST http://localhost:14345/v1/ml/anomaly/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "live-sensor",
    "data": {"temperature": 72.5, "humidity": 45.2},
    "backend": "ecod",
    "min_samples": 50,
    "retrain_interval": 100,
    "window_size": 1000,
    "threshold": 0.5
  }'
```

### Streaming Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | "default-stream" | Unique detector identifier |
| `data` | required | Single dict or batch of dicts |
| `backend` | "ecod" | PyOD backend (ecod recommended for streaming) |
| `min_samples` | 50 | Samples before first training (cold start) |
| `retrain_interval` | 100 | Retrain after this many new samples |
| `window_size` | 1000 | Sliding window size (keeps recent N samples) |
| `threshold` | 0.5 | Anomaly score threshold |
| `contamination` | 0.1 | Expected outlier proportion |
| `rolling_windows` | null | Optional: [5, 10, 20] for rolling features |
| `include_lags` | false | Include lag features |

### Response Statuses

- **collecting**: Cold start phase, building initial dataset
- **ready**: Model trained, returning anomaly scores
- **retraining**: Background retrain in progress (still scoring)

```json
{
  "object": "streaming_result",
  "model": "live-sensor",
  "status": "ready",
  "results": [
    {
      "index": 0,
      "score": 0.42,
      "is_anomaly": false,
      "raw_score": 0.38,
      "samples_until_ready": 0
    }
  ],
  "model_version": 3,
  "samples_collected": 350,
  "samples_until_ready": 0,
  "threshold": 0.5
}
```

### Managing Streaming Detectors

```bash
# List all active detectors
curl http://localhost:14345/v1/ml/anomaly/stream/detectors

# Get specific detector stats
curl http://localhost:14345/v1/ml/anomaly/stream/live-sensor

# Reset detector (clears data, restarts cold start)
curl -X POST http://localhost:14345/v1/ml/anomaly/stream/live-sensor/reset

# Delete detector
curl -X DELETE http://localhost:14345/v1/ml/anomaly/stream/live-sensor
```

### Recommended Backends for Streaming

| Backend | Why |
|---------|-----|
| `ecod` | **Recommended** - Fast, parameter-free, handles drift |
| `hbos` | Fastest, good for high-throughput streams |
| `loda` | Designed for streaming, lightweight |
| `isolation_forest` | Reliable, good baseline |

---

## Polars Buffer API

For advanced use cases, LlamaFarm exposes direct access to Polars-based data buffers. These provide:
- **High-performance columnar storage** using Polars DataFrames
- **Automatic sliding window** truncation
- **Rolling feature computation** (mean, std, min, max)
- **Lag features** for temporal patterns

:::tip When to Use Polars Buffers
Most users should use the streaming anomaly detection API, which manages Polars buffers automatically. Use the direct Polars API when you need:
- Custom feature engineering pipelines
- Integration with other ML systems
- Manual control over data lifecycle
:::

### Create a Buffer

```bash
curl -X POST http://localhost:14345/v1/ml/polars/buffers \
  -H "Content-Type: application/json" \
  -d '{"buffer_id": "sensor-data", "window_size": 1000}'
```

### Append Data

```bash
# Single record
curl -X POST http://localhost:14345/v1/ml/polars/append \
  -H "Content-Type: application/json" \
  -d '{
    "buffer_id": "sensor-data",
    "data": {"temperature": 72.5, "humidity": 45.2}
  }'

# Batch append
curl -X POST http://localhost:14345/v1/ml/polars/append \
  -H "Content-Type: application/json" \
  -d '{
    "buffer_id": "sensor-data",
    "data": [
      {"temperature": 72.5, "humidity": 45.2},
      {"temperature": 73.1, "humidity": 44.8},
      {"temperature": 71.9, "humidity": 46.0}
    ]
  }'
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
    "tail": 10
  }'
```

Response includes computed columns like:
- `temperature_rolling_mean_5`, `temperature_rolling_std_10`
- `humidity_rolling_min_20`, `humidity_rolling_max_5`
- `temperature_lag_1`, `humidity_lag_2`

### Buffer Management

```bash
# List all buffers
curl http://localhost:14345/v1/ml/polars/buffers

# Get buffer stats
curl http://localhost:14345/v1/ml/polars/buffers/sensor-data

# Get raw data
curl "http://localhost:14345/v1/ml/polars/buffers/sensor-data/data?tail=10"

# Get data with features
curl "http://localhost:14345/v1/ml/polars/buffers/sensor-data/data?with_features=true&tail=10"

# Clear buffer (keep structure)
curl -X POST http://localhost:14345/v1/ml/polars/buffers/sensor-data/clear

# Delete buffer
curl -X DELETE http://localhost:14345/v1/ml/polars/buffers/sensor-data
```

### Buffer Statistics

```json
{
  "buffer_id": "sensor-data",
  "size": 500,
  "window_size": 1000,
  "columns": ["temperature", "humidity"],
  "numeric_columns": ["temperature", "humidity"],
  "memory_bytes": 8000,
  "append_count": 500,
  "avg_append_ms": 0.15
}
```

---

## API Reference

### POST /v1/ml/anomaly/fit

Train an anomaly detector on data assumed to be mostly normal.

**Request Body:**
```json
{
  "model": "string",           // Model identifier (for caching)
  "backend": "string",         // isolation_forest | one_class_svm | local_outlier_factor | autoencoder
  "data": [[...]] | [{...}],   // Training data (numeric arrays or dicts)
  "schema": {...},             // Feature encoding schema (required for dict data)
  "contamination": 0.1,        // Expected proportion of anomalies
  "normalization": "zscore",   // standardization | zscore | raw
  "epochs": 100,               // Training epochs (autoencoder only)
  "batch_size": 32             // Batch size (autoencoder only)
}
```

**Response:**
```json
{
  "object": "fit_result",
  "model": "api-detector",
  "backend": "isolation_forest",
  "samples_fitted": 1000,
  "training_time_ms": 123.45,
  "model_params": {
    "backend": "isolation_forest",
    "contamination": 0.1,
    "threshold": 2.23,
    "input_dim": 4
  },
  "encoder": {"schema": {...}, "features": [...]},
  "status": "fitted"
}
```

### POST /v1/ml/anomaly/score

Score data points for anomalies. Returns all points with scores.

**Request Body:**
```json
{
  "model": "string",
  "backend": "string",
  "data": [[...]] | [{...}],
  "schema": {...},             // Optional (uses cached encoder if available)
  "normalization": "zscore",   // Must match training normalization
  "threshold": 2.0             // Override default threshold
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {"index": 0, "score": -0.22, "is_anomaly": false, "raw_score": 0.49},
    {"index": 1, "score": 3.25, "is_anomaly": true, "raw_score": 0.65}
  ],
  "summary": {
    "total_points": 2,
    "anomaly_count": 1,
    "anomaly_rate": 0.5,
    "threshold": 2.0
  }
}
```

### POST /v1/ml/anomaly/detect

Detect anomalies (returns only anomalous points).

Same request format as `/v1/ml/anomaly/score`, but response only includes points classified as anomalies.
The response does not include an `is_anomaly` field since all returned points are anomalies.

**Response:**
```json
{
  "object": "list",
  "data": [
    {"index": 1, "score": 3.25, "raw_score": 0.65}
  ],
  "total_count": 1,
  "summary": {
    "anomalies_detected": 1,
    "threshold": 2.0
  }
}
```

### POST /v1/ml/anomaly/save

Save a fitted model to disk. Models are saved to `~/.llamafarm/models/anomaly/` with auto-generated filenames.

**Request Body:**
```json
{
  "model": "string",           // Model identifier (must be fitted)
  "backend": "string"          // Backend type
}
```

### POST /v1/ml/anomaly/load

Load a pre-trained model from disk. The file is automatically located based on model name and backend.

**Request Body:**
```json
{
  "model": "string",           // Model identifier to load/cache as
  "backend": "string"          // Backend type
}
```

### GET /v1/ml/anomaly/models

List all saved models.

### DELETE /v1/ml/anomaly/models/\{filename\}

Delete a saved model.

---

## Use Cases

### API Log Monitoring

Detect suspicious API requests (attacks, abuse, anomalies):

```json
{
  "backend": "isolation_forest",
  "normalization": "zscore",
  "data": [
    {"response_time_ms": 100, "bytes": 1024, "status": 200, "method": "GET", "endpoint": "/api/users", "user_agent": "Mozilla/5.0"},
    ...
  ],
  "schema": {
    "response_time_ms": "numeric",
    "bytes": "numeric",
    "status": "label",
    "method": "label",
    "endpoint": "label",
    "user_agent": "hash"
  }
}
```

**Detects:**
- SQL injection attempts (unusual user agents like `sqlmap`)
- Data exfiltration (high bytes transferred)
- DoS attempts (many requests, unusual patterns)
- Scanning (unusual endpoints, methods)

**Recommended settings:**
- Backend: `isolation_forest` (fast, handles mixed data well)
- Normalization: `zscore` (easy to explain: "this request is 5 std devs from normal")
- Threshold: 2.0-3.0 depending on false positive tolerance

### Sensor Monitoring (IoT)

Detect faulty sensors or unusual conditions:

```json
{
  "backend": "isolation_forest",
  "normalization": "zscore",
  "data": [[temperature, pressure, humidity, vibration], ...],
  "contamination": 0.05
}
```

**Detects:**
- Sensor failures (stuck values, spikes)
- Equipment issues (correlated anomalies)
- Environmental anomalies

**Recommended settings:**
- Backend: `isolation_forest` or `local_outlier_factor` (for local anomalies)
- Normalization: `zscore` (engineers understand std deviations)
- Threshold: 3.0 (industrial settings often use 3-sigma)

### Financial Transactions

Detect fraudulent transactions:

```json
{
  "backend": "one_class_svm",
  "normalization": "standardization",
  "data": [
    {"amount": 50.00, "merchant_category": "grocery", "hour": 14, "country": "US"},
    ...
  ],
  "schema": {
    "amount": "numeric",
    "merchant_category": "label",
    "hour": "numeric",
    "country": "label"
  }
}
```

**Detects:**
- Unusual amounts for category
- Unusual times
- Geographic anomalies

**Recommended settings:**
- Backend: `one_class_svm` (tight boundaries around normal behavior)
- Normalization: `standardization` (0-1 scores for risk scoring)
- Contamination: 0.01-0.05 (fraud is rare)

### Network Intrusion Detection

Detect malicious network activity:

```json
{
  "backend": "local_outlier_factor",
  "normalization": "zscore",
  "data": [
    {"bytes_in": 1024, "bytes_out": 512, "packets": 10, "duration_ms": 100, "protocol": "TCP", "port": 443},
    ...
  ],
  "schema": {
    "bytes_in": "numeric",
    "bytes_out": "numeric",
    "packets": "numeric",
    "duration_ms": "numeric",
    "protocol": "label",
    "port": "label"
  }
}
```

**Detects:**
- Port scanning (unusual port patterns)
- Data exfiltration (high outbound bytes)
- C2 communication (unusual timing patterns)

**Recommended settings:**
- Backend: `local_outlier_factor` (detects local anomalies in network clusters)
- Normalization: `zscore` (clear severity indication)
- Threshold: 2.0-3.0

---

## Best Practices

### Training Data

1. **Use mostly normal data**: The `contamination` parameter tells the algorithm what proportion of anomalies to expect
2. **Include variety**: Cover different normal patterns (weekday/weekend, seasonal, etc.)
3. **Sufficient samples**: At least 100-1000 samples for good results
4. **Clean data**: Remove known bad data if possible before training

### Feature Selection

1. **Include relevant features**: All features that might indicate anomalies
2. **Normalize scales**: Features are automatically scaled, but extreme ranges can affect sensitivity
3. **Choose appropriate encodings**: Use `hash` for high-cardinality, `label` for ordered categories

### Threshold Tuning

1. **Use the learned threshold**: The runtime automatically computes a threshold during training based on the `contamination` parameter (percentile of normalized scores). This learned threshold is returned in the fit response and used by default.
2. **Override when needed**: You can pass a custom `threshold` parameter to `/v1/ml/anomaly/score` or `/v1/ml/anomaly/detect` endpoints.
3. **Match normalization to threshold**:
   - `standardization`: threshold 0.5-0.9
   - `zscore`: threshold 2.0-4.0
   - `raw`: threshold depends on backend
4. **Test with known anomalies**: Tune based on your false positive tolerance

### Production Deployment

1. **Save models**: Don't retrain on every restart
2. **Version models**: Use descriptive filenames like `api_detector_v2_2024_01`
3. **Monitor performance**: Track false positive/negative rates
4. **Retrain periodically**: Normal patterns may drift over time
5. **Log raw scores**: Even if using normalized scores, log `raw_score` for debugging

---

## Troubleshooting

### Scores are all around 0.5 (standardization)

**Cause:** Training data has low variance, or test data is very similar to training.

**Solutions:**
- Switch to `zscore` normalization for more spread
- Check that training data covers the full range of normal behavior
- Verify test data actually contains anomalies

### Z-scores are extremely large (100+)

**Cause:** Test point is far outside the training distribution.

**Solutions:**
- This is actually correct behavior - the point is genuinely extreme
- Consider capping scores for display purposes
- Use `standardization` if you need bounded scores

### Different backends give different scores

**Cause:** Each backend has different native score ranges.

**Solutions:**
- Use `zscore` normalization to make scores comparable
- Don't compare raw scores across backends
- Choose one backend and stick with it for consistency

### Model not detecting obvious anomalies

**Causes:**
1. Contamination too high (model thinks anomalies are normal)
2. Training data already contains similar anomalies
3. Threshold too high

**Solutions:**
- Lower contamination (e.g., 0.01-0.05)
- Curate cleaner training data
- Lower threshold
- Try `local_outlier_factor` for local anomalies

---

## Environment Variables

```bash
# Base data directory (default: ~/.llamafarm)
# Anomaly models are saved to $LF_DATA_DIR/models/anomaly/
LF_DATA_DIR=/path/to/llamafarm/data
```

---

## Next Steps

- [Specialized ML Models](./specialized-ml.md) - Overview of all ML endpoints
- [Universal Runtime](./index.md#universal-runtime) - General runtime configuration
- [API Reference](../api/index.md) - Full API documentation
