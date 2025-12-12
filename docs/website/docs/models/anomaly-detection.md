---
title: Anomaly Detection Guide
sidebar_position: 3
---

# Anomaly Detection Guide

The Universal Runtime provides production-ready anomaly detection for monitoring APIs, sensors, financial transactions, and any time-series or tabular data.

## Overview

Anomaly detection learns what "normal" looks like from training data and then identifies deviations in new data. LlamaFarm supports:

- **Multiple algorithms**: Isolation Forest, One-Class SVM, Local Outlier Factor, Autoencoder
- **Mixed data types**: Numeric values, categorical features, and combinations
- **Production workflow**: Train, save, load, and score with persistence
- **Feature encoding**: Automatic encoding of categorical data (hash, label, one-hot, etc.)

## Quick Start

### 1. Train on Normal Data

```bash
curl -X POST http://localhost:11540/v1/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sensor-monitor",
    "backend": "isolation_forest",
    "data": [
      [100, 1024, 10],
      [105, 1100, 12],
      [98, 980, 9],
      [102, 1050, 11],
      [103, 1080, 10]
    ],
    "contamination": 0.1
  }'
```

### 2. Detect Anomalies

```bash
curl -X POST http://localhost:11540/v1/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sensor-monitor",
    "data": [
      [100, 1024, 10],
      [9999, 50000, 1000],
      [103, 1080, 11]
    ]
  }'
```

Response:
```json
{
  "object": "list",
  "data": [
    {"index": 1, "score": 0.92, "raw_score": -0.45}
  ],
  "summary": {
    "anomalies_detected": 1,
    "threshold": 0.5
  }
}
```

---

## Algorithms

### Isolation Forest (Recommended)

Best general-purpose algorithm. Fast, works well out of the box.

```json
{
  "backend": "isolation_forest",
  "contamination": 0.1
}
```

- **How it works**: Isolates anomalies by randomly partitioning data. Anomalies require fewer splits to isolate.
- **Best for**: General purpose, large datasets, unknown anomaly patterns
- **Strengths**: Fast training, handles high dimensions well
- **Contamination**: Expected proportion of anomalies (0.01-0.5)

### One-Class SVM

Support vector machine for outlier detection.

```json
{
  "backend": "one_class_svm"
}
```

- **How it works**: Learns a boundary around normal data in high-dimensional space
- **Best for**: Small to medium datasets, well-defined normal regions
- **Strengths**: Works well when normal data is tightly clustered
- **Limitations**: Slower on large datasets

### Local Outlier Factor

Density-based anomaly detection.

```json
{
  "backend": "local_outlier_factor"
}
```

- **How it works**: Compares local density of points to their neighbors
- **Best for**: Data with varying densities, clustered anomalies
- **Strengths**: Detects anomalies relative to local context
- **Limitations**: Requires setting number of neighbors

### Autoencoder (Neural Network)

Deep learning approach for complex patterns.

```json
{
  "backend": "autoencoder",
  "epochs": 100,
  "batch_size": 32
}
```

- **How it works**: Neural network learns to compress and reconstruct normal data. Anomalies have high reconstruction error.
- **Best for**: Complex patterns, large datasets, time-series
- **Strengths**: Captures non-linear relationships
- **Requirements**: More training data, GPU recommended

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
|-----------|--------------------------|
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
curl -X POST http://localhost:11540/v1/anomaly/fit \
  -d '{"model": "test", "data": [...], "contamination": 0.05}'

# Test on data with known anomalies
curl -X POST http://localhost:11540/v1/anomaly/score \
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
curl -X POST http://localhost:11540/v1/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-log-detector",
    "backend": "isolation_forest",
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
curl -X POST http://localhost:11540/v1/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-log-detector",
    "data": [
      {"response_time_ms": 100, "bytes": 1024, "method": "GET", "user_agent": "Mozilla/5.0"},
      {"response_time_ms": 9000, "bytes": 500000, "method": "POST", "user_agent": "sqlmap/1.0"}
    ]
  }'
```

The encoder is automatically cached with the model - no need to pass the schema again for detection.

---

## Production Workflow

### Save Trained Model

After training, save the model for production use:

```bash
curl -X POST http://localhost:11540/v1/anomaly/save \
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

Models are saved to `~/.llamafarm/models/anomaly/` with auto-generated filenames based on the model name and backend.

### Load Saved Model

Load a pre-trained model (e.g., after server restart):

```bash
curl -X POST http://localhost:11540/v1/anomaly/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-log-detector",
    "backend": "isolation_forest"
  }'
```

The model is loaded from the standard location based on its name. The encoder is automatically loaded if it exists.

### List Saved Models

```bash
curl http://localhost:11540/v1/anomaly/models
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
curl -X DELETE http://localhost:11540/v1/anomaly/models/api_detector_v1.joblib
```

---

## API Reference

### POST /v1/anomaly/fit

Train an anomaly detector on data assumed to be mostly normal.

**Request Body:**
```json
{
  "model": "string",           // Model identifier (for caching)
  "backend": "string",         // isolation_forest | one_class_svm | local_outlier_factor | autoencoder
  "data": [[...]] | [{...}],   // Training data (numeric arrays or dicts)
  "schema": {...},             // Feature encoding schema (required for dict data)
  "contamination": 0.1,        // Expected proportion of anomalies
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
  "model_params": {...},
  "encoder": {"schema": {...}, "features": [...]},
  "status": "fitted"
}
```

### POST /v1/anomaly/score

Score data points for anomalies. Returns all points with scores.

**Request Body:**
```json
{
  "model": "string",
  "backend": "string",
  "data": [[...]] | [{...}],
  "schema": {...},             // Optional (uses cached encoder if available)
  "threshold": 0.5             // Override default threshold
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {"index": 0, "score": 0.23, "is_anomaly": false, "raw_score": 0.12},
    {"index": 1, "score": 0.89, "is_anomaly": true, "raw_score": -0.45}
  ],
  "summary": {
    "total_points": 2,
    "anomaly_count": 1,
    "anomaly_rate": 0.5,
    "threshold": 0.5
  }
}
```

### POST /v1/anomaly/detect

Detect anomalies (returns only anomalous points).

Same request format as `/score`, but response only includes points classified as anomalies.
The response does not include an `is_anomaly` field since all returned points are anomalies.

**Response:**
```json
{
  "object": "list",
  "data": [
    {"index": 1, "score": 0.89, "raw_score": -0.45}
  ],
  "total_count": 1,
  "summary": {
    "anomalies_detected": 1,
    "threshold": 0.5
  }
}
```

### POST /v1/anomaly/save

Save a fitted model to disk. Models are saved to `~/.llamafarm/models/anomaly/` with auto-generated filenames.

**Request Body:**
```json
{
  "model": "string",           // Model identifier (must be fitted)
  "backend": "string"          // Backend type
}
```

### POST /v1/anomaly/load

Load a pre-trained model from disk. The file is automatically located based on model name and backend.

**Request Body:**
```json
{
  "model": "string",           // Model identifier to load/cache as
  "backend": "string"          // Backend type
}
```

### GET /v1/anomaly/models

List all saved models.

### DELETE /v1/anomaly/models/\{filename\}

Delete a saved model.

---

## Use Cases

### API Log Monitoring

Detect suspicious API requests (attacks, abuse, anomalies):

```json
{
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

### Sensor Monitoring

Detect faulty sensors or unusual conditions:

```json
{
  "data": [[temperature, pressure, humidity, vibration], ...],
  "contamination": 0.05
}
```

**Detects:**
- Sensor failures (stuck values, spikes)
- Equipment issues (correlated anomalies)
- Environmental anomalies

### Financial Transactions

Detect fraudulent transactions:

```json
{
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
2. **Override when needed**: You can pass a custom `threshold` parameter (0-1 range) to `/v1/anomaly/score` or `/v1/anomaly/detect` endpoints.
3. **Lower threshold**: More sensitive (more false positives)
4. **Higher threshold**: Less sensitive (may miss anomalies)
5. **Test with known anomalies**: Tune based on your false positive tolerance

### Production Deployment

1. **Save models**: Don't retrain on every restart
2. **Version models**: Use descriptive filenames like `api_detector_v2_2024_01`
3. **Monitor performance**: Track false positive/negative rates
4. **Retrain periodically**: Normal patterns may drift over time

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
