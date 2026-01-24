# Anomaly Detection Guide

LlamaFarm Universal Runtime provides powerful anomaly detection powered by [PyOD](https://pyod.readthedocs.io/), the most comprehensive Python library for outlier detection with 40+ algorithms.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Available Backends (12 Algorithms)](#available-backends)
4. [API Reference](#api-reference)
5. [Streaming Detection](#streaming-detection)
6. [Training Best Practices](#training-best-practices)
7. [Model Persistence](#model-persistence)
8. [Feature Engineering](#feature-engineering)
9. [Production Deployment](#production-deployment)

---

## Quick Start

```python
import httpx

# Training data (normal behavior)
train = [[1, 2], [1.1, 2.1], [0.9, 1.9]] * 10  # Need 10+ samples

# Test data (includes anomaly)
test = [[1, 2], [100, 200], [1.1, 2.1]]

client = httpx.Client(timeout=30)

# 1. Train the model
client.post("http://localhost:11545/v1/anomaly/fit", json={
    "data": train,
    "backend": "ecod",  # Fast, parameter-free
    "model": "my-detector",
})

# 2. Score new data
response = client.post("http://localhost:11545/v1/anomaly/score", json={
    "data": test,
    "backend": "ecod",
    "model": "my-detector",
})

for item in response.json()["data"]:
    if item["is_anomaly"]:
        print(f"Anomaly detected at index {item['index']}: score={item['score']:.3f}")
```

---

## Core Concepts

### Anomaly Score

Every data point receives a **normalized anomaly score** between 0 and 1:
- **0.0 - 0.5**: Normal (low anomaly likelihood)
- **0.5 - 0.7**: Borderline (worth investigating)
- **0.7 - 1.0**: Anomaly (high anomaly likelihood)

### Contamination

The `contamination` parameter (default: 0.1) tells the algorithm what fraction of your training data might be anomalous. This affects the anomaly threshold:

```python
# Expect 5% of data to be anomalies
{"contamination": 0.05}

# Expect 20% of data to be anomalies
{"contamination": 0.2}
```

Lower contamination = higher threshold = fewer false positives.

### Threshold

The `threshold` parameter determines the anomaly cutoff:
- Scores above threshold = Anomaly
- Scores below threshold = Normal

The threshold is automatically computed during training based on contamination.

---

## Available Backends

### Fast (Parameter-Free) - RECOMMENDED

These algorithms require no tuning and work well for most use cases.

#### ECOD - Empirical Cumulative Distribution

```python
{"backend": "ecod", "contamination": 0.1}
```

**Best for:** General purpose, first choice for any dataset
**Speed:** Fast | **Memory:** Low
**How it works:** Uses empirical cumulative distribution functions to measure how extreme each dimension is.

**Pros:**
- No hyperparameters to tune
- Scales to millions of samples
- Handles high-dimensional data well

**Cons:**
- May miss local anomalies in clustered data

---

#### HBOS - Histogram-Based Outlier Score

```python
{"backend": "hbos", "n_bins": 10, "contamination": 0.1}
```

**Best for:** Speed-critical applications, high-dimensional data
**Speed:** Very Fast | **Memory:** Very Low
**How it works:** Builds histograms for each feature and scores based on bin density.

**Pros:**
- Fastest algorithm available
- Minimal memory usage
- Good for streaming

**Cons:**
- Assumes feature independence
- Sensitive to bin size choice

---

#### COPOD - Copula-Based Outlier Detection

```python
{"backend": "copod", "contamination": 0.1}
```

**Best for:** Interpretable results, when you need to explain detections
**Speed:** Fast | **Memory:** Low
**How it works:** Uses copulas to model multivariate distributions.

**Pros:**
- Parameter-free
- Interpretable scores
- Handles dependencies between features

**Cons:**
- Less effective for very high dimensions

---

### Legacy (Well-Tested)

Classic algorithms with proven track records.

#### Isolation Forest

```python
{
    "backend": "isolation_forest",
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": 0.1
}
```

**Best for:** General purpose, tree-based detection
**Speed:** Fast | **Memory:** Medium
**How it works:** Builds random trees; anomalies are isolated with fewer splits.

**Pros:**
- Works well on high-dimensional data
- Robust to irrelevant features
- Interpretable (feature importance)

**Cons:**
- Needs tuning for optimal results
- Can miss local anomalies

---

#### Local Outlier Factor (LOF)

```python
{
    "backend": "local_outlier_factor",
    "n_neighbors": 20,
    "contamination": 0.1
}
```

**Best for:** Clustered data, local anomalies
**Speed:** Medium | **Memory:** High
**How it works:** Compares local density to neighbors' density.

**Pros:**
- Detects local anomalies
- Works with clusters of varying density

**Cons:**
- Memory-intensive for large datasets
- Sensitive to `n_neighbors` choice

---

#### One-Class SVM

```python
{
    "backend": "one_class_svm",
    "kernel": "rbf",
    "nu": 0.1,
    "gamma": "auto"
}
```

**Best for:** Small datasets with clear boundaries
**Speed:** Slow | **Memory:** High
**How it works:** Finds a hyperplane that separates normal data from the origin.

**Pros:**
- Works well with limited data
- Can learn complex boundaries

**Cons:**
- Very slow on large datasets
- Sensitive to kernel choice

---

### Distance-Based

Algorithms based on distance metrics.

#### KNN - K-Nearest Neighbors

```python
{
    "backend": "knn",
    "n_neighbors": 5,
    "method": "mean",  # or "largest", "median"
    "contamination": 0.1
}
```

**Best for:** Distance-based anomalies
**Speed:** Medium | **Memory:** High
**How it works:** Anomaly score based on distance to k-nearest neighbors.

---

#### MCD - Minimum Covariance Determinant

```python
{"backend": "mcd", "contamination": 0.1}
```

**Best for:** Multivariate Gaussian data
**Speed:** Medium | **Memory:** Medium
**How it works:** Robust covariance estimation using a clean subset.

---

### Clustering-Based

#### CBLOF - Clustering-Based Local Outlier Factor

```python
{
    "backend": "cblof",
    "n_clusters": 8,
    "contamination": 0.1
}
```

**Best for:** Grouped/clustered data
**Speed:** Medium | **Memory:** Medium
**How it works:** Combines clustering with outlier factor computation.

---

### Ensemble

#### SUOD - Scalable Unsupervised Outlier Detection

```python
{
    "backend": "suod",
    "base_estimators": ["ecod", "hbos", "isolation_forest"],
    "contamination": 0.1
}
```

**Best for:** Most robust detection, production critical applications
**Speed:** Slow | **Memory:** High
**How it works:** Ensemble of multiple algorithms with parallel training.

**Note:** Requires additional PyOD dependencies.

---

### Streaming

#### LODA - Lightweight Online Detector of Anomalies

```python
{
    "backend": "loda",
    "n_bins": 10,
    "n_random_cuts": 100,
    "contamination": 0.1
}
```

**Best for:** Streaming data, online detection
**Speed:** Fast | **Memory:** Low
**How it works:** Ensemble of sparse random projections.

---

### Deep Learning

#### AutoEncoder

```python
{
    "backend": "autoencoder",
    "hidden_neurons": [64, 32, 32, 64],
    "epochs": 50,
    "batch_size": 32,
    "contamination": 0.1
}
```

**Best for:** Complex patterns, large datasets
**Speed:** Slow | **Memory:** High
**How it works:** Neural network learns to reconstruct normal data; high reconstruction error = anomaly.

**Note:** Requires PyTorch or TensorFlow.

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/anomaly/fit` | POST | Train a model |
| `/v1/anomaly/score` | POST | Score data points |
| `/v1/anomaly/detect` | POST | One-shot fit + detect |
| `/v1/anomaly/load` | POST | Load saved model |
| `/v1/anomaly/models` | GET | List saved models |
| `/v1/anomaly/backends` | GET | List available backends |
| `/v1/anomaly/stream` | POST | Stream data point |
| `/v1/anomaly/stream/detectors` | GET | List streaming detectors |

### Fit Request

```json
{
    "data": [[1.0, 2.0], [1.1, 2.1], ...],
    "model": "my-model",
    "backend": "ecod",
    "contamination": 0.1,
    "schema": {
        "fields": ["amount", "time"],
        "types": ["numeric", "numeric"]
    }
}
```

### Score Response

```json
{
    "object": "list",
    "data": [
        {
            "index": 0,
            "score": 0.45,
            "is_anomaly": false,
            "raw_score": 1.23
        },
        {
            "index": 1,
            "score": 0.85,
            "is_anomaly": true,
            "raw_score": 5.67
        }
    ],
    "summary": {
        "total_points": 2,
        "anomaly_count": 1,
        "anomaly_rate": 0.5,
        "threshold": 0.65
    }
}
```

---

## Streaming Detection

For real-time monitoring, use the streaming API with automatic cold-start handling and background retraining.

### Architecture (Tick-Tock Pattern)

```
Data Point → Buffer (Polars) → Score (Tick) → Response
                    ↓
              [When buffer full]
                    ↓
            Retrain (Tock, Background)
```

### Usage

```python
import httpx

client = httpx.Client(timeout=30)

# Stream a data point (detector created automatically)
response = client.post("http://localhost:11545/v1/anomaly/stream", json={
    "model_id": "sensor-monitor",
    "backend": "ecod",
    "data": {"temperature": 45.0, "vibration": 0.5},
    "min_samples": 50,      # Cold start threshold
    "retrain_interval": 100, # Retrain frequency
    "rolling_windows": [5, 10],  # Optional rolling features
})

result = response.json()
if result["status"] == "ready" and result["is_anomaly"]:
    alert(f"Anomaly: score={result['score']}")
```

### Cold Start

During cold start (collecting initial samples), the response includes:
```json
{
    "status": "collecting",
    "samples_collected": 25,
    "samples_until_ready": 25,
    "score": null,
    "is_anomaly": null
}
```

Once ready:
```json
{
    "status": "ready",
    "samples_collected": 50,
    "samples_until_ready": 0,
    "score": 0.72,
    "is_anomaly": true,
    "model_version": 1
}
```

---

## Training Best Practices

### Data Requirements

| Dataset Size | Recommendation |
|--------------|----------------|
| < 50 samples | Use `one_class_svm` or `mcd` |
| 50-1000 | Any algorithm works |
| 1000-100k | Use `ecod`, `hbos`, `isolation_forest` |
| > 100k | Use `hbos` or `loda` |

### Feature Scaling

LlamaFarm automatically normalizes scores, but input data should be reasonably scaled:

```python
# Good: Features on similar scales
[[100, 0.5], [105, 0.6], [98, 0.4]]

# Bad: Features on wildly different scales
[[100000000, 0.0001], [105000000, 0.0002]]
```

### Training on Normal Data Only

For best results, train on normal data only:

```python
# Recommended: Train on known-good data
client.post("/v1/anomaly/fit", json={
    "data": normal_data_only,
    "contamination": 0.01  # Low contamination
})

# Less ideal: Train on mixed data
client.post("/v1/anomaly/fit", json={
    "data": mixed_data,
    "contamination": 0.1  # Expect 10% anomalies
})
```

---

## Model Persistence

Models are automatically saved during training to `~/.llamafarm/models/anomaly/`.

### Filename Format

```
{model_name}_{backend}.joblib
```

Example: `fraud-detector_ecod.joblib`

### Loading a Model

```bash
# List available models
curl http://localhost:11545/v1/anomaly/models

# Load a specific model
curl -X POST http://localhost:11545/v1/anomaly/load \
  -H "Content-Type: application/json" \
  -d '{"model": "fraud-detector", "backend": "ecod"}'
```

### Model Versioning

For production, use versioned model names:

```python
from datetime import datetime

version = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"fraud-detector-v{version}"

client.post("/v1/anomaly/fit", json={
    "data": training_data,
    "model": model_name,
    "backend": "ecod"
})
```

---

## Feature Engineering

### Rolling Features (Streaming Only)

Enable automatic rolling feature computation for temporal patterns:

```python
client.post("/v1/anomaly/stream", json={
    "model_id": "sensor-monitor",
    "data": {"value": 45.0},
    "rolling_windows": [5, 10],  # 5-sample and 10-sample windows
    "include_lags": true,
    "lag_periods": [1, 2]  # t-1 and t-2 values
})
```

This automatically computes:
- `value_mean_5`, `value_std_5` (5-sample rolling stats)
- `value_mean_10`, `value_std_10` (10-sample rolling stats)
- `value_lag_1`, `value_lag_2` (lagged values)

### Schema-Based Feature Extraction

Define a schema for automatic feature encoding:

```python
client.post("/v1/anomaly/fit", json={
    "data": [
        {"amount": 100, "merchant_type": "retail", "hour": 14},
        {"amount": 150, "merchant_type": "online", "hour": 10},
    ],
    "schema": {
        "fields": ["amount", "merchant_type", "hour"],
        "types": ["numeric", "categorical", "numeric"]
    }
})
```

Categorical features are automatically one-hot encoded.

---

## Production Deployment

### Health Checks

```bash
curl http://localhost:11545/health
```

### Metrics to Monitor

1. **Anomaly Rate**: Track rate over time; sudden spikes may indicate drift
2. **Model Version**: Ensure correct model is loaded
3. **Latency**: Score time should be < 10ms for fast backends
4. **Memory Usage**: Monitor for memory leaks with streaming

### Recommended Setup

```yaml
# Production configuration
anomaly:
  backend: ecod  # Fast, robust
  model: production-detector-v1

streaming:
  min_samples: 100  # Higher for production
  retrain_interval: 1000  # Less frequent retraining
  window_size: 10000  # Larger sliding window
  threshold: 0.8  # Higher threshold for fewer false positives
```

### Alerting Thresholds

| Score Range | Action |
|-------------|--------|
| 0.0 - 0.5 | No action |
| 0.5 - 0.7 | Log for review |
| 0.7 - 0.9 | Alert + automated investigation |
| 0.9 - 1.0 | Critical alert + immediate response |

---

## Troubleshooting

### Common Issues

**"Model not fitted"**
- Load the model first: `POST /v1/anomaly/load`
- Or fit a new model: `POST /v1/anomaly/fit`

**High false positive rate**
- Increase threshold
- Lower contamination
- Use more training data
- Try different backend

**Slow performance**
- Use `hbos` or `ecod` instead of `one_class_svm`
- Reduce feature dimensions
- Use batch scoring instead of single-point scoring

**Memory issues**
- Use `hbos` or `ecod` (lowest memory)
- Reduce `window_size` for streaming
- Use `loda` for streaming applications
