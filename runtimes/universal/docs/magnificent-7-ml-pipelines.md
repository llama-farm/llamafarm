# Magnificent 7 ML Pipelines

LlamaFarm Universal Runtime provides seven specialized ML capabilities for production-grade machine learning workflows. This guide covers all seven "Magnificent 7" pipelines.

## Table of Contents

1. [Overview](#overview)
2. [Anomaly Detection (PyOD)](#1-anomaly-detection-pyod)
3. [Time-Series Anomaly Detection (ADTK)](#2-time-series-anomaly-detection-adtk)
4. [Data Drift Detection (Alibi Detect)](#3-data-drift-detection-alibi-detect)
5. [CatBoost Classifier](#4-catboost-classifier)
6. [Time-Series Forecasting (Darts)](#5-time-series-forecasting-darts)
7. [SHAP Explainability](#6-shap-explainability)
8. [Streaming ML (Polars Buffer)](#7-streaming-ml-polars-buffer)

---

## Overview

| Pipeline | Library | Use Case | Port |
|----------|---------|----------|------|
| Anomaly Detection | PyOD | Detect outliers in tabular data | 8005 |
| Time-Series Anomaly | ADTK | Level shifts, spikes, seasonality | 8005 |
| Drift Detection | Alibi Detect | Monitor distribution changes | 8005 |
| CatBoost | CatBoost | Classification with incremental learning | 8005 |
| Forecasting | Darts | Time-series prediction | 8005 |
| Explainability | SHAP | Explain model predictions | 8005 |
| Streaming | Polars | Real-time feature engineering | 8005 |

All endpoints are accessible through the LlamaFarm server at `http://localhost:8005/v1/`.

---

## 1. Anomaly Detection (PyOD)

See [anomaly-detection.md](anomaly-detection.md) for comprehensive documentation.

### Quick Start

```bash
# Train a detector
curl -X POST http://localhost:8005/v1/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[1,2], [1.1,2.1], [0.9,1.9], [1,2], [1.1,2.1]],
    "backend": "ecod",
    "model": "my-detector"
  }'

# Score new data
curl -X POST http://localhost:8005/v1/anomaly/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-detector",
    "data": [[1,2], [100,200]]
  }'
```

### Available Backends

- **Fast**: `ecod`, `hbos`, `copod`
- **Legacy**: `isolation_forest`, `local_outlier_factor`, `one_class_svm`
- **Streaming**: `loda`

---

## 2. Time-Series Anomaly Detection (ADTK)

ADTK (Anomaly Detection Toolkit) excels at detecting time-series-specific patterns that general-purpose detectors miss.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/adtk/detectors` | GET | List available detectors |
| `/v1/adtk/fit` | POST | Fit detector on time series |
| `/v1/adtk/detect` | POST | Detect anomalies |
| `/v1/adtk/models` | GET | List saved models |
| `/v1/adtk/load` | POST | Load saved model |
| `/v1/adtk/models/{name}` | DELETE | Delete model |

### Detector Types

| Detector | Description | Best For |
|----------|-------------|----------|
| `level_shift` | Sudden level changes | Step changes in metrics |
| `spike` | Sudden spikes/drops | One-time anomalies |
| `seasonal` | Seasonal pattern deviations | Periodic data |
| `volatility_shift` | Variance changes | Stability monitoring |
| `persist` | Changes that persist | Regime changes |
| `threshold` | Simple threshold | Known bounds |
| `interquartile_range` | IQR-based outliers | Statistical outliers |

### Example: Detect Level Shifts

```bash
# List detectors
curl http://localhost:8005/v1/adtk/detectors

# Fit a level shift detector
curl -X POST http://localhost:8005/v1/adtk/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cpu-monitor",
    "detector": "level_shift",
    "data": [
      {"timestamp": "2024-01-01T00:00:00", "value": 50.0},
      {"timestamp": "2024-01-01T01:00:00", "value": 52.0},
      {"timestamp": "2024-01-01T02:00:00", "value": 48.0},
      {"timestamp": "2024-01-01T03:00:00", "value": 51.0},
      {"timestamp": "2024-01-01T04:00:00", "value": 49.0}
    ],
    "params": {"c": 3.0, "window": 3}
  }'

# Detect anomalies in new data
curl -X POST http://localhost:8005/v1/adtk/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cpu-monitor",
    "data": [
      {"timestamp": "2024-01-02T00:00:00", "value": 50.0},
      {"timestamp": "2024-01-02T01:00:00", "value": 90.0},
      {"timestamp": "2024-01-02T02:00:00", "value": 88.0}
    ]
  }'
```

### Detector Parameters

#### Level Shift
```json
{
  "detector": "level_shift",
  "params": {
    "c": 3.0,     // Sensitivity (lower = more sensitive)
    "side": "both", // "positive", "negative", or "both"
    "window": 5    // Comparison window size
  }
}
```

#### Spike
```json
{
  "detector": "spike",
  "params": {
    "c": 1.5  // Sensitivity (lower = more sensitive)
  }
}
```

#### Volatility Shift
```json
{
  "detector": "volatility_shift",
  "params": {
    "c": 3.0,
    "side": "both",
    "window": 5
  }
}
```

---

## 3. Data Drift Detection (Alibi Detect)

Monitor production data for distribution drift from training data.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/drift/detectors` | GET | List available detectors |
| `/v1/drift/fit` | POST | Fit on reference data |
| `/v1/drift/detect` | POST | Check for drift |
| `/v1/drift/status/{name}` | GET | Get detector status |
| `/v1/drift/models` | GET | List saved models |
| `/v1/drift/reset/{name}` | POST | Reset detector |
| `/v1/drift/models/{name}` | DELETE | Delete model |

### Detector Types

| Detector | Description | Data Type |
|----------|-------------|-----------|
| `ks` | Kolmogorov-Smirnov test | Univariate numeric |
| `mmd` | Maximum Mean Discrepancy | Multivariate numeric |
| `chi2` | Chi-squared test | Categorical |

### Example: Detect Feature Drift

```bash
# Fit KS detector on reference data (training distribution)
curl -X POST http://localhost:8005/v1/drift/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "feature-monitor",
    "detector": "ks",
    "reference_data": [
      [1.0, 2.0, 3.0],
      [1.1, 2.1, 3.1],
      [0.9, 1.9, 2.9],
      [1.0, 2.0, 3.0]
    ],
    "feature_names": ["feature_a", "feature_b", "feature_c"],
    "description": "Production feature monitor"
  }'

# Check for drift in new data
curl -X POST http://localhost:8005/v1/drift/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "feature-monitor",
    "data": [
      [5.0, 6.0, 7.0],
      [5.1, 6.1, 7.1],
      [4.9, 5.9, 6.9]
    ]
  }'
```

### Response Interpretation

```json
{
  "model": "feature-monitor",
  "result": {
    "is_drift": true,      // Drift detected!
    "p_value": 0.000001,   // Very low = significant drift
    "threshold": 0.0167,   // Bonferroni-corrected threshold
    "feature_scores": {
      "feature_a": {"p_value": 0.001, "is_drift": true},
      "feature_b": {"p_value": 0.002, "is_drift": true},
      "feature_c": {"p_value": 0.001, "is_drift": true}
    }
  }
}
```

### Categorical Drift (Chi-squared)

```bash
curl -X POST http://localhost:8005/v1/drift/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "category-monitor",
    "detector": "chi2",
    "reference_data": [[0], [1], [2], [0], [1], [2], [0], [1]],
    "feature_names": ["category"]
  }'
```

---

## 4. CatBoost Classifier

High-performance gradient boosting with incremental learning support.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/catboost/info` | GET | CatBoost availability |
| `/v1/catboost/fit` | POST | Train classifier |
| `/v1/catboost/predict` | POST | Make predictions |
| `/v1/catboost/update` | POST | Incremental update |
| `/v1/catboost/models` | GET | List models |
| `/v1/catboost/{id}/importance` | GET | Feature importance |
| `/v1/catboost/{id}` | DELETE | Delete model |

### Example: Train and Predict

```bash
# Train classifier
curl -X POST http://localhost:8005/v1/catboost/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "fraud-classifier",
    "data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
    "labels": [0, 0, 1, 1],
    "feature_names": ["amount", "frequency"],
    "iterations": 100
  }'

# Make predictions
curl -X POST http://localhost:8005/v1/catboost/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "fraud-classifier",
    "data": [[2.0, 3.0], [6.0, 7.0]]
  }'

# Get feature importance
curl http://localhost:8005/v1/catboost/fraud-classifier/importance
```

### Incremental Learning

CatBoost supports incremental updates without full retraining:

```bash
# Update with new data (adds more trees)
curl -X POST http://localhost:8005/v1/catboost/update \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "fraud-classifier",
    "data": [[9.0, 10.0], [11.0, 12.0]],
    "labels": [1, 0],
    "iterations": 50
  }'
```

---

## 5. Time-Series Forecasting (Darts)

Multiple forecasting backends from statistical to deep learning models.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/timeseries/backends` | GET | List backends |
| `/v1/timeseries/fit` | POST | Train model |
| `/v1/timeseries/predict` | POST | Generate forecast |
| `/v1/timeseries/models` | GET | List models |
| `/v1/timeseries/load` | POST | Load model |
| `/v1/timeseries/models/{name}` | DELETE | Delete model |

### Available Backends

| Backend | Description | Speed |
|---------|-------------|-------|
| `arima` | Auto-ARIMA (requires StatsForecast) | Medium |
| `exponential_smoothing` | Holt-Winters method | Fast |
| `theta` | Theta forecasting | Fast |
| `chronos` | Amazon's foundation model | Medium |
| `chronos-bolt` | Faster Chronos variant | Fast |

### Example: Forecast

```bash
# Train exponential smoothing model
curl -X POST http://localhost:8005/v1/timeseries/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sales-forecast",
    "backend": "exponential_smoothing",
    "data": [
      {"timestamp": "2024-01-01", "value": 100},
      {"timestamp": "2024-01-02", "value": 110},
      {"timestamp": "2024-01-03", "value": 105},
      {"timestamp": "2024-01-04", "value": 115}
    ]
  }'

# Generate 7-day forecast
curl -X POST http://localhost:8005/v1/timeseries/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sales-forecast",
    "horizon": 7
  }'
```

### Response with Confidence Intervals

```json
{
  "model": "sales-forecast",
  "predictions": [
    {
      "timestamp": "2024-01-05T00:00:00",
      "value": 112.5,
      "lower_95": 105.2,
      "upper_95": 119.8
    }
  ]
}
```

---

## 6. SHAP Explainability

Explain model predictions with SHAP (SHapley Additive exPlanations).

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/explain/explainers` | GET | List explainers |
| `/v1/explain/shap` | POST | Generate explanation |
| `/v1/explain/importance` | POST | Feature importance |

### Available Explainers

| Explainer | Best For | Speed |
|-----------|----------|-------|
| `tree` | IForest, CatBoost, XGBoost | Fast |
| `linear` | Linear, Logistic, Ridge | Fast |
| `kernel` | Any model | Slow |

### Example: Explain Predictions

```bash
# Get available explainers
curl http://localhost:8005/v1/explain/explainers

# Generate SHAP explanation
curl -X POST http://localhost:8005/v1/explain/shap \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "anomaly",
    "model_id": "my-detector",
    "data": [[5.0, 6.0, 7.0]],
    "feature_names": ["feature_a", "feature_b", "feature_c"],
    "top_k": 5,
    "generate_narrative": true
  }'
```

### Response

```json
{
  "model_type": "anomaly",
  "model_id": "my-detector",
  "explainer_type": "tree",
  "explanations": [
    {
      "sample_index": 0,
      "base_value": 0.5,
      "prediction": 0.85,
      "contributions": [
        {"feature": "feature_b", "value": 6.0, "shap_value": 0.32, "direction": "increases"},
        {"feature": "feature_a", "value": 5.0, "shap_value": 0.15, "direction": "increases"},
        {"feature": "feature_c", "value": 7.0, "shap_value": -0.12, "direction": "decreases"}
      ]
    }
  ],
  "narrative": {
    "summary": "The prediction is significantly higher than average, primarily due to feature_b.",
    "details": [
      "feature_b (value=6.00) strongly increases the prediction (contribution: +0.320)",
      "feature_a (value=5.00) moderately increases the prediction (contribution: +0.150)"
    ]
  },
  "explain_time_ms": 12.5
}
```

---

## 7. Streaming ML (Polars Buffer)

High-performance streaming data processing with Polars.

### Features

- **O(1) append**: Constant-time data insertion
- **Automatic truncation**: Rolling window maintains fixed size
- **Lazy computation**: Rolling features computed on-demand
- **Sub-millisecond**: &lt;1ms per append operation

### Usage via Streaming Anomaly Detection

```bash
curl -X POST http://localhost:8005/v1/anomaly/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "sensor-monitor",
    "backend": "ecod",
    "data": {"temperature": 45.0, "humidity": 0.6},
    "min_samples": 50,
    "retrain_interval": 100,
    "rolling_windows": [5, 10],
    "include_lags": true,
    "lag_periods": [1, 2]
  }'
```

### Rolling Features

When `rolling_windows` is specified, these features are computed:

| Feature | Description |
|---------|-------------|
| `{col}_rolling_mean_{window}` | Rolling average |
| `{col}_rolling_std_{window}` | Rolling standard deviation |
| `{col}_rolling_min_{window}` | Rolling minimum |
| `{col}_rolling_max_{window}` | Rolling maximum |
| `{col}_lag_{period}` | Lagged values |

### Streaming Detector States

| State | Description |
|-------|-------------|
| `collecting` | Cold start, gathering initial samples |
| `ready` | Model fitted, scoring new points |
| `retraining` | Background retraining in progress |

---

## Server Proxy Architecture

All endpoints are proxied through the LlamaFarm server (port 8005) to the Universal Runtime (port 11545):

```
Client → LlamaFarm Server (8005) → Universal Runtime (11545)
                ↓
        Unified API Gateway
```

### Benefits

- Single entry point for all ML capabilities
- Consistent authentication and rate limiting
- Request/response logging
- Error handling standardization

### Starting Services

```bash
# Start runtime
nx start universal-runtime &

# Start server (includes proxy)
nx start server &
```

---

## Troubleshooting

### Model Not Found

```json
{"detail": "Model not found: anomaly/my-model"}
```

**Solution**: Fit or load the model first:
```bash
curl -X POST http://localhost:8005/v1/anomaly/fit -d '...'
```

### Dependency Not Installed

```json
{"detail": "Required dependency not installed: StatsForecast"}
```

**Solution**: Install the optional dependency:
```bash
cd runtimes/universal
uv add statsforecast
```

### Request Timeout

Long-running operations may timeout. Use async patterns:

```bash
# Increase timeout
curl --max-time 120 -X POST http://localhost:8005/v1/catboost/fit -d '...'
```

---

## Quick Reference

### Anomaly Detection
```bash
POST /v1/anomaly/fit      # Train
POST /v1/anomaly/score    # Detect
```

### ADTK (Time-Series Anomaly)
```bash
POST /v1/adtk/fit         # Train
POST /v1/adtk/detect      # Detect
```

### Drift Detection
```bash
POST /v1/drift/fit        # Learn reference
POST /v1/drift/detect     # Check drift
```

### CatBoost
```bash
POST /v1/catboost/fit     # Train
POST /v1/catboost/predict # Predict
POST /v1/catboost/update  # Incremental
```

### Forecasting
```bash
POST /v1/timeseries/fit   # Train
POST /v1/timeseries/predict # Forecast
```

### Explainability
```bash
POST /v1/explain/shap     # SHAP values
POST /v1/explain/importance # Feature importance
```

### Streaming
```bash
POST /v1/anomaly/stream   # Stream + detect
```
