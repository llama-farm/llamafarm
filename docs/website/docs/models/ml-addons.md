---
title: ML Addons Guide
sidebar_position: 6
---

# ML Addons Guide

ML Addons extend the Universal Runtime with specialized machine learning capabilities. Each addon is installed separately to keep the base runtime lightweight.

## Installing Addons

```bash
# Install individual addons
lf addons install timeseries
lf addons install adtk
lf addons install catboost
lf addons install drift

# Or install dependencies manually
pip install darts>=0.29.0 chronos-forecasting>=1.2.0   # timeseries
pip install adtk>=0.6.2                                 # adtk
pip install catboost>=1.2.0                             # catboost
pip install alibi-detect>=0.12.0                        # drift
```

---

## Time-Series Forecasting

Forecast future values using classical methods (ARIMA, Exponential Smoothing, Theta) or zero-shot foundation models (Amazon Chronos).

### Available Backends

| Backend | Training Required | Confidence Intervals | Speed | Description |
|---------|-------------------|---------------------|-------|-------------|
| `arima` | Yes | Yes | Fast | Classical ARIMA model |
| `exponential_smoothing` | Yes | Yes | Fast | Holt-Winters exponential smoothing |
| `theta` | Yes | Yes | Fast | Theta forecasting method |
| `chronos` | No (zero-shot) | Yes | Slow | Amazon Chronos foundation model |
| `chronos-bolt` | No (zero-shot) | Yes | Medium | Faster Chronos variant |

### List Backends

```bash
curl http://localhost:11540/v1/timeseries/backends
```

### Fit a Model

```bash
curl -X POST http://localhost:11540/v1/timeseries/fit \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "arima",
    "model": "sales-forecast",
    "data": [
      {"timestamp": "2024-01-01", "value": 100},
      {"timestamp": "2024-01-02", "value": 105},
      {"timestamp": "2024-01-03", "value": 98},
      {"timestamp": "2024-01-04", "value": 110},
      {"timestamp": "2024-01-05", "value": 103}
    ],
    "frequency": "D",
    "description": "Daily sales forecast"
  }'
```

Response:
```json
{
  "model": "sales-forecast",
  "backend": "arima",
  "saved_path": "~/.llamafarm/models/timeseries/sales-forecast_arima.joblib",
  "training_time_ms": 234.5,
  "samples_fitted": 5,
  "description": "Daily sales forecast"
}
```

### Predict

```bash
# From a fitted model
curl -X POST http://localhost:11540/v1/timeseries/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sales-forecast",
    "horizon": 7,
    "confidence_level": 0.95
  }'
```

Response:
```json
{
  "model_id": "sales-forecast",
  "backend": "arima",
  "predictions": [
    {"timestamp": "2024-01-06", "value": 107.2, "lower": 95.1, "upper": 119.3},
    {"timestamp": "2024-01-07", "value": 104.8, "lower": 90.5, "upper": 119.1}
  ],
  "fit_time_ms": 0.0,
  "predict_time_ms": 12.3
}
```

### Zero-Shot Forecasting (Chronos)

No training needed — just provide historical data:

```bash
curl -X POST http://localhost:11540/v1/timeseries/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chronos",
    "horizon": 7,
    "data": [
      {"timestamp": "2024-01-01", "value": 100},
      {"timestamp": "2024-01-02", "value": 105},
      {"timestamp": "2024-01-03", "value": 98}
    ],
    "confidence_level": 0.9
  }'
```

### Manage Models

```bash
# List saved models
curl http://localhost:11540/v1/timeseries/models

# Load a specific model
curl -X POST http://localhost:11540/v1/timeseries/load \
  -H "Content-Type: application/json" \
  -d '{"model": "sales-forecast"}'

# Load latest version
curl -X POST http://localhost:11540/v1/timeseries/load \
  -H "Content-Type: application/json" \
  -d '{"model": "sales-forecast-latest"}'

# Delete a model
curl -X DELETE http://localhost:11540/v1/timeseries/models/sales-forecast
```

### Python Example

```python
import requests

BASE = "http://localhost:11540"

# Fit
requests.post(f"{BASE}/v1/timeseries/fit", json={
    "backend": "exponential_smoothing",
    "model": "sensor-temp",
    "data": [{"timestamp": f"2024-01-{d:02d}", "value": v}
             for d, v in enumerate([22.1, 23.5, 21.8, 24.2, 22.7], 1)],
    "frequency": "D"
})

# Predict
resp = requests.post(f"{BASE}/v1/timeseries/predict", json={
    "model": "sensor-temp",
    "horizon": 3,
    "confidence_level": 0.95
})

for p in resp.json()["predictions"]:
    print(f"{p['timestamp']}: {p['value']:.1f} [{p['lower']:.1f}, {p['upper']:.1f}]")
```

---

## Time-Series Anomaly Detection (ADTK)

Detect temporal patterns that point anomaly detectors miss: level shifts, seasonal violations, spikes, volatility changes, and stuck values.

### Available Detectors

| Detector | Training | Description |
|----------|----------|-------------|
| `level_shift` | Yes | Sudden baseline changes |
| `spike` | No | IQR-based outlier spikes/dips |
| `seasonal` | Yes | Seasonal pattern violations |
| `volatility` | Yes | Variance/volatility shifts |
| `persist` | No | Stuck/constant value detection |
| `threshold` | No | Simple threshold-based detection |

### List Detectors

```bash
curl http://localhost:11540/v1/adtk/detectors
```

### Fit a Detector

```bash
curl -X POST http://localhost:11540/v1/adtk/fit \
  -H "Content-Type: application/json" \
  -d '{
    "detector": "level_shift",
    "model": "server-baseline",
    "data": [
      {"timestamp": "2024-01-01T00:00:00", "value": 100},
      {"timestamp": "2024-01-01T01:00:00", "value": 102},
      {"timestamp": "2024-01-01T02:00:00", "value": 98},
      {"timestamp": "2024-01-01T03:00:00", "value": 101}
    ],
    "params": {"window": 5},
    "description": "Server CPU baseline"
  }'
```

Response:
```json
{
  "model": "server-baseline",
  "detector": "level_shift",
  "saved_path": "~/.llamafarm/models/adtk/server-baseline_level_shift.joblib",
  "training_time_ms": 15.2,
  "samples_fitted": 4,
  "requires_training": true
}
```

### Detect Anomalies

```bash
curl -X POST http://localhost:11540/v1/adtk/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "server-baseline",
    "detector": "level_shift",
    "data": [
      {"timestamp": "2024-01-01T04:00:00", "value": 100},
      {"timestamp": "2024-01-01T05:00:00", "value": 150},
      {"timestamp": "2024-01-01T06:00:00", "value": 155},
      {"timestamp": "2024-01-01T07:00:00", "value": 148}
    ]
  }'
```

Response:
```json
{
  "model": "server-baseline",
  "detector": "level_shift",
  "anomalies": [
    {
      "timestamp": "2024-01-01T05:00:00",
      "value": 150.0,
      "anomaly_type": "level_shift",
      "score": 0.95
    }
  ],
  "total_points": 4,
  "anomaly_count": 1,
  "detection_time_ms": 3.2
}
```

### Ad-Hoc Detection (No Saved Model)

Use detectors without fitting first (for detectors that don't require training):

```bash
curl -X POST http://localhost:11540/v1/adtk/detect \
  -H "Content-Type: application/json" \
  -d '{
    "detector": "spike",
    "data": [
      {"timestamp": "2024-01-01T00:00:00", "value": 10},
      {"timestamp": "2024-01-01T01:00:00", "value": 12},
      {"timestamp": "2024-01-01T02:00:00", "value": 500},
      {"timestamp": "2024-01-01T03:00:00", "value": 11}
    ]
  }'
```

### Manage Models

```bash
# List saved ADTK models
curl http://localhost:11540/v1/adtk/models

# Load a saved model
curl -X POST http://localhost:11540/v1/adtk/load \
  -H "Content-Type: application/json" \
  -d '{"model": "server-baseline"}'

# Delete a model
curl -X DELETE http://localhost:11540/v1/adtk/models/server-baseline
```

:::tip ADTK vs Standard Anomaly Detection
Use [standard anomaly detection](./anomaly-detection.md) (`/v1/ml/anomaly/*`) for point anomalies in tabular data. Use ADTK for **temporal patterns** — level shifts, seasonal deviations, and volatility changes that require time-series context.
:::

---

## Data Drift Detection

Monitor production data for distribution drift using statistical tests. Detect when incoming data has shifted from your training distribution.

### Available Detectors

| Detector | Type | Description |
|----------|------|-------------|
| `ks` | Univariate | Kolmogorov-Smirnov test for numeric features |
| `chi2` | Univariate | Chi-squared test for categorical features |
| `mmd` | Multivariate | Maximum Mean Discrepancy for multi-feature drift |

### List Detectors

```bash
curl http://localhost:11540/v1/drift/detectors
```

### Train Reference Distribution

```bash
curl -X POST http://localhost:11540/v1/drift/fit \
  -H "Content-Type: application/json" \
  -d '{
    "detector": "ks",
    "model": "model-input-monitor",
    "reference_data": [
      [1.0, 2.0, 3.0],
      [1.1, 2.1, 2.9],
      [0.9, 1.9, 3.1],
      [1.2, 2.2, 2.8]
    ],
    "feature_names": ["feature_a", "feature_b", "feature_c"],
    "description": "Production model input baseline"
  }'
```

Response:
```json
{
  "model": "model-input-monitor",
  "detector": "ks",
  "saved_path": "~/.llamafarm/models/drift/model-input-monitor_ks.joblib",
  "training_time_ms": 8.5,
  "reference_size": 4,
  "n_features": 3
}
```

### Detect Drift

```bash
curl -X POST http://localhost:11540/v1/drift/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-input-monitor",
    "data": [
      [5.0, 8.0, 1.0],
      [4.8, 7.5, 0.5],
      [5.2, 8.2, 1.2]
    ]
  }'
```

Response:
```json
{
  "model": "model-input-monitor",
  "detector": "ks",
  "result": {
    "is_drift": true,
    "p_value": 0.002,
    "threshold": 0.05,
    "distance": 0.85,
    "p_values": [0.001, 0.003, 0.01]
  },
  "detection_time_ms": 5.1
}
```

### Monitor Status

```bash
# Check detector status
curl http://localhost:11540/v1/drift/status/model-input-monitor

# Reset detector
curl -X POST http://localhost:11540/v1/drift/reset/model-input-monitor
```

### Manage Models

```bash
# List drift models
curl http://localhost:11540/v1/drift/models

# Load a saved model
curl -X POST http://localhost:11540/v1/drift/load \
  -H "Content-Type: application/json" \
  -d '{"model": "model-input-monitor"}'

# Delete a model
curl -X DELETE http://localhost:11540/v1/drift/models/model-input-monitor
```

### Python Example

```python
import requests

BASE = "http://localhost:11540"

# Train on reference data
requests.post(f"{BASE}/v1/drift/fit", json={
    "detector": "ks",
    "model": "api-inputs",
    "reference_data": training_features.tolist(),
    "feature_names": ["latency", "payload_size", "error_rate"]
})

# Check new batch for drift
resp = requests.post(f"{BASE}/v1/drift/detect", json={
    "model": "api-inputs",
    "data": new_batch.tolist()
})

result = resp.json()["result"]
if result["is_drift"]:
    print(f"⚠️ Drift detected! p-value: {result['p_value']:.4f}")
```

---

## CatBoost Gradient Boosting

Train gradient boosting models with native categorical feature support, incremental learning, and feature importance.

### Capabilities

```bash
curl http://localhost:11540/v1/catboost/info
```

### Train a Model

```bash
curl -X POST http://localhost:11540/v1/catboost/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "churn-predictor",
    "model_type": "classifier",
    "data": [
      [25, "premium", 12],
      [45, "basic", 3],
      [35, "premium", 24],
      [28, "basic", 1]
    ],
    "labels": [0, 1, 0, 1],
    "feature_names": ["age", "plan", "months"],
    "cat_features": [1],
    "iterations": 100,
    "learning_rate": 0.1,
    "depth": 6,
    "validation_fraction": 0.2
  }'
```

Response:
```json
{
  "model_id": "churn-predictor",
  "model_type": "classifier",
  "samples_fitted": 3,
  "n_features": 3,
  "iterations": 100,
  "best_iteration": 85,
  "classes": [0, 1],
  "saved_path": "~/.llamafarm/models/catboost/churn-predictor.joblib",
  "fit_time_ms": 456.2
}
```

### Predict

```bash
curl -X POST http://localhost:11540/v1/catboost/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "churn-predictor",
    "data": [[30, "premium", 18], [50, "basic", 2]],
    "return_proba": true
  }'
```

Response:
```json
{
  "model_id": "churn-predictor",
  "predictions": [
    {"sample_index": 0, "prediction": 0, "probabilities": {"0": 0.85, "1": 0.15}},
    {"sample_index": 1, "prediction": 1, "probabilities": {"0": 0.25, "1": 0.75}}
  ],
  "predict_time_ms": 2.1
}
```

### Incremental Learning

Update a model with new data without retraining from scratch:

```bash
curl -X POST http://localhost:11540/v1/catboost/update \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "churn-predictor",
    "data": [[33, "premium", 6], [41, "basic", 8]],
    "labels": [0, 1]
  }'
```

Response:
```json
{
  "model_id": "churn-predictor",
  "samples_added": 2,
  "trees_before": 100,
  "trees_after": 150,
  "update_time_ms": 123.4
}
```

### Feature Importance

```bash
curl http://localhost:11540/v1/catboost/churn-predictor/importance
```

Response:
```json
{
  "model_id": "churn-predictor",
  "importances": [
    {"feature": "months", "importance": 45.2},
    {"feature": "plan", "importance": 32.8},
    {"feature": "age", "importance": 22.0}
  ],
  "importance_type": "FeatureImportance"
}
```

### Manage Models

```bash
# List models
curl http://localhost:11540/v1/catboost/models

# Load a model
curl -X POST http://localhost:11540/v1/catboost/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "churn-predictor"}'

# Delete a model
curl -X DELETE http://localhost:11540/v1/catboost/churn-predictor
```

---

## SHAP Explainability

Generate SHAP (SHapley Additive exPlanations) explanations for any ML model predictions. Understand which features drive model decisions.

### Standalone SHAP Explanation

```bash
curl -X POST http://localhost:11540/v1/explain/shap \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "catboost",
    "model_id": "churn-predictor",
    "data": [[30, 1, 18], [50, 0, 2]],
    "feature_names": ["age", "plan", "months"],
    "top_k": 3,
    "generate_narrative": true
  }'
```

Response:
```json
{
  "model_type": "catboost",
  "model_id": "churn-predictor",
  "explainer_type": "tree",
  "explanations": [
    {
      "sample_index": 0,
      "base_value": 0.5,
      "prediction": 0.15,
      "contributions": [
        {"feature": "months", "value": 18, "shap_value": -0.25, "direction": "decreases"},
        {"feature": "plan", "value": 1, "shap_value": -0.08, "direction": "decreases"},
        {"feature": "age", "value": 30, "shap_value": -0.02, "direction": "decreases"}
      ]
    }
  ],
  "narrative": {
    "summary": "The prediction is primarily driven by high tenure (18 months), which strongly decreases churn risk.",
    "details": ["months=18 contributes -0.25 to the prediction", "..."]
  },
  "explain_time_ms": 45.3
}
```

### Global Feature Importance

```bash
curl -X POST http://localhost:11540/v1/explain/importance \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "anomaly",
    "model_id": "sensor-monitor",
    "data": [[22.1, 100], [23.5, 105], [21.8, 98]],
    "feature_names": ["temperature", "pressure"]
  }'
```

### SHAP with Anomaly Detection

Add `explain: true` to any anomaly detection call to get SHAP explanations inline:

```bash
curl -X POST http://localhost:14345/v1/ml/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sensor-monitor",
    "data": [[22.0, 100], [5.0, 500], [50.0, 10]],
    "explain": true,
    "feature_names": ["temperature", "pressure"]
  }'
```

The response includes SHAP explanations alongside each anomaly, showing which features caused the anomaly flag.

### Available Explainer Types

```bash
curl http://localhost:11540/v1/explain/explainers
```

| Explainer | Description | Best For |
|-----------|-------------|----------|
| `tree` | TreeExplainer | CatBoost, Isolation Forest, XGBoost |
| `linear` | LinearExplainer | Linear models, One-Class SVM |
| `kernel` | KernelExplainer | Any model (slower, model-agnostic) |

The explainer type is auto-detected based on the model.

---

## Addon API Reference

### Timeseries Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/timeseries/backends` | GET | List forecasting backends |
| `/v1/timeseries/fit` | POST | Train a forecaster |
| `/v1/timeseries/predict` | POST | Generate predictions |
| `/v1/timeseries/load` | POST | Load a saved model |
| `/v1/timeseries/models` | GET | List saved models |
| `/v1/timeseries/models/{name}` | GET | Get model info |
| `/v1/timeseries/models/{name}` | DELETE | Delete a model |

### ADTK Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/adtk/detectors` | GET | List detector types |
| `/v1/adtk/fit` | POST | Fit a detector |
| `/v1/adtk/detect` | POST | Detect anomalies |
| `/v1/adtk/load` | POST | Load a saved model |
| `/v1/adtk/models` | GET | List saved models |
| `/v1/adtk/models/{name}` | DELETE | Delete a model |

### Drift Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/drift/detectors` | GET | List detector types |
| `/v1/drift/fit` | POST | Fit on reference data |
| `/v1/drift/detect` | POST | Check for drift |
| `/v1/drift/load` | POST | Load a saved model |
| `/v1/drift/status/{name}` | GET | Get detector status |
| `/v1/drift/reset/{name}` | POST | Reset a detector |
| `/v1/drift/models` | GET | List saved models |
| `/v1/drift/models/{name}` | DELETE | Delete a model |

### CatBoost Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/catboost/info` | GET | CatBoost capabilities |
| `/v1/catboost/fit` | POST | Train a model |
| `/v1/catboost/predict` | POST | Make predictions |
| `/v1/catboost/update` | POST | Incremental update |
| `/v1/catboost/load` | POST | Load a model |
| `/v1/catboost/models` | GET | List models |
| `/v1/catboost/{id}` | DELETE | Delete a model |
| `/v1/catboost/{id}/importance` | GET | Feature importance |

### Explainability Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/explain/explainers` | GET | List explainer types |
| `/v1/explain/shap` | POST | SHAP explanation |
| `/v1/explain/importance` | POST | Global feature importance |

---

## Next Steps

- [Anomaly Detection Guide](./anomaly-detection.md) — Point anomaly detection with SHAP integration
- [Vision Pipeline](./vision.md) — Object detection, classification, and streaming
- [Specialized ML Models](./specialized-ml.md) — OCR, NER, classification, and more
