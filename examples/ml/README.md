# Machine Learning Examples

This directory contains examples demonstrating LlamaFarm's ML capabilities beyond basic inference:

## Examples

### 01_classifier_setfit.py - Text Classification (SetFit)
Few-shot text classification using SetFit:
- Train with as few as 8-16 examples per class
- Intent classification, sentiment analysis
- No large dataset required

```bash
cd server && uv run python ../examples/ml/01_classifier_setfit.py
```

### 02_timeseries_chronos.py - Time-Series Forecasting (Chronos)
Zero-shot time-series forecasting:
- Amazon Chronos foundation model (no training needed!)
- Also supports ARIMA, Exponential Smoothing, Theta
- Confidence intervals for planning

```bash
cd server && uv run python ../examples/ml/02_timeseries_chronos.py
```

### 03_timeseries_anomaly_adtk.py - Time-Series Anomaly Detection (ADTK)
Specialized time-series anomaly detection:
- Level shift detection (sudden baseline changes)
- Spike detection (short-term outliers)
- Persist detection (stuck sensor values)
- Seasonal anomalies

```bash
cd server && uv run python ../examples/ml/03_timeseries_anomaly_adtk.py
```

### 04_drift_detection.py - Data Drift Detection (Alibi Detect)
Monitor data distribution changes:
- KS test (univariate drift)
- MMD test (multivariate drift)
- Chi-squared (categorical drift)
- Production monitoring workflow

```bash
cd server && uv run python ../examples/ml/04_drift_detection.py
```

### 05_catboost_incremental.py - CatBoost with Incremental Learning
Gradient boosting with unique features:
- Native categorical feature support (no one-hot encoding)
- Incremental learning (update model without full retrain)
- GPU acceleration (when available)
- Feature importance

```bash
cd server && uv run python ../examples/ml/05_catboost_incremental.py
```

## Prerequisites

Start the LlamaFarm servers:

```bash
# From the llamafarm repo root
nx start universal-runtime &
nx start server &
```

## API Endpoints

| Feature | Endpoint | Description |
|---------|----------|-------------|
| **Classifier** | `POST /v1/ml/classifier/fit` | Train SetFit classifier |
| | `POST /v1/ml/classifier/predict` | Classify texts |
| **Time-Series** | `GET /v1/timeseries/backends` | List forecasting methods |
| | `POST /v1/timeseries/fit` | Fit traditional models |
| | `POST /v1/timeseries/predict` | Generate forecasts |
| **ADTK** | `GET /v1/adtk/detectors` | List anomaly detectors |
| | `POST /v1/adtk/detect` | Detect time-series anomalies |
| **Drift** | `GET /v1/drift/detectors` | List drift detectors |
| | `POST /v1/drift/fit` | Fit on reference data |
| | `POST /v1/drift/detect` | Detect drift |
| **CatBoost** | `GET /v1/catboost/info` | Check CatBoost availability |
| | `POST /v1/catboost/fit` | Train classifier/regressor |
| | `POST /v1/catboost/predict` | Make predictions |
| | `POST /v1/catboost/update` | Incremental model update |
| | `GET /v1/catboost/{model}/importance` | Feature importance |

Note: Time-series, ADTK, drift, and CatBoost endpoints are on the Universal Runtime (port 11540).

## Comparison: Which Tool for What?

| Use Case | Tool | When to Use |
|----------|------|-------------|
| Point anomalies | PyOD (`/v1/ml/anomaly/*`) | Individual data points, no time context |
| Time-series anomalies | ADTK (`/v1/adtk/*`) | Temporal patterns, level shifts, spikes |
| Forecasting | Darts/Chronos (`/v1/timeseries/*`) | Predict future values |
| Distribution drift | Alibi Detect (`/v1/drift/*`) | ML monitoring, data quality |
| Text classification | SetFit (`/v1/ml/classifier/*`) | Few-shot text tasks |
| Tabular classification | CatBoost (`/v1/catboost/*`) | Categorical features, incremental learning |

## See Also

- `examples/anomaly/` - Detailed anomaly detection examples with SHAP
- `examples/anomaly/full/` - Complete anomaly detection pipeline demos
