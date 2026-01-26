# Anomaly Detection Examples

LlamaFarm provides powerful anomaly detection powered by [PyOD](https://pyod.readthedocs.io/) with 12+ algorithms and streaming support via Polars.

## Quick Start

```bash
# Start the Universal Runtime
nx start universal-runtime

# Run the quick start example
cd /path/to/llamafarm
uv run python examples/anomaly/01_quick_start.py
```

## Examples

| Example | Description | Key Features |
|---------|-------------|--------------|
| `01_quick_start.py` | Simplest possible example | Detect anomalies in 5 lines |
| `02_fraud_detection.py` | Real-world fraud detection | Training, saving, loading models |
| `03_streaming_sensors.py` | IoT sensor monitoring | Streaming detection, rolling features |
| `04_backend_comparison.py` | Compare all backends | Choose the right algorithm |

## Available Backends (12 Algorithms)

### Fast (Parameter-Free) - Recommended

| Backend | Speed | Best For |
|---------|-------|----------|
| `ecod` | Fast | General purpose, no tuning needed |
| `hbos` | Very Fast | Speed-critical, high dimensions |
| `copod` | Fast | Interpretable results |

### Legacy (Well-Tested)

| Backend | Speed | Best For |
|---------|-------|----------|
| `isolation_forest` | Fast | General purpose, tree-based |
| `one_class_svm` | Slow | Small datasets |
| `local_outlier_factor` | Medium | Clustered anomalies |

### Advanced

| Backend | Speed | Best For |
|---------|-------|----------|
| `knn` | Medium | Distance-based anomalies |
| `mcd` | Medium | Gaussian data |
| `cblof` | Medium | Clustered data |
| `suod` | Slow | Most robust, ensemble |
| `loda` | Fast | Streaming data |
| `autoencoder` | Slow | Complex patterns |

## API Endpoints

### Batch Detection

```bash
# Fit a model
curl -X POST http://localhost:8000/v1/ml/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{"data": [[1,2], [1,2.1], [10,20]], "backend": "ecod"}'

# Score new data
curl -X POST http://localhost:8000/v1/ml/anomaly/score \
  -H "Content-Type: application/json" \
  -d '{"data": [[1,2], [100,200]], "backend": "ecod"}'

# One-shot detect (fit + score)
curl -X POST http://localhost:8000/v1/ml/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{"data": [[1,2], [1,2.1], [100,200]], "backend": "ecod"}'
```

### Streaming Detection

```bash
# Stream a data point (creates detector on first call)
curl -X POST http://localhost:8000/v1/ml/anomaly/stream \
  -H "Content-Type: application/json" \
  -d '{"model_id": "my-sensor", "data": {"temp": 45.0, "vibration": 0.5}}'

# List active detectors
curl http://localhost:8000/v1/ml/anomaly/stream/detectors

# Get detector stats
curl http://localhost:8000/v1/ml/anomaly/stream/my-sensor

# Reset a detector
curl -X POST http://localhost:8000/v1/ml/anomaly/stream/my-sensor/reset
```

### Model Management

```bash
# List saved models
curl http://localhost:8000/v1/ml/anomaly/models

# Load a saved model
curl -X POST http://localhost:8000/v1/ml/anomaly/load \
  -H "Content-Type: application/json" \
  -d '{"model": "fraud-detector", "backend": "ecod"}'

# Get model info
curl http://localhost:8000/v1/ml/anomaly/models/fraud-detector_ecod.joblib
```

## Architecture

```
JSON/Dict Input
      │
      ▼
┌─────────────────────────────────────┐
│  Polars Buffer (Data Substrate)     │
│  - Sliding window                   │
│  - Rolling features (mean, std)     │
│  - Lag features                     │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  PyOD Backend (ML Utensils)         │
│  - 12+ algorithms                   │
│  - Fit / Score / Detect             │
│  - Auto-save models                 │
└─────────────────────────────────────┘
      │
      ▼
Anomaly Scores + Labels
```

## Streaming Detection (Tick-Tock Pattern)

For real-time monitoring, use the `StreamingAnomalyDetector`:

```python
from models.streaming_anomaly import StreamingAnomalyDetector

detector = StreamingAnomalyDetector(
    model_id="sensor-monitor",
    backend="ecod",
    min_samples=50,        # Cold start threshold
    retrain_interval=100,  # Auto-retrain frequency
    window_size=1000,      # Sliding window size
    rolling_windows=[5, 10],  # Rolling feature windows
)

# Stream data points
for reading in sensor_readings:
    result = await detector.process({"temp": reading.temp})
    if result.is_anomaly:
        alert(f"Anomaly detected: {result.score}")
```

The streaming detector:
1. **Cold Start**: Collects `min_samples` before first model training
2. **Tick**: Fast inference on current model
3. **Tock**: Background retraining every `retrain_interval` samples
4. **Auto-Rolling**: Maintains sliding window, drops old data

## Choosing a Backend

```
                    Speed
                      │
      HBOS ──────────►├────────── Very Fast
                      │
      ECOD ──────────►├────────── Fast
      COPOD ─────────►│
      LODA ──────────►│
      Isolation Forest►│
                      │
      KNN ───────────►├────────── Medium
      MCD ───────────►│
      CBLOF ─────────►│
      LOF ───────────►│
                      │
      One-Class SVM ─►├────────── Slow
      SUOD ──────────►│
      AutoEncoder ───►│
                      │
                      ▼
                 Complexity/Accuracy
```

**Recommendation**: Start with `ecod`. It's fast, parameter-free, and works well for most use cases.
