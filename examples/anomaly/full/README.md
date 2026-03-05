# Full Anomaly Detection Demo

This demo showcases LlamaFarm's complete anomaly detection capabilities:

1. **Training data generation** - Create realistic sensor data
2. **Model training** - Train with different backends
3. **Streaming detection** - Real-time anomaly detection with auto-retraining
4. **Polars feature engineering** - Rolling features and lags
5. **Anomaly injection** - Demonstrate detection of injected anomalies

## Prerequisites

Start the LlamaFarm servers:

```bash
# From the llamafarm repo root
nx start universal-runtime &
sleep 5
nx start server &
sleep 5
```

## Running the Demo

### Option 1: Run Everything

```bash
./run_full_demo.sh
```

This will:
1. Generate training data
2. Train a model
3. Run streaming detection with anomaly injection
4. Display results

### Option 2: Run Individual Scripts

```bash
# Step 1: Generate training data
python 01_generate_training_data.py

# Step 2: Train the model
python 02_train_model.py

# Step 3: Run streaming detection
python 03_streaming_detection.py

# Step 4: (Optional) Use Polars buffers directly
python 04_polars_features.py
```

## What Each Script Does

### 01_generate_training_data.py

Generates realistic factory sensor data:
- Temperature: 72°F ± 2°F
- Humidity: 45% ± 3%
- Pressure: 1013 hPa ± 5 hPa
- Motor RPM: 3000 RPM ± 50 RPM

Saves to `training_data.json`.

### 02_train_model.py

Trains anomaly detection models:
- **ECOD** (default) - Fast, parameter-free
- **Isolation Forest** - Robust, general purpose
- **HBOS** - Fastest, histogram-based

Demonstrates:
- Model training API
- Model saving/loading
- Listing saved models

### 03_streaming_detection.py

Runs real-time anomaly detection:
- Cold start handling
- Auto-retraining
- Anomaly injection every 50 samples
- Score interpretation
- Detector statistics

### 04_polars_features.py

Demonstrates Polars buffer API:
- Creating buffers
- Appending data
- Computing rolling features
- Getting buffer statistics

## Backend Selection Guide

| Backend | Speed | Use Case |
|---------|-------|----------|
| `ecod` | ⚡⚡⚡⚡ | Default, streaming, parameter-free |
| `hbos` | ⚡⚡⚡⚡⚡ | Ultra-fast, high throughput |
| `isolation_forest` | ⚡⚡⚡ | Complex patterns, multi-variate |
| `loda` | ⚡⚡⚡⚡ | Memory-constrained, streaming |
| `autoencoder` | ⚡⚡ | Complex non-linear patterns |

## Understanding the Output

### During Cold Start
```
⏳ Collecting... 35 samples until ready
⏳ Collecting... 25 samples until ready
⏳ Collecting... 15 samples until ready
```

### Normal Operation
```
✅ Sample 100: temp=72.3, score=0.28
✅ Sample 101: temp=71.8, score=0.31
```

### Anomaly Detected
```
💉 Injecting anomaly: temp=150.0
🚨 ANOMALY at sample 150: score=0.89, is_anomaly=True
```

### Model Retraining
```
🔄 Model retrained: version 2 (samples: 200)
```

## Customizing the Demo

Edit the configuration at the top of each script:

```python
# Streaming configuration
DETECTOR_CONFIG = {
    "model": "factory-demo",
    "backend": "ecod",          # Try: isolation_forest, hbos, loda
    "min_samples": 50,          # Samples before first training
    "retrain_interval": 100,    # Retrain frequency
    "window_size": 1000,        # History to keep
    "threshold": 0.5,           # Anomaly threshold (0-1)
    "contamination": 0.05,      # Expected anomaly rate
}
```

## API Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/ml/anomaly/fit` | Train model on batch data |
| `POST /v1/ml/anomaly/save` | Save model to disk |
| `POST /v1/ml/anomaly/load` | Load model from disk |
| `GET /v1/ml/anomaly/models` | List saved models |
| `POST /v1/ml/anomaly/stream` | Process streaming data |
| `GET /v1/ml/anomaly/stream/detectors` | List active detectors |
| `POST /v1/ml/polars/buffers` | Create data buffer |
| `POST /v1/ml/polars/append` | Append to buffer |
| `POST /v1/ml/polars/features` | Compute rolling features |

## Troubleshooting

### "Connection refused"
Start the servers:
```bash
nx start universal-runtime &
nx start server &
```

### "Model not found"
Run the training script first:
```bash
python 02_train_model.py
```

### "Buffer not found"
Create the buffer first:
```bash
python 04_polars_features.py
```
