# ML Examples

Machine learning integration examples for LlamaFarm.

## Timeseries Forecasting

**02_timeseries_chronos.py** - Demonstrates time series forecasting using the Chronos model:
- Load and prepare time series data
- Train a Chronos-based forecasting model
- Generate predictions for future time steps
- Visualize results

Supports multiple backends including:
- Chronos (Amazon's foundation model for time series)
- ARIMA (AutoRegressive Integrated Moving Average)
- Exponential Smoothing
- Theta method

## Running Examples

```bash
cd examples/ml
uv run python 02_timeseries_chronos.py
```

## Requirements

Timeseries examples require the `darts` package:
```bash
uv pip install darts
```
