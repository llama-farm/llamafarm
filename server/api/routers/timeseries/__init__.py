"""
Timeseries Router Module - Proxy endpoints to Universal Runtime's time-series forecasting.

Provides access to:
- Classical forecasting (ARIMA, ExponentialSmoothing, Theta)
- Zero-shot forecasting (Chronos, Chronos-Bolt)
- Model management (list, load, delete)
"""

from .router import router as timeseries_router

__all__ = ["timeseries_router"]
