"""
ADTK Router Module - Proxy endpoints to Universal Runtime's ADTK time series anomaly detection.

Provides access to:
- Specialized time series anomaly detectors (Level Shift, Seasonal, Persist, etc.)
- Unsupervised detection for unlabeled time series data
"""

from .router import router

__all__ = ["router"]
