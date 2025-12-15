"""
ML Router Module - Proxy endpoints to Universal Runtime's ML capabilities.

Provides access to:
- Custom Text Classification (SetFit few-shot learning)
- Anomaly Detection (train and detect anomalies)

Note: OCR and Document extraction have moved to the vision router (/v1/vision/*).
"""

from .router import router

__all__ = ["router"]
