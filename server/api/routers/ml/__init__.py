"""
ML Router Module - Proxy endpoints to Universal Runtime's specialized ML capabilities.

Provides access to:
- OCR (text extraction from images/PDFs)
- Document Extraction (structured data from forms/invoices)
- Custom Text Classification (SetFit few-shot learning)
- Anomaly Detection (train and detect anomalies)
"""

from .router import router

__all__ = ["router"]
