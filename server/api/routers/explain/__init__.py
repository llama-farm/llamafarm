"""
Explain Router Module - Proxy endpoints to Universal Runtime's SHAP explainability.

Provides access to:
- SHAP explanations for model predictions
- Global feature importance computation
"""

from .router import router

__all__ = ["router"]
