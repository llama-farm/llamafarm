"""
CatBoost Router Module - Proxy endpoints to Universal Runtime's CatBoost ML.

Provides access to:
- CatBoost classification and regression
- Native categorical feature support
- Incremental learning
- GPU acceleration (when available)
"""

from .router import router as catboost_router

__all__ = ["catboost_router"]
