"""
Drift Router Module - Proxy endpoints to Universal Runtime's drift detection.

Provides access to data drift monitoring:
- KS test: Univariate numeric drift
- MMD: Multivariate drift
- Chi-squared: Categorical drift
"""

from .router import router

__all__ = ["router"]
