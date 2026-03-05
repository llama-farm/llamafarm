"""SHAP Explainability router."""

from routers.explain.router import router, set_explain_state, set_model_getter

__all__ = ["router", "set_explain_state", "set_model_getter"]
