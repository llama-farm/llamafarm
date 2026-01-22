"""Anomaly detection router for fit, score, detect, load, and model management."""

from .router import router, set_anomaly_loader, set_models_dir, set_state

__all__ = ["router", "set_anomaly_loader", "set_models_dir", "set_state"]
