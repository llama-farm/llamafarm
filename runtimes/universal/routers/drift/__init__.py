"""Drift detection router using Alibi Detect."""

from routers.drift.router import router, set_drift_loader, set_drift_state

__all__ = ["router", "set_drift_loader", "set_drift_state"]
