"""ADTK (time-series anomaly detection) router."""

from routers.adtk.router import router, set_adtk_loader, set_adtk_state

__all__ = ["router", "set_adtk_loader", "set_adtk_state"]
