"""Timeseries forecasting router for fit, predict, load, and model management."""

from .router import (
    get_timeseries_loader,
    router,
    set_models_dir,
    set_state,
    set_timeseries_loader,
)

__all__ = [
    "router",
    "set_timeseries_loader",
    "get_timeseries_loader",
    "set_models_dir",
    "set_state",
]
