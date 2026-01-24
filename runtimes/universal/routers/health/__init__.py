"""Health router for health check and models list endpoints."""

from .router import router, set_device_info_getter, set_models_cache

__all__ = ["router", "set_models_cache", "set_device_info_getter"]
