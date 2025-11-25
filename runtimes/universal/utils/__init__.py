"""
Utility modules for transformers runtime.
"""

from .file_utils import save_image_with_metadata

# Lazy import device functions to avoid requiring torch at import time
# This allows model_format to be imported without triggering device.py imports
def __getattr__(name: str):
    if name in ("get_optimal_device", "get_device_info"):
        from .device import get_optimal_device, get_device_info
        return get_optimal_device if name == "get_optimal_device" else get_device_info
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["get_optimal_device", "get_device_info", "save_image_with_metadata"]
