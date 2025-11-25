"""
Utility modules for transformers runtime.
"""

# Lazy import all functions to avoid requiring dependencies (torch, PIL) at import time
# This allows model_format to be imported without triggering device.py or file_utils.py imports
# which are only needed in the universal-runtime venv, not the server venv
def __getattr__(name: str):
    if name == "get_optimal_device":
        from .device import get_optimal_device
        return get_optimal_device
    elif name == "get_device_info":
        from .device import get_device_info
        return get_device_info
    elif name == "save_image_with_metadata":
        from .file_utils import save_image_with_metadata
        return save_image_with_metadata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["get_optimal_device", "get_device_info", "save_image_with_metadata"]
