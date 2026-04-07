"""Re-export from llamafarm_common — single source of truth."""
from llamafarm_common.device import (
    get_device_info,
    get_gguf_gpu_layers,
    get_optimal_device,
    is_torch_available,
)

__all__ = ["get_optimal_device", "get_device_info", "is_torch_available", "get_gguf_gpu_layers"]
