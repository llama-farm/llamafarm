"""
Utility modules for transformers runtime.
"""

from .device import get_device_info, get_gguf_gpu_layers, get_optimal_device
from .feature_encoder import (
    ENCODER_REGISTRY,
    FeatureEncoder,
    FeatureSchema,
    register_encoder,
)
from .file_utils import save_image_with_metadata
from .training_executor import (
    get_training_executor,
    get_training_executor_stats,
    shutdown_training_executor,
)

__all__ = [
    "get_optimal_device",
    "get_device_info",
    "get_gguf_gpu_layers",
    "save_image_with_metadata",
    "FeatureEncoder",
    "FeatureSchema",
    "register_encoder",
    "ENCODER_REGISTRY",
    "get_training_executor",
    "get_training_executor_stats",
    "shutdown_training_executor",
]
