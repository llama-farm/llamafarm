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
from .polars_buffer import BufferStats, PolarsBuffer
from .rolling_features import (
    RollingFeatureConfig,
    compute_anomaly_features,
    compute_features,
    get_feature_names,
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
    # Polars buffer and rolling features
    "PolarsBuffer",
    "BufferStats",
    "RollingFeatureConfig",
    "compute_features",
    "compute_anomaly_features",
    "get_feature_names",
]
