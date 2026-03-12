"""
LlamaFarm Common Utilities

Shared utilities used across all LlamaFarm Python services (server, rag, runtimes).
"""

__version__ = "0.1.0"

from .model_utils import (
    GGUF_QUANTIZATION_PREFERENCE_ORDER,
    get_gguf_file_path,
    get_mmproj_file_path,
    list_gguf_files,
    parse_model_with_quantization,
    parse_quantization_from_filename,
    select_gguf_file,
    select_gguf_file_with_logging,
)

# Submodules also importable as llamafarm_common.safe_home, etc.
# Kept as submodule imports to avoid adding their deps to the top-level namespace.
# Usage:
#   from llamafarm_common.safe_home import safe_home, get_data_dir
#   from llamafarm_common.device import get_optimal_device, get_device_info
#   from llamafarm_common.model_cache import ModelCache
#   from llamafarm_common.model_format import detect_model_format

__all__ = [
    "GGUF_QUANTIZATION_PREFERENCE_ORDER",
    "get_gguf_file_path",
    "get_mmproj_file_path",
    "list_gguf_files",
    "parse_model_with_quantization",
    "parse_quantization_from_filename",
    "select_gguf_file",
    "select_gguf_file_with_logging",
]
